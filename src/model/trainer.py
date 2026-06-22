"""
LightGBM 学習スクリプト。

学習データ（過去成績の特徴量DF）を受け取り、
1着入線確率を予測するモデルを訓練・保存する。
"""

from __future__ import annotations

import json
import joblib
from itertools import groupby as _itertools_groupby
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

from config.settings import settings
from src.features.engineer import FeatureEngineer


# ──────────────────────────────────────────────────────────────────────────────
# LTR ユーティリティ（LTRTrainer から使用）
# ──────────────────────────────────────────────────────────────────────────────

def _ltr_group_sizes(race_ids: np.ndarray) -> list[int]:
    """連続する race_id の連長を返す（LightGBM group 引数用）。
    df が race_id でソート済みであることを前提とする。"""
    return [sum(1 for _ in g) for _, g in _itertools_groupby(race_ids)]


def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """単一レースの NDCG@k を計算する。"""
    order = np.argsort(-scores)
    top_labels = labels[order[:k]]
    dcg  = sum((2.0 ** lbl - 1.0) / np.log2(i + 2.0) for i, lbl in enumerate(top_labels))
    ideal = np.sort(labels)[::-1][:k]
    idcg = sum((2.0 ** lbl - 1.0) / np.log2(i + 2.0) for i, lbl in enumerate(ideal))
    return dcg / idcg if idcg > 0.0 else 0.0


def _mean_ndcg_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    race_ids: np.ndarray,
    k: int = 3,
) -> float:
    """全レースにわたる平均 NDCG@k を計算する。"""
    races_seen: list = []
    seen_set: set = set()
    for r in race_ids:
        if r not in seen_set:
            races_seen.append(r)
            seen_set.add(r)
    ndcgs = []
    for r in races_seen:
        mask = race_ids == r
        ndcgs.append(_ndcg_at_k(scores[mask], labels[mask], k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


class ModelTrainer:
    """
    LightGBM の学習・評価・保存を担当する。

    Usage
    -----
    >>> trainer = ModelTrainer()
    >>> trainer.fit(feature_df, label_series, groups=race_id_series)
    >>> trainer.save()
    """

    # ----------------------------------------------------------------
    # 改善②: ハイパーパラメータ調整
    #   - learning_rate: 0.05 → 0.02（細かいステップで収束・木数増加）
    #   - num_leaves: 63 → 127（全会場データ増加に対応した表現力向上）
    #   - EARLY_STOPPING_ROUNDS: 50 → 100（早期終了を緩和して十分探索）
    # 改善⑨ 撤回 (2026-04): odds_log/popularity_rank_norm を FEATURE_COLUMNS に復帰
    #   2026年4月実績で model が actual の 1.6x を過信していることが判明。
    #   市場情報をモデルに組み込み + Platt scaling で確率を較正する方針に変更。
    # ----------------------------------------------------------------
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,   # odds_log過学習抑制のため強化（0.1→1.0）
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 100

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        self.place_model: lgb.Booster | None = None   # is_top3 (3着以内) モデル
        self.feature_columns = FeatureEngineer.FEATURE_COLUMNS
        # Platt scaling: OOF 予測から較正した確率変換器
        self.calibrator: LogisticRegression | None = None       # 勝利モデル用（base rate ≈ 8%）
        self.place_calibrator: LogisticRegression | None = None  # 複勝モデル用（base rate ≈ 30-35%）

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = "is_win",
        group_col: str = "race_id",
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """
        GroupKFold（race_id でグループ分け）で CV しながら学習。

        Parameters
        ----------
        df : pd.DataFrame
            特徴量 + ラベル + race_id を含む DataFrame
        label_col : str
            目的変数のカラム名
        group_col : str
            グループ分けに使うカラム名（data leakage 防止）
        sample_weight : np.ndarray | None
            各サンプルの学習重み。None の場合は全サンプル均等重み。
            時系列重み付け（最近のデータを重視）に使用。
        """
        X = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[label_col].values
        groups = df[group_col].values
        weights = sample_weight if sample_weight is not None else np.ones(len(y))

        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(X))
        models: list[lgb.Booster] = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = weights[train_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

            booster = lgb.train(
                self.LGBM_PARAMS,
                train_set,
                num_boost_round=self.NUM_BOOST_ROUND,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(self.EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )
            oof_preds[val_idx] = booster.predict(X_val)
            models.append(booster)
            logger.info(
                f"Fold {fold+1} | best_iter={booster.best_iteration} "
                f"logloss={log_loss(y_val, oof_preds[val_idx]):.4f} "
                f"auc={roc_auc_score(y_val, oof_preds[val_idx]):.4f}"
            )

        oof_logloss = log_loss(y, oof_preds)
        oof_auc     = roc_auc_score(y, oof_preds)
        logger.info(f"OOF logloss={oof_logloss:.4f}  auc={oof_auc:.4f}")

        # ── Platt Scaling（確率較正） ──────────────────────────────────
        # OOF 予測（未知データへの近似）をもとにロジスティック較正器を学習する。
        # LightGBM の生確率は過信気味（実際の的中率より高い）なため、
        # この1変量ロジスティック回帰で補正する。
        calib = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000)
        calib.fit(oof_preds.reshape(-1, 1), y)
        calib_preds = calib.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
        calib_mean_pos = calib_preds[y == 1].mean()
        calib_mean_neg = calib_preds[y == 0].mean()
        logger.info(
            f"Platt scaling 較正完了: 1着馬平均確率 {calib_mean_pos:.3f}  "
            f"非1着平均確率 {calib_mean_neg:.3f}  "
            f"logloss after calib={log_loss(y, calib_preds):.4f}"
        )
        self.calibrator = calib

        # 最終モデルは全データで学習（サンプル重みも反映）
        full_dataset = lgb.Dataset(X, label=y, weight=weights)
        best_rounds = max(m.best_iteration for m in models)
        self.model = lgb.train(
            self.LGBM_PARAMS,
            full_dataset,
            num_boost_round=best_rounds,
        )
        logger.info(f"Final model trained with {best_rounds} rounds.")

        # 特徴量重要度をログ出力（改善③）
        self._log_feature_importance(self.model, label="win_model")

        # is_placed モデル（3着以内確率）を追加学習
        if "is_placed" in df.columns:
            self._fit_place_model(df, group_col, sample_weight=weights)

    def _fit_place_model(
        self,
        df: pd.DataFrame,
        group_col: str = "race_id",
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """is_placed（3着以内）を目的変数としたモデルを学習する。"""
        X = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df["is_placed"].values
        groups = df[group_col].values
        weights = sample_weight if sample_weight is not None else np.ones(len(y))

        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(X))
        models: list[lgb.Booster] = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = weights[train_idx]
            train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            booster = lgb.train(
                self.LGBM_PARAMS,
                train_set,
                num_boost_round=self.NUM_BOOST_ROUND,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(self.EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(200),
                ],
            )
            oof_preds[val_idx] = booster.predict(X_val)
            models.append(booster)

        oof_logloss = log_loss(y, oof_preds)
        oof_auc     = roc_auc_score(y, oof_preds)
        logger.info(
            f"Place model OOF logloss={oof_logloss:.4f} "
            f"auc={oof_auc:.4f}"
        )

        # ── Place Platt Scaling（複勝確率較正）────────────────────────────
        # 勝利モデルの calibrator とは別に、複勝モデル専用の較正器を学習する。
        # 複勝の base rate は約 30-35%（勝利の約 8% とは大きく異なる）ため
        # 勝利用 calibrator を流用すると大きな較正エラーが発生する。
        place_calib = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000)
        place_calib.fit(oof_preds.reshape(-1, 1), y)  # y=is_placed
        place_calib_preds = place_calib.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
        logger.info(
            f"Place Platt scaling 較正完了: 3着以内馬平均確率 {place_calib_preds[y==1].mean():.3f}  "
            f"非3着以内平均確率 {place_calib_preds[y==0].mean():.3f}  "
            f"logloss after calib={log_loss(y, place_calib_preds):.4f}"
        )
        self.place_calibrator = place_calib

        full_dataset = lgb.Dataset(X, label=y, weight=weights)
        best_rounds = max(m.best_iteration for m in models)
        self.place_model = lgb.train(
            self.LGBM_PARAMS,
            full_dataset,
            num_boost_round=best_rounds,
        )
        logger.info(f"Place model trained with {best_rounds} rounds.")
        self._log_feature_importance(self.place_model, label="place_model")

    # ------------------------------------------------------------------
    # 改善③: 特徴量重要度の可視化・保存
    # ------------------------------------------------------------------

    @staticmethod
    def _log_feature_importance(model: lgb.Booster, label: str = "model") -> None:
        """特徴量重要度をログに出力する。"""
        features = model.feature_name()
        gain = model.feature_importance("gain")
        split = model.feature_importance("split")
        total_gain = gain.sum() or 1
        pairs = sorted(zip(features, gain, split), key=lambda x: -x[1])
        logger.info(f"--- 特徴量重要度 [{label}] ---")
        for f, g, s in pairs:
            logger.info(f"  {f:<24} gain={g:>8,} ({g/total_gain*100:>5.1f}%)  split={s:>6,}")

    def save_feature_importance(self, path: Path | None = None) -> None:
        """
        特徴量重要度を JSON で保存する。
        GitHub Actions から commit して docs/ に置くことで
        Python バージョン非依存で参照できる。
        """
        if self.model is None:
            return
        path = path or (settings.model_path.parent / "feature_importance.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _importance_dict(model: lgb.Booster) -> list[dict]:
            features = model.feature_name()
            gain = model.feature_importance("gain")
            split = model.feature_importance("split")
            total_gain = gain.sum() or 1
            pairs = sorted(zip(features, gain.tolist(), split.tolist()), key=lambda x: -x[1])
            return [
                {
                    "feature": f,
                    "gain": int(g),
                    "gain_pct": round(g / total_gain * 100, 2),
                    "split": int(s),
                }
                for f, g, s in pairs
            ]

        payload = {
            "win_model": {
                "num_trees": self.model.num_trees(),
                "importance": _importance_dict(self.model),
            }
        }
        if self.place_model is not None:
            payload["place_model"] = {
                "num_trees": self.place_model.num_trees(),
                "importance": _importance_dict(self.place_model),
            }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info(f"Feature importance saved to {path}")

    # ------------------------------------------------------------------
    # 保存 / 読み込み
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None, org: str = "jra") -> None:
        if path is None:
            path = settings.nar_model_path if org == "nar" else settings.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "place_model": self.place_model,
            "calibrator": self.calibrator,         # 勝利モデル Platt scaling 較正器
            "place_calibrator": self.place_calibrator,  # 複勝モデル専用 Platt scaling 較正器
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")

        # 特徴量重要度を JSON でも保存（改善③）
        importance_name = f"{org}_feature_importance.json" if org != "jra" else "feature_importance.json"
        self.save_feature_importance(path.parent / importance_name)

    @classmethod
    def load(cls, path: Path | None = None, org: str = "jra") -> "ModelTrainer":
        if path is None:
            path = settings.nar_model_path if org == "nar" else settings.model_path
        instance = cls()
        raw = joblib.load(path)
        if isinstance(raw, dict):
            instance.model = raw.get("model")
            instance.place_model = raw.get("place_model")
            instance.calibrator = raw.get("calibrator")                 # 旧モデルは None
            instance.place_calibrator = raw.get("place_calibrator")     # 旧モデルは None
        else:
            # 旧フォーマット（Booster 直接保存）に対する後方互換
            instance.model = raw
        calib_info = []
        if instance.calibrator is not None:
            calib_info.append("win_calib=あり")
        if instance.place_calibrator is not None:
            calib_info.append("place_calib=あり")
        calib_str = ", ".join(calib_info) if calib_info else "Platt scaling なし（旧モデル）"
        logger.info(f"Model loaded from {path} (org={org}, {calib_str})")
        return instance


class LTRTrainer:
    """
    LightGBM LambdaRank による Learning-to-Rank モデル。

    【現行バイナリ分類との根本的な違い】
    - バイナリ分類 (ModelTrainer): 各馬を独立に評価 → レース間の相対強度を無視
    - LTR (LTRTrainer)          : レース内の相対的な強さを直接学習 → 同一レース内比較

    【ラベル設計（4段階関連度スコア）】
    - 1着 = 3  (最高関連度)
    - 2着 = 2
    - 3着 = 1
    - 4着以下 = 0

    【最適化指標】
    NDCG@3 → top3 に入る馬の順位精度を最大化 → ワイド的中率に直結

    【確率変換】
    LTR スコア → softmax (Plackett-Luce モデル) → レース内確率
    → Harville 公式に入力して ワイド/3連複 確率を計算
    """

    # ── ハイパーパラメータ ────────────────────────────────────────────
    LGBM_PARAMS: dict = {
        "objective":                   "lambdarank",
        "metric":                      "ndcg",
        "ndcg_eval_at":                [3],
        "lambdarank_truncation_level": 5,
        "learning_rate":               0.05,
        "num_leaves":                  63,
        "max_depth":                   -1,
        "min_child_samples":           20,
        "feature_fraction":            0.8,
        "bagging_fraction":            0.8,
        "bagging_freq":                5,
        "lambda_l1":                   0.1,
        "lambda_l2":                   0.5,
        "verbose":                     -1,
        "n_jobs":                      -1,
        "random_state":                42,
    }

    NUM_BOOST_ROUND     = 1000
    EARLY_STOPPING_ROUNDS = 50
    N_CV_FOLDS          = 5

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        self.feature_columns: list[str] = FeatureEngineer.FEATURE_COLUMNS
        self.oof_ndcg3: float = 0.0    # CV 時の平均 NDCG@3（品質指標）
        self.temperature: float = 1.0  # Temperature Scaling パラメーター（τ*）

    # ── ラベル生成 ────────────────────────────────────────────────────

    @staticmethod
    def make_rank_labels(df: pd.DataFrame, pos_col: str = "finish_pos_num") -> np.ndarray:
        """
        着順 → 関連度スコア (int32: 0/1/2/3) に変換。

        Parameters
        ----------
        df : DataFrame
            finish_pos_num カラムを含む DataFrame
        pos_col : str
            着順（数値）カラム名

        Returns
        -------
        np.ndarray of int32
        """
        pos = pd.to_numeric(df[pos_col], errors="coerce")
        labels = np.zeros(len(df), dtype=np.int32)
        labels[pos == 1] = 3
        labels[pos == 2] = 2
        labels[pos == 3] = 1
        # 4着以下・着外・除外は 0 のまま
        return labels

    # ── 確率変換 ─────────────────────────────────────────────────────

    @staticmethod
    def scores_to_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Plackett-Luce モデル: LTR スコア → レース内確率。

        softmax(scores / temperature) を適用し、レース内で合計 1 になる
        確率ベクトルを返す。Harville 公式への入力として使用。

        Parameters
        ----------
        scores : np.ndarray
            1レース分の LTR raw スコア
        temperature : float
            > 1 → 確率を均等化（不確実性増加）
            < 1 → 確率を集中（最有力馬に集中）
            デフォルト 1.0（Step 3 でチューニング）

        Returns
        -------
        np.ndarray  shape=(n_horses,)  合計 ≈ 1.0
        """
        s = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-9)
        exp_s = np.exp(s - s.max())   # overflow 防止
        return exp_s / exp_s.sum()

    # ── Temperature Scaling 較正 ──────────────────────────────────────

    @staticmethod
    def _calibrate_temperature(
        oof_scores: np.ndarray,
        race_ids:   np.ndarray,
        y_win:      np.ndarray,
    ) -> float:
        """
        OOF スコアで negative log-likelihood を最小化し、最適温度 τ* を返す。

        softmax(scores / τ) の τ を [0.5, 5.0] で探索。
        τ > 1 → 均等化（モデル過信を緩和）
        τ < 1 → 集中（モデルが過小評価している場合）

        Parameters
        ----------
        oof_scores : OOF フォールドで蓄積した LTR raw スコア（全馬）
        race_ids   : 各サンプルの race_id（同一レースが連続している必要はない）
        y_win      : 1着馬=1、それ以外=0 のバイナリラベル

        Returns
        -------
        float
            最適温度 τ*
        """
        from scipy.optimize import minimize_scalar

        unique_races = np.unique(race_ids)

        def neg_ll(tau: float) -> float:
            if tau < 1e-9:
                return 1e9
            loss = 0.0
            for rid in unique_races:
                mask = race_ids == rid
                probs = LTRTrainer.scores_to_probs(oof_scores[mask], temperature=tau)
                loss -= float(np.sum(y_win[mask] * np.log(probs + 1e-9)))
            return loss

        result = minimize_scalar(neg_ll, bounds=(0.5, 5.0), method="bounded")
        return float(result.x)

    # ── 学習 ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, group_col: str = "race_id") -> None:
        """
        LambdaRank モデルを GroupKFold CV で学習し、全データで最終モデルを構築する。

        Parameters
        ----------
        df : DataFrame
            FEATURE_COLUMNS + finish_pos_num + race_id を含む
        group_col : str
            グループ分けカラム（race_id）
        """
        # ── データ前処理 ──────────────────────────────────────────────
        # LightGBM LambdaRank の要件: 同一グループのサンプルが連続している必要あり
        df_s = df.sort_values(group_col).reset_index(drop=True)

        X = (
            df_s[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        y         = self.make_rank_labels(df_s)       # 0/1/2/3
        race_ids  = df_s[group_col].values
        grp_sizes = _ltr_group_sizes(race_ids)        # LightGBM group 配列

        logger.info(
            f"LTR 学習データ: {len(df_s):,} サンプル  "
            f"{len(grp_sizes):,} レース  "
            f"label 分布: 0={( y==0).sum():,} 1={(y==1).sum():,} "
            f"2={(y==2).sum():,} 3={(y==3).sum():,}"
        )

        # ── GroupKFold CV ─────────────────────────────────────────────
        gkf = GroupKFold(n_splits=self.N_CV_FOLDS)
        fold_ndcg3: list[float] = []
        models: list[lgb.Booster] = []
        oof_scores = np.zeros(len(df_s), dtype=np.float64)   # temperature 較正用

        for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X, y, race_ids)):
            X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]
            r_tr, r_vl = race_ids[tr_idx], race_ids[vl_idx]

            grp_tr = _ltr_group_sizes(r_tr)
            grp_vl = _ltr_group_sizes(r_vl)

            train_ds = lgb.Dataset(X_tr, label=y_tr, group=grp_tr)
            val_ds   = lgb.Dataset(X_vl, label=y_vl, group=grp_vl,
                                   reference=train_ds)

            booster = lgb.train(
                self.LGBM_PARAMS,
                train_ds,
                num_boost_round=self.NUM_BOOST_ROUND,
                valid_sets=[val_ds],
                callbacks=[
                    lgb.early_stopping(self.EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(200),
                ],
            )

            vl_scores = booster.predict(X_vl)
            oof_scores[vl_idx] = vl_scores            # OOF スコアを蓄積
            ndcg3 = _mean_ndcg_at_k(vl_scores, y_vl, r_vl, k=3)
            fold_ndcg3.append(ndcg3)
            models.append(booster)
            logger.info(
                f"  Fold {fold+1}/{self.N_CV_FOLDS} | "
                f"best_iter={booster.best_iteration:4d} | "
                f"NDCG@3={ndcg3:.4f}"
            )

        self.oof_ndcg3 = float(np.mean(fold_ndcg3))
        logger.info(
            f"OOF NDCG@3 = {self.oof_ndcg3:.4f} "
            f"± {np.std(fold_ndcg3):.4f}  "
            f"(fold wise: {[round(v,4) for v in fold_ndcg3]})"
        )

        # ── Temperature Scaling Calibration ──────────────────────────
        # OOF スコアで softmax(scores/τ) の τ を最適化し、
        # モデルの過信（確率集中）を緩和する。
        y_win = (y == 3).astype(np.float64)  # 1着馬フラグ（ラベル=3）
        tau_star = self._calibrate_temperature(oof_scores, race_ids, y_win)
        self.temperature = tau_star
        logger.info(
            f"Temperature Scaling 較正完了: τ* = {tau_star:.4f}  "
            f"（τ>1→均等化, τ<1→集中）"
        )

        # ── 全データで最終モデルを学習 ────────────────────────────────
        best_rounds = max(m.best_iteration for m in models)
        full_ds = lgb.Dataset(X, label=y, group=grp_sizes)
        self.model = lgb.train(
            self.LGBM_PARAMS,
            full_ds,
            num_boost_round=best_rounds,
        )
        logger.info(f"Final LTR model: {best_rounds} rounds  (NDCG@3={self.oof_ndcg3:.4f})")
        self._log_feature_importance()

    # ── 推論 ─────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        各馬の LTR raw スコアを返す。
        確率ではないため、レース内確率への変換は scores_to_probs() を使うこと。
        """
        assert self.model is not None, "fit() を先に実行してください"
        X_num = (
            X[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        return self.model.predict(X_num)

    # ── 特徴量重要度 ──────────────────────────────────────────────────

    def _log_feature_importance(self, top_n: int = 20) -> None:
        """特徴量重要度を gain ベースでログ出力する。"""
        if self.model is None:
            return
        gain  = self.model.feature_importance("gain")
        names = self.model.feature_name()
        total = gain.sum() or 1.0
        pairs = sorted(zip(names, gain), key=lambda x: -x[1])
        logger.info(f"=== LTR Feature Importance (top {top_n}) ===")
        for feat, g in pairs[:top_n]:
            logger.info(f"  {feat:<30} gain={g:>8,}  ({g/total*100:>5.1f}%)")

    def save_feature_importance(self, path: Path) -> None:
        """特徴量重要度を JSON で保存する。"""
        if self.model is None:
            return
        gain  = self.model.feature_importance("gain")
        split = self.model.feature_importance("split")
        names = self.model.feature_name()
        total = gain.sum() or 1.0
        pairs = sorted(zip(names, gain.tolist(), split.tolist()), key=lambda x: -x[1])
        payload = {
            "ltr_model": {
                "num_trees": self.model.num_trees(),
                "oof_ndcg3": round(self.oof_ndcg3, 4),
                "importance": [
                    {"feature": f, "gain": int(g),
                     "gain_pct": round(g / total * 100, 2), "split": int(s)}
                    for f, g, s in pairs
                ],
            }
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info(f"LTR feature importance saved: {path}")

    # ── 保存 / 読み込み ──────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """モデルを joblib で保存する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":           self.model,
            "feature_columns": self.feature_columns,
            "oof_ndcg3":       self.oof_ndcg3,
            "temperature":     self.temperature,   # Temperature Scaling τ*
        }
        joblib.dump(payload, path)
        logger.info(
            f"LTR model saved: {path}  "
            f"(NDCG@3={self.oof_ndcg3:.4f}  τ={self.temperature:.4f})"
        )
        # 特徴量重要度 JSON
        imp_path = path.parent / "ltr_feature_importance.json"
        self.save_feature_importance(imp_path)

    @classmethod
    def load(cls, path: Path) -> "LTRTrainer":
        """保存済みモデルを読み込む。"""
        raw = joblib.load(path)
        instance = cls()
        if isinstance(raw, dict):
            instance.model           = raw["model"]
            instance.feature_columns = raw.get("feature_columns",
                                               FeatureEngineer.FEATURE_COLUMNS)
            instance.oof_ndcg3       = raw.get("oof_ndcg3", 0.0)
            instance.temperature     = raw.get("temperature", 1.0)  # 旧モデルは 1.0
        else:
            instance.model = raw
        logger.info(
            f"LTR model loaded: {path}  "
            f"(NDCG@3={instance.oof_ndcg3:.4f}  "
            f"τ={instance.temperature:.4f}  "
            f"trees={instance.model.num_trees() if instance.model else 0})"
        )
        return instance


def main() -> None:
    """CLI エントリーポイント: keiba-train"""
    import argparse

    parser = argparse.ArgumentParser(description="LightGBM 学習スクリプト")
    parser.add_argument("--input", required=True, help="学習用 CSV パス")
    parser.add_argument("--output", default=None, help="モデル保存先（省略時は設定値）")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    trainer = ModelTrainer()
    trainer.fit(df)
    trainer.save(Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
