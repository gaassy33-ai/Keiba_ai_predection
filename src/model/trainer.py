"""
LightGBM 学習スクリプト。

学習データ（過去成績の特徴量DF）を受け取り、
1着入線確率を予測するモデルを訓練・保存する。
"""

from __future__ import annotations

import json
import joblib
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
        """
        X = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[label_col].values
        groups = df[group_col].values

        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(X))
        models: list[lgb.Booster] = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr)
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

        # 最終モデルは全データで学習
        full_dataset = lgb.Dataset(X, label=y)
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
            self._fit_place_model(df, group_col)

    def _fit_place_model(self, df: pd.DataFrame, group_col: str = "race_id") -> None:
        """is_placed（3着以内）を目的変数としたモデルを学習する。"""
        X = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df["is_placed"].values
        groups = df[group_col].values

        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(X))
        models: list[lgb.Booster] = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            train_set = lgb.Dataset(X_tr, label=y_tr)
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

        full_dataset = lgb.Dataset(X, label=y)
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
