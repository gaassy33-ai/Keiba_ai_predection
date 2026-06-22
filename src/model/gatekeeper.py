"""
gatekeeper.py
=============
Two-Brain System の「市場側の脳」。
LTR（市場情報なし・純粋な能力評価）が選んだ軸馬候補に対し、
「3着以内に来る安全な軸か」を市場情報（オッズ）込みで判定する
軽量な二値分類モデル。

【LTRとの役割分担】
  LTR（src/model/trainer.py LTRTrainer）
    - 市場情報（odds_log, popularity_rank_norm）を除外
    - 「市場が見落としているアルファ」を見つけるための純粋な能力評価
  Gatekeeper（本ファイル）
    - 市場情報を含む全特徴量を使用
    - 「LTRが推す馬が、実際に馬券内に来る信頼性があるか」のリスク管理
    - ターゲット: is_top3（1着〜3着なら1、それ以外0）

学習データは LTRTrainer と同じ training_df（FeatureEngineer.build_training_dataset()
の出力 + finish_pos_num）を再利用する（retrain_pipeline.py から呼び出し）。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold

from src.features.engineer import FeatureEngineer


class GatekeeperTrainer:
    """軸馬適性判定（is_top3 二値分類）モデル。市場情報を含む全特徴量を使用する。"""

    LGBM_PARAMS = {
        "objective":        "binary",
        "metric":           "auc",
        "learning_rate":    0.03,
        "num_leaves":       31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbosity":        -1,
    }
    NUM_BOOST_ROUND       = 1000
    EARLY_STOPPING_ROUNDS = 50
    N_CV_FOLDS            = 5

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        # Gatekeeper は LTR と逆に、市場情報（odds_log, popularity_rank_norm）を
        # 「除外しない」。FEATURE_COLUMNS をそのまま使用する。
        self.feature_columns: list[str] = FeatureEngineer.FEATURE_COLUMNS
        self.calibrator: IsotonicRegression | None = None
        self.oof_auc: float = 0.0
        self.oof_logloss: float = 0.0
        self.oof_brier: float = 0.0

    @staticmethod
    def make_labels(df: pd.DataFrame, pos_col: str = "finish_pos_num") -> np.ndarray:
        """is_top3 ラベル（1〜3着=1, それ以外=0）。"""
        pos = pd.to_numeric(df[pos_col], errors="coerce")
        return (pos <= 3).astype(np.int32).values

    def fit(self, df: pd.DataFrame, group_col: str = "race_id") -> None:
        df_s = df.sort_values(group_col).reset_index(drop=True)

        X = (
            df_s[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        y = self.make_labels(df_s)
        race_ids = df_s[group_col].values

        logger.info(
            f"Gatekeeper 学習データ: {len(df_s):,} サンプル  "
            f"is_top3=1: {y.sum():,}件 ({y.mean()*100:.1f}%)"
        )

        gkf = GroupKFold(n_splits=self.N_CV_FOLDS)
        oof_pred = np.zeros(len(df_s), dtype=np.float64)
        fold_auc: list[float] = []
        models: list[lgb.Booster] = []

        for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X, y, race_ids)):
            X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]

            train_ds = lgb.Dataset(X_tr, label=y_tr)
            val_ds   = lgb.Dataset(X_vl, label=y_vl, reference=train_ds)

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
            vl_pred = booster.predict(X_vl)
            oof_pred[vl_idx] = vl_pred
            auc = roc_auc_score(y_vl, vl_pred)
            fold_auc.append(auc)
            models.append(booster)
            logger.info(f"  Fold {fold+1}/{self.N_CV_FOLDS} | best_iter={booster.best_iteration:4d} | AUC={auc:.4f}")

        self.oof_auc = float(np.mean(fold_auc))
        self.oof_logloss = float(log_loss(y, oof_pred))
        self.oof_brier = float(brier_score_loss(y, oof_pred))
        logger.info(
            f"OOF AUC = {self.oof_auc:.4f}  logloss={self.oof_logloss:.4f}  "
            f"brier={self.oof_brier:.4f}"
        )

        # ── Isotonic Calibration（OOF予測の事後補正）────────────────
        # 二値分類・サンプル数が多いため Isotonic を採用（LTRのPair Calibratorとは異なり
        # 過学習リスクが低い）。
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(oof_pred, y)
        calibrated_oof = self.calibrator.predict(oof_pred)
        logger.info(
            f"キャリブレーション補正: 補正前平均={oof_pred.mean():.4f}  "
            f"補正後平均={calibrated_oof.mean():.4f}  実際の的中率={y.mean():.4f}"
        )

        # ── 全データで最終モデルを学習 ────────────────────────────────
        best_rounds = max(m.best_iteration for m in models)
        full_ds = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.LGBM_PARAMS, full_ds, num_boost_round=best_rounds)
        logger.info(f"Final Gatekeeper model: {best_rounds} rounds")
        self._log_feature_importance()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P_axis_safe（3着以内に入る確率、キャリブレーション済み）を返す。"""
        assert self.model is not None, "fit() を先に実行してください"
        X_num = (
            X[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        raw = self.model.predict(X_num)
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return raw

    def _log_feature_importance(self, top_n: int = 20) -> None:
        if self.model is None:
            return
        gain  = self.model.feature_importance("gain")
        names = self.model.feature_name()
        total = gain.sum() or 1.0
        pairs = sorted(zip(names, gain), key=lambda x: -x[1])
        logger.info(f"=== Gatekeeper Feature Importance (top {top_n}) ===")
        for feat, g in pairs[:top_n]:
            logger.info(f"  {feat:<30} gain={g:>8,}  ({g/total*100:>5.1f}%)")

    def save_feature_importance(self, path: Path) -> None:
        if self.model is None:
            return
        gain  = self.model.feature_importance("gain")
        split = self.model.feature_importance("split")
        names = self.model.feature_name()
        total = gain.sum() or 1.0
        pairs = sorted(zip(names, gain.tolist(), split.tolist()), key=lambda x: -x[1])
        payload = {
            "gatekeeper_model": {
                "num_trees": self.model.num_trees(),
                "oof_auc": round(self.oof_auc, 4),
                "oof_logloss": round(self.oof_logloss, 4),
                "oof_brier": round(self.oof_brier, 4),
                "importance": [
                    {"feature": f, "gain": int(g),
                     "gain_pct": round(g / total * 100, 2), "split": int(s)}
                    for f, g, s in pairs
                ],
            }
        }
        import json
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info(f"Gatekeeper feature importance saved: {path}")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":           self.model,
            "feature_columns": self.feature_columns,
            "calibrator":      self.calibrator,
            "oof_auc":         self.oof_auc,
            "oof_logloss":     self.oof_logloss,
            "oof_brier":       self.oof_brier,
        }, path)
        logger.info(
            f"Gatekeeper model saved: {path}  "
            f"(AUC={self.oof_auc:.4f}  logloss={self.oof_logloss:.4f}  brier={self.oof_brier:.4f})"
        )
        imp_path = path.parent / "gatekeeper_feature_importance.json"
        self.save_feature_importance(imp_path)

    @classmethod
    def load(cls, path: Path) -> "GatekeeperTrainer":
        raw = joblib.load(path)
        instance = cls()
        instance.model           = raw["model"]
        instance.feature_columns = raw.get("feature_columns", FeatureEngineer.FEATURE_COLUMNS)
        instance.calibrator      = raw.get("calibrator")
        instance.oof_auc         = raw.get("oof_auc", 0.0)
        instance.oof_logloss     = raw.get("oof_logloss", 0.0)
        instance.oof_brier       = raw.get("oof_brier", 0.0)
        logger.info(
            f"Gatekeeper model loaded: {path}  "
            f"(AUC={instance.oof_auc:.4f}  "
            f"trees={instance.model.num_trees() if instance.model else 0})"
        )
        return instance
