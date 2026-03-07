"""
LightGBM 学習スクリプト。

学習データ（過去成績の特徴量DF）を受け取り、
1着入線確率を予測するモデルを訓練・保存する。
"""

from __future__ import annotations

import joblib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
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

    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    NUM_BOOST_ROUND = 1000
    EARLY_STOPPING_ROUNDS = 50

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        self.feature_columns = FeatureEngineer.FEATURE_COLUMNS

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
        X = df[self.feature_columns].copy()
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
                f"Fold {fold+1} | logloss={log_loss(y_val, oof_preds[val_idx]):.4f} "
                f"auc={roc_auc_score(y_val, oof_preds[val_idx]):.4f}"
            )

        logger.info(
            f"OOF logloss={log_loss(y, oof_preds):.4f} "
            f"auc={roc_auc_score(y, oof_preds):.4f}"
        )

        # 最終モデルは全データで学習
        full_dataset = lgb.Dataset(X, label=y)
        best_rounds = max(m.best_iteration for m in models)
        self.model = lgb.train(
            self.LGBM_PARAMS,
            full_dataset,
            num_boost_round=best_rounds,
        )
        logger.info(f"Final model trained with {best_rounds} rounds.")

    # ------------------------------------------------------------------
    # 保存 / 読み込み
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        path = path or settings.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelTrainer":
        path = path or settings.model_path
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
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
