"""
利用可能な全データを結合してJRAモデルを再学習するスクリプト。

処理内容:
1. train_results.csv (2024) + test_results.csv の早期データ (2025-03-15 まで) を結合
2. 結合データから特徴量を生成
3. LightGBM モデルを学習・保存

実行例:
    python expand_and_train.py
    nohup .venv/bin/python expand_and_train.py > logs/expand_train.log 2>&1 &
"""

from __future__ import annotations

import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from config.settings import settings

Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/expand_train.log", level="DEBUG", format=_fmt, rotation="20 MB")

# バックテスト開始日（この日以降はテスト期間として使わない）
# 2025年末までを学習データとして利用し、2026年でバックテスト
BACKTEST_START = "2026-01-01"


def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("JRA モデル拡張再学習")
    logger.info(f"  学習期間: 2024年〜2025年末 (バックテスト: 2026年〜)")
    logger.info("=" * 60)

    # ── 既存訓練データ読み込み ────────────────────────────────────
    train_results = pd.read_csv("data/raw/train_results.csv", dtype=str)
    train_meta    = pd.read_csv("data/raw/train_meta.csv",    dtype=str)
    logger.info(f"訓練データ: {len(train_results):,} 行  {train_meta['race_id'].nunique()} レース")

    # ── テストデータから早期分を抽出 ─────────────────────────────
    test_results = pd.read_csv("data/raw/test_results.csv", dtype=str)
    test_meta    = pd.read_csv("data/raw/test_meta.csv",    dtype=str)

    # バックテスト開始日より前のレースを抽出
    test_meta_clean = test_meta.dropna(subset=["race_date"])
    early_meta = test_meta_clean[test_meta_clean["race_date"] < BACKTEST_START].copy()
    early_ids  = set(early_meta["race_id"].tolist())
    early_results = test_results[test_results["race_id"].isin(early_ids)].copy()

    logger.info(
        f"早期テストデータ (< {BACKTEST_START}): "
        f"{len(early_results):,} 行  {len(early_ids)} レース"
    )

    # ── データ結合（重複排除）────────────────────────────────────
    all_ids_in_train = set(train_results["race_id"].tolist())
    early_results_new = early_results[~early_results["race_id"].isin(all_ids_in_train)]
    early_meta_new    = early_meta[~early_meta["race_id"].isin(
        set(train_meta["race_id"].tolist())
    )]

    combined_results = pd.concat([train_results, early_results_new], ignore_index=True)
    combined_meta    = pd.concat([train_meta,    early_meta_new],    ignore_index=True)

    logger.info(
        f"結合後: {len(combined_results):,} 行  "
        f"{combined_meta['race_id'].nunique()} レース"
    )
    years = sorted(combined_results["race_id"].str[:4].unique())
    venues = sorted(combined_results["race_id"].str[4:6].unique())
    logger.info(f"  対象年: {years}")
    logger.info(f"  対象会場: {venues}")

    # ── 特徴量生成 ─────────────────────────────────────────────────
    TRAINING_CACHE = ROOT / "data/processed/train_2024_2025.csv"

    if TRAINING_CACHE.exists():
        logger.info(f"[STEP 1] キャッシュから特徴量を読み込み: {TRAINING_CACHE}")
        training_df = pd.read_csv(TRAINING_CACHE)
        # fe は統計計算のために引き続き必要
        fe = FeatureEngineer(combined_results)
        fe.precompute_aggregations()
        logger.info(f"  読み込み完了: {len(training_df):,} 行")
    else:
        logger.info("[STEP 1] 特徴量生成（per-race ルックバック方式）")
        logger.info(f"  対象レース数: {combined_meta['race_id'].nunique():,}")

        fe = FeatureEngineer(combined_results)
        fe.precompute_aggregations()

        training_df = fe.build_training_dataset(
            combined_meta, output_path=str(TRAINING_CACHE)
        )

    if training_df.empty:
        logger.error("特徴量データセットが空です。")
        sys.exit(1)

    elapsed1 = (time.time() - t0) / 60
    logger.info(f"[STEP 1] 完了: {len(training_df):,} 行  {elapsed1:.1f}分")
    logger.info(f"  勝ち馬率: {training_df['is_win'].mean():.2%}")
    logger.info(f"  レース数: {training_df['race_id'].nunique():,}")

    # ── LightGBM 学習 ───────────────────────────────────────────
    logger.info("[STEP 2] LightGBM 学習（GroupKFold 5-fold）")
    trainer = ModelTrainer()
    trainer.fit(training_df)

    elapsed2 = (time.time() - t0) / 60
    logger.info(f"[STEP 2] 完了: {elapsed2:.1f}分")

    # ── 保存 ────────────────────────────────────────────────────
    logger.info("[STEP 3] モデル・特徴量統計保存")
    trainer.save(settings.model_path, org="jra")
    fe.save_stats(settings.stats_path)

    total = (time.time() - t0) / 60
    logger.info("=" * 60)
    logger.info(f"完了: {total:.1f}分")
    logger.info(f"  モデル: {settings.model_path}")
    logger.info(f"  統計  : {settings.stats_path}")
    logger.info("")
    logger.info("次のステップ: バックテストで精度確認")
    logger.info("  python backtest_1year.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
