"""
JRA モデル再学習スクリプト。

data/raw/train_results.csv + train_meta.csv から特徴量を生成し
LightGBM モデルを学習・保存する。

実行例:
    # デフォルト（train_results.csv + train_meta.csv → data/models/lgbm_model.pkl）
    python train_jra_model.py

    # バックグラウンド実行（長時間になる場合）
    nohup .venv/bin/python train_jra_model.py > logs/train_jra.log 2>&1 &
    tail -f logs/train_jra.log
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

# ── ロガー設定 ─────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/train_jra.log", level="DEBUG", format=_fmt, rotation="20 MB")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="JRA モデル再学習")
    parser.add_argument("--results-csv", default="data/raw/train_results.csv",
                        help="学習用結果CSV (デフォルト: data/raw/train_results.csv)")
    parser.add_argument("--meta-csv",    default="data/raw/train_meta.csv",
                        help="学習用メタCSV (デフォルト: data/raw/train_meta.csv)")
    parser.add_argument("--model-output", default=None,
                        help="モデル保存先 (デフォルト: data/models/lgbm_model.pkl)")
    parser.add_argument("--training-csv", default=None,
                        help="特徴量CSVの保存先（省略時は保存しない）")
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    meta_path    = Path(args.meta_csv)
    model_path   = Path(args.model_output) if args.model_output else settings.model_path

    if not results_path.exists():
        logger.error(f"results CSV が見つかりません: {results_path}")
        sys.exit(1)
    if not meta_path.exists():
        logger.error(f"meta CSV が見つかりません: {meta_path}")
        sys.exit(1)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("JRA モデル学習開始")
    logger.info(f"  results: {results_path}")
    logger.info(f"  meta   : {meta_path}")
    logger.info(f"  output : {model_path}")
    logger.info("=" * 60)

    # ── データ読み込み ─────────────────────────────────────────────
    res_df  = pd.read_csv(results_path, dtype=str)
    meta_df = pd.read_csv(meta_path,    dtype=str)
    logger.info(f"データ読み込み完了: results={len(res_df):,} 行  meta={len(meta_df):,} レース")
    logger.info(f"  対象年: {sorted(res_df['race_id'].str[:4].unique())}")
    logger.info(f"  対象会場: {sorted(res_df['race_id'].str[4:6].unique())}")

    # ── 特徴量生成（データリーク防止ルックバック付き）──────────────
    logger.info("[STEP 1] 特徴量生成（per-race ルックバック方式）")
    logger.info(f"  対象レース数: {meta_df['race_id'].nunique():,}")
    logger.info("  ※ 多数のレースがある場合は数十分〜1時間以上かかる場合があります")

    fe = FeatureEngineer(res_df)
    fe.precompute_aggregations()

    training_df = fe.build_training_dataset(
        meta_df,
        output_path=args.training_csv,
    )

    if training_df.empty:
        logger.error("特徴量データセットが空です。データを確認してください。")
        sys.exit(1)

    elapsed1 = (time.time() - t0) / 60
    logger.info(f"[STEP 1] 完了: {len(training_df):,} 行  {elapsed1:.1f}分")
    logger.info(f"  勝ち馬率: {training_df['is_win'].mean():.2%}")
    logger.info(f"  レース数: {training_df['race_id'].nunique():,}")

    # ── モデル学習 ─────────────────────────────────────────────────
    logger.info("[STEP 2] LightGBM 学習（GroupKFold 5-fold）")
    trainer = ModelTrainer()
    trainer.fit(training_df)

    elapsed2 = (time.time() - t0) / 60
    logger.info(f"[STEP 2] 完了: {elapsed2:.1f}分")

    # ── モデル保存 ─────────────────────────────────────────────────
    logger.info("[STEP 3] モデル・特徴量統計保存")
    trainer.save(model_path, org="jra")
    fe.save_stats(settings.stats_path)

    total = (time.time() - t0) / 60
    logger.info("=" * 60)
    logger.info(f"JRA モデル学習完了: {total:.1f}分")
    logger.info(f"  モデル: {model_path}")
    logger.info(f"  統計  : {settings.stats_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
