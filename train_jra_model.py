"""
JRA モデル再学習スクリプト。

data/raw/train_results.csv + train_meta.csv から特徴量を生成し
LightGBM モデルを学習・保存する。

実行例:
    # デフォルト（全期間 → data/models/lgbm_model.pkl）
    python train_jra_model.py

    # 不調期専用モデル（1,7-9,11-12月のみ → data/models/lgbm_model_bad_season.pkl）
    python train_jra_model.py --bad-season

    # バックグラウンド実行（長時間になる場合）
    nohup .venv/bin/python train_jra_model.py > logs/train_jra.log 2>&1 &
    nohup .venv/bin/python train_jra_model.py --bad-season > logs/train_bad_season.log 2>&1 &
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


BAD_SEASON_MONTHS = {1, 7, 8, 9, 11, 12}
BAD_SEASON_MODEL_PATH = Path("data/models/lgbm_model_bad_season.pkl")


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
    parser.add_argument("--bad-season", action="store_true",
                        help="不調期専用モデルを学習（1,7-9,11-12月のみ）")
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    meta_path    = Path(args.meta_csv)
    if args.model_output:
        model_path = Path(args.model_output)
    elif args.bad_season:
        model_path = BAD_SEASON_MODEL_PATH
    else:
        model_path = settings.model_path

    if not results_path.exists():
        logger.error(f"results CSV が見つかりません: {results_path}")
        sys.exit(1)
    if not meta_path.exists():
        logger.error(f"meta CSV が見つかりません: {meta_path}")
        sys.exit(1)

    t0 = time.time()
    logger.info("=" * 60)
    if args.bad_season:
        logger.info("JRA 不調期専用モデル学習開始")
        logger.info(f"  対象月: {sorted(BAD_SEASON_MONTHS)}")
    else:
        logger.info("JRA モデル学習開始（全期間）")
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

    # ── 不調期フィルタ ────────────────────────────────────────────
    if args.bad_season:
        meta_df["_race_date"] = pd.to_datetime(meta_df["race_date"], errors="coerce")
        meta_df["_month"] = meta_df["_race_date"].dt.month
        meta_before = len(meta_df)
        meta_df = meta_df[meta_df["_month"].isin(BAD_SEASON_MONTHS)].copy()
        meta_df = meta_df.drop(columns=["_race_date", "_month"])
        logger.info(f"不調期フィルタ後: {len(meta_df):,} レース（全{meta_before:,}中）")
        # results もフィルタ（不調期レースのみ学習データとして使用）
        bad_race_ids = set(meta_df["race_id"].tolist())
        res_df = res_df[res_df["race_id"].isin(bad_race_ids)].copy()
        logger.info(f"  results 絞り込み後: {len(res_df):,} 行")

    # ── 特徴量生成（データリーク防止ルックバック付き）──────────────
    logger.info("[STEP 1] 特徴量生成（per-race ルックバック方式）")
    logger.info(f"  対象レース数: {meta_df['race_id'].nunique():,}")
    logger.info("  ※ 多数のレースがある場合は数十分〜1時間以上かかる場合があります")

    # 不調期モデルの場合: 全履歴を使って特徴量を生成（ルックバック精度を維持）
    # 不調期レースのみを学習するが、特徴量計算には全データを参照する
    if args.bad_season:
        full_res_df = pd.read_csv(results_path, dtype=str)
        fe = FeatureEngineer(full_res_df)
    else:
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
    logger.info("[STEP 3] モデル保存")
    trainer.save(model_path, org="jra")
    # 不調期モデルは feature_stats を上書きしない（全期間統計を維持）
    if not args.bad_season:
        # test_results_new.csv が存在する場合、horse_recent_form に含める
        # （直近成績の精度向上: 2025年以降のレース結果を反映）
        extra_results_path = ROOT / "data/raw/test_results_new.csv"
        extra_df: pd.DataFrame | None = None
        if extra_results_path.exists():
            extra_df = pd.read_csv(extra_results_path, dtype=str)
            logger.info(f"  追加履歴読み込み: {extra_results_path} ({len(extra_df):,} 行)")
        fe.save_stats(settings.stats_path, extra_history_df=extra_df)
        logger.info(f"  統計  : {settings.stats_path}")
    else:
        logger.info("  統計: 全期間モデルの feature_stats.pkl をそのまま使用")

    total = (time.time() - t0) / 60
    logger.info("=" * 60)
    if args.bad_season:
        logger.info(f"JRA 不調期専用モデル学習完了: {total:.1f}分")
    else:
        logger.info(f"JRA モデル学習完了: {total:.1f}分")
    logger.info(f"  モデル: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
