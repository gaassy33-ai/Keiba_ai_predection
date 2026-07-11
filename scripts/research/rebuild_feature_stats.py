"""
feature_stats.pkl を学習データのみで再構築するスクリプト。

【目的】
train_jra_model.py は feature_stats.pkl 構築時に test_results_new.csv（2025-2026 データ）
を horse_recent_form に追加する。これにより 2025 年バックテストで時系列リーク
（将来の horse_recent_form を使って過去レースを予測）が発生し、学習時と推論時の特徴量が
不一致になる問題がある。

本スクリプトは extra_history_df を使わずに feature_stats.pkl を再構築することで
推論特徴量の時系列整合性を確保する。

実行例:
    .venv/bin/python rebuild_feature_stats.py
    .venv/bin/python rebuild_feature_stats.py --backup  # 既存をバックアップしてから置換
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from config.settings import settings

Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/rebuild_feature_stats.log", level="DEBUG", format=_fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="feature_stats.pkl を学習データのみで再構築")
    parser.add_argument("--results-csv", default="data/raw/train_results.csv")
    parser.add_argument("--meta-csv",    default="data/raw/train_meta.csv")
    parser.add_argument("--backup", action="store_true",
                        help="既存の feature_stats.pkl を .bak にバックアップしてから置換")
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    meta_path    = Path(args.meta_csv)
    stats_path   = settings.stats_path

    if not results_path.exists():
        logger.error(f"results CSV が見つかりません: {results_path}")
        sys.exit(1)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("feature_stats.pkl 再構築（学習データのみ・時系列リーク除去）")
    logger.info(f"  入力: {results_path}")
    logger.info(f"  出力: {stats_path}")
    logger.info("=" * 60)

    # ── データ読み込み ──────────────────────────────────────────────
    res_df = pd.read_csv(results_path, dtype=str)
    logger.info(f"学習データ: {len(res_df):,} 行")
    years = sorted(res_df["race_id"].str[:4].unique())
    logger.info(f"  対象年: {years}")

    # 年別レース数を確認
    year_counts = res_df.groupby(res_df["race_id"].str[:4])["race_id"].nunique()
    for yr, cnt in year_counts.items():
        logger.info(f"    {yr}: {cnt:,} レース")

    # ── FeatureEngineer 初期化（学習データのみ） ─────────────────────
    logger.info("[STEP 1] FeatureEngineer 初期化（学習データのみ）")
    fe = FeatureEngineer(res_df)

    # ── 集計計算（騎手・調教師・会場別勝率など） ──────────────────────
    logger.info("[STEP 2] 集計計算（jockey/trainer/horse stats）")
    fe.precompute_aggregations()
    elapsed1 = (time.time() - t0) / 60
    logger.info(f"  完了: {elapsed1:.1f}分")

    # ── feature_stats.pkl 保存（extra_history_df なし = 将来データ混入なし） ─
    logger.info("[STEP 3] feature_stats.pkl 保存（extra_history なし）")

    if args.backup and stats_path.exists():
        bak_path = stats_path.with_suffix(".pkl.bak")
        shutil.copy(stats_path, bak_path)
        logger.info(f"  バックアップ: {bak_path}")

    fe.save_stats(stats_path)  # extra_history_df=None → 学習データのみで horse_recent_form 計算
    elapsed2 = (time.time() - t0) / 60
    logger.info(f"  完了: {elapsed2:.1f}分")

    # ── 確認 ──────────────────────────────────────────────────────
    import pickle
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    hrf = stats.get("horse_recent_form")
    if hrf is not None:
        dates = pd.to_datetime(hrf.get("last_race_date", pd.Series()), errors="coerce").dropna()
        logger.info(f"horse_recent_form: {len(hrf):,} 頭")
        if len(dates) > 0:
            logger.info(f"  最新レース日: {dates.max().date()} ← 学習データのみのため将来データなし")
            logger.info(f"  2025年以降: {(dates.dt.year >= 2025).sum()} 頭（学習データに2025年が含まれる場合のみ）")
    jcs = stats.get("jockey_course_stats")
    if jcs is not None:
        logger.info(f"jockey_course_stats: {len(jcs):,} 件")

    total = (time.time() - t0) / 60
    logger.info("=" * 60)
    logger.info(f"完了: {total:.1f}分")
    logger.info(f"  {stats_path} を更新しました（学習データのみ版）")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
