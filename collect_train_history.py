"""
過去JRAデータをtrain_results.csv / train_meta.csv に追記収集するスクリプト。

使用例:
    # 2022-2023年、全10会場を収集（デフォルト）
    python collect_train_history.py

    # 期間・会場を指定
    python collect_train_history.py --start 2022-01-01 --end 2023-12-31 --jyo 05,06,08,09

    # バックグラウンド実行
    nohup .venv/bin/python collect_train_history.py > logs/collect_train.log 2>&1 &
    tail -f logs/collect_train.log
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.scraper.netkeiba_scraper import NetkeibaScraper

# ── 保存先 ────────────────────────────────────────────────────────────
TRAIN_RESULTS_CSV = ROOT / "data" / "raw" / "train_results.csv"
TRAIN_META_CSV    = ROOT / "data" / "raw" / "train_meta.csv"

# ── ロガー ───────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/collect_train.log", level="DEBUG", format=_fmt, rotation="10 MB")


def _append_dedup(new_df: pd.DataFrame, csv_path: Path, key: str = "race_id") -> int:
    """new_df を csv_path に追記（key 列で重複排除）。追記件数を返す。"""
    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype=str)
        existing_keys = set(existing[key].tolist())
        to_add = new_df[~new_df[key].isin(existing_keys)]
        if to_add.empty:
            logger.info(f"  追記なし（全件既存）: {csv_path.name}")
            return 0
        combined = pd.concat([existing, to_add], ignore_index=True)
    else:
        combined = new_df
        to_add = new_df

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    n = len(to_add)
    logger.info(f"  追記完了: {csv_path.name} +{n}行 (合計 {len(combined)}行)")
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JRA過去データ収集 (train用)")
    parser.add_argument("--start", default="2022-01-01", metavar="YYYY-MM-DD",
                        help="収集開始日 (デフォルト: 2022-01-01)")
    parser.add_argument("--end",   default="2023-12-31", metavar="YYYY-MM-DD",
                        help="収集終了日 (デフォルト: 2023-12-31)")
    parser.add_argument("--jyo", default="01,02,03,04,05,06,07,08,09,10",
                        help="会場コード カンマ区切り (デフォルト: 全10会場)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    jyo_codes = [c.strip() for c in args.jyo.split(",") if c.strip()]

    logger.info("=" * 60)
    logger.info(f"JRA訓練データ収集: {start} → {end}")
    logger.info(f"対象会場: {jyo_codes}")
    logger.info("=" * 60)

    # 既に収集済みのrace_idを読み込む
    existing_ids: set[str] = set()
    for csv_path in (TRAIN_RESULTS_CSV, TRAIN_META_CSV):
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, dtype=str)
                existing_ids = set(df["race_id"].dropna().tolist())
                logger.info(f"既存 {csv_path.name}: {len(existing_ids)} race_ids")
                break
            except Exception:
                pass

    with NetkeibaScraper() as scraper:
        # Step 1: race_id 収集
        logger.info("=== Step 1: race_id 収集 ===")
        race_ids = scraper.collect_race_ids_for_period(
            start_date=start,
            end_date=end,
            jyo_codes=jyo_codes,
            save_path="data/raw/train_race_ids_new.csv",
        )
        logger.info(f"合計 race_ids: {len(race_ids)}")

        # 未収集分のみ
        new_ids = [r for r in race_ids if r not in existing_ids]
        logger.info(f"未収集 race_ids: {len(new_ids)}")

        if not new_ids:
            logger.info("全 race_id が既に収集済みです。終了します。")
            return

        # Step 2: results + meta 取得
        logger.info(f"=== Step 2: スクレイピング ({len(new_ids)} レース) ===")
        logger.info(f"  推定時間: {len(new_ids) * 3 / 3600:.1f}時間")
        results_df, meta_df = scraper.fetch_bulk_results_and_meta(
            race_ids=new_ids,
            checkpoint_path="data/raw/train_checkpoint_new",
        )
        logger.info(f"取得完了: results={len(results_df)}行, meta={len(meta_df)}レース")

    # Step 3: CSV追記
    logger.info("=== Step 3: CSV追記 ===")
    added_r = _append_dedup(results_df, TRAIN_RESULTS_CSV, key="race_id")
    added_m = _append_dedup(meta_df,    TRAIN_META_CSV,    key="race_id")
    logger.info(f"完了: results +{added_r}行, meta +{added_m}件")
    logger.info("")
    logger.info("次のステップ: モデルを再学習")
    logger.info("  python train_jra_model.py")


if __name__ == "__main__":
    main()
