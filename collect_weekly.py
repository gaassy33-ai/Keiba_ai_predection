"""
週次レース結果収集スクリプト
- 直近の土日のレース結果を収集し data/raw/test_results.csv / test_meta.csv に追記する
- モデルは再学習しない（特徴量計算のルックバックデータのみ更新）

実行例:
    # 自動（直前の土曜〜日曜を対象）
    python collect_weekly.py

    # 日付指定
    python collect_weekly.py --start 2026-03-15 --end 2026-03-16

    # 会場指定
    python collect_weekly.py --jyo 05,06,08,09
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.scraper.netkeiba_scraper import NetkeibaScraper

# ── ロガー ─────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/collect_weekly.log", level="DEBUG", format=_fmt, rotation="10 MB")

# ── 保存先 ────────────────────────────────────────────────────
TEST_RESULTS_CSV = ROOT / "data" / "raw" / "test_results.csv"
TEST_META_CSV    = ROOT / "data" / "raw" / "test_meta.csv"


def _last_weekend() -> tuple[date, date]:
    """直前の土曜〜日曜を返す（月〜金に実行した場合は直前の土日）"""
    today = date.today()
    # 今日の曜日(0=月, 6=日)
    dow = today.weekday()
    if dow == 0:   # 月曜: 直前の土日
        sat = today - timedelta(days=2)
        sun = today - timedelta(days=1)
    elif dow == 6: # 日曜: 今日
        sat = today - timedelta(days=1)
        sun = today
    elif dow == 5: # 土曜: 今日
        sat = today
        sun = today
    else:          # 火〜金: 直前の土日
        sat = today - timedelta(days=dow + 2)
        sun = today - timedelta(days=dow + 1)
    return sat, sun


def _parse_args() -> argparse.Namespace:
    sat, sun = _last_weekend()
    parser = argparse.ArgumentParser(description="週次レース結果収集")
    parser.add_argument("--start", default=str(sat), metavar="YYYY-MM-DD",
                        help=f"収集開始日 (デフォルト: {sat})")
    parser.add_argument("--end",   default=str(sun), metavar="YYYY-MM-DD",
                        help=f"収集終了日 (デフォルト: {sun})")
    parser.add_argument("--jyo", default="01,02,03,04,05,06,07,08,09,10",
                        help="会場コード (カンマ区切り)")
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    jyo_codes = [c.strip() for c in args.jyo.split(",")]

    logger.info(f"収集期間: {start} 〜 {end}  会場: {jyo_codes}")

    with NetkeibaScraper() as scraper:
        # race_id 収集
        logger.info("race_id 収集中...")
        race_ids = scraper.collect_race_ids_for_period(
            start_date=start,
            end_date=end,
            jyo_codes=jyo_codes,
        )
        if not race_ids:
            logger.warning("対象レースが見つかりません。終了します。")
            return
        logger.info(f"対象レース数: {len(race_ids)}")

        # 既存の race_id と照合して未収集分だけ取得
        new_ids = race_ids
        for csv_path in (TEST_RESULTS_CSV, TEST_META_CSV):
            if csv_path.exists():
                existing_ids = set(
                    pd.read_csv(csv_path, dtype=str)["race_id"].tolist()
                )
                new_ids = [r for r in race_ids if r not in existing_ids]
                break  # どちらかで確認すれば十分

        if not new_ids:
            logger.info("全 race_id が既に収集済みです。終了します。")
            return
        logger.info(f"新規収集対象: {len(new_ids)} レース")

        # results + meta 取得
        logger.info("レース結果・メタ情報を取得中...")
        results_df, meta_df = scraper.fetch_bulk_results_and_meta(new_ids)
        logger.info(f"取得完了: results={len(results_df)}行, meta={len(meta_df)}レース")

    # CSV 追記
    added_r = _append_dedup(results_df, TEST_RESULTS_CSV, key="race_id")
    added_m = _append_dedup(meta_df,    TEST_META_CSV,    key="race_id")

    logger.info(f"完了: results +{added_r}行, meta +{added_m}件")


if __name__ == "__main__":
    main()
