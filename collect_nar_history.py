"""
NAR（地方競馬）過去レースデータ一括収集スクリプト。

チェックポイント付きで途中再開可能。
scrape_interval を短縮して収集速度を上げる（ローカル実行用）。

実行:
    python collect_nar_history.py
    python collect_nar_history.py --start 2025-03-17 --end 2026-03-15
    python collect_nar_history.py --interval 1.5   # 秒/リクエスト
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── パス定数 ──────────────────────────────────────────────────
RACE_IDS_CSV   = ROOT / "data" / "raw" / "nar_race_ids.csv"
RESULTS_CSV    = ROOT / "data" / "raw" / "nar_results.csv"
META_CSV       = ROOT / "data" / "raw" / "nar_meta.csv"
CHECKPOINT_PFX = str(ROOT / "data" / "raw" / "nar_checkpoint")

# NAR 主要会場コード（netkeiba 内部コード）
NAR_JYO_DEFAULT = ["30","35","36","42","43","44","45","46","47","48","50","51","54","55","65"]

# ── ロガー ─────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/collect_nar_history.log", level="DEBUG", format=_fmt, rotation="20 MB")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAR 過去レースデータ一括収集")
    parser.add_argument("--start", default="2025-03-17", metavar="YYYY-MM-DD")
    parser.add_argument("--end",   default="2026-03-15", metavar="YYYY-MM-DD")
    parser.add_argument("--jyo",   default=",".join(NAR_JYO_DEFAULT),
                        help="会場コード (カンマ区切り)")
    parser.add_argument("--interval", type=float, default=1.5,
                        help="リクエスト間隔 秒 (デフォルト: 1.5)")
    parser.add_argument("--skip-ids", action="store_true",
                        help="race_id 収集をスキップ (既存 nar_race_ids.csv を使用)")
    return parser.parse_args()


def _append_dedup(new_df: pd.DataFrame, csv_path: Path, key: str = "race_id") -> int:
    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype=str)
        to_add = new_df[~new_df[key].isin(set(existing[key].tolist()))]
        if to_add.empty:
            return 0
        combined = pd.concat([existing, to_add], ignore_index=True)
    else:
        combined = new_df
        to_add = new_df

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return len(to_add)


def main() -> None:
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    jyo_codes = [c.strip() for c in args.jyo.split(",")]

    # ── NARScraper をカスタムインターバルで初期化 ────────────────
    from config.settings import settings
    # interval を一時的に上書き
    settings.scrape_interval_seconds = args.interval

    from src.scraper.nar_scraper import NARScraper
    scraper = NARScraper()

    # ── STEP 1: race_id 収集 ─────────────────────────────────────
    if args.skip_ids and RACE_IDS_CSV.exists():
        race_ids = pd.read_csv(RACE_IDS_CSV, dtype=str)["race_id"].tolist()
        logger.info(f"既存 race_ids を使用: {len(race_ids)} 件 ({RACE_IDS_CSV})")
    else:
        logger.info(f"[STEP 1] NAR race_id 収集: {start} 〜 {end}  jyo={jyo_codes}")
        race_ids = scraper.collect_race_ids_for_period(
            start_date=start,
            end_date=end,
            jyo_codes=jyo_codes,
            save_path=str(RACE_IDS_CSV),
        )
        logger.info(f"  race_ids: {len(race_ids)} 件")

    # ── STEP 2: 既収集分を除外 ──────────────────────────────────
    done_ids: set[str] = set()
    if RESULTS_CSV.exists():
        done_ids = set(pd.read_csv(RESULTS_CSV, dtype=str)["race_id"].unique())
        logger.info(f"既収集済み: {len(done_ids)} レース → スキップ")

    pending = [r for r in race_ids if r not in done_ids]
    if not pending:
        logger.info("全レース収集済み。終了します。")
        scraper.close()
        return

    logger.info(f"[STEP 2] 結果+メタ取得: {len(pending)} レース")
    eta_min = len(pending) * args.interval / 60
    logger.info(f"  推定所要時間: {eta_min:.0f}分 ({args.interval}秒/レース)")

    # ── STEP 3: 一括取得（チェックポイント付き）────────────────────
    t0 = time.time()
    results_df, meta_df = scraper.fetch_bulk_results_and_meta(
        pending,
        checkpoint_path=CHECKPOINT_PFX,
    )
    scraper.close()

    elapsed = (time.time() - t0) / 60
    logger.info(f"取得完了: results={len(results_df)}行, meta={len(meta_df)}件 ({elapsed:.1f}分)")

    # ── STEP 4: CSV 追記 ─────────────────────────────────────────
    if not results_df.empty:
        _append_dedup(results_df, RESULTS_CSV, key="race_id")
    if not meta_df.empty:
        _append_dedup(meta_df, META_CSV, key="race_id")

    logger.info(f"保存完了: {RESULTS_CSV} / {META_CSV}")

    # チェックポイントファイル削除
    for suffix in ("_results.csv", "_meta.csv"):
        p = Path(CHECKPOINT_PFX + suffix)
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    main()
