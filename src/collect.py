"""
過去レースデータの収集 → 学習用CSVの生成を一括実行するCLI。

使用例:
  # 直近2年分（東京・中山・京都・阪神）を収集して学習CSV を生成
  keiba-collect --years 2

  # 期間指定
  keiba-collect --start 2023-01-01 --end 2024-12-31 --jyo 05,06,08,09

  # race_id の収集だけ（既存ファイルへ追記・再開可）
  keiba-collect --only-ids --output data/raw/race_ids.csv

  # 収集済み race_id から特徴量生成のみ再実行
  keiba-collect --from-ids data/raw/race_ids.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.scraper.netkeiba_scraper import NetkeibaScraper
from src.features.engineer import FeatureEngineer


def _setup_logger() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(settings.log_file, level=settings.log_level,
               rotation="1 day", retention="7 days", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="競馬データ収集 & 学習CSV生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 期間指定（--years か --start/--end のどちらか）
    period = parser.add_mutually_exclusive_group()
    period.add_argument(
        "--years",
        type=float,
        default=2.0,
        help="直近 N 年分を収集 (デフォルト: 2)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="収集開始日（--years より優先）",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="収集終了日（デフォルト: 今日）",
    )

    # 競馬場フィルタ
    parser.add_argument(
        "--jyo",
        type=str,
        default=None,
        metavar="CODES",
        help="競馬場コード（カンマ区切り）例: 05,06,08,09。省略時は設定ファイルの値を使用",
    )

    # 出力先
    parser.add_argument(
        "--ids-file",
        type=str,
        default="data/raw/race_ids.csv",
        metavar="PATH",
        help="race_id の中間ファイル保存先（再開時にも使用）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/training.csv",
        metavar="PATH",
        help="学習CSV の出力先",
    )

    # 実行モード
    parser.add_argument(
        "--only-ids",
        action="store_true",
        help="race_id の収集のみ実行（特徴量生成はしない）",
    )
    parser.add_argument(
        "--from-ids",
        type=str,
        default=None,
        metavar="PATH",
        help="既存の race_id CSV から読み込み、特徴量生成のみ実行",
    )

    return parser.parse_args()


def main() -> None:
    """CLI エントリーポイント: keiba-collect"""
    _setup_logger()
    args = _parse_args()

    import pandas as pd

    # ------------------------------------------------------------------
    # 既存 race_id から特徴量生成のみ
    # ------------------------------------------------------------------
    if args.from_ids:
        ids_path = Path(args.from_ids)
        if not ids_path.exists():
            logger.error(f"race_id ファイルが見つかりません: {ids_path}")
            sys.exit(1)

        race_ids = pd.read_csv(ids_path, dtype=str)["race_id"].tolist()
        logger.info(f"Loaded {len(race_ids)} race_ids from {ids_path}")

        with NetkeibaScraper() as scraper:
            FeatureEngineer.build_from_scratch(
                race_ids=race_ids,
                scraper=scraper,
                output_path=args.output,
                checkpoint_prefix="data/raw/checkpoint",
            )
        logger.info("Done.")
        return

    # ------------------------------------------------------------------
    # 期間の決定
    # ------------------------------------------------------------------
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = end_date - timedelta(days=int(args.years * 365))

    # 競馬場コード
    if args.jyo:
        jyo_codes = [c.strip() for c in args.jyo.split(",") if c.strip()]
    else:
        jyo_codes = settings.target_jyo_code_list or None

    logger.info(
        f"Collection period: {start_date} → {end_date} | "
        f"jyo_codes={jyo_codes or 'ALL'}"
    )

    ids_file = args.ids_file  # 途中保存・再開用

    # ------------------------------------------------------------------
    # Step 1: race_id 収集
    # ------------------------------------------------------------------
    with NetkeibaScraper() as scraper:
        logger.info("=== Step 1: race_id 収集 ===")
        race_ids = scraper.collect_race_ids_for_period(
            start_date=start_date,
            end_date=end_date,
            jyo_codes=jyo_codes,
            save_path=ids_file,
        )

        if args.only_ids:
            logger.info(f"--only-ids モード: {len(race_ids)} race_ids を {ids_file} に保存して終了。")
            return

        # ------------------------------------------------------------------
        # Step 2: 過去成績 + レースメタ → 学習CSV
        # ------------------------------------------------------------------
        logger.info("=== Step 2: 学習データセット生成 ===")
        FeatureEngineer.build_from_scratch(
            race_ids=race_ids,
            scraper=scraper,
            output_path=args.output,
            checkpoint_prefix="data/raw/checkpoint",
        )

    logger.info(f"完了。学習CSV: {args.output}")
    logger.info("次のステップ: keiba-train --input " + args.output)


if __name__ == "__main__":
    main()
