"""
データ収集 → 学習 を一括実行するスクリプト（バックグラウンド実行用）。

Usage:
  nohup .venv/bin/python run_pipeline.py > logs/pipeline.log 2>&1 &
  tail -f logs/pipeline.log
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from config.settings import settings
from src.scraper.netkeiba_scraper import NetkeibaScraper
from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer


def setup_logger():
    log_path = Path("logs/pipeline.log")
    log_path.parent.mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(str(log_path), level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
               rotation="50 MB", retention=3)


def main():
    setup_logger()

    # ----------------------------------------------------------------
    # 収集期間: 5年分（2020年1月〜2024年12月）
    # ----------------------------------------------------------------
    START = date(2020, 1, 1)
    END = date(2024, 12, 31)
    JYO_CODES = ["05", "06", "08", "09"]  # 東京・中山・京都・阪神

    IDS_FILE = "data/raw/race_ids.csv"
    CHECKPOINT = "data/raw/checkpoint"
    TRAINING_CSV = "data/processed/training.csv"
    MODEL_PATH = Path("data/models/lgbm_model.pkl")

    logger.info("=" * 60)
    logger.info("競馬AI パイプライン開始")
    logger.info(f"期間: {START} → {END}")
    logger.info(f"対象場: {JYO_CODES}")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Phase 1: race_id 収集
    # ----------------------------------------------------------------
    t0 = time.time()
    logger.info("[Phase 1] race_id 収集")

    with NetkeibaScraper() as scraper:
        race_ids = scraper.collect_race_ids_for_period(
            start_date=START,
            end_date=END,
            jyo_codes=JYO_CODES,
            save_path=IDS_FILE,
        )

    elapsed = (time.time() - t0) / 60
    logger.info(f"[Phase 1] 完了: {len(race_ids):,} races in {elapsed:.1f}分")

    # ----------------------------------------------------------------
    # Phase 2: 結果 + メタ スクレイピング（最長フェーズ）
    # ----------------------------------------------------------------
    t1 = time.time()
    logger.info("[Phase 2] result+meta スクレイピング開始")
    logger.info(f"  推定時間: {len(race_ids) * 3 / 3600:.1f}時間")

    with NetkeibaScraper() as scraper:
        history_df, race_meta_df = scraper.fetch_bulk_results_and_meta(
            race_ids=race_ids,
            checkpoint_path=CHECKPOINT,
        )

    elapsed = (time.time() - t1) / 3600
    logger.info(f"[Phase 2] 完了: {len(history_df):,} 行, {len(race_meta_df):,} レース in {elapsed:.1f}h")

    # ----------------------------------------------------------------
    # Phase 3: 特徴量エンジニアリング → 学習CSV 生成
    # ----------------------------------------------------------------
    t2 = time.time()
    logger.info("[Phase 3] 特徴量エンジニアリング & 学習CSV生成")

    fe = FeatureEngineer(history_df)
    training_df = fe.build_training_dataset(race_meta_df, output_path=TRAINING_CSV)

    elapsed = (time.time() - t2) / 60
    logger.info(f"[Phase 3] 完了: {len(training_df):,} 行 in {elapsed:.1f}分")
    logger.info(f"  勝ち馬率: {training_df['is_win'].mean():.2%}")

    # ----------------------------------------------------------------
    # Phase 4: LightGBM 学習
    # ----------------------------------------------------------------
    t3 = time.time()
    logger.info("[Phase 4] LightGBM 学習")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    trainer = ModelTrainer()
    trainer.fit(training_df)
    trainer.save(MODEL_PATH)

    elapsed = (time.time() - t3) / 60
    logger.info(f"[Phase 4] 完了 in {elapsed:.1f}分")

    total_elapsed = (time.time() - t0) / 3600
    logger.info("=" * 60)
    logger.info(f"全パイプライン完了: {total_elapsed:.1f}時間")
    logger.info(f"  学習CSV  → {TRAINING_CSV}")
    logger.info(f"  モデル   → {MODEL_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
