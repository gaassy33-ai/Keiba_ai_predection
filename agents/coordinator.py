"""
agents/coordinator.py - エージェントコーディネーター
=====================================================
役割:
  - 時刻に応じて各サブエージェントを協調実行
  - 朝フェーズ: DataMaster（週次統計）
  - 夕方フェーズ: Reviewer（結果照合）→ Critic（Claude評価）

実行モード:
    python -m agents.coordinator --phase morning [--date YYYY-MM-DD] [--dry-run]
    python -m agents.coordinator --phase evening [--date YYYY-MM-DD] [--dry-run]
    python -m agents.coordinator --phase all     [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.data_master import DataMaster
from agents.reviewer import Reviewer
from agents.critic import Critic


class Coordinator:
    """エージェントオーケストレーター"""

    def __init__(self) -> None:
        self.data_master = DataMaster()
        self.reviewer    = Reviewer()
        self.critic      = Critic()

    # ------------------------------------------------------------------ #
    # フェーズ実行
    # ------------------------------------------------------------------ #

    def run_morning(self, target_date: date, dry_run: bool = False) -> dict:
        """
        朝フェーズ（土日 07:00 JST 前後）
          DataMaster: 直近8週の精度統計・特徴量重要度をまとめてLINE送信
        """
        logger.info(f"=== Coordinator 朝フェーズ: {target_date} ===")
        context = self.data_master.run(target_date, dry_run=dry_run)
        logger.info("朝フェーズ完了")
        return {"phase": "morning", "date": str(target_date), "context": context}

    def run_evening(self, target_date: date, dry_run: bool = False) -> dict:
        """
        夕方フェーズ（土日 17:00 JST 前後）
          1. Reviewer: レース結果を照合・精度分析・LINE送信
          2. Critic: Claude API で客観評価・フィードバック LINE送信
        """
        logger.info(f"=== Coordinator 夕方フェーズ: {target_date} ===")

        # ① 振り返り君
        logger.info("-- Reviewer 起動 --")
        review = self.reviewer.run(target_date, dry_run=dry_run)

        # ② 評論家（振り返りデータが揃ってから）
        logger.info("-- Critic 起動 --")
        time.sleep(2)  # ファイル書き込み完了を待つ
        feedback = self.critic.run(target_date, dry_run=dry_run)

        logger.info("夕方フェーズ完了")
        return {
            "phase": "evening",
            "date": str(target_date),
            "review": review,
            "feedback": feedback,
        }

    def run_all(self, target_date: date, dry_run: bool = False) -> dict:
        """全フェーズを順次実行（テスト・手動実行用）"""
        logger.info(f"=== Coordinator 全フェーズ: {target_date} ===")
        morning_result = self.run_morning(target_date, dry_run=dry_run)
        evening_result = self.run_evening(target_date, dry_run=dry_run)
        return {
            "phase": "all",
            "date": str(target_date),
            **morning_result,
            **evening_result,
        }


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="エージェントコーディネーター")
    parser.add_argument(
        "--phase",
        choices=["morning", "evening", "all"],
        default="all",
        help="実行フェーズ (morning=朝バッチ前, evening=レース後, all=全フェーズ)",
    )
    parser.add_argument("--date", default=str(date.today()), help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="LINE 送信しない")
    args = parser.parse_args()

    coordinator = Coordinator()
    target_date = date.fromisoformat(args.date)

    if args.phase == "morning":
        coordinator.run_morning(target_date, dry_run=args.dry_run)
    elif args.phase == "evening":
        coordinator.run_evening(target_date, dry_run=args.dry_run)
    else:
        coordinator.run_all(target_date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
