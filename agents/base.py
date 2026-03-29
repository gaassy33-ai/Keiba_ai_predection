"""
agents/base.py - 全エージェント共通の基底クラス
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings

AGENTS_DIR = ROOT / "data" / "agents"
AGENTS_DIR.mkdir(parents=True, exist_ok=True)


class AgentBase:
    """全エージェント共通の基底クラス。"""

    name: str = "BaseAgent"

    def __init__(self) -> None:
        self._setup_logger()

    def _setup_logger(self) -> None:
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.remove()
        fmt = "{time:HH:mm:ss} | {level:<7} | " + self.name + " | {message}"
        logger.add(sys.stdout, level="INFO", format=fmt, colorize=True)
        logger.add(
            log_dir / "agents.log",
            level="DEBUG",
            format=fmt,
            rotation="20 MB",
        )

    # ------------------------------------------------------------------ #
    # JSON I/O
    # ------------------------------------------------------------------ #

    def save(self, data: dict[str, Any], filename: str) -> Path:
        """data/agents/{filename} に JSON 保存。"""
        path = AGENTS_DIR / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        logger.info(f"保存: {path}")
        return path

    def load(self, filename: str) -> dict[str, Any] | None:
        """data/agents/{filename} を読み込む。存在しなければ None。"""
        path = AGENTS_DIR / filename
        if not path.exists():
            logger.warning(f"ファイルが見つかりません: {path}")
            return None
        return json.loads(path.read_text())

    # ------------------------------------------------------------------ #
    # ファイル名ヘルパー
    # ------------------------------------------------------------------ #

    @staticmethod
    def context_file(d: date) -> str:
        return f"{d.strftime('%Y%m%d')}_context.json"

    @staticmethod
    def review_file(d: date) -> str:
        return f"{d.strftime('%Y%m%d')}_review.json"

    @staticmethod
    def feedback_file(d: date) -> str:
        return f"{d.strftime('%Y%m%d')}_feedback.json"

    # ------------------------------------------------------------------ #
    # LINE 送信
    # ------------------------------------------------------------------ #

    def send_line(self, message: str, dry_run: bool = False) -> None:
        """テキストメッセージを LINE に送信する。"""
        if dry_run:
            logger.info(f"[DRY-RUN] LINE:\n{message}")
            return
        if not settings.line_channel_access_token or not settings.line_target_user_id:
            logger.warning("LINE 認証情報が未設定。送信をスキップします。")
            return
        try:
            resp = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.line_channel_access_token}",
                },
                json={
                    "to": settings.line_target_user_id,
                    "messages": [{"type": "text", "text": message}],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                logger.info("LINE 送信完了")
            else:
                logger.error(f"LINE 送信失敗: {resp.status_code} {resp.text}")
        except Exception as exc:
            logger.error(f"LINE 送信エラー: {exc}")

    def send_line_flex(self, flex_msg: dict, dry_run: bool = False) -> None:
        """Flex Message を LINE に送信する。"""
        if dry_run:
            logger.info(f"[DRY-RUN] LINE Flex:\n{json.dumps(flex_msg, ensure_ascii=False, indent=2)}")
            return
        if not settings.line_channel_access_token or not settings.line_target_user_id:
            logger.warning("LINE 認証情報が未設定。送信をスキップします。")
            return
        try:
            resp = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.line_channel_access_token}",
                },
                json={
                    "to": settings.line_target_user_id,
                    "messages": [flex_msg],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                logger.info("LINE Flex 送信完了")
            else:
                logger.error(f"LINE Flex 送信失敗: {resp.status_code} {resp.text}")
        except Exception as exc:
            logger.error(f"LINE Flex 送信エラー: {exc}")
