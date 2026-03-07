"""
LINE Messaging API を使って予測結果を Flex Message で送信する。
"""

from __future__ import annotations

from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    FlexMessage,
    FlexContainer,
)
from loguru import logger

from config.settings import settings
from src.model.predictor import PredictionResult


def _build_flex_container(
    result: PredictionResult,
    shap_text: str,
    ground_condition: str,
    weather: str,
) -> dict:
    """
    Flex Message の JSON 構造を組み立てる（bubble 形式）。
    LINE Flex Message Simulator で確認可能。
    """

    def horse_row(label: str, color: str, horse: dict) -> dict:
        name = horse.get("horse_name", "---")
        num = horse.get("horse_number", "-")
        prob = horse.get("win_prob", 0)
        return {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {
                    "type": "text",
                    "text": label,
                    "size": "sm",
                    "color": "#ffffff",
                    "align": "center",
                    "gravity": "center",
                    "flex": 1,
                    "backgroundColor": color,
                    "decoration": "none",
                },
                {
                    "type": "text",
                    "text": f"#{num}",
                    "size": "sm",
                    "color": "#555555",
                    "flex": 1,
                    "align": "center",
                },
                {
                    "type": "text",
                    "text": name,
                    "size": "sm",
                    "color": "#111111",
                    "flex": 3,
                    "weight": "bold",
                },
                {
                    "type": "text",
                    "text": f"{prob:.1%}",
                    "size": "sm",
                    "color": "#888888",
                    "flex": 2,
                    "align": "end",
                },
            ],
            "margin": "sm",
            "paddingAll": "sm",
            "backgroundColor": "#f9f9f9",
            "cornerRadius": "md",
        }

    # SHAP 根拠を複数行のテキストコンポーネントに変換
    shap_lines = [line for line in shap_text.split("\n") if line.strip()]
    shap_components = [
        {
            "type": "text",
            "text": line,
            "size": "xs",
            "color": "#555555",
            "wrap": True,
        }
        for line in shap_lines
    ]

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#1a237e",
            "paddingAll": "md",
            "contents": [
                {
                    "type": "text",
                    "text": "馬券AI予想",
                    "color": "#ffffff",
                    "size": "xs",
                    "weight": "bold",
                },
                {
                    "type": "text",
                    "text": result.race_name,
                    "color": "#ffffff",
                    "size": "lg",
                    "weight": "bold",
                    "wrap": True,
                },
                {
                    "type": "text",
                    "text": f"天候: {weather}　馬場: {ground_condition}",
                    "color": "#aaaaff",
                    "size": "xs",
                    "margin": "sm",
                },
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                # 予想馬セクション
                {
                    "type": "text",
                    "text": "今日の予想",
                    "size": "sm",
                    "weight": "bold",
                    "color": "#333333",
                },
                horse_row("本命", "#e53935", result.honmei),
                horse_row("対抗", "#1e88e5", result.taikou),
                horse_row("穴馬", "#43a047", result.ana),
                # 区切り
                {"type": "separator", "margin": "md"},
                # SHAP 根拠
                {
                    "type": "text",
                    "text": "予測根拠（AI説明）",
                    "size": "sm",
                    "weight": "bold",
                    "color": "#333333",
                    "margin": "md",
                },
                *shap_components,
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "※ 予想は参考情報です。馬券購入は自己責任でお願いします。",
                    "size": "xxs",
                    "color": "#aaaaaa",
                    "wrap": True,
                    "align": "center",
                }
            ],
        },
        "styles": {
            "footer": {"backgroundColor": "#f5f5f5"},
        },
    }


class LineNotifier:
    """
    Usage
    -----
    >>> notifier = LineNotifier()
    >>> notifier.send_prediction(result, shap_text, "良", "晴")
    """

    def __init__(self) -> None:
        config = Configuration(access_token=settings.line_channel_access_token)
        self._api_client = ApiClient(config)
        self._messaging_api = MessagingApi(self._api_client)

    def send_prediction(
        self,
        result: PredictionResult,
        shap_text: str,
        ground_condition: str = "不明",
        weather: str = "不明",
    ) -> None:
        """
        Flex Message として予測結果を LINE に送信する。

        Parameters
        ----------
        result : PredictionResult
        shap_text : str
            PredictionExplainer.explain_text() の出力
        ground_condition : str
            馬場状態
        weather : str
            天候
        """
        container_dict = _build_flex_container(result, shap_text, ground_condition, weather)

        flex_msg = FlexMessage(
            alt_text=f"【AI競馬予想】{result.race_name} 本命: {result.honmei.get('horse_name', '')}",
            contents=FlexContainer.from_dict(container_dict),
        )

        push_req = PushMessageRequest(
            to=settings.line_target_user_id,
            messages=[flex_msg],
        )

        try:
            self._messaging_api.push_message(push_req)
            logger.info(f"LINE notification sent for {result.race_name}")
        except Exception as e:
            logger.error(f"Failed to send LINE notification: {e}")
            raise

    def send_text(self, text: str) -> None:
        """シンプルなテキストメッセージ送信（エラー通知用）。"""
        from linebot.v3.messaging import TextMessage

        push_req = PushMessageRequest(
            to=settings.line_target_user_id,
            messages=[TextMessage(text=text)],
        )
        try:
            self._messaging_api.push_message(push_req)
        except Exception as e:
            logger.error(f"Failed to send LINE text: {e}")
