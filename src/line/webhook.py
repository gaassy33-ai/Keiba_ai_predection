"""
LINE Webhook サーバー（Flask）。

起動:
    python -m src.line.webhook

必要な環境変数（.env または GitHub Secrets）:
    LINE_CHANNEL_ACCESS_TOKEN
    LINE_CHANNEL_SECRET
    WEBHOOK_BASE_URL   例: https://your-app.railway.app
                       ※ 収支グラフ画像の配信 URL として使用

デプロイ先の候補:
    Railway / Render / Fly.io (無料プランあり)
    Ngrok (ローカル開発用)
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from pathlib import Path

from flask import Flask, abort, request, send_file
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    FlexContainer,
    FlexMessage,
    ImageMessage,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from loguru import logger

from config.settings import settings
from src.line.chart import create_today_results_chart, log_prediction

app = Flask(__name__)
handler = WebhookHandler(settings.line_channel_secret)

# 一時画像保存ディレクトリ
_IMG_DIR = Path("data/tmp_images")
_IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Flask ルート
# ---------------------------------------------------------------------------

@app.route("/img/<filename>")
def serve_image(filename: str):
    """一時生成画像を HTTPS で公開するエンドポイント。"""
    path = _IMG_DIR / filename
    if not path.exists() or not filename.endswith(".png"):
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    logger.debug(f"Webhook received: {body[:200]}")
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature")
        abort(400)
    return "OK"


# ---------------------------------------------------------------------------
# メッセージハンドラー
# ---------------------------------------------------------------------------

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent) -> None:
    text = event.message.text.strip().lower()
    reply_token = event.reply_token

    dispatch = {
        "main_race":    _handle_main_race,
        "today_result": _handle_today_result,
        "help_config":  _handle_help,
    }

    handler_fn = dispatch.get(text)
    if handler_fn:
        handler_fn(reply_token)


# ---------------------------------------------------------------------------
# 各ボタンの応答ロジック
# ---------------------------------------------------------------------------

def _reply(reply_token: str, messages: list) -> None:
    config = Configuration(access_token=settings.line_channel_access_token)
    with ApiClient(config) as api_client:
        api = MessagingApi(api_client)
        try:
            api.reply_message(ReplyMessageRequest(
                reply_token=reply_token,
                messages=messages,
            ))
        except Exception as e:
            logger.error(f"Reply failed: {e}")


# ── ① 今日のメインレース ────────────────────────────────────────────────────

def _handle_main_race(reply_token: str) -> None:
    """
    当日のメインレース（最大レース番号）を特定し、
    予想 Flex Message を返信する。
    """
    from datetime import datetime
    from src.scraper.race_schedule import RaceScheduleFetcher
    from src.line.notifier import create_prediction_message
    from src.betting.strategy import generate_betting_strategies

    try:
        fetcher = RaceScheduleFetcher()
        races = fetcher.fetch_race_list(date.today())
        races = fetcher.filter_by_jyo(races)

        if not races:
            _reply(reply_token, [TextMessage(text="本日の対象レースが見つかりませんでした。")])
            return

        # レース番号が最大のものをメインレースとして選択
        main_race = max(races, key=lambda r: int(r.get("race_number", 0)))
        race_id   = main_race.get("race_id", "")
        race_name = main_race.get("race_name", "メインレース")
        start_time = main_race.get("start_time")

        # --- 出走表・特徴量・予測 ---
        # ここでは軽量版（モデルが利用可能な場合のみ予想を実行）
        try:
            from src.model.predictor import RacePredictor
            from src.features.engineer import FeatureEngineer
            import pandas as pd

            predictor = RacePredictor.from_saved_model()

            # 出走表取得（簡易）
            from src.scraper.netkeiba_scraper import NetkeibaScraper
            with NetkeibaScraper() as scraper:
                race_info = scraper.fetch_today_entries(race_id)

            entry_records = [{
                "horse_id":      e.horse_id,
                "horse_name":    e.horse_name,
                "horse_number":  e.horse_number,
                "frame_number":  e.frame_number,
                "jockey_name":   e.jockey_name,
                "weight_carried": e.weight_carried,
            } for e in race_info.entries]
            entry_df = pd.DataFrame(entry_records)

            fe = FeatureEngineer(pd.DataFrame())
            feature_df = fe.build_entry_features(
                entry_df=entry_df,
                course_type=race_info.course_type,
                distance=race_info.distance,
                ground_condition_code="1",
                weather_code="1",
            )

            result = predictor.predict(race_id, race_name, feature_df)

            # 予想ログに記録
            log_prediction(
                race_id=race_id,
                race_name=race_name,
                honmei_num=result.honmei.get("horse_number", 0),
                honmei_name=result.honmei.get("horse_name", ""),
            )

            # 買い目生成
            horses_for_strategy = [
                {**h, "win_odds": None}
                for h in [result.honmei, result.taikou, result.ana]
                if h
            ]
            strategies = generate_betting_strategies(horses_for_strategy)

            # Flex Message 組み立て
            deadline = start_time.strftime("%H:%M") if start_time else ""
            race_data = {
                "race_name":        race_name,
                "race_id":          race_id,
                "course_type":      race_info.course_type,
                "distance":         race_info.distance,
                "weather":          "不明",
                "ground_condition": "不明",
                "deadline":         deadline,
                "horses": [
                    {**h, "tags": [], "jockey_name": h.get("jockey_name", "")}
                    for h in [result.honmei, result.taikou, result.ana]
                    if h
                ],
                "strategies": strategies,
                "budget": 10_000,
            }
            from src.line.notifier import create_prediction_message
            container = create_prediction_message(race_data)
            flex_msg = FlexMessage(
                alt_text=f"【AI予想】{race_name}",
                contents=FlexContainer.from_dict(container),
            )
            _reply(reply_token, [flex_msg])

        except FileNotFoundError:
            # モデル未学習 → テキストで概要のみ返信
            deadline_str = start_time.strftime("%H:%M") if start_time else "未定"
            _reply(reply_token, [TextMessage(
                text=(
                    f"🏇 {race_name}\n"
                    f"発走: {deadline_str}\n\n"
                    "※ モデルファイルが見つかりません。\n"
                    "先に keiba-train を実行してください。"
                )
            )])

    except Exception as e:
        logger.exception(e)
        _reply(reply_token, [TextMessage(text=f"エラーが発生しました:\n{e}")])


# ── ③ 今日の成績 ────────────────────────────────────────────────────────────

def _handle_today_result(reply_token: str) -> None:
    """
    当日の収支グラフを PNG 画像として返信する。
    画像は Flask の /img/<filename> エンドポイント経由で提供する。

    必要: 環境変数 WEBHOOK_BASE_URL に公開 HTTPS URL を設定。
    """
    base_url = os.environ.get("WEBHOOK_BASE_URL", "").rstrip("/")
    if not base_url:
        _reply(reply_token, [TextMessage(
            text=(
                "⚠️ WEBHOOK_BASE_URL が未設定です。\n"
                "環境変数に公開 URL を設定してください。\n"
                "例: https://your-app.railway.app"
            )
        )])
        return

    try:
        img_bytes = create_today_results_chart(date.today())
        filename  = f"result_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}.png"
        (_IMG_DIR / filename).write_bytes(img_bytes)

        img_url = f"{base_url}/img/{filename}"
        logger.info(f"Chart image URL: {img_url}")

        _reply(reply_token, [
            ImageMessage(
                original_content_url=img_url,
                preview_image_url=img_url,
            )
        ])

    except Exception as e:
        logger.exception(e)
        _reply(reply_token, [TextMessage(text=f"グラフ生成エラー:\n{e}")])


# ── ④ ヘルプ/設定 ────────────────────────────────────────────────────────────

def _handle_help(reply_token: str) -> None:
    help_text = (
        "【馬券AI予想  使い方ガイド】\n"
        "─────────────────\n"
        "① 今日のメインレース\n"
        "   本日のメインレース（11R等）の\n"
        "   AI予想と推奨買い目を表示します。\n\n"
        "② 開催スケジュール\n"
        "   netkeiba の当日開催一覧を開きます。\n\n"
        "③ 今日の成績\n"
        "   本日の的中状況と月間累計収支を\n"
        "   グラフで表示します。\n\n"
        "④ ヘルプ/設定\n"
        "   このメッセージを表示します。\n\n"
        "⑤ お問い合わせ/SNS\n"
        "   開発者の X アカウントを開きます。\n"
        "─────────────────\n"
        "🔔 予想は土日のレース20分前に自動送信\n"
        "📌 予想は参考情報です。\n"
        "   馬券購入は自己責任でお願いします。"
    )
    _reply(reply_token, [TextMessage(text=help_text)])


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting webhook server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
