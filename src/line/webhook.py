"""
LINE Webhook サーバー（Flask）。

起動:
    python -m src.line.webhook

環境変数:
    LINE_CHANNEL_ACCESS_TOKEN  (必須)
    LINE_CHANNEL_SECRET        (必須)
    LINE_TARGET_USER_ID        (必須)
    WEBHOOK_BASE_URL           (任意) 例: https://your-app.railway.app
                               設定時は today_result で収支グラフ画像を追加送信

デプロイ先:
    Railway / Render / Fly.io (無料プランあり) / ngrok (開発用)
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
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

_IMG_DIR = Path("data/tmp_images")
_IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Flask ルート
# ---------------------------------------------------------------------------

@app.route("/img/<filename>")
def serve_image(filename: str):
    path = _IMG_DIR / filename
    if not path.exists() or not filename.endswith(".png"):
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


# ---------------------------------------------------------------------------
# ディスパッチ
# ---------------------------------------------------------------------------

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent) -> None:
    text = event.message.text.strip().lower()
    dispatch = {
        "main_race":    _handle_main_race,
        "today_result": _handle_today_result,
    }
    fn = dispatch.get(text)
    if fn:
        fn(event.reply_token)


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------

def _reply(reply_token: str, messages: list) -> None:
    config = Configuration(access_token=settings.line_channel_access_token)
    with ApiClient(config) as api_client:
        try:
            MessagingApi(api_client).reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=messages)
            )
        except Exception as e:
            logger.error(f"Reply failed: {e}")


# ---------------------------------------------------------------------------
# ① 今日のメインレース
# ---------------------------------------------------------------------------

def _handle_main_race(reply_token: str) -> None:
    """
    当日のメインレース（最大R番号）を特定し、AI予想 Flex Message を返信する。
    モデル未学習・スクレイピングエラー時はテキストでフォールバック。
    """
    from src.scraper.race_schedule import RaceScheduleFetcher, select_main_race

    try:
        # ── レーススケジュール取得 ─────────────────────────────────────
        with RaceScheduleFetcher() as fetcher:
            races = fetcher.fetch_race_list(date.today())
            races = fetcher.filter_by_jyo(races)

        if not races:
            _reply(reply_token, [TextMessage(text=(
                "🏇 本日の対象レースが見つかりませんでした。\n"
                "土日の開催日にご利用ください。"
            ))])
            return

        main_race  = select_main_race(races)
        race_id    = main_race.get("race_id", "")
        race_name  = main_race.get("race_name", "メインレース")
        start_time = main_race.get("start_time")
        deadline   = start_time.strftime("%H:%M") if start_time else "不明"

        # ── モデル存在確認 ─────────────────────────────────────────────
        if not settings.model_path.exists():
            _reply(reply_token, [TextMessage(text=(
                f"🏇 {race_name}\n"
                f"発走: {deadline}\n\n"
                "⚠️ モデルが見つかりません。\n"
                "Actions の「モデル週次再学習」を実行してください。"
            ))])
            return

        # ── 出走表取得 ─────────────────────────────────────────────────
        from src.scraper.netkeiba_scraper import NetkeibaScraper
        with NetkeibaScraper() as scraper:
            race_info = scraper.fetch_today_entries(race_id)
            pedigree_map: dict[str, dict] = {}
            recent_form_map: dict[str, dict] = {}
            for e in race_info.entries:
                if e.horse_id:
                    pedigree_map[e.horse_id] = scraper.fetch_horse_pedigree(e.horse_id)
                    recent_form_map[e.horse_id] = scraper.fetch_horse_recent_form(e.horse_id)

        entry_records = []
        for e in race_info.entries:
            ped  = pedigree_map.get(e.horse_id, {})
            form = recent_form_map.get(e.horse_id, {})
            entry_records.append({
                "horse_id":          e.horse_id,
                "horse_name":        e.horse_name,
                "horse_number":      e.horse_number,
                "frame_number":      e.frame_number,
                "sex":               getattr(e, "sex", ""),
                "age":               getattr(e, "age", 0),
                "jockey_id":         getattr(e, "jockey_id", ""),
                "jockey_name":       getattr(e, "jockey_name", ""),
                "weight_carried":    getattr(e, "weight_carried", 55),
                "father":            ped.get("father", ""),
                "mother_father":     ped.get("mother_father", ""),
                "recent_avg_pos":    form.get("recent_avg_pos", float("nan")),
                "recent_avg_last3f": form.get("recent_avg_last3f", float("nan")),
            })
        entry_df = pd.DataFrame(entry_records)

        # ── 特徴量エンジニアリング ─────────────────────────────────────
        from src.features.engineer import FeatureEngineer
        from config.settings import settings as _settings
        fe = FeatureEngineer.from_stats(_settings.stats_path)
        feature_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition_code=getattr(race_info, "ground_condition_code", "1"),
            weather_code=getattr(race_info, "weather_code", "1"),
        )

        # ── 予測 ───────────────────────────────────────────────────────
        from src.model.predictor import RacePredictor
        predictor = RacePredictor.from_saved_model()
        result = predictor.predict(race_id, race_name, feature_df)

        # ── 予想ログ記録 ───────────────────────────────────────────────
        log_prediction(
            race_id=race_id,
            race_name=race_name,
            honmei_num=result.honmei.get("horse_number", 0),
            honmei_name=result.honmei.get("horse_name", ""),
        )

        # ── 買い目生成 ─────────────────────────────────────────────────
        from src.betting.strategy import generate_betting_strategies
        top_horses = [h for h in [result.honmei, result.taikou, result.ana] if h]
        for h in top_horses:
            h.setdefault("win_odds", None)
        strategies = generate_betting_strategies(top_horses)

        # ── Flex Message 組み立て ──────────────────────────────────────
        from src.line.notifier import create_prediction_message
        race_data = {
            "race_name":        race_name,
            "race_id":          race_id,
            "course_type":      race_info.course_type,
            "distance":         race_info.distance,
            "weather":          getattr(race_info, "weather", "不明"),
            "ground_condition": getattr(race_info, "ground_condition", "不明"),
            "deadline":         deadline,
            "horses": [
                {
                    **h,
                    "tags":        [],
                    "jockey_name": h.get("jockey_name", ""),
                }
                for h in top_horses
            ],
            "strategies": strategies,
            "budget":     10_000,
        }
        container = create_prediction_message(race_data)
        honmei_name = result.honmei.get("horse_name", "")
        _reply(reply_token, [
            FlexMessage(
                alt_text=f"【AI予想】{race_name}  本命: {honmei_name}",
                contents=FlexContainer.from_dict(container),
            )
        ])

    except Exception as e:
        logger.exception(e)
        _reply(reply_token, [TextMessage(
            text=f"⚠️ 予想取得中にエラーが発生しました。\n{type(e).__name__}: {e}"
        )])


# ---------------------------------------------------------------------------
# ③ 今日の成績
# ---------------------------------------------------------------------------

def _handle_today_result(reply_token: str) -> None:
    """
    当日の成績をテキストサマリーで返信する。
    WEBHOOK_BASE_URL が設定されている場合は収支グラフ画像を追加送信。
    """
    today = date.today()
    messages: list = []

    # ── テキストサマリー（常に送信）────────────────────────────────────
    try:
        summary = _build_result_summary(today)
        messages.append(TextMessage(text=summary))
    except Exception as e:
        logger.exception(e)
        messages.append(TextMessage(text=f"⚠️ 成績データ取得エラー:\n{e}"))

    # ── 収支グラフ画像（WEBHOOK_BASE_URL 設定時のみ）────────────────────
    base_url = os.environ.get("WEBHOOK_BASE_URL", "").rstrip("/")
    if base_url:
        try:
            img_bytes = create_today_results_chart(today)
            filename  = f"result_{today.isoformat()}_{uuid.uuid4().hex[:8]}.png"
            (_IMG_DIR / filename).write_bytes(img_bytes)
            img_url = f"{base_url}/img/{filename}"
            messages.append(ImageMessage(
                original_content_url=img_url,
                preview_image_url=img_url,
            ))
        except Exception as e:
            logger.warning(f"Chart image skipped: {e}")

    _reply(reply_token, messages[:5])  # LINE は1返信5メッセージまで


def _build_result_summary(target_date: date) -> str:
    """predictions_log.csv から当日成績のテキストサマリーを生成する。"""
    from src.line.chart import PREDICTIONS_LOG

    header = f"📊 今日の成績  {target_date.strftime('%-m/%-d')}\n{'─' * 22}\n"

    if not PREDICTIONS_LOG.exists():
        return header + "まだ予想データがありません。\n土日のレース20分前に自動送信されます。"

    df = pd.read_csv(PREDICTIONS_LOG, parse_dates=["date"])
    today_df = df[df["date"].dt.date == target_date]

    if today_df.empty:
        return header + "本日の予想記録がありません。"

    lines = []
    for _, row in today_df.iterrows():
        hit     = bool(row.get("hit", False))
        payout  = int(row.get("payout", 0))
        net     = payout - 100
        mark    = "✅" if hit else "❌"
        amount  = f"¥{net:+,}" if hit else "¥-100"
        lines.append(f"{mark} {row['race_name'][:8]}  ◎{int(row['honmei_num'])} {row['honmei_name']}\n"
                     f"   {amount}")

    body = "\n".join(lines)

    # 集計
    hit_count   = int(today_df["hit"].sum())
    total_races = len(today_df)
    hit_rate    = hit_count / total_races * 100 if total_races else 0
    net_today   = int(today_df["payout"].sum()) - total_races * 100

    # 月間累計
    month_start = target_date.replace(day=1)
    df["date_only"] = df["date"].dt.date
    month_df  = df[df["date_only"] >= month_start]
    net_month = int(month_df["payout"].sum()) - len(month_df) * 100
    month_label = target_date.strftime("%-m月")

    footer = (
        f"\n{'─' * 22}\n"
        f"本日: {hit_count}/{total_races}的中 ({hit_rate:.0f}%)  {'🟢' if net_today >= 0 else '🔴'} ¥{net_today:+,}\n"
        f"{month_label}累計: {'🟢' if net_month >= 0 else '🔴'} ¥{net_month:+,}"
    )

    return header + body + footer


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting webhook server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
