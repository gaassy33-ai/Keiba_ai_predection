"""
odds_notify.py
発走30分前にオッズを取得してEVフィルタを適用し、LINEに通知するスクリプト。

実行方法:
    python odds_notify.py              # 当日・常駐ループ
    python odds_notify.py --dry-run    # LINE送信なし（ターミナル確認用）
    python odds_notify.py --date 2026-03-29  # 日付指定

動作フロー:
    1. 起動時に当日レーススケジュールを取得（発走時刻付き）
    2. 1分おきにループ: 発走25〜35分前のレースを検出
    3. 対象レースについてオッズ付き出走表を取得し予測を実行
    4. EVフィルタ（model_prob × odds ≥ 1.05）を適用
    5. 条件を満たすレースのみ LINE Flex Message を送信
    6. 同一レースへの重複通知を防止

環境変数 (.env):
    LINE_CHANNEL_ACCESS_TOKEN
    LINE_CHANNEL_SECRET
    LINE_TARGET_USER_ID
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from src.scraper.netkeiba_scraper import NetkeibaScraper
from src.scraper.race_schedule import RaceScheduleFetcher
from config.settings import settings

# ======================================================================
# 定数
# ======================================================================
# 発走X分前に通知する（25〜35分前のレースを対象）
NOTIFY_BEFORE_MIN  = 30      # 中心時間
NOTIFY_WINDOW_MIN  = 5       # ±5分のウィンドウ
CHECK_INTERVAL_SEC = 60      # ループ間隔（秒）

# 予測フィルタ（daily_batch.py と統一）
MIN_HONMEI_PROB   = 0.15
MIN_GAP           = 0.05
EV_THRESHOLD      = 1.05     # model_prob × odds ≥ 1.05

# NAR/JRA 判定（平日→NAR, 土日→JRA）
def get_org(d: date) -> str:
    return "nar" if d.weekday() < 5 else "jra"

# ======================================================================
# ロガー
# ======================================================================
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/odds_notify.log", level="DEBUG", format=_fmt, rotation="20 MB")


# ======================================================================
# LINE 送信（1レース用シンプルメッセージ）
# ======================================================================

def _send_line_message(msg: dict, dry_run: bool = False) -> None:
    """LINE Messaging API で Push Message を送信する"""
    if dry_run:
        import json
        logger.info("[DRY-RUN] LINE 送信メッセージ:")
        logger.info(json.dumps(msg, ensure_ascii=False, indent=2))
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.line_channel_access_token}",
    }
    payload = {
        "to": settings.line_target_user_id,
        "messages": [msg],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        logger.error(f"LINE 送信失敗: {resp.status_code} {resp.text}")
    else:
        logger.info(f"LINE 送信完了 (status={resp.status_code})")


def build_race_notify_flex(
    race_info_dict: dict,
    result: dict,
    start_time: datetime,
) -> dict:
    """
    1レース分の通知用 Flex Message を構築する。

    Parameters
    ----------
    race_info_dict : dict
        RaceScheduleFetcher.fetch_race_list() の1要素
    result : dict
        predict_single_race() の戻り値
    start_time : datetime
        発走時刻
    """
    race_name  = race_info_dict.get("race_name") or f"{result['race_num']}"
    jyo_name   = race_info_dict.get("jyo_name", "")
    start_str  = start_time.strftime("%H:%M")
    mark       = result["mark"]
    hon_num    = result["honmei_num"]
    hon_name   = result["honmei_name"]
    hon_prob   = result["honmei_prob"]
    t_prob     = result.get("taikou_prob", 0.0)
    gap        = result.get("gap", 0.0)
    hon_odds   = result.get("honmei_odds", float("nan"))
    hon_ev     = result.get("honmei_ev", float("nan"))

    color = {
        "◎": "#1a5533",
        "○": "#7a5500",
    }.get(mark, "#888888")

    odds_str = f"オッズ {hon_odds:.1f}倍" if not np.isnan(hon_odds) else "オッズ未取得"
    ev_str   = f"EV={hon_ev:.2f}" if not np.isnan(hon_ev) else "EV=--"
    prob_str = f"確率 {hon_prob:.1%}  差+{gap:.1%}  対抗{t_prob:.1%}"

    baren  = result.get("baren_partners", [])
    baren_str  = f"馬連: {hon_num}-{' / '.join(baren)}" if baren else "馬連: なし"

    um_part = result.get("umatan_partners", [])
    umatan_str = "馬単: " + " / ".join(f"{hon_num}→{p}" for p in um_part) if um_part else "馬単: なし"

    sf_combos = result.get("sanrenfuku_combos", [])
    sf_str = (
        "3連複: " + " / ".join(f"{hon_num}-{a}-{b}" for a, b in sf_combos)
        if sf_combos else "3連複: なし"
    )
    st_combos = result.get("sanrentan_combos", [])
    st_str = (
        "3連単: " + " / ".join(f"{hon_num}→{n2}→{n3}" for n2, n3 in st_combos)
        if st_combos else "3連単: なし"
    )

    contents: list[dict] = [
        {"type": "text", "text": f"{mark} {hon_num} {hon_name}",
         "weight": "bold", "size": "lg", "color": color, "wrap": True},
        {"type": "text", "text": prob_str,
         "size": "xs", "color": "#666666", "margin": "xs", "wrap": True},
        {"type": "text", "text": f"{odds_str}  {ev_str}",
         "size": "xs", "color": "#888888", "margin": "xs"},
        {"type": "separator", "margin": "sm"},
        {"type": "text", "text": baren_str,
         "size": "xs", "color": "#555555", "margin": "xs", "wrap": True},
        {"type": "text", "text": umatan_str,
         "size": "xs", "color": "#7a5500", "margin": "xs", "wrap": True},
    ]
    if sf_combos:
        contents.append({"type": "text", "text": sf_str,
                         "size": "xs", "color": "#4a4a8a", "margin": "xs", "wrap": True})
    if st_combos:
        contents.append({"type": "text", "text": st_str,
                         "size": "xs", "color": "#7a3a3a", "margin": "xs", "wrap": True})

    bubble = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": color,
            "paddingAll": "14px",
            "contents": [
                {"type": "text",
                 "text": f"🔔 発走{start_str} {jyo_name} {result['race_num']}",
                 "color": "#ffffff", "size": "sm", "weight": "bold"},
                {"type": "text",
                 "text": race_name,
                 "color": "#ffffffcc", "size": "xs", "margin": "xs", "wrap": True},
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "14px",
            "contents": contents,
        },
    }

    return {
        "type": "flex",
        "altText": f"🔔 {start_str} {jyo_name}{result['race_num']} {mark}{hon_name} {odds_str}",
        "contents": bubble,
    }


# ======================================================================
# 1レース予測（daily_batch.predict_and_bet と同等ロジック）
# ======================================================================

def predict_single_race(
    race_info,  # src.scraper.base_scraper.RaceInfo
    fe: FeatureEngineer,
    trainer: ModelTrainer,
    org: str = "jra",
) -> dict | None:
    """
    1レースの予測と買い目を生成する。
    daily_batch.predict_and_bet() と同じロジックを使用し、
    honmei_odds / honmei_ev を追加して返す。

    Returns None if the race should be skipped entirely (insufficient data).
    Returns dict with result including 'mark' (◎/○/△).
    """
    from daily_batch import predict_and_bet
    result = predict_and_bet(race_info, fe, trainer, org=org)
    if result is None:
        return None

    # honmei_odds / honmei_ev を補完（通知メッセージ表示用）
    hon_num = result.get("honmei_num", "")
    odds_map = {e.horse_number: e.odds for e in race_info.entries if e.odds}
    try:
        raw_odds = odds_map.get(int(hon_num))
        honmei_odds = float(raw_odds) if raw_odds is not None else float("nan")
    except Exception:
        honmei_odds = float("nan")

    honmei_prob = result.get("honmei_prob", 0.0)
    honmei_ev = (honmei_prob * honmei_odds
                 if not np.isnan(honmei_odds) else float("nan"))

    result["honmei_odds"] = honmei_odds
    result["honmei_ev"]   = honmei_ev
    return result


# ======================================================================
# スケジュール取得・ループ
# ======================================================================

def run_notify_loop(
    target_date: date,
    dry_run: bool = False,
) -> None:
    """
    発走30分前通知の常駐ループ。

    1. 当日のレーススケジュールを取得
    2. 60秒ごとに発走25〜35分前のレースを検出
    3. 対象レースを予測 → EVフィルタ → LINE通知
    """
    org = get_org(target_date)
    logger.info("=" * 60)
    logger.info(f"odds_notify 起動: {target_date}  [org={org.upper()}]")
    logger.info(f"  EV閾値={EV_THRESHOLD}  通知タイミング=発走{NOTIFY_BEFORE_MIN}分前±{NOTIFY_WINDOW_MIN}分")
    logger.info("=" * 60)

    # ── モデル・FeatureEngineer 読み込み ────────────────────────
    model_path = settings.nar_model_path if org == "nar" else settings.model_path
    stats_path  = settings.nar_stats_path  if org == "nar" else settings.stats_path

    if org == "nar" and not model_path.exists():
        logger.warning("NAR モデルが見つかりません。JRA モデルで代替します。")
        model_path = settings.model_path
        stats_path  = settings.stats_path

    logger.info("[1/3] モデル読み込み中...")
    trainer = ModelTrainer.load(model_path, org=org)

    logger.info("[2/3] FeatureEngineer 読み込み中...")
    fe = FeatureEngineer.from_stats(stats_path)

    # ── スケジュール取得 ────────────────────────────────────────
    logger.info("[3/3] レーススケジュール取得中...")
    schedule: list[dict] = []
    with RaceScheduleFetcher() as fetcher:
        all_races = fetcher.fetch_race_list(target_date)
        # JRA/NAR の対象会場に絞る
        schedule = fetcher.filter_by_jyo(all_races) if settings.target_jyo_code_list else all_races

    if not schedule:
        logger.warning("対象レースが見つかりません。終了します。")
        return

    logger.info(f"取得レース数: {len(schedule)} レース")
    for r in schedule:
        logger.info(f"  {r['race_id']}  {r.get('jyo_name','')} {r['race_number']}R  "
                    f"発走: {r['start_time'].strftime('%H:%M')}  {r.get('race_name','')}")

    # ── 通知済み管理 ────────────────────────────────────────────
    notified:   set[str] = set()   # 通知済み race_id
    checked:    set[str] = set()   # 確認試行済み race_id（エラー含む）

    # start_time が 00:00（未取得）のレースは除外
    valid_schedule = [r for r in schedule if r["start_time"].hour != 0]
    if len(valid_schedule) < len(schedule):
        skipped_n = len(schedule) - len(valid_schedule)
        logger.warning(f"  発走時刻不明のレースを {skipped_n} 件除外")
        schedule = valid_schedule

    if not schedule:
        logger.warning("有効なレーススケジュールがありません。終了します。")
        return

    last_race_time = max(r["start_time"] for r in schedule)

    # ── メインループ ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("監視ループ開始（Ctrl+C で終了）")
    logger.info("=" * 60)

    with NetkeibaScraper() as scraper:
        while True:
            now = datetime.now()

            # 全レース終了後（最終発走+30分）は終了
            if now > last_race_time + timedelta(minutes=30):
                logger.info("全レース終了。odds_notify を終了します。")
                break

            # 発走25〜35分前のレースを検出
            lower = timedelta(minutes=NOTIFY_BEFORE_MIN - NOTIFY_WINDOW_MIN)
            upper = timedelta(minutes=NOTIFY_BEFORE_MIN + NOTIFY_WINDOW_MIN)

            targets = [
                r for r in schedule
                if r["race_id"] not in notified
                and r["race_id"] not in checked
                and lower <= (r["start_time"] - now) <= upper
            ]

            if targets:
                for race_sched in targets:
                    race_id    = race_sched["race_id"]
                    start_time = race_sched["start_time"]
                    remaining  = int((start_time - now).total_seconds() / 60)
                    logger.info(
                        f"[発走{remaining}分前] {race_id}  "
                        f"{race_sched.get('jyo_name','')} {race_sched['race_number']}R  "
                        f"発走: {start_time.strftime('%H:%M')}"
                    )

                    checked.add(race_id)

                    try:
                        # オッズ付き出走表を取得
                        logger.info(f"  出走表取得中: {race_id}")
                        race_info = scraper.fetch_today_entries(race_id)

                        if race_info is None:
                            logger.warning(f"  出走表取得失敗: {race_id}")
                            continue

                        # 予測実行
                        result = predict_single_race(race_info, fe, trainer, org=org)
                        if result is None:
                            logger.info(f"  スキップ（データ不足）: {race_id}")
                            continue

                        mark       = result.get("mark", "△")
                        hon_name   = result["honmei_name"]
                        hon_prob   = result["honmei_prob"]
                        hon_ev     = result.get("honmei_ev", float("nan"))
                        is_buy     = result["is_buy"]

                        ev_disp = f"{hon_ev:.2f}" if not np.isnan(hon_ev) else "--"
                        logger.info(
                            f"  予測: {mark} {hon_name}  確率={hon_prob:.1%}  "
                            f"EV={ev_disp}  is_buy={is_buy}"
                        )

                        if not is_buy:
                            logger.info(f"  フィルタ不通過（△）: {race_id} → 通知しません")
                            notified.add(race_id)  # △も送信済みとして記録して重複防止
                            continue

                        # ── LINE 通知 ──────────────────────────────
                        flex_msg = build_race_notify_flex(race_sched, result, start_time)
                        _send_line_message(flex_msg, dry_run=dry_run)
                        notified.add(race_id)
                        logger.info(f"  ✅ 通知完了: {race_id}  {mark}{hon_name}")

                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        logger.exception(f"  エラー ({race_id}): {exc}")
                        # エラー時は checked に残し、次のループで再試行しない
                        # → 再試行したい場合は checked.discard(race_id)
            else:
                # 直近の未通知レースを表示（デバッグ用）
                upcoming = [
                    r for r in schedule
                    if r["race_id"] not in notified
                    and r["start_time"] > now
                ]
                if upcoming:
                    next_r = min(upcoming, key=lambda r: r["start_time"])
                    mins_left = int((next_r["start_time"] - now).total_seconds() / 60)
                    logger.debug(
                        f"次の対象: {next_r['race_id']}  "
                        f"発走まで {mins_left} 分  "
                        f"通知タイミングまで {max(0, mins_left - (NOTIFY_BEFORE_MIN + NOTIFY_WINDOW_MIN))} 分"
                    )
                else:
                    logger.debug("未通知レースなし")

            time.sleep(CHECK_INTERVAL_SEC)


# ======================================================================
# エントリーポイント
# ======================================================================

def _check_settings(dry_run: bool) -> bool:
    """
    起動前に必須設定の確認を行う。
    問題があれば警告を表示して False を返す（dry-run では警告のみ）。
    """
    ok = True
    placeholder = "placeholder_fill_before_run"

    if not dry_run:
        missing = []
        if not settings.line_channel_access_token or settings.line_channel_access_token == placeholder:
            missing.append("LINE_CHANNEL_ACCESS_TOKEN")
        if not settings.line_target_user_id or settings.line_target_user_id == placeholder:
            missing.append("LINE_TARGET_USER_ID")

        if missing:
            logger.error("=" * 60)
            logger.error("❌ LINE 認証情報が未設定です。LINE通知が送信できません。")
            logger.error("")
            logger.error("【設定手順】")
            logger.error("  1. https://developers.line.biz/console/ にアクセス")
            logger.error("  2. 対象チャンネル → Messaging API タブを開く")
            logger.error("  3. 「チャンネルアクセストークン（長期）」を発行・コピー")
            logger.error("  4. keiba-prediction/.env を開いて以下を本物の値に書き換える:")
            for key in missing:
                logger.error(f"     {key}=<実際のトークンをここに貼り付け>")
            logger.error("")
            logger.error("  ※ LINE_TARGET_USER_ID は LINEアプリ→設定→プロフィール")
            logger.error("     またはWebhook経由で確認してください (Uで始まる文字列)")
            logger.error("=" * 60)
            logger.error("  --dry-run オプションを付けると通知なしで動作確認できます:")
            logger.error("  python odds_notify.py --dry-run")
            logger.error("=" * 60)
            ok = False
    else:
        # dry-run でも設定状況を確認して警告表示
        if settings.line_channel_access_token == placeholder:
            logger.warning("⚠️  [DRY-RUN] LINE_CHANNEL_ACCESS_TOKEN が未設定です（dry-run なので送信はしません）")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="発走30分前にオッズを取得してLINE通知するスクリプト"
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="対象日 (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LINE 送信をスキップしてターミナルに出力するだけ",
    )
    args = parser.parse_args()
    target_date = date.fromisoformat(args.date)

    if not _check_settings(dry_run=args.dry_run):
        logger.error("設定を修正してから再実行してください。")
        sys.exit(1)

    try:
        run_notify_loop(target_date, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Ctrl+C で終了しました。")


if __name__ == "__main__":
    main()
