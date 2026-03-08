"""
土日朝バッチ / 20分前通知 / 週末まとめレポートのエントリーポイント。

用途:
  --morning          朝5時バッチ: 開催日チェック → 予想ページ生成
  --notify           20分前通知: 開催日チェック → レース通知パイプライン
  --weekly-summary   日曜17時: 土日2日間のまとめ LINE 送信

GitHub Actions での早期終了（Early Exit）:
  --morning / --notify は先に is_jra_race_day() を呼び、
  非開催日の場合は sys.exit(0)（緑チェック）で即終了する。
  これにより Chrome セットアップ後のムダな処理を回避できる。

開催日チェック実装:
  netkeiba のレースリストページに race_id が含まれるか urllib で確認。
  JS レンダリング前のソースのため結果が不確かな場合は True（開催あり）にフォールバック。
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from datetime import date, timedelta

from loguru import logger

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────
# JRA 開催日チェック
# ─────────────────────────────────────────────────────────────────────

def is_jra_race_day(target_date: date | None = None) -> bool:
    """
    指定日に JRA レースが開催されているかを軽量チェックする。

    netkeiba のレースリストページの初期 HTML に race_id パターンが
    含まれているかで判定する（Selenium 不要）。

    Returns
    -------
    bool
        True  = 開催あり（または判定不能のためフォールバック）
        False = 開催なし（確実に race_id が見つからない場合のみ）
    """
    if target_date is None:
        target_date = date.today()

    url = (
        "https://race.netkeiba.com/top/race_list.html"
        f"?kaisai_date={target_date.strftime('%Y%m%d')}"
    )
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": settings.user_agent}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        has_races = bool(re.search(r'race_id=\d{12}', html))
        logger.info(
            f"JRA開催日チェック: {target_date} → "
            f"{'開催あり' if has_races else '開催なし（JSレンダリング必要な可能性あり → 続行）'}"
        )
        # JS レンダリングが必要な場合は race_id が見つからないことがある。
        # その場合 False にすると正常な開催日でもスキップされてしまうため、
        # 見つかったときだけ確定的に True を返し、
        # 見つからないときはレース一覧 URL の存在確認を行う。
        if has_races:
            return True

        # セカンダリチェック: レース一覧ページ自体が存在するか
        # （kaisai_date を持つ URL が 200 で返れば何らかのレースがある可能性）
        return _secondary_check(target_date)

    except Exception as e:
        logger.warning(f"開催日チェック失敗 ({type(e).__name__}: {e}) → 開催ありとして続行")
        return True


def _secondary_check(target_date: date) -> bool:
    """
    db.netkeiba.com の開催カレンダーで二次確認する。
    失敗時は True（フォールバック）。
    """
    try:
        url = f"https://db.netkeiba.com/?pid=race_list&date={target_date.strftime('%Y%m%d')}"
        req = urllib.request.Request(
            url, headers={"User-Agent": settings.user_agent}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        has_races = bool(re.search(r'/race/result/\d{12}', html))
        logger.info(f"セカンダリチェック: {target_date} → {'開催あり' if has_races else '開催なし'}")
        return has_races
    except Exception as e:
        logger.warning(f"セカンダリチェック失敗 ({e}) → 開催ありとして続行")
        return True


# ─────────────────────────────────────────────────────────────────────
# 週末まとめレポート
# ─────────────────────────────────────────────────────────────────────

def send_weekend_summary() -> None:
    """
    土日 2 日間の集計をテキストで LINE 送信する。

    呼び出しタイミング: 日曜 17:00 JST
    集計対象: 今週の土曜〜今日（日曜）の predictions_log.csv
    """
    import pandas as pd
    from src.line.chart import PREDICTIONS_LOG
    from src.line.notifier import LineNotifier

    today = date.today()
    # 今日が日曜（weekday=6）なら土曜は -1 日、それ以外は当日のみ
    sat = today - timedelta(days=1) if today.weekday() == 6 else today
    days = [sat, today] if sat != today else [today]

    sat_label   = sat.strftime("%-m/%-d")
    today_label = today.strftime("%-m/%-d")
    header = (
        f"📊 週末まとめ  {sat_label}(土)〜{today_label}(日)\n"
        f"{'─' * 24}\n"
    )

    if not PREDICTIONS_LOG.exists():
        LineNotifier().send_text(header + "今週の予想データがありません。")
        return

    df = pd.read_csv(PREDICTIONS_LOG, parse_dates=["date"])
    df["date_only"] = df["date"].dt.date
    weekend_df = df[df["date_only"].isin(days)]

    if weekend_df.empty:
        LineNotifier().send_text(header + "今週の予想データがありません。")
        return

    # ── 日別サマリー行 ──────────────────────────────────────────────
    lines: list[str] = []
    for day in days:
        day_df = weekend_df[weekend_df["date_only"] == day]
        if day_df.empty:
            continue
        hits  = int(day_df["hit"].sum())
        total = len(day_df)
        net   = int(day_df["payout"].sum()) - total * 100
        dow   = "(土)" if day.weekday() == 5 else "(日)"
        color = "🟢" if net >= 0 else "🔴"
        lines.append(
            f"{day.strftime('%-m/%-d')}{dow}  {hits}/{total}的中  {color} ¥{net:+,}"
        )

    # ── 土日合計 ─────────────────────────────────────────────────────
    total_hits   = int(weekend_df["hit"].sum())
    total_races  = len(weekend_df)
    total_payout = int(weekend_df["payout"].sum())
    total_cost   = total_races * 100
    total_net    = total_payout - total_cost
    hit_rate     = total_hits / total_races * 100 if total_races else 0
    roi          = total_payout / total_cost * 100 if total_cost else 0

    footer = (
        f"\n{'─' * 24}\n"
        f"【土日合計】\n"
        f"  的中: {total_hits}/{total_races}R  ({hit_rate:.0f}%)\n"
        f"  収支: {'🟢' if total_net >= 0 else '🔴'} ¥{total_net:+,}\n"
        f"  回収率: {roi:.0f}%"
    )

    text = header + "\n".join(lines) + footer
    LineNotifier().send_text(text)
    logger.info("週末まとめレポートを送信しました。")


# ─────────────────────────────────────────────────────────────────────
# ロガー設定
# ─────────────────────────────────────────────────────────────────────

def _setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="1 day",
        retention="7 days",
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _setup_logger()

    parser = argparse.ArgumentParser(description="競馬 土日バッチ処理")
    parser.add_argument("--morning",        action="store_true",
                        help="朝バッチ: 開催日チェック → 予想ページ生成")
    parser.add_argument("--notify",         action="store_true",
                        help="20分前通知: 開催日チェック → レース通知パイプライン")
    parser.add_argument("--weekly-summary", action="store_true",
                        help="週末まとめ: 土日2日間の成績をLINE送信")
    args = parser.parse_args()

    # ── 開催日チェック（--weekly-summary は除く）──────────────────────
    if args.morning or args.notify:
        if not is_jra_race_day():
            logger.info("本日は JRA 非開催日です。処理を終了します。")
            sys.exit(0)   # exit 0 = GitHub Actions では緑チェック（正常終了）

    if args.morning:
        logger.info("朝バッチ開始（予想ページ生成）")
        from src.scheduler.runner import run_morning_pages
        run_morning_pages()

    elif args.notify:
        logger.info("20分前通知パイプライン開始")
        from src.scheduler.runner import run_once_for_date
        run_once_for_date(date.today())

    elif args.weekly_summary:
        logger.info("週末まとめレポート送信開始")
        send_weekend_summary()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
