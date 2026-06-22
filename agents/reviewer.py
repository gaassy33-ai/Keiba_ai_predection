"""
agents/reviewer.py - サブ③「振り返り君」
=====================================================
役割:
  - 当日の予想結果（馬連・軸馬流し方式）を実際の結果と照合
  - 的中率・ROI・失注パターン（軸飛び/ヒモ抜け）を計算
  - 改善点のヒントをルールベースで抽出
  - 出力: data/agents/YYYYMMDD_review.json + LINE通知

2026-06-22: 旧システム（単勝/馬単・honmei形式, docs/predictions_log.csv）から
新システム（馬連・軸馬流し方式, data/logs/predictions/）への参照に移行。

実行:
    python -m agents.reviewer [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentBase
from src.results.store import load_predictions, enrich


class Reviewer(AgentBase):
    """サブ③: 振り返り君 — 結果照合・精度分析（馬連・軸馬流し方式）"""

    name = "Reviewer"

    def run(self, target_date: date, dry_run: bool = False) -> dict:
        logger.info(f"Reviewer 起動: {target_date}")

        df = load_predictions(start_date=target_date, end_date=target_date)
        if df.empty:
            logger.warning("本日の予想が見つかりません")
            review = {"date": str(target_date), "status": "no_predictions"}
            self.save(review, self.review_file(target_date))
            return review

        n_races = df["race_id"].nunique()
        logger.info(f"本日の予想: {n_races} レース（{len(df)}点）")

        df = enrich(df, fetch_missing=True)
        analysis = self._analyze(df)

        review = {
            "generated_at": datetime.now().isoformat(),
            "date": str(target_date),
            "bets": df.to_dict(orient="records"),
            "analysis": analysis,
        }
        self.save(review, self.review_file(target_date))

        msg = self._build_line_message(analysis, target_date)
        self.send_line(msg, dry_run=dry_run)

        logger.info("振り返り完了")
        return review

    # ------------------------------------------------------------------ #
    # 精度分析
    # ------------------------------------------------------------------ #

    def _analyze(self, df) -> dict:
        ok = df[df["hit"].notna()].copy()
        if ok.empty:
            return {"buy_bets": len(df), "note": "結果未確定 or 照合不可（次回バッチで再試行）"}

        n_bets = len(ok)
        hits = int(ok["hit"].sum())
        cost = n_bets * 100
        payout = int(ok["payout"].sum())
        roi = payout / cost if cost else 0.0

        # レース単位の的中（そのレースの買い目のうち1点でも当たればレース的中）
        race_hit = ok.groupby("race_id")["hit"].max()
        race_n = len(race_hit)
        race_hits = int(race_hit.sum())

        # 失注パターン分析: 不的中レースを「軸は来たがヒモ抜け」vs「軸が飛んだ」に分解
        missed_races = race_hit[~race_hit].index
        miss_df = ok[ok["race_id"].isin(missed_races)]
        axis_survived = miss_df.groupby("race_id")["axis_in_top2"].max()
        partner_missed_races = int(axis_survived.sum())
        axis_missed_races = int((~axis_survived).sum())

        hints: list[str] = []
        if n_bets > 0 and hits / n_bets < 0.03:
            hints.append("的中率が非常に低い水準です。EV閾値・PairCalibratorの再検証を推奨。")
        if axis_missed_races > partner_missed_races:
            hints.append(
                f"不的中の主因は軸馬選定（軸が飛んだ{axis_missed_races}R "
                f"> ヒモ抜け{partner_missed_races}R）。Gatekeeper閾値の見直しを検討。"
            )
        elif partner_missed_races > 0:
            hints.append(
                f"軸は的中domainに残ったがヒモが抜けたレースが{partner_missed_races}R。"
                "partner_top_nの拡大を検討。"
            )

        return {
            "buy_bets": n_bets,
            "buy_races": race_n,
            "hit_bets": hits,
            "hit_bet_rate": round(hits / n_bets, 3) if n_bets else 0,
            "hit_races": race_hits,
            "hit_race_rate": round(race_hits / race_n, 3) if race_n else 0,
            "roi": round(roi, 3),
            "cost": cost,
            "payout": payout,
            "axis_missed_races": axis_missed_races,
            "partner_missed_races": partner_missed_races,
            "hints": hints,
        }

    # ------------------------------------------------------------------ #
    # LINE メッセージ生成
    # ------------------------------------------------------------------ #

    def _build_line_message(self, analysis: dict, target_date: date) -> str:
        n = analysis.get("buy_bets", 0)
        if n == 0 or "hit_bets" not in analysis:
            note = analysis.get("note", "買い対象なし or 結果取得失敗")
            return f"📋 振り返り君 ({target_date})\n{note}"

        lines = [
            f"📋 振り返り君レポート ({target_date})",
            "",
            "【本日の成績（馬連）】",
            f"  購入: {n}点（{analysis['buy_races']}R）  的中: {analysis['hit_bets']}点（{analysis['hit_races']}R）",
            f"  点数的中率: {analysis['hit_bet_rate']:.1%}　レース的中率: {analysis['hit_race_rate']:.1%}",
            f"  ROI: {analysis['roi']:.1%}　収支: {analysis['payout'] - analysis['cost']:+,}円",
            "",
            "【失注パターン】",
            f"  軸が飛んだ: {analysis['axis_missed_races']}R",
            f"  軸は残ったがヒモ抜け: {analysis['partner_missed_races']}R",
        ]

        hints = analysis.get("hints", [])
        if hints:
            lines.append("")
            lines.append("【改善ヒント】")
            for h in hints:
                lines.append(f"  ・{h}")

        return "\n".join(lines)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="振り返り君: レース結果照合・精度分析")
    parser.add_argument("--date", default=str(date.today()), help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="LINE 送信・結果取得しない")
    args = parser.parse_args()

    agent = Reviewer()
    agent.run(date.fromisoformat(args.date), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
