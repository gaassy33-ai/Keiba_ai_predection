"""
agents/reviewer.py - サブ③「振り返り君」
=====================================================
役割:
  - 当日の予想結果を取得して実際の結果と照合
  - 的中率・ROI・ミス分析を計算
  - 改善点のヒントをルールベースで抽出
  - 出力: data/agents/YYYYMMDD_review.json + LINE通知

実行:
    python -m agents.reviewer [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentBase

PREDICTIONS_LOG = ROOT / "docs" / "predictions_log.csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}


class Reviewer(AgentBase):
    """サブ③: 振り返り君 — 結果照合・精度分析"""

    name = "Reviewer"

    def run(self, target_date: date, dry_run: bool = False) -> dict:
        logger.info(f"Reviewer 起動: {target_date}")

        # 本日の予想を取得
        predictions = self._load_today_predictions(target_date)
        if not predictions:
            logger.warning("本日の予想が見つかりません")
            review = {"date": str(target_date), "status": "no_predictions"}
            self.save(review, self.review_file(target_date))
            return review

        logger.info(f"本日の予想: {len(predictions)} レース（うち買い: {sum(1 for p in predictions if p.get('is_buy'))}件）")

        # 結果を取得・照合
        results = self._fetch_and_match(predictions)

        # 分析
        analysis = self._analyze(results)

        # predictions_log.csv を更新
        if not dry_run:
            self._update_predictions_log(results, target_date)

        review = {
            "generated_at": datetime.now().isoformat(),
            "date": str(target_date),
            "predictions": results,
            "analysis": analysis,
        }

        self.save(review, self.review_file(target_date))

        # LINE 通知
        msg = self._build_line_message(analysis, target_date)
        self.send_line(msg, dry_run=dry_run)

        logger.info("振り返り完了")
        return review

    # ------------------------------------------------------------------ #
    # 予想データ読み込み
    # ------------------------------------------------------------------ #

    def _load_today_predictions(self, target_date: date) -> list[dict]:
        if not PREDICTIONS_LOG.exists():
            return []
        df = pd.read_csv(PREDICTIONS_LOG, dtype=str)
        df = df[df["date"] == str(target_date)]
        return df.to_dict(orient="records")

    # ------------------------------------------------------------------ #
    # 結果取得（Netkeiba スクレイピング）
    # ------------------------------------------------------------------ #

    def _fetch_result(self, race_id: str) -> dict | None:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.warning(f"結果取得失敗 {race_id}: {e}")
            return None

        table = soup.select_one("table.RaceTable01")
        if not table:
            return None

        finish_order: dict[str, int] = {}
        for tr in table.select("tr")[1:]:
            tds = tr.select("td")
            if len(tds) < 4:
                continue
            try:
                pos = int(tds[0].get_text(strip=True))
                num = tds[2].get_text(strip=True)
                finish_order[num] = pos
            except (ValueError, IndexError):
                continue

        # 単勝払戻
        tansho_ret = None
        for row in soup.select(".Payout_Detail_Table tr"):
            cells = row.select("td")
            if not cells:
                continue
            label = row.select_one("th")
            if label and "単勝" in label.get_text():
                try:
                    num_cell = cells[0].get_text(strip=True)
                    pay_cell = cells[1].get_text(strip=True).replace(",", "")
                    tansho_ret = int(pay_cell)
                    tansho_winner = num_cell
                except Exception:
                    pass

        return {"finish_order": finish_order, "tansho_ret": tansho_ret}

    def _fetch_and_match(self, predictions: list[dict]) -> list[dict]:
        results = []
        buy_preds = [p for p in predictions if p.get("is_buy") == "True"]

        for pred in buy_preds:
            race_id    = pred.get("race_id", "")
            race_name  = pred.get("race_name", "")
            hon_num    = str(pred.get("honmei_num", ""))
            hon_name   = pred.get("honmei_name", "")
            hon_prob   = float(pred.get("honmei_prob", 0) or 0)
            mark       = pred.get("mark", "△")

            logger.info(f"  結果取得: {race_id} {race_name}")
            time.sleep(1.5)

            result_data = self._fetch_result(race_id)
            if result_data is None:
                results.append({
                    "race_id": race_id,
                    "race_name": race_name,
                    "honmei_num": hon_num,
                    "honmei_name": hon_name,
                    "honmei_prob": hon_prob,
                    "mark": mark,
                    "status": "fetch_failed",
                })
                continue

            finish = result_data["finish_order"]
            actual_pos = finish.get(hon_num, 99)
            tansho_hit = actual_pos == 1
            tansho_ret = result_data["tansho_ret"] if tansho_hit else 0

            # 馬単照合
            umatan_str = pred.get("umatan_str", "")
            umatan_hit = False
            umatan_ret_val = 0
            if umatan_str:
                for combo in umatan_str.split(","):
                    parts = combo.strip().split("→")
                    if len(parts) == 2:
                        n1, n2 = parts[0].strip(), parts[1].strip()
                        if finish.get(n1) == 1 and finish.get(n2) == 2:
                            umatan_hit = True

            results.append({
                "race_id":    race_id,
                "race_name":  race_name,
                "honmei_num": hon_num,
                "honmei_name": hon_name,
                "honmei_prob": hon_prob,
                "mark":       mark,
                "actual_pos": actual_pos,
                "tansho_hit": tansho_hit,
                "tansho_ret": tansho_ret,
                "umatan_hit": umatan_hit,
                "status": "ok",
            })

        return results

    # ------------------------------------------------------------------ #
    # 精度分析
    # ------------------------------------------------------------------ #

    def _analyze(self, results: list[dict]) -> dict:
        ok = [r for r in results if r.get("status") == "ok"]
        if not ok:
            return {"buy_races": 0, "note": "照合可能なレースなし"}

        n = len(ok)
        hits = sum(1 for r in ok if r["tansho_hit"])
        roi  = sum(r["tansho_ret"] for r in ok) / (n * 100) if n else 0

        # ミス分析: 実際の着順が2〜3着（惜しい）
        near_miss = [r for r in ok if r["actual_pos"] in (2, 3)]
        upset     = [r for r in ok if r["actual_pos"] > 5 and r["honmei_prob"] > 0.25]

        # マーク別精度
        mark_stats: dict[str, dict] = {}
        for r in ok:
            m = r.get("mark", "?")
            if m not in mark_stats:
                mark_stats[m] = {"n": 0, "hit": 0}
            mark_stats[m]["n"] += 1
            if r["tansho_hit"]:
                mark_stats[m]["hit"] += 1

        # 改善ヒント（ルールベース）
        hints: list[str] = []
        if n > 0 and hits / n < 0.3:
            hints.append("単勝的中率が30%を下回っています。確率閾値（MIN_HONMEI_PROB）の引き上げを検討。")
        if near_miss:
            near_names = [r["honmei_name"] for r in near_miss]
            hints.append(f"惜しいレース（2〜3着）: {', '.join(near_names)} → 上位拮抗レースのフィルタ強化を検討。")
        if upset:
            upset_names = [r["honmei_name"] for r in upset]
            hints.append(f"大敗（高確率→6着以下）: {', '.join(upset_names)} → EV フィルタ・gap フィルタの見直しを検討。")

        return {
            "buy_races": n,
            "tansho_hit": hits,
            "tansho_hit_rate": round(hits / n, 3) if n else 0,
            "tansho_roi": round(roi, 3),
            "near_miss_races": len(near_miss),
            "upset_races": len(upset),
            "mark_stats": mark_stats,
            "hints": hints,
        }

    # ------------------------------------------------------------------ #
    # predictions_log.csv 更新
    # ------------------------------------------------------------------ #

    def _update_predictions_log(self, results: list[dict], target_date: date) -> None:
        if not PREDICTIONS_LOG.exists():
            return
        df = pd.read_csv(PREDICTIONS_LOG, dtype=str)
        today_str = str(target_date)

        for r in results:
            if r.get("status") != "ok":
                continue
            mask = (df["date"] == today_str) & (df["race_id"] == r["race_id"])
            if mask.sum() == 0:
                continue
            df.loc[mask, "tansho_hit"] = str(r["tansho_hit"])
            df.loc[mask, "tansho_ret"] = str(r["tansho_ret"]) if r["tansho_hit"] else "0"
            df.loc[mask, "umatan_hit"] = str(r["umatan_hit"])

        df.to_csv(PREDICTIONS_LOG, index=False)
        logger.info(f"predictions_log.csv 更新完了")

    # ------------------------------------------------------------------ #
    # LINE メッセージ生成
    # ------------------------------------------------------------------ #

    def _build_line_message(self, analysis: dict, target_date: date) -> str:
        n = analysis.get("buy_races", 0)
        if n == 0:
            return f"📋 振り返り君 ({target_date})\n買い対象レースなし or 結果取得失敗"

        hit  = analysis.get("tansho_hit", 0)
        rate = analysis.get("tansho_hit_rate", 0)
        roi  = analysis.get("tansho_roi", 0)
        nm   = analysis.get("near_miss_races", 0)
        up   = analysis.get("upset_races", 0)

        lines = [
            f"📋 振り返り君レポート ({target_date})",
            "",
            f"【本日の成績】",
            f"  買い: {n}R  的中: {hit}R  的中率: {rate:.1%}",
            f"  単勝ROI: {roi:.1%}",
            f"  惜しいレース(2〜3着): {nm}R",
            f"  大敗(6着以下): {up}R",
        ]

        marks = analysis.get("mark_stats", {})
        if marks:
            lines.append("")
            lines.append("【マーク別成績】")
            for m, s in sorted(marks.items()):
                lines.append(f"  {m}: {s['hit']}/{s['n']} ({s['hit']/s['n']:.0%})")

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
    parser.add_argument("--dry-run", action="store_true", help="LINE 送信・CSV更新しない")
    args = parser.parse_args()

    agent = Reviewer()
    agent.run(date.fromisoformat(args.date), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
