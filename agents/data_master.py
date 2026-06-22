"""
agents/data_master.py - サブ①「データマスター」
=====================================================
役割:
  - 過去の予想精度（直近8週・馬連）を集計して予想屋に提供
  - LTRモデルの特徴量重要度サマリーを整理
  - 会場別の傾向を分析
  - 出力: data/agents/YYYYMMDD_context.json

2026-06-22: 旧システム（単勝/馬単・honmei形式, docs/predictions_log.csv）から
新システム（馬連・軸馬流し方式, data/logs/predictions/）への参照に移行。
結果取得は Reviewer が既に当日分のキャッシュ(data/logs/results_cache.json)を
作成済みであることを前提に、ここではネットワークアクセスせずキャッシュのみ参照する
（高速・週次集計に専念するための役割分担）。

実行:
    python -m agents.data_master [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentBase
from src.results.store import load_predictions, enrich

LTR_FEATURE_IMPORTANCE = ROOT / "data" / "models" / "ltr_feature_importance.json"
LOOKBACK_WEEKS = 8   # 直近何週分を集計するか

JYO_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


class DataMaster(AgentBase):
    """サブ①: データマスター — 学習データ整備・週次統計提供（馬連）"""

    name = "DataMaster"

    # ------------------------------------------------------------------ #
    # メインエントリ
    # ------------------------------------------------------------------ #

    def run(self, target_date: date, dry_run: bool = False) -> dict:
        logger.info(f"DataMaster 起動: {target_date}")

        recent = self._load_recent(target_date)

        context = {
            "generated_at": datetime.now().isoformat(),
            "target_date": str(target_date),
            "accuracy_summary": self._accuracy_summary(recent),
            "venue_accuracy": self._venue_accuracy(recent),
            "feature_importance": self._feature_importance_summary(),
            "data_health": self._data_health(recent, target_date),
        }

        path = self.save(context, self.context_file(target_date))
        logger.info(f"コンテキスト保存完了: {path}")

        msg = self._build_line_message(context, target_date)
        self.send_line(msg, dry_run=dry_run)

        return context

    # ------------------------------------------------------------------ #
    # データ読み込み（キャッシュのみ参照、ネットワークアクセスなし）
    # ------------------------------------------------------------------ #

    def _load_recent(self, target_date: date) -> pd.DataFrame:
        cutoff = target_date - timedelta(weeks=LOOKBACK_WEEKS)
        df = load_predictions(start_date=cutoff, end_date=target_date)
        if df.empty:
            return df
        return enrich(df, fetch_missing=False)

    # ------------------------------------------------------------------ #
    # 精度集計
    # ------------------------------------------------------------------ #

    def _accuracy_summary(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        ok = df[df["hit"].notna()].copy()
        buy_n = len(ok)
        if buy_n == 0:
            return {"total_bets": len(df), "buy_bets": 0, "note": "結果照合可能なデータなし"}

        hits = int(ok["hit"].sum())
        cost = buy_n * 100
        payout = int(ok["payout"].sum())

        # 週別推移
        ok["week"] = pd.to_datetime(ok["src_date"]).dt.isocalendar().week
        weekly = (
            ok.groupby("week")
            .agg(bets=("hit", "count"), hits=("hit", "sum"), payout=("payout", "sum"))
            .tail(LOOKBACK_WEEKS)
            .to_dict(orient="index")
        )

        return {
            "total_bets": len(df),
            "buy_bets": buy_n,
            "hit_rate": round(hits / buy_n, 3),
            "roi": round(payout / cost, 3) if cost else None,
            "weekly_trend": {str(k): v for k, v in weekly.items()},
        }

    def _venue_accuracy(self, df: pd.DataFrame) -> dict:
        """会場別の馬連的中率を集計"""
        if df.empty or "race_id" not in df.columns:
            return {}
        ok = df[df["hit"].notna()].copy()
        if ok.empty:
            return {}

        ok["venue"] = ok["race_id"].astype(str).str[4:6].map(JYO_MAP).fillna("?")

        result = {}
        for venue, grp in ok.groupby("venue"):
            n = len(grp)
            hits = int(grp["hit"].sum())
            cost = n * 100
            payout = int(grp["payout"].sum())
            result[venue] = {
                "bets": n,
                "hit_rate": round(hits / n, 3) if n else 0,
                "roi": round(payout / cost, 3) if cost else None,
            }
        return result

    # ------------------------------------------------------------------ #
    # 特徴量重要度サマリー（LTRモデル）
    # ------------------------------------------------------------------ #

    def _feature_importance_summary(self) -> dict:
        if not LTR_FEATURE_IMPORTANCE.exists():
            return {}
        raw = json.loads(LTR_FEATURE_IMPORTANCE.read_text())
        ltr = raw.get("ltr_model", {})
        importance = ltr.get("importance", [])
        top5 = [{
            "feature": f["feature"],
            "gain_pct": f["gain_pct"],
        } for f in sorted(importance, key=lambda x: -x["gain_pct"])[:5]]
        return {
            "top5": top5,
            "num_trees": ltr.get("num_trees"),
            "oof_ndcg3": ltr.get("oof_ndcg3"),
        }

    # ------------------------------------------------------------------ #
    # データ健全性チェック
    # ------------------------------------------------------------------ #

    def _data_health(self, df: pd.DataFrame, target_date: date) -> dict:
        if df.empty:
            return {"status": "ERROR", "message": "直近8週の予測ログが見つかりません"}

        total = len(df)
        result_filled = int(df["hit"].notna().sum())
        missing_result = total - result_filled

        latest = pd.to_datetime(df["src_date"]).max()
        days_since = (pd.Timestamp(target_date) - latest).days if pd.notna(latest) else 999

        return {
            "total_records": total,
            "result_filled": result_filled,
            "missing_result": missing_result,
            "latest_date": str(latest.date()) if pd.notna(latest) else None,
            "days_since_latest": days_since,
            "status": "OK" if days_since <= 14 else "STALE",
        }

    # ------------------------------------------------------------------ #
    # LINE メッセージ生成
    # ------------------------------------------------------------------ #

    def _build_line_message(self, ctx: dict, target_date: date) -> str:
        acc = ctx.get("accuracy_summary", {})
        imp = ctx.get("feature_importance", {})
        health = ctx.get("data_health", {})

        lines = [
            f"📊 データマスター週次レポート ({target_date})",
            "",
            "【直近8週の予想精度（馬連）】",
        ]

        buy_n = acc.get("buy_bets", 0)
        if buy_n:
            roi = acc.get("roi")
            lines += [
                f"  購入: {buy_n}点",
                f"  的中率: {acc.get('hit_rate', 0):.1%}",
                f"  ROI: {roi:.1%}" if roi is not None else "  ROI: -",
            ]
        else:
            lines.append("  データなし")

        if imp.get("top5"):
            lines.append("")
            lines.append(f"【LTR特徴量重要度 Top5】(NDCG@3={imp.get('oof_ndcg3', '?')})")
            for f in imp["top5"]:
                lines.append(f"  {f['feature']}: {f['gain_pct']:.1f}%")

        lines.append("")
        lines.append("【データ健全性】")
        lines.append(f"  総レコード: {health.get('total_records', 0)}件")
        lines.append(f"  最終更新: {health.get('latest_date', '不明')}")
        lines.append(f"  ステータス: {health.get('status', '?')}")

        venue_acc = ctx.get("venue_accuracy", {})
        if venue_acc:
            top_venues = sorted(venue_acc.items(), key=lambda x: -x[1]["hit_rate"])[:3]
            lines.append("")
            lines.append("【会場別馬連的中率 Top3】")
            for v, d in top_venues:
                lines.append(f"  {v}: {d['hit_rate']:.1%} ({d['bets']}点)")

        return "\n".join(lines)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="データマスター: 週次データ整備・統計提供")
    parser.add_argument("--date", default=str(date.today()), help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="LINE 送信しない")
    args = parser.parse_args()

    agent = DataMaster()
    agent.run(date.fromisoformat(args.date), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
