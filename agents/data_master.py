"""
agents/data_master.py - サブ①「データマスター」
=====================================================
役割:
  - 過去の予想精度（直近8週）を集計して予想屋に提供
  - モデルの特徴量重要度サマリーを整理
  - 開催日・会場・馬場の傾向を分析
  - 出力: data/agents/YYYYMMDD_context.json

実行:
    python -m agents.data_master [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentBase

PREDICTIONS_LOG = ROOT / "docs" / "predictions_log.csv"
FEATURE_IMPORTANCE = ROOT / "data" / "models" / "feature_importance.json"
LOOKBACK_WEEKS = 8   # 直近何週分を集計するか


class DataMaster(AgentBase):
    """サブ①: データマスター — 学習データ整備・週次統計提供"""

    name = "DataMaster"

    # ------------------------------------------------------------------ #
    # メインエントリ
    # ------------------------------------------------------------------ #

    def run(self, target_date: date, dry_run: bool = False) -> dict:
        logger.info(f"DataMaster 起動: {target_date}")

        context = {
            "generated_at": datetime.now().isoformat(),
            "target_date": str(target_date),
            "accuracy_summary": self._accuracy_summary(target_date),
            "venue_accuracy": self._venue_accuracy(target_date),
            "mark_accuracy": self._mark_accuracy(target_date),
            "feature_importance": self._feature_importance_summary(),
            "data_health": self._data_health(),
        }

        path = self.save(context, self.context_file(target_date))
        logger.info(f"コンテキスト保存完了: {path}")

        # LINE サマリー通知
        msg = self._build_line_message(context, target_date)
        self.send_line(msg, dry_run=dry_run)

        return context

    # ------------------------------------------------------------------ #
    # 精度集計
    # ------------------------------------------------------------------ #

    def _load_log(self) -> pd.DataFrame:
        if not PREDICTIONS_LOG.exists():
            return pd.DataFrame()
        df = pd.read_csv(PREDICTIONS_LOG, dtype=str)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("honmei_prob", "taikou_prob", "gap", "tansho_ret", "umatan_ret"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("is_buy", "tansho_hit", "umatan_hit"):
            if col in df.columns:
                df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
        return df

    def _accuracy_summary(self, target_date: date) -> dict:
        df = self._load_log()
        if df.empty:
            return {}

        cutoff = pd.Timestamp(target_date) - timedelta(weeks=LOOKBACK_WEEKS)
        recent = df[df["date"] >= cutoff].copy()
        bought = recent[recent["is_buy"] == True]

        total  = len(recent)
        buy_n  = len(bought)
        if buy_n == 0:
            return {"total_races": total, "buy_races": 0, "note": "買い対象なし"}

        hit_tan  = bought["tansho_hit"].sum()
        hit_uma  = bought["umatan_hit"].sum() if "umatan_hit" in bought else 0
        roi_tan  = bought["tansho_ret"].mean()
        roi_uma  = bought["umatan_ret"].mean() if "umatan_ret" in bought else np.nan

        # 週別推移
        recent["week"] = recent["date"].dt.isocalendar().week
        weekly = (
            recent[recent["is_buy"] == True]
            .groupby("week")
            .agg(
                buy=("is_buy", "count"),
                hit_tan=("tansho_hit", "sum"),
                avg_ret=("tansho_ret", "mean"),
            )
            .tail(LOOKBACK_WEEKS)
            .to_dict(orient="index")
        )

        return {
            "total_races": total,
            "buy_races": buy_n,
            "tansho_hit_rate": round(hit_tan / buy_n, 3),
            "tansho_roi_avg": round(float(roi_tan), 1) if not np.isnan(roi_tan) else None,
            "umatan_hit_rate": round(hit_uma / buy_n, 3) if buy_n else None,
            "umatan_roi_avg": round(float(roi_uma), 1) if not np.isnan(roi_uma) else None,
            "weekly_trend": {str(k): v for k, v in weekly.items()},
        }

    def _venue_accuracy(self, target_date: date) -> dict:
        """会場別の単勝的中率を集計"""
        df = self._load_log()
        if df.empty or "race_id" not in df.columns:
            return {}

        cutoff = pd.Timestamp(target_date) - timedelta(weeks=LOOKBACK_WEEKS)
        recent = df[(df["date"] >= cutoff) & (df["is_buy"] == True)].copy()
        if recent.empty:
            return {}

        JYO_MAP = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
            "05": "東京", "06": "中山", "07": "中京", "08": "京都",
            "09": "阪神", "10": "小倉",
        }
        recent["venue"] = recent["race_id"].str[4:6].map(JYO_MAP).fillna("?")

        result = {}
        for venue, grp in recent.groupby("venue"):
            n = len(grp)
            hits = grp["tansho_hit"].sum()
            roi  = grp["tansho_ret"].mean()
            result[venue] = {
                "races": n,
                "hit_rate": round(hits / n, 3) if n else 0,
                "roi_avg": round(float(roi), 1) if n and not np.isnan(roi) else None,
            }
        return result

    def _mark_accuracy(self, target_date: date) -> dict:
        """◎/○/△ 別の精度"""
        df = self._load_log()
        if df.empty or "mark" not in df.columns:
            return {}

        cutoff = pd.Timestamp(target_date) - timedelta(weeks=LOOKBACK_WEEKS)
        recent = df[df["date"] >= cutoff].copy()
        result = {}
        for mark, grp in recent.groupby("mark"):
            bought = grp[grp["is_buy"] == True]
            n = len(bought)
            if n == 0:
                continue
            hits = bought["tansho_hit"].sum()
            result[mark] = {
                "races": n,
                "hit_rate": round(hits / n, 3),
                "avg_prob": round(float(bought["honmei_prob"].mean()), 3),
            }
        return result

    # ------------------------------------------------------------------ #
    # 特徴量重要度サマリー
    # ------------------------------------------------------------------ #

    def _feature_importance_summary(self) -> dict:
        if not FEATURE_IMPORTANCE.exists():
            return {}
        raw = json.loads(FEATURE_IMPORTANCE.read_text())
        win = raw.get("win_model", {}).get("importance", [])
        top5 = [{
            "feature": f["feature"],
            "gain_pct": f["gain_pct"],
        } for f in sorted(win, key=lambda x: -x["gain_pct"])[:5]]
        return {
            "top5_win": top5,
            "num_trees": raw.get("win_model", {}).get("num_trees"),
        }

    # ------------------------------------------------------------------ #
    # データ健全性チェック
    # ------------------------------------------------------------------ #

    def _data_health(self) -> dict:
        if not PREDICTIONS_LOG.exists():
            return {"status": "ERROR", "message": "predictions_log.csv が存在しません"}

        df = self._load_log()
        total = len(df)
        result_filled = df["tansho_hit"].notna().sum() if "tansho_hit" in df.columns else 0
        missing_result = total - result_filled

        latest = df["date"].max()
        days_since = (pd.Timestamp.now() - latest).days if pd.notna(latest) else 999

        return {
            "total_records": total,
            "result_filled": int(result_filled),
            "missing_result": int(missing_result),
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
            "【直近8週の予想精度】",
        ]

        buy_n = acc.get("buy_races", 0)
        if buy_n:
            lines += [
                f"  買い対象: {buy_n}R",
                f"  単勝的中率: {acc.get('tansho_hit_rate', 0):.1%}",
                f"  単勝平均回収: {acc.get('tansho_roi_avg', '-')}円",
                f"  馬単的中率: {acc.get('umatan_hit_rate', '-') if acc.get('umatan_hit_rate') else '-'}",
            ]
        else:
            lines.append("  データなし")

        lines.append("")
        lines.append("【特徴量重要度 Top5】")
        for f in imp.get("top5_win", []):
            lines.append(f"  {f['feature']}: {f['gain_pct']:.1f}%")

        lines.append("")
        lines.append("【データ健全性】")
        lines.append(f"  総レコード: {health.get('total_records', 0)}件")
        lines.append(f"  最終更新: {health.get('latest_date', '不明')}")
        lines.append(f"  ステータス: {health.get('status', '?')}")

        # 会場別精度（上位3会場）
        venue_acc = ctx.get("venue_accuracy", {})
        if venue_acc:
            top_venues = sorted(venue_acc.items(), key=lambda x: -x[1]["hit_rate"])[:3]
            lines.append("")
            lines.append("【会場別単勝的中率 Top3】")
            for v, d in top_venues:
                lines.append(f"  {v}: {d['hit_rate']:.1%} ({d['races']}R)")

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
