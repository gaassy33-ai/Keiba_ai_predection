"""
data/logs/predictions/*.csv（馬連・軸馬流し方式）の実際の結果を取得し、
docs/stats.html（GitHub Pages公開ページ）を再生成する。

週次まとめバッチ（日曜 17:00 JST）に組み込み、直近分の結果を取得して
results_cache を更新し、stats.html を再生成する。

2026-06-22: 旧システム（単勝/馬単・honmei形式, docs/predictions_log.csv）から
新システム（馬連・軸馬流し方式, data/logs/predictions/）への参照に移行。

実行:
    python -m src.line.update_results
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.results.store import load_predictions, enrich


def update_results() -> None:
    df = load_predictions()
    if df.empty:
        logger.info("予測ログが見つかりません。スキップします。")
        return

    logger.info(f"対象: {len(df)}点（{df['race_id'].nunique()}レース）")
    df = enrich(df, fetch_missing=True)

    updated = int(df["hit"].notna().sum())
    logger.info(f"結果照合済み: {updated}点")

    try:
        from src.line.stats_page import generate_stats_page
        generate_stats_page(df)
        logger.info("stats.html 再生成完了")
    except Exception as e:
        logger.warning(f"stats.html 再生成失敗: {e}")


if __name__ == "__main__":
    update_results()
