"""
predictions_log.csv の hit / payout を実際のレース結果で更新する。

週次まとめバッチ（日曜 17:00 JST）に組み込み、
当週分の is_buy=True レースについて単勝結果を取得して CSV を更新し、
stats.html を再生成する。

実行:
    python -m src.line.update_results
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PREDICTIONS_LOG = ROOT / "docs" / "predictions_log.csv"


def update_results() -> None:
    """
    predictions_log.csv のうち hit が空欄の行に対して
    db.netkeiba.com から実際の着順・払戻を取得して更新する。
    """
    if not PREDICTIONS_LOG.exists():
        logger.info("predictions_log.csv が存在しません。スキップします。")
        return

    df = pd.read_csv(PREDICTIONS_LOG, dtype=str)

    # 更新対象: is_buy=True かつ hit が未入力の行
    target_mask = (
        (df["is_buy"].str.lower() == "true") &
        (df["hit"].isna() | (df["hit"] == ""))
    )
    targets = df[target_mask].copy()

    if targets.empty:
        logger.info("更新対象レースなし（全て結果入力済み）")
        return

    logger.info(f"結果取得対象: {len(targets)} レース")

    from src.scraper.netkeiba_scraper import NetkeibaScraper
    scraper = NetkeibaScraper()

    updated = 0
    for idx, row in targets.iterrows():
        race_id = str(row["race_id"])
        honmei_num = str(row["honmei_num"]).strip()
        try:
            # 着順データ取得（静的HTML: db.netkeiba.com）
            result_df = scraper.fetch_race_result(race_id)
            if result_df.empty:
                logger.warning(f"  {race_id}: 結果取得失敗（空DataFrame）")
                continue

            # 1着馬の馬番を取得
            winner_rows = result_df[
                pd.to_numeric(result_df.get("finish_position", pd.Series()), errors="coerce") == 1
            ]
            if winner_rows.empty:
                # finish_position 列名が違う場合のフォールバック
                pos_col = next(
                    (c for c in result_df.columns
                     if "着" in c or "finish" in c.lower() or "pos" in c.lower()),
                    None
                )
                if pos_col:
                    winner_rows = result_df[
                        pd.to_numeric(result_df[pos_col], errors="coerce") == 1
                    ]

            if winner_rows.empty:
                logger.warning(f"  {race_id}: 1着馬を特定できず")
                continue

            winner_num = str(int(pd.to_numeric(
                winner_rows.iloc[0].get("horse_number", winner_rows.iloc[0].get("馬番", "0")),
                errors="coerce"
            ) or 0))

            hit = (winner_num == honmei_num)

            # 単勝払戻取得
            payout_val = 0
            try:
                payouts = scraper.fetch_race_payouts(race_id)
                tansho = payouts.get("単勝", [])
                if tansho:
                    # 本命馬番の単勝払戻を探す
                    for entry in tansho:
                        nums = [str(n) for n in entry.get("horses", [])]
                        if honmei_num in nums:
                            payout_val = int(entry.get("payout", 0))
                            break
                    # 見つからなければ1着馬の払戻を使用
                    if payout_val == 0 and hit and tansho:
                        payout_val = int(tansho[0].get("payout", 0))
            except Exception as e:
                logger.warning(f"  {race_id}: 払戻取得失敗 ({e})")

            df.at[idx, "hit"] = str(hit)
            df.at[idx, "payout"] = str(payout_val)
            updated += 1
            logger.info(
                f"  {race_id}: ◎{honmei_num} {'✅ 的中' if hit else '❌ 外れ'}"
                f"  払戻={payout_val:,}円"
            )
            time.sleep(2)

        except Exception as e:
            logger.error(f"  {race_id}: 結果更新エラー ({e})")

    scraper.close()

    if updated > 0:
        df.to_csv(PREDICTIONS_LOG, index=False)
        logger.info(f"predictions_log.csv 更新完了 ({updated} 件)")

        # stats.html 再生成
        try:
            from src.line.stats_page import generate_stats_page
            generate_stats_page()
            logger.info("stats.html 再生成完了")
        except Exception as e:
            logger.warning(f"stats.html 再生成失敗: {e}")
    else:
        logger.info("更新対象なし（結果未確定 or 取得失敗）")


if __name__ == "__main__":
    update_results()
