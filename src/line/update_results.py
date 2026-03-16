"""
predictions_log.csv の tansho_hit/ret, umatan_hit/ret を
実際のレース結果で更新する。

週次まとめバッチ（日曜 17:00 JST）に組み込み、
当週分の is_buy=True レースについて結果を取得して CSV を更新し、
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
    if not PREDICTIONS_LOG.exists():
        logger.info("predictions_log.csv が存在しません。スキップします。")
        return

    df = pd.read_csv(PREDICTIONS_LOG, dtype=str)

    # 旧スキーマ（hit/payout列）を新スキーマに変換
    if "hit" in df.columns and "tansho_hit" not in df.columns:
        df["tansho_hit"] = df["hit"]
        df["tansho_ret"] = df.get("payout", "")
        df["umatan_str"] = ""
        df["umatan_hit"] = ""
        df["umatan_ret"] = ""
        df = df.drop(columns=[c for c in ("hit", "payout") if c in df.columns])

    # 新スキーマ列が不足している場合は補完
    for col in ("tansho_hit", "tansho_ret", "umatan_str", "umatan_hit", "umatan_ret"):
        if col not in df.columns:
            df[col] = ""

    # 更新対象: is_buy=True かつ tansho_hit が未入力
    target_mask = (
        (df["is_buy"].str.lower() == "true") &
        (df["tansho_hit"].isna() | (df["tansho_hit"] == ""))
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
        race_id    = str(row["race_id"])
        honmei_num = str(row["honmei_num"]).strip()
        umatan_str = str(row.get("umatan_str", "") or "")

        try:
            result_df = scraper.fetch_race_result(race_id)
            if result_df.empty:
                logger.warning(f"  {race_id}: 結果取得失敗（空DataFrame）")
                continue

            fp_col = next(
                (c for c in result_df.columns
                 if "finish" in c.lower() or "着" in c or "pos" in c.lower()),
                None
            )
            if fp_col is None:
                logger.warning(f"  {race_id}: 着順列が見つからず")
                continue

            result_df["_fp"] = pd.to_numeric(result_df[fp_col], errors="coerce")

            num_col = next(
                (c for c in result_df.columns
                 if "horse_number" in c.lower() or "馬番" in c),
                None
            )
            if num_col is None:
                logger.warning(f"  {race_id}: 馬番列が見つからず")
                continue

            def _num(fp):
                rows = result_df[result_df["_fp"] == fp]
                return str(int(pd.to_numeric(rows.iloc[0][num_col], errors="coerce") or 0)) \
                    if not rows.empty else ""

            winner_num = _num(1)
            second_num = _num(2)

            # 単勝的中
            tansho_hit = (winner_num == honmei_num)

            # 払戻取得
            tansho_ret = 0
            umatan_ret = 0
            try:
                payouts = scraper.fetch_race_payouts(race_id)

                # 単勝払戻
                for entry in payouts.get("単勝", []):
                    if honmei_num in [str(n) for n in entry.get("horses", [])]:
                        tansho_ret = int(entry.get("payout", 0))
                        break
                if tansho_ret == 0 and tansho_hit:
                    entries = payouts.get("単勝", [])
                    if entries:
                        tansho_ret = int(entries[0].get("payout", 0))

                # 馬単払戻（◎1着かつ相手が2着のとき）
                if tansho_hit and second_num and umatan_str:
                    bought = {
                        combo.replace("→", "-")
                        for combo in umatan_str.split(",")
                        if combo.strip()
                    }
                    actual_key = f"{honmei_num}-{second_num}"
                    if actual_key in bought:
                        for entry in payouts.get("馬単", []):
                            horses = [str(n) for n in entry.get("horses", [])]
                            if horses == [honmei_num, second_num]:
                                umatan_ret = int(entry.get("payout", 0))
                                break

            except Exception as e:
                logger.warning(f"  {race_id}: 払戻取得失敗 ({e})")

            umatan_hit = bool(
                tansho_hit and second_num and umatan_str and
                f"{honmei_num}-{second_num}" in {
                    c.replace("→", "-") for c in umatan_str.split(",") if c.strip()
                }
            )

            df.at[idx, "tansho_hit"] = str(tansho_hit)
            df.at[idx, "tansho_ret"] = str(tansho_ret)
            df.at[idx, "umatan_hit"] = str(umatan_hit)
            df.at[idx, "umatan_ret"] = str(umatan_ret)
            updated += 1

            um_label = f"馬単{'✅' if umatan_hit else '❌'}(¥{umatan_ret:,})" if umatan_str else ""
            logger.info(
                f"  {race_id}: ◎{honmei_num} 単勝{'✅' if tansho_hit else '❌'}(¥{tansho_ret:,})"
                + (f"  {um_label}" if um_label else "")
            )
            time.sleep(2)

        except Exception as e:
            logger.error(f"  {race_id}: 結果更新エラー ({e})")

    scraper.close()

    if updated > 0:
        df.to_csv(PREDICTIONS_LOG, index=False)
        logger.info(f"predictions_log.csv 更新完了 ({updated} 件)")
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
