"""
review_results.py
=================
馬連予想の結果振り返りスクリプト。
data/logs/predictions/{date}.csv を読み込み、
各レースの着順と馬連払戻を netkeiba から取得して集計する。

実行方法:
    .venv/bin/python review_results.py              # 当日
    .venv/bin/python review_results.py 2026-05-30   # 日付指定
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT    = Path(__file__).resolve().parents[2]
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# 馬番 → 囲み数字
_CIRCLED = {
    1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
    6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
    11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮",
    16: "⑯", 17: "⑰", 18: "⑱",
}


def _circled(num_str: str) -> str:
    try:
        return _CIRCLED.get(int(num_str), f"({num_str})")
    except (ValueError, TypeError):
        return f"({num_str})"


def fetch_result(race_id: str) -> tuple[list[dict], int]:
    """
    netkeiba の結果ページから着順（上位5着）と馬連払戻を取得する。

    Returns
    -------
    top5 : list[dict]  各要素 {"rank": int, "num": str, "name": str}
    quinella_payout : int  馬連払戻（円、0 = 未取得）
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    # ── 着順テーブル ─────────────────────────────────────────────────
    top5: list[dict] = []
    table = soup.select_one("table.RaceTable01")
    if table:
        for tr in table.select("tr.HorseList"):
            cols = tr.select("td")
            if len(cols) < 4:
                continue
            rank_str = cols[0].get_text(strip=True)
            num      = cols[2].get_text(strip=True)
            name     = cols[3].get_text(strip=True)
            try:
                rank = int(rank_str)
            except ValueError:
                continue
            if rank <= 5:
                top5.append({"rank": rank, "num": num, "name": name})

    # ── 払戻テーブル（馬連）───────────────────────────────────────────
    quinella_payout = 0

    # 方法①: table.Payout_Detail_Table
    for t in soup.select("table.Payout_Detail_Table"):
        for tr in t.select("tr"):
            th = tr.select_one("th")
            tds = tr.select("td")
            if not th or len(tds) < 2:
                continue
            if "馬連" in th.get_text(strip=True):
                raw = tds[1].decode_contents()
                # 最初の <br/> 前の払戻額を取得
                p_lines = [x.strip() for x in raw.split("<br/>") if x.strip()]
                if p_lines:
                    m = re.search(r"([\d,]+)", p_lines[0])
                    if m:
                        quinella_payout = int(m.group(1).replace(",", ""))
                break
        if quinella_payout:
            break

    # 方法②: Payback クラスのテキストから正規表現でパース（フォールバック）
    if not quinella_payout:
        for tag in soup.find_all(
            ["tr", "div", "td"],
            class_=lambda c: c and "Payback" in c,
        ):
            txt = tag.get_text(" ", strip=True)
            # "馬連 <num1> <num2> 1,690円" のようなパターン
            m = re.search(r"馬連\s+\d+\s+\d+\s+([\d,]+)円", txt)
            if m:
                quinella_payout = int(m.group(1).replace(",", ""))
                break

    return sorted(top5, key=lambda x: x["rank"]), quinella_payout


def run(target_date_str: str) -> None:
    pred_file = ROOT / "data" / "logs" / "predictions" / f"{target_date_str}.csv"
    if not pred_file.exists():
        print(f"予測ファイルが見つかりません: {pred_file}")
        sys.exit(1)

    df = pd.read_csv(pred_file, dtype=str)

    print(f"\n{'━'*55}")
    print(f"  {target_date_str} 馬連予想 結果振り返り")
    print(f"{'━'*55}")

    total_bets    = len(df)
    total_cost    = total_bets * 100
    total_payout  = 0
    hits          = 0

    for race_id in df["race_id"].unique():
        rdf         = df[df["race_id"] == race_id].copy()
        race_name   = rdf.iloc[0]["race_name"]
        course_type = rdf.iloc[0]["course_type"]
        distance    = rdf.iloc[0]["distance"]

        print(f"\n【{race_name}】{course_type}{distance}m")

        try:
            top5, q_pay = fetch_result(race_id)
        except Exception as e:
            print(f"  ⚠ 結果取得エラー: {e}")
            time.sleep(1)
            continue

        if not top5:
            print("  ⚠ 結果未確定（レース未了）")
            time.sleep(1)
            continue

        # 着順表示（上位3着）
        for h in top5[:3]:
            print(f"  {h['rank']}着: {_circled(h['num'])}{h['name']}")
        print(f"  馬連払戻: {q_pay:,}円" if q_pay else "  馬連払戻: 取得失敗")

        top2_nums = {h["num"] for h in top5 if h["rank"] <= 2}
        top3_nums = {h["num"] for h in top5 if h["rank"] <= 3}

        for _, row in rdf.iterrows():
            ni      = str(row["horse_num_i"]).strip()
            nj      = str(row["horse_num_j"]).strip()
            ni_name = row["horse_name_i"]
            nj_name = row["horse_name_j"]
            ev      = float(row["ev"])
            est     = float(row["est_quinella_odds"])

            hit = (ni in top2_nums) and (nj in top2_nums)

            if hit:
                total_payout += q_pay
                hits += 1
                mark   = "✅ 的中"
                suffix = f"  → 払戻 {q_pay:,}円"
            elif (ni in top3_nums) or (nj in top3_nums):
                mark   = "△ 惜しい"
                suffix = ""
            else:
                mark   = "❌"
                suffix = ""

            print(
                f"  {mark}  {_circled(ni)}{ni_name} × {_circled(nj)}{nj_name}"
                f"  EV={ev:.2f}  想定{est:.1f}倍{suffix}"
            )

        time.sleep(1)

    # ── 集計 ──────────────────────────────────────────────────────────
    balance = total_payout - total_cost
    roi     = total_payout / total_cost * 100 if total_cost else 0.0

    print(f"\n{'━'*55}")
    print(f"  集計")
    print(f"{'━'*55}")
    print(f"  購入     : {total_bets}点 × 100円 = {total_cost:,}円")
    print(f"  的中     : {hits}点")
    print(f"  払戻合計 : {total_payout:,}円")
    print(f"  収支     : {balance:+,}円")
    print(f"  ROI      : {roi:.1f}%")
    print(f"{'━'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="馬連予想結果振り返り")
    parser.add_argument(
        "date",
        nargs="?",
        default=date.today().strftime("%Y-%m-%d"),
        help="対象日 (YYYY-MM-DD)  省略時は当日",
    )
    args = parser.parse_args()
    run(args.date)
