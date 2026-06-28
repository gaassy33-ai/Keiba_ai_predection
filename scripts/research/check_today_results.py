"""
2026-03-28 の予想結果を race.netkeiba.com から取得して比較する一時スクリプト。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PREDICTIONS_LOG = ROOT / "docs" / "predictions_log.csv"
TARGET_DATE = "2026-03-28"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
VENUE_MAP = {"06": "中山", "09": "阪神", "07": "中京"}


def fetch_result(race_id: str):
    """race.netkeiba.com から着順と払戻を取得する"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, "lxml")

    # 着順テーブル
    table = soup.select_one("table.RaceTable01")
    if table is None:
        return None, {}

    rows = []
    for tr in table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 7:
            continue
        rows.append({
            "finish_position": tds[0].get_text(strip=True),
            "horse_number":    tds[2].get_text(strip=True),
            "horse_name":      tds[3].get_text(strip=True),
            "jockey_name":     tds[6].get_text(strip=True),
            "odds":            tds[10].get_text(strip=True) if len(tds) > 10 else "",
        })
    result_df = pd.DataFrame(rows)

    # 払戻テーブル: <br/> 区切りで複数行をパース
    payout_data: dict[str, dict] = {}
    for t in soup.select("table.Payout_Detail_Table"):
        for tr in t.select("tr"):
            th = tr.select_one("th")
            tds = tr.select("td")
            if not th or len(tds) < 2:
                continue
            bet_type = th.get_text(strip=True)
            # <br/> 区切りで分割（複勝・ワイドは複数行）
            h_lines = [x.strip() for x in tds[0].decode_contents().split("<br/>") if x.strip()]
            p_lines = [x.strip() for x in tds[1].decode_contents().split("<br/>") if x.strip()]
            pays_list = [int(re.sub(r"[^\d]", "", p)) for p in p_lines if re.search(r"\d", p)]
            # 馬番は各行の数字列（連結されている場合もあるが払戻順序は着順依存）
            payout_data[bet_type] = {"horses": h_lines, "payout": pays_list}

    return result_df, payout_data


def run():
    df = pd.read_csv(PREDICTIONS_LOG, dtype=str)
    today    = df[df["date"] == TARGET_DATE].copy()
    buy_races = today[today["is_buy"].str.lower() == "true"].copy()

    print(f"\n{'='*62}")
    print(f"  {TARGET_DATE} 予想サマリー")
    print(f"{'='*62}")
    print(f"  予想対象レース: {len(today)} レース（中山{(today['race_id'].str[4:6]=='06').sum()}・"
          f"阪神{(today['race_id'].str[4:6]=='09').sum()}・"
          f"中京{(today['race_id'].str[4:6]=='07').sum()}）")
    print(f"  馬券購入対象 : {len(buy_races)} レース")
    print(f"{'='*62}\n")

    results = []

    for _, row in buy_races.iterrows():
        race_id     = str(row["race_id"])
        race_name   = str(row["race_name"])
        honmei_num  = str(row["honmei_num"]).strip()
        honmei_name = str(row["honmei_name"])
        honmei_prob = float(row["honmei_prob"])
        umatan_str  = str(row.get("umatan_str", "") or "")
        mark        = str(row.get("mark", ""))
        venue_code  = race_id[4:6]
        venue_name  = VENUE_MAP.get(venue_code, venue_code)
        race_num    = race_id[-2:]

        print(f"【{venue_name}{int(race_num)}R】{race_name}  "
              f"◎{honmei_num}.{honmei_name} ({honmei_prob:.1%}) {mark}")
        if umatan_str:
            print(f"  馬単買い目: {umatan_str}")

        try:
            result_df, payouts = fetch_result(race_id)
            if result_df is None or result_df.empty:
                print("  → 結果未確定（レース未了）\n")
                results.append({"race_id": race_id, "race_name": race_name,
                                 "honmei_num": honmei_num, "status": "pending"})
                time.sleep(1)
                continue

            result_df["_fp"] = pd.to_numeric(result_df["finish_position"], errors="coerce")

            def get_by_rank(fp):
                r = result_df[result_df["_fp"] == fp]
                if r.empty:
                    return "", ""
                return r.iloc[0]["horse_number"], r.iloc[0]["horse_name"]

            w1_num, w1_name = get_by_rank(1)
            w2_num, w2_name = get_by_rank(2)
            w3_num, w3_name = get_by_rank(3)
            top3 = [w1_num, w2_num, w3_num]

            tansho_hit = (w1_num == honmei_num)
            place_hit  = (honmei_num in top3)

            # 払戻取得
            def get_payout(bet_type, target_nums=None):
                entry = payouts.get(bet_type, {})
                if not entry:
                    return 0
                horses = entry.get("horses", [])
                pays   = entry.get("payout", [])
                if target_nums:
                    # 特定馬番の払戻を探す
                    for i, h in enumerate(horses):
                        if str(h) == str(target_nums):
                            return pays[i] if i < len(pays) else (pays[0] if pays else 0)
                return pays[0] if pays else 0

            tansho_ret = get_payout("単勝", w1_num)
            # 複勝払戻: 着順(1/2/3着)→インデックス対応
            fukusho_ret_honmei = 0
            if place_hit:
                fukusho_pays = payouts.get("複勝", {}).get("payout", [])
                # top3 の中での◎の順位（0-indexed）を払戻のインデックスとして使う
                for idx, num in enumerate(top3):
                    if num == honmei_num and idx < len(fukusho_pays):
                        fukusho_ret_honmei = fukusho_pays[idx]
                        break

            umatan_hit = False
            umatan_ret = 0
            if tansho_hit and w2_num and umatan_str:
                bought = {c.replace("→", "-") for c in umatan_str.split(",") if c.strip()}
                actual_key = f"{honmei_num}-{w2_num}"
                if actual_key in bought:
                    umatan_hit = True
                    umatan_entry = payouts.get("馬単", {})
                    umatan_pays  = umatan_entry.get("payout", [])
                    umatan_ret   = umatan_pays[0] if umatan_pays else 0

            sanrenpuku_ret = 0
            if honmei_num in top3:
                sanrenpuku_entry = payouts.get("3連複", {})
                sp = sanrenpuku_entry.get("payout", [])
                sanrenpuku_ret = sp[0] if sp else 0

            tansho_mark = "✅" if tansho_hit else "❌"
            place_mark  = "✅" if place_hit  else "❌"
            umatan_mark = "✅" if umatan_hit else ("－" if not umatan_str else "❌")

            print(f"  結果: 1着={w1_num}.{w1_name}  2着={w2_num}.{w2_name}  3着={w3_num}.{w3_name}")
            print(f"  単勝 {tansho_mark} ¥{tansho_ret:,}  "
                  f"複勝 {place_mark} ¥{fukusho_ret_honmei:,}")
            if umatan_str:
                print(f"  馬単 {umatan_mark} ¥{umatan_ret:,}")
            if honmei_num in top3:
                print(f"  3連複払戻 ¥{sanrenpuku_ret:,}")
            print()

            results.append({
                "race_id": race_id, "race_name": race_name, "venue": venue_name,
                "honmei_num": honmei_num, "honmei_name": honmei_name,
                "winner_num": w1_num, "winner_name": w1_name,
                "tansho_hit": tansho_hit, "tansho_ret": tansho_ret,
                "place_hit": place_hit,   "fukusho_ret": fukusho_ret_honmei,
                "umatan_hit": umatan_hit, "umatan_ret": umatan_ret,
                "sanrenpuku_ret": sanrenpuku_ret,
                "status": "done",
            })
            time.sleep(2)

        except Exception as e:
            print(f"  → エラー: {e}\n")
            results.append({"race_id": race_id, "race_name": race_name,
                             "honmei_num": honmei_num, "status": "error"})

    # ── 集計 ─────────────────────────────────────────────────────────
    valid = [r for r in results if r.get("status") == "done"]
    pending = [r for r in results if r.get("status") == "pending"]

    if not valid:
        print(f"\n確定済みレースなし。未確定: {len(pending)} レース")
        return

    tansho_hits = [r for r in valid if r["tansho_hit"]]
    place_hits  = [r for r in valid if r["place_hit"]]
    umatan_hits = [r for r in valid if r.get("umatan_hit")]

    tansho_rate = len(tansho_hits) / len(valid) * 100
    place_rate  = len(place_hits)  / len(valid) * 100

    total_tansho  = sum(r["tansho_ret"]    for r in tansho_hits)
    total_fukusho = sum(r["fukusho_ret"]   for r in place_hits)
    total_umatan  = sum(r["umatan_ret"]    for r in umatan_hits)

    print(f"{'='*62}")
    print(f"  {TARGET_DATE} 集計（確定 {len(valid)} / 購入対象 {len(results)} レース）")
    print(f"{'='*62}")
    print(f"  単勝的中 : {len(tansho_hits)}/{len(valid)} = {tansho_rate:.1f}%")
    print(f"  複勝的中 : {len(place_hits)}/{len(valid)} = {place_rate:.1f}%")
    print(f"  馬単的中 : {len(umatan_hits)}/{len(valid)}")
    print(f"  単勝払戻合計 : ¥{total_tansho:,}")
    print(f"  複勝払戻合計 : ¥{total_fukusho:,}")
    print(f"  馬単払戻合計 : ¥{total_umatan:,}")
    if pending:
        print(f"  未確定レース : {len(pending)} レース（結果待ち）")
    print(f"{'='*62}")

    print("\n  ◎ 単勝的中:")
    for r in tansho_hits:
        print(f"    ✅ [{r['venue']}] {r['race_name']}  ◎{r['honmei_num']}.{r['honmei_name']} → ¥{r['tansho_ret']:,}")
    if not tansho_hits:
        print("    なし")

    print("\n  ◎ 複勝的中（単勝外・3着以内）:")
    only_place = [r for r in place_hits if not r["tansho_hit"]]
    for r in only_place:
        print(f"    ○ [{r['venue']}] {r['race_name']}  ◎{r['honmei_num']}.{r['honmei_name']}  "
              f"（1着={r['winner_num']}.{r['winner_name']}） 複勝¥{r['fukusho_ret']:,}")
    if not only_place:
        print("    なし（または全て単勝的中）")

    print("\n  ◎ 馬単的中:")
    for r in umatan_hits:
        print(f"    ✅ [{r['venue']}] {r['race_name']}  → ¥{r['umatan_ret']:,}")
    if not umatan_hits:
        print("    なし")


if __name__ == "__main__":
    run()
