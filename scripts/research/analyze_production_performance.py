"""
analyze_production_performance.py
==================================
本稼働(2026-04-25〜)の予測ログを実際のレース結果と突合し、
KPI・セグメント別パフォーマンス・失注パターンを集計する。

実行方法:
    .venv/bin/python analyze_production_performance.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "data" / "logs" / "predictions"
CACHE_PATH = ROOT / "data" / "analysis" / "result_cache.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_result(race_id: str) -> dict | None:
    """
    netkeiba 結果ページから着順(上位5着・人気・オッズ)・馬連払戻・グレードを取得する。
    レース未確定の場合は None を返す。
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "lxml")

    table = soup.select_one("table.RaceTable01")
    if not table:
        return None

    top5 = []
    for tr in table.select("tr.HorseList"):
        tds = tr.select("td")
        if len(tds) < 11:
            continue
        rank_str = tds[0].get_text(strip=True)
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        if rank > 5:
            continue
        num = tds[2].get_text(strip=True)
        name = tds[3].get_text(strip=True)
        pop_str = tds[9].get_text(strip=True)
        odds_str = tds[10].get_text(strip=True)
        try:
            popularity = int(pop_str)
        except ValueError:
            popularity = None
        try:
            win_odds = float(odds_str)
        except ValueError:
            win_odds = None
        top5.append({
            "rank": rank, "num": num, "name": name,
            "popularity": popularity, "win_odds": win_odds,
        })

    if not top5:
        return None

    # グレード判定（当該レース見出し領域に限定）
    race_header = soup.select_one(".RaceList_Item02")
    is_g1 = bool(race_header.select_one(".Icon_GradeType1")) if race_header else False

    # 馬連払戻
    quinella_payout = 0
    for t in soup.select("table.Payout_Detail_Table"):
        for tr in t.select("tr"):
            th = tr.select_one("th")
            tds = tr.select("td")
            if not th or len(tds) < 2:
                continue
            if "馬連" in th.get_text(strip=True):
                raw = tds[1].decode_contents()
                p_lines = [x.strip() for x in raw.split("<br/>") if x.strip()]
                if p_lines:
                    m = re.search(r"([\d,]+)", BeautifulSoup(p_lines[0], "lxml").get_text())
                    if m:
                        quinella_payout = int(m.group(1).replace(",", ""))
                break
        if quinella_payout:
            break

    return {
        "race_id": race_id,
        "top5": top5,
        "quinella_payout": quinella_payout,
        "is_g1": is_g1,
    }


def main() -> None:
    cache = load_cache()

    pred_files = sorted(PRED_DIR.glob("*.csv"))
    all_rows = []
    for f in pred_files:
        target_date = f.stem
        df = pd.read_csv(f, dtype=str)
        df["src_date"] = target_date
        all_rows.append(df)
    full_df = pd.concat(all_rows, ignore_index=True)

    unique_race_ids = full_df["race_id"].unique().tolist()
    print(f"対象race_id数: {len(unique_race_ids)}  (予測行数: {len(full_df)})")

    new_fetches = 0
    for i, race_id in enumerate(unique_race_ids):
        if race_id in cache and cache[race_id] is not None:
            continue
        try:
            res = fetch_result(race_id)
        except Exception as e:
            print(f"  ⚠ {race_id} 取得エラー: {e}")
            res = None
        cache[race_id] = res
        new_fetches += 1
        status = "未確定/取得失敗" if res is None else f"払戻={res['quinella_payout']} G1={res['is_g1']}"
        print(f"  [{i+1}/{len(unique_race_ids)}] {race_id}: {status}")
        time.sleep(0.8)

    save_cache(cache)
    print(f"\n新規取得: {new_fetches}件  キャッシュ保存先: {CACHE_PATH}")

    full_df.to_pickle(ROOT / "data" / "analysis" / "full_predictions.pkl")
    print(f"予測データ結合保存: data/analysis/full_predictions.pkl  ({len(full_df)} 行)")


if __name__ == "__main__":
    main()
