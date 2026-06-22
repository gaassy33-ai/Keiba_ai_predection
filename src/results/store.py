"""
src/results/store.py
=====================
新システム（LTR + Two-Brain, 馬連軸馬流し方式）の予測ログ
(data/logs/predictions/*.csv) と実際のレース結果を突合する共通ユーティリティ。

旧システム（docs/predictions_log.csv, 単勝/馬単中心・honmei形式）から
agents/reviewer.py・agents/data_master.py・src/line/update_results.py が
それぞれ独自にスクレイピングしていた処理を統一する。

【予測ログのスキーマ（data/logs/predictions/YYYY-MM-DD.csv）】
    date, race_id, race_name, course_type, distance,
    horse_num_i, horse_id_i, horse_name_i,   ← 軸馬（axis）
    horse_num_j, horse_id_j, horse_name_j,   ← パートナー（partner）
    odds_i, odds_j, est_quinella_odds, p_model, p_market, ev, ev_rank

enrich() でレース結果を突合し、以下の列を追加する:
    hit (bool)            : 軸・パートナー両方が1-2着 → 馬連的中
    payout (int)          : 的中時の馬連払戻（円）。不的中時0
    axis_in_top2 (bool)   : 軸馬が1-2着に入ったか
    is_g1 (bool)          : 当該レースがG1か
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "data" / "logs" / "predictions"
RESULTS_CACHE_PATH = ROOT / "data" / "logs" / "results_cache.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


# ──────────────────────────────────────────────────────────────────────────────
# 予測ログ読み込み
# ──────────────────────────────────────────────────────────────────────────────

def load_predictions(start_date=None, end_date=None) -> pd.DataFrame:
    """
    data/logs/predictions/*.csv を結合して返す。

    Parameters
    ----------
    start_date, end_date : date | None
        指定した場合、ファイル名（YYYY-MM-DD.csv）でフィルタする。
    """
    if not PREDICTIONS_DIR.exists():
        return pd.DataFrame()

    frames = []
    for f in sorted(PREDICTIONS_DIR.glob("*.csv")):
        try:
            d = pd.Timestamp(f.stem).date()
        except ValueError:
            continue
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        df = pd.read_csv(f, dtype=str)
        if df.empty:
            continue
        df["src_date"] = f.stem
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    for col in ("odds_i", "odds_j", "est_quinella_odds", "p_model", "p_market", "ev"):
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")
    return full


# ──────────────────────────────────────────────────────────────────────────────
# レース結果取得（キャッシュ付き）
# ──────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if RESULTS_CACHE_PATH.exists():
        return json.loads(RESULTS_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    RESULTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_result(race_id: str) -> dict | None:
    """
    netkeiba結果ページから着順上位5頭・馬連払戻・グレードを取得する。
    レース未確定・取得失敗時は None。
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "lxml")

    table = soup.select_one("table.RaceTable01")
    if not table:
        return None

    top2: set[str] = set()
    for tr in table.select("tr.HorseList"):
        tds = tr.select("td")
        if len(tds) < 3:
            continue
        try:
            rank = int(tds[0].get_text(strip=True))
        except ValueError:
            continue
        if rank <= 2:
            top2.add(tds[2].get_text(strip=True))

    if not top2:
        return None

    race_header = soup.select_one(".RaceList_Item02")
    is_g1 = bool(race_header.select_one(".Icon_GradeType1")) if race_header else False

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
                    import re
                    m = re.search(r"([\d,]+)", BeautifulSoup(p_lines[0], "lxml").get_text())
                    if m:
                        quinella_payout = int(m.group(1).replace(",", ""))
                break
        if quinella_payout:
            break

    return {"top2": sorted(top2), "quinella_payout": quinella_payout, "is_g1": is_g1}


def enrich(df: pd.DataFrame, fetch_missing: bool = True, sleep_sec: float = 0.8) -> pd.DataFrame:
    """
    予測DataFrameにhit/payout/axis_in_top2/is_g1列を追加する。

    Parameters
    ----------
    fetch_missing : True の場合、キャッシュに無いrace_idをnetkeibaへ取得しに行く。
        False の場合はキャッシュのみ参照する（ネットワークアクセスなし・高速）。
    """
    if df.empty:
        return df

    cache = _load_cache()
    race_ids = df["race_id"].astype(str).unique().tolist()

    if fetch_missing:
        new_count = 0
        for rid in race_ids:
            if rid in cache and cache[rid] is not None:
                continue
            try:
                res = fetch_result(rid)
            except Exception as e:
                logger.warning(f"  ⚠ {rid} 結果取得エラー: {e}")
                res = None
            cache[rid] = res
            new_count += 1
            time.sleep(sleep_sec)
        if new_count:
            _save_cache(cache)
            logger.info(f"  結果キャッシュ更新: {new_count}件")

    hits, payouts, axis_in_top2, is_g1_list = [], [], [], []
    for _, row in df.iterrows():
        res = cache.get(str(row["race_id"]))
        if res is None:
            hits.append(None)
            payouts.append(None)
            axis_in_top2.append(None)
            is_g1_list.append(None)
            continue
        top2 = set(res["top2"])
        ni = str(row["horse_num_i"])
        nj = str(row["horse_num_j"])
        hit = (ni in top2) and (nj in top2)
        hits.append(hit)
        payouts.append(res["quinella_payout"] if hit else 0)
        axis_in_top2.append(ni in top2)
        is_g1_list.append(res["is_g1"])

    out = df.copy()
    out["hit"] = hits
    out["payout"] = payouts
    out["axis_in_top2"] = axis_in_top2
    out["is_g1"] = is_g1_list
    return out
