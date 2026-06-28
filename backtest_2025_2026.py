"""
backtest_2025_2026.py
=====================
2025〜2026年レースのバックテスト。
daily_batch.py と同一ロジック（モデル確率基準の買い目選択）を使用。

- 特徴量: feature_stats.pkl（inference モード）
- 買い条件: daily_batch.py と同一（季節フィルタ・EV フィルタ含む）
- 買い目選択: モデル確率降順（市場オッズ非依存・時刻ブレなし）
- オッズ推定: Harville 法（単勝オッズから推定）

実行:
    .venv/bin/python backtest_2025_2026.py
    .venv/bin/python backtest_2025_2026.py --year 2025
    .venv/bin/python backtest_2025_2026.py --year 2026
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations as _comb, permutations as _perm
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from config.settings import settings

# ── 定数（daily_batch.py と完全一致） ────────────────────────
MIN_HONMEI_PROB      = 0.30
MARK_STRONG_PROB     = 0.35
MIN_CONFIDENCE_GAP   = 0.05
EV_THRESHOLD         = 1.05
ENTRIES_ADJ          = 0.003
ENTRIES_BASE         = 10
ENTRIES_CAP          = 0.05
MAIDEN_KEYWORDS      = ("新馬", "未勝利")

_SEASON_PROB_THRESHOLD = {
    1: 0.38, 2: 0.38, 3: 0.33,
    4: 0.30, 5: 0.30, 6: 0.30,
    7: 0.38, 8: 0.38, 9: 0.38,
    10: 0.33, 11: 0.33, 12: 0.38,
}
_SUMMER_DIRT_SKIP_MONTHS = {7, 8, 9}
_MARK_O_SKIP_MONTHS      = {1, 2, 10}
_HIGH_VALUE_ODDS_MIN     = 8.0
_HIGH_VALUE_ODDS_MAX     = 15.0
_HIGH_VALUE_EV_THRESHOLD = 0.95
_BAD_SEASON_MONTHS       = {1, 7, 8, 9, 11, 12}
_BAD_VENUE_SKIP_CODES    = {"02", "04", "09"}
_BAD_SEASON_MAX_DISTANCE = 1800
_BAD_MODEL_PROB_THRESHOLD = 0.25
_BAD_MODEL_MARK_STRONG    = 0.28

EV_PARTNER_TOP_N       = 7
MAX_BAREN_TICKETS      = 3
MAX_SANRENFUKU_TICKETS = 7
MAX_SANRENTAN_TICKETS  = 7

JRA_TAKE = {"馬連": 0.225, "馬単": 0.25, "3連複": 0.225, "3連単": 0.275}

# ── ロガー ────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO",  format=_fmt, colorize=True)
logger.add("logs/backtest_2025_2026.log", level="DEBUG", format=_fmt, rotation="20 MB")


# ── ヘルパー ─────────────────────────────────────────────────
def _parse_odds(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")

def _market_probs(odds_list):
    raw = np.array([1.0 / max(o, 1.01) for o in odds_list])
    s = raw.sum()
    return (raw / s).tolist() if s > 0 else raw.tolist()

def _harville(probs, order):
    p, rem = 1.0, 1.0
    for idx in order:
        if rem < 1e-9:
            return 0.0
        p *= probs[idx] / rem
        rem -= probs[idx]
    return p

def _prob_quinella(probs, i, j):
    return _harville(probs, [i, j]) + _harville(probs, [j, i])

def _prob_trio(probs, i, j, k):
    return sum(_harville(probs, list(o)) for o in _perm([i, j, k]))

def _prob_sanrentan(probs, i, j, k):
    return _harville(probs, [i, j, k])

def _synth_odds(odds_list):
    denom = sum(1.0 / max(o, 1e-9) for o in odds_list)
    return 1.0 / denom if denom > 0 else 0.0

def _est_odds(prob, bet_type="3連複"):
    take = JRA_TAKE.get(bet_type, 0.225)
    return (1.0 - take) / max(prob, 0.001)


def is_buy_race(h1_prob, gap, race_month, is_maiden, n_entries,
                h1_ev, h1_odds, course_type, jyo_code, distance, using_bad_model):
    if is_maiden:
        return False

    if using_bad_model:
        prob_thr = _BAD_MODEL_PROB_THRESHOLD
    else:
        base = _SEASON_PROB_THRESHOLD.get(race_month, MIN_HONMEI_PROB)
        adj  = min(max(0, n_entries - ENTRIES_BASE) * ENTRIES_ADJ, ENTRIES_CAP)
        prob_thr = base + adj

    if gap < MIN_CONFIDENCE_GAP:
        return False
    if h1_prob < prob_thr:
        return False

    if np.isnan(h1_ev):
        ev_ok = True
    elif _HIGH_VALUE_ODDS_MIN <= h1_odds <= _HIGH_VALUE_ODDS_MAX:
        ev_ok = h1_ev >= _HIGH_VALUE_EV_THRESHOLD
    else:
        ev_ok = h1_ev >= EV_THRESHOLD
    if not ev_ok:
        return False

    if race_month in _SUMMER_DIRT_SKIP_MONTHS and course_type == "ダート":
        return False
    if race_month in _BAD_SEASON_MONTHS and jyo_code in _BAD_VENUE_SKIP_CODES:
        return False
    if race_month in _BAD_SEASON_MONTHS and distance >= _BAD_SEASON_MAX_DISTANCE:
        return False

    return True


def run_backtest(year_filter: int | None = None) -> pd.DataFrame:
    t0 = time.time()
    label = f"{year_filter}年" if year_filter else "2025-2026年"
    logger.info("=" * 65)
    logger.info(f"バックテスト開始: {label}")
    logger.info("=" * 65)

    # データ読み込み
    results_csv = ROOT / "data/raw/test_results_new.csv"
    meta_csv    = ROOT / "data/raw/test_meta_new.csv"
    results_df  = pd.read_csv(results_csv, dtype=str)
    meta_df     = pd.read_csv(meta_csv,    dtype=str)
    meta_df["race_date"] = pd.to_datetime(
        meta_df["race_id"].str[:8].apply(
            lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
        ), errors="coerce"
    )
    meta_df = meta_df.dropna(subset=["race_date"]).sort_values("race_date")

    if year_filter:
        meta_df    = meta_df[meta_df["race_date"].dt.year == year_filter]
        results_df = results_df[results_df["race_id"].isin(meta_df["race_id"])]

    logger.info(f"対象: {len(results_df):,}行 / {meta_df['race_id'].nunique():,}R "
                f"({meta_df['race_date'].min().date()} 〜 {meta_df['race_date'].max().date()})")

    # モデル読み込み
    trainer     = ModelTrainer.load(settings.model_path)
    fe          = FeatureEngineer.from_stats(settings.stats_path)
    bad_trainer = None
    bad_path    = ROOT / "data/models/lgbm_model_bad_season.pkl"
    if bad_path.exists():
        bad_trainer = ModelTrainer.load(bad_path)
        logger.info("  不調期専用モデル読み込み完了")

    records = []
    target_ids = meta_df["race_id"].tolist()
    buy_count = 0

    for i, race_id in enumerate(target_ids):
        entries = results_df[results_df["race_id"] == race_id].copy()
        if len(entries) < 4:
            continue

        meta_row    = meta_df[meta_df["race_id"] == race_id].iloc[0]
        race_date   = meta_row["race_date"]
        course_type = str(meta_row.get("course_type", ""))
        race_name   = str(meta_row.get("race_name", ""))
        distance    = int(meta_row.get("distance", 0) or 0)
        gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
        wx_code     = int(meta_row.get("weather_code", -1) or -1)
        jyo_code    = str(race_id)[4:6]
        race_month  = race_date.month

        if not course_type or distance == 0:
            continue

        entry_df = entries[["horse_id","horse_name","horse_number","frame_number",
                             "jockey_id","sex_age","weight_carried"]].copy()
        entry_df["sex"] = entry_df["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entry_df["sex_age"].str[1:], errors="coerce")
        entry_df["weight_carried"] = pd.to_numeric(entry_df["weight_carried"], errors="coerce")
        entry_df["father"] = ""
        entry_df["mother_father"] = ""
        if "odds" in entries.columns:
            entry_df["odds"] = pd.to_numeric(entries["odds"].values, errors="coerce")

        race_class_code = FeatureEngineer._race_name_to_class_code(race_name)
        try:
            venue_code = int(race_id[4:6])
        except Exception:
            venue_code = -1

        try:
            feat_df = fe.build_entry_features(
                entry_df=entry_df,
                course_type=course_type,
                distance=distance,
                ground_condition_code=gc_code,
                weather_code=wx_code,
                race_class_code=race_class_code,
                venue_code=venue_code,
            )
        except Exception as e:
            logger.debug(f"skip {race_id}: {e}")
            continue

        using_bad = (bad_trainer is not None and race_month in _BAD_SEASON_MONTHS)
        act = bad_trainer if using_bad else trainer
        X = feat_df[FeatureEngineer.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
        win_probs = act.model.predict(X, num_threads=1)
        if act.place_model is not None:
            place_probs = act.place_model.predict(X, num_threads=1)
            probs = 0.7 * win_probs + 0.3 * place_probs
        else:
            probs = win_probs

        pred_df = feat_df[["horse_id","horse_name","horse_number"]].copy()
        pred_df["win_prob"] = probs
        pred_df = pred_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

        if len(pred_df) < 3:
            continue

        h1_prob = float(pred_df.iloc[0]["win_prob"])
        h2_prob = float(pred_df.iloc[1]["win_prob"])
        gap     = h1_prob - h2_prob
        n_ent   = len(entries)
        is_maiden = any(k in race_name for k in MAIDEN_KEYWORDS)

        odds_map = {}
        if "odds" in feat_df.columns:
            odds_map = feat_df.set_index("horse_id")["odds"].dropna().to_dict()

        h1_id   = str(pred_df.iloc[0]["horse_id"])
        h1_odds = float(odds_map.get(h1_id, float("nan")))
        h1_ev   = h1_prob * h1_odds if not np.isnan(h1_odds) else float("nan")

        buy = is_buy_race(h1_prob, gap, race_month, is_maiden, n_ent,
                          h1_ev, h1_odds, course_type, jyo_code, distance, using_bad)

        ms = _BAD_MODEL_MARK_STRONG if using_bad else MARK_STRONG_PROB
        mark = "◎" if buy and h1_prob >= ms else ("○" if buy else "△")
        if mark == "○" and race_month in _MARK_O_SKIP_MONTHS:
            mark = "△"
            buy = False

        if not buy:
            continue

        buy_count += 1

        # 実際の着順
        actual = entries[["horse_id","horse_number","finish_position","odds"]].copy()
        actual["pos"] = pd.to_numeric(
            actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        actual["odds_f"] = actual["odds"].apply(_parse_odds)
        actual = actual.dropna(subset=["pos"]).sort_values("pos")
        if len(actual) < 3:
            continue

        top3 = actual.iloc[:3]
        pos1_id  = str(top3.iloc[0]["horse_id"])
        pos2_id  = str(top3.iloc[1]["horse_id"])
        pos3_id  = str(top3.iloc[2]["horse_id"])
        pos1_num = str(int(float(top3.iloc[0]["horse_number"])))
        pos2_num = str(int(float(top3.iloc[1]["horse_number"])))
        pos3_num = str(int(float(top3.iloc[2]["horse_number"])))
        win_odds = float(top3.iloc[0]["odds_f"])

        honmei_id  = str(pred_df.iloc[0]["horse_id"])
        honmei_num = str(int(float(pred_df.iloc[0]["horse_number"])))

        # 市場確率（Harville用）
        valid_ids = pred_df["horse_id"].astype(str).tolist()
        odds_list = [float(odds_map.get(hid, float("nan"))) for hid in valid_ids]
        odds_list = [o if not np.isnan(o) and o > 1.0 else 5.0 for o in odds_list]
        mkt_probs = _market_probs(odds_list)

        # モデル確率（正規化）
        probs_raw = pred_df["win_prob"].tolist()
        ps = sum(probs_raw)
        model_probs = [p / ps for p in probs_raw] if ps > 0 else probs_raw

        # ◎のインデックス
        hi = valid_ids.index(honmei_id) if honmei_id in valid_ids else 0

        # パートナープール（◎除く上位7頭）
        partner_rows = pred_df[pred_df["horse_id"] != honmei_id].head(EV_PARTNER_TOP_N)
        pool = []
        for _, row in partner_rows.iterrows():
            hid = str(row["horse_id"])
            num = str(int(float(row["horse_number"])))
            vi  = valid_ids.index(hid) if hid in valid_ids else None
            pool.append((num, vi))

        partner_idx = {num: i + 1 for i, (num, _) in enumerate(pool)}

        # ─── 単勝 ───
        tansho_hit = int(honmei_id == pos1_id)
        tansho_ret = win_odds * 100 if tansho_hit else 0.0

        # ─── 馬連（モデル確率順上位3頭から組み合わせ） ───
        baren_pool = [num for num, _ in pool[:3]]
        baren_combos = [(honmei_num, p) for p in baren_pool]
        actual_baren = {pos1_num, pos2_num}
        baren_hit = int(any(set([h, p]) == actual_baren for h, p in baren_combos))

        baren_ret = 0.0
        if baren_hit:
            vi1 = valid_ids.index(pos1_id) if pos1_id in valid_ids else None
            vi2 = valid_ids.index(pos2_id) if pos2_id in valid_ids else None
            if vi1 is not None and vi2 is not None:
                mkt_p = _prob_quinella(mkt_probs, vi1, vi2)
                baren_ret = _est_odds(mkt_p, "馬連") * 100

        baren_cost = len(baren_combos) * 100

        # ─── 3連複（モデル確率降順） ───
        sf_all = []
        for (na, vi_a), (nb, vi_b) in _comb(pool, 2):
            pa = partner_idx.get(na, 1)
            pb = partner_idx.get(nb, 2)
            model_p = _prob_trio(model_probs, 0, pa, pb)
            if hi is not None and vi_a is not None and vi_b is not None:
                mkt_p = _prob_trio(mkt_probs, hi, vi_a, vi_b)
                e_od  = _est_odds(mkt_p, "3連複")
            else:
                e_od  = _est_odds(model_p, "3連複")
            sf_all.append((na, nb, model_p, e_od))

        sf_all.sort(key=lambda x: -x[2])  # モデル確率降順
        sf_sel = sf_all[:MAX_SANRENFUKU_TICKETS]
        sf_est = [od for _, _, _, od in sf_sel]
        sf_combos = (
            [(a, b) for a, b, _, _ in sf_sel]
            if (not sf_est or _synth_odds(sf_est) >= 1.0)
            else []
        )

        actual_trio = {pos1_num, pos2_num, pos3_num}
        sf_hit = int(any(set([honmei_num, a, b]) == actual_trio for a, b in sf_combos))
        sf_ret = 0.0
        if sf_hit:
            vi1 = valid_ids.index(pos1_id) if pos1_id in valid_ids else None
            vi2 = valid_ids.index(pos2_id) if pos2_id in valid_ids else None
            vi3 = valid_ids.index(pos3_id) if pos3_id in valid_ids else None
            if vi1 is not None and vi2 is not None and vi3 is not None:
                mkt_p = _prob_trio(mkt_probs, vi1, vi2, vi3)
                sf_ret = _est_odds(mkt_p, "3連複") * 100
        sf_cost = len(sf_combos) * 100

        # ─── 3連単（モデル確率降順） ───
        st_all = []
        for (n2, vi_2), (n3, vi_3) in _perm(pool, 2):
            p2 = partner_idx.get(n2, 1)
            p3 = partner_idx.get(n3, 2)
            model_p = _prob_sanrentan(model_probs, 0, p2, p3)
            if hi is not None and vi_2 is not None and vi_3 is not None:
                mkt_p = _prob_sanrentan(mkt_probs, hi, vi_2, vi_3)
                e_od  = _est_odds(mkt_p, "3連単")
            else:
                e_od  = _est_odds(model_p, "3連単")
            st_all.append((n2, n3, model_p, e_od))

        st_all.sort(key=lambda x: -x[2])  # モデル確率降順
        st_sel = st_all[:MAX_SANRENTAN_TICKETS]
        st_est = [od for _, _, _, od in st_sel]
        st_combos = (
            [(n2, n3) for n2, n3, _, _ in st_sel]
            if (not st_est or _synth_odds(st_est) >= 1.0)
            else []
        )

        st_hit = int(
            honmei_num == pos1_num and
            any(n2 == pos2_num and n3 == pos3_num for n2, n3 in st_combos)
        )
        st_ret = 0.0
        if st_hit:
            vi2 = valid_ids.index(pos2_id) if pos2_id in valid_ids else None
            vi3 = valid_ids.index(pos3_id) if pos3_id in valid_ids else None
            if vi2 is not None and vi3 is not None:
                mkt_p = _prob_sanrentan(mkt_probs, hi, vi2, vi3)
                st_ret = _est_odds(mkt_p, "3連単") * 100
        st_cost = len(st_combos) * 100

        total_cost = 100 + baren_cost + sf_cost + st_cost
        total_ret  = tansho_ret + baren_ret + sf_ret + st_ret

        records.append({
            "race_id":     race_id,
            "race_date":   str(race_date)[:10],
            "race_name":   race_name,
            "race_month":  race_month,
            "year":        race_date.year,
            "course_type": course_type,
            "distance":    distance,
            "mark":        mark,
            "n_entries":   n_ent,
            "h1_prob":     round(h1_prob, 4),
            "h2_prob":     round(h2_prob, 4),
            "gap":         round(gap, 4),
            "h1_odds":     round(h1_odds, 1) if not np.isnan(h1_odds) else None,
            "honmei_num":  honmei_num,
            "honmei_name": pred_df.iloc[0]["horse_name"],
            "actual_1st":  pos1_num,
            "actual_2nd":  pos2_num,
            "actual_3rd":  pos3_num,
            "tansho_hit":  tansho_hit,
            "tansho_ret":  round(tansho_ret, 1),
            "baren_hit":   baren_hit,
            "baren_cost":  baren_cost,
            "baren_ret":   round(baren_ret, 1),
            "sf_hit":      sf_hit,
            "sf_cost":     sf_cost,
            "sf_ret":      round(sf_ret, 1),
            "st_hit":      st_hit,
            "st_cost":     st_cost,
            "st_ret":      round(st_ret, 1),
            "total_cost":  total_cost,
            "total_ret":   round(total_ret, 1),
        })

        if (i + 1) % 500 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(f"  {i+1}/{len(target_ids)} レース処理済み  買い{buy_count}R  ({elapsed:.1f}分)")

    df = pd.DataFrame(records)
    elapsed = (time.time() - t0) / 60
    logger.info(f"完了: 買い対象 {len(df):,}R  ({elapsed:.1f}分)")
    return df


def print_summary(df: pd.DataFrame, title: str) -> None:
    border = "=" * 68
    print(f"\n{border}")
    print(f"  {title}")
    print(f"  期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  買い対象: {len(df):,} R")
    print(border)

    def roi(cost, ret):
        return ret / cost * 100 if cost > 0 else 0.0

    # 総合
    tc = df["total_cost"].sum()
    tr = df["total_ret"].sum()
    print(f"\n  【総合（単勝+馬連+3連複+3連単）】")
    print(f"    投票: {int(tc):,}円  回収: {int(tr):,}円  ROI: {roi(tc,tr):.1f}%")

    # 馬券種別
    print(f"\n  【馬券種別 ROI】")
    for bet, cost_col, ret_col in [
        ("単勝", None, "tansho_ret"),
        ("馬連", "baren_cost", "baren_ret"),
        ("3連複", "sf_cost", "sf_ret"),
        ("3連単", "st_cost", "st_ret"),
    ]:
        if cost_col:
            c = df[cost_col].sum()
            r = df[ret_col].sum()
            h = df[ret_col.replace("_ret", "_hit")].sum()
        else:
            c = len(df) * 100
            r = df[ret_col].sum()
            h = df["tansho_hit"].sum()
        print(f"    {bet:<5}: 投票{int(c):>8,}円  回収{int(r):>9,}円  "
              f"ROI {roi(c,r):>6.1f}%  的中{int(h):>3}/{len(df)}R ({h/len(df):.1%})")

    # 月別
    print(f"\n  【月別 ROI（総合）】")
    print(f"  {'月':>3}  {'R数':>4}  {'単勝':>7}  {'馬連':>7}  {'3連複':>7}  {'3連単':>7}  {'総合':>7}")
    for m, g in df.groupby("race_month"):
        r_ts = roi(len(g)*100, g["tansho_ret"].sum())
        r_br = roi(g["baren_cost"].sum(), g["baren_ret"].sum())
        r_sf = roi(g["sf_cost"].sum(), g["sf_ret"].sum())
        r_st = roi(g["st_cost"].sum(), g["st_ret"].sum())
        r_to = roi(g["total_cost"].sum(), g["total_ret"].sum())
        print(f"  {m:>3}月  {len(g):>4}R  {r_ts:>6.0f}%  {r_br:>6.0f}%  {r_sf:>6.0f}%  "
              f"{r_st:>6.0f}%  {r_to:>6.0f}%")

    # コースタイプ別
    print(f"\n  【芝/ダート別 ROI（総合）】")
    for ct, g in df.groupby("course_type"):
        r = roi(g["total_cost"].sum(), g["total_ret"].sum())
        h_sf = g["sf_hit"].sum()
        print(f"    {ct}: {len(g)}R  ROI {r:.1f}%  3連複的中{int(h_sf)}R")

    # gap 別
    print(f"\n  【確率差(gap)別 ROI（総合）】")
    df["gap_bin"] = pd.cut(df["gap"], bins=[0,.05,.10,.15,.25,1.],
                           labels=["0-5%","5-10%","10-15%","15-25%","25%+"])
    for gb, g in df.groupby("gap_bin", observed=True):
        if len(g) == 0:
            continue
        r_sf = roi(g["sf_cost"].sum(), g["sf_ret"].sum())
        r_st = roi(g["st_cost"].sum(), g["st_ret"].sum())
        r_to = roi(g["total_cost"].sum(), g["total_ret"].sum())
        print(f"    {str(gb):>7}  {len(g):>4}R  3連複:{r_sf:>6.1f}%  3連単:{r_st:>6.1f}%  総合:{r_to:>6.1f}%")

    # 的中した3連複上位配当
    sf_hits = df[df["sf_hit"] == 1].sort_values("sf_ret", ascending=False)
    if len(sf_hits) > 0:
        print(f"\n  【3連複 的中上位10件（推定配当）】")
        for _, row in sf_hits.head(10).iterrows():
            print(f"    {row['race_date']} {row['race_name']}  "
                  f"◎{row['honmei_num']}番{row['honmei_name']}  "
                  f"推定{row['sf_ret']:.0f}円")

    # 的中した3連単上位配当
    st_hits = df[df["st_hit"] == 1].sort_values("st_ret", ascending=False)
    if len(st_hits) > 0:
        print(f"\n  【3連単 的中上位10件（推定配当）】")
        for _, row in st_hits.head(10).iterrows():
            print(f"    {row['race_date']} {row['race_name']}  "
                  f"◎{row['honmei_num']}番{row['honmei_name']}  "
                  f"推定{row['st_ret']:.0f}円")

    print(f"\n{border}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        choices=[2025, 2026], help="絞り込む年（省略時は全期間）")
    args = parser.parse_args()

    df = run_backtest(year_filter=args.year)

    if len(df) == 0:
        print("買い対象レースなし")
        sys.exit(0)

    year_label = f"{args.year}年" if args.year else "2025-2026年"
    print_summary(df, f"バックテスト結果: {year_label}（モデル確率基準）")

    # 年別サマリー（全期間時）
    if args.year is None and df["year"].nunique() > 1:
        for yr, g in df.groupby("year"):
            print_summary(g, f"バックテスト結果: {yr}年")

    # CSV 保存
    suffix = f"_{args.year}" if args.year else ""
    out = ROOT / f"data/processed/backtest_2025_2026{suffix}.csv"
    df.to_csv(out, index=False)
    logger.info(f"詳細CSV保存: {out}")
