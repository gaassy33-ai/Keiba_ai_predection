"""
backtest_full.py
================
新モデル（Platt scaling + odds_log/popularity_rank_norm）対応
2024〜2026年 統合バックテスト。

- 2024年: train_results.csv を lookback モードで使用（データリーク防止）
- 2025-2026年: test_results_new.csv + feature_stats.pkl で inference モード

買い条件は daily_batch.py の現行ロジックと完全一致。

実行例:
    .venv/bin/python backtest_full.py              # 2024-2026 全期間
    .venv/bin/python backtest_full.py --year 2024
    .venv/bin/python backtest_full.py --year 2025
    .venv/bin/python backtest_full.py --year 2026
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

# ── 買い条件（daily_batch.py 2026-04-20 現行値と完全一致） ──────────────
MIN_HONMEI_PROB       = 0.30
MARK_STRONG_PROB      = 0.40
MIN_CONFIDENCE_GAP    = 0.07
CALIBRATION_FACTOR    = 1.0    # Platt scaling 済みモデルのため 1.0
EV_THRESHOLD          = 1.05
ENTRIES_ADJ           = 0.003
ENTRIES_BASE          = 10
ENTRIES_CAP           = 0.05
MAIDEN_KEYWORDS       = ("新馬", "未勝利")

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
_HIGH_VALUE_EV_THRESHOLD = 1.00
_MAX_LONGSHOT_ODDS       = 20.0
_BAD_SEASON_MONTHS       = {1, 7, 8, 9, 11, 12}
_BAD_VENUE_SKIP_CODES    = {"02", "04", "09"}
_BAD_SEASON_MAX_DISTANCE = 1800
_BAD_MODEL_PROB_THRESHOLD = 0.25
_BAD_MODEL_MARK_STRONG    = 0.28

EV_PARTNER_TOP_N       = 7
MAX_BAREN_TICKETS      = 2    # quinella EV 上位2点（EV > EV_BAREN_THRESHOLD のみ）
MAX_SANRENFUKU_TICKETS = 5    # EV選択・5点
EV_BAREN_THRESHOLD     = 0.0   # 馬連: quinella EV 閾値（0.0=フィルタなし、確率順で選択）
MAX_SANRENTAN_TICKETS  = 3    # 確率上位3点（5→3、コスト削減でROI改善 2026-04-27）
JRA_TAKE = {"馬連": 0.225, "馬単": 0.25, "3連複": 0.225, "3連単": 0.275}

# ── ロガー ────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/backtest_full.log", level="DEBUG", format=_fmt, rotation="30 MB")


# ── ヘルパー ────────────────────────────────────────────────────────
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


def _apply_model(act: ModelTrainer, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Platt scaling 込みの確率を返す。

    Returns
    -------
    win_probs : np.ndarray
        較正済み勝利確率（Harville モデルに使用）
    blended_probs : np.ndarray
        0.7×勝利 + 0.3×複勝 のアンサンブル確率（馬の順位付けに使用）
    """
    win_raw = act.model.predict(X, num_threads=1)
    if act.calibrator is not None:
        win_probs = act.calibrator.predict_proba(win_raw.reshape(-1, 1))[:, 1]
    else:
        win_probs = win_raw

    if act.place_model is not None:
        place_raw = act.place_model.predict(X, num_threads=1)
        # place_model には専用の place_calibrator を使用
        # （base rate: 勝利≈8% vs 複勝≈32% のため win calibrator の流用は不正確）
        if getattr(act, "place_calibrator", None) is not None:
            place_probs = act.place_calibrator.predict_proba(place_raw.reshape(-1, 1))[:, 1]
        elif act.calibrator is not None:
            # 後方互換: 旧モデル（place_calibrator なし）は win calibrator で代替
            place_probs = act.calibrator.predict_proba(place_raw.reshape(-1, 1))[:, 1]
        else:
            place_probs = place_raw
        blended = 0.7 * win_probs + 0.3 * place_probs
        return win_probs, blended, place_probs
    return win_probs, win_probs, win_probs  # place_probs なし時は win_probs で代替


def is_buy_race(h1_prob, gap, month, is_maiden, n_entries,
                h1_ev, h1_odds, course_type, jyo_code, distance, using_bad):
    """daily_batch.py の is_buy 条件と完全一致。"""
    if is_maiden:
        return False

    prob_thr = (
        _BAD_MODEL_PROB_THRESHOLD
        if using_bad
        else _SEASON_PROB_THRESHOLD.get(month, MIN_HONMEI_PROB)
             + min(max(0, n_entries - ENTRIES_BASE) * ENTRIES_ADJ, ENTRIES_CAP)
    )
    if gap < MIN_CONFIDENCE_GAP:
        return False
    if h1_prob < prob_thr:
        return False

    # ロングショットフィルタ
    if not np.isnan(h1_odds) and h1_odds > _MAX_LONGSHOT_ODDS:
        return False

    if np.isnan(h1_ev):
        ev_ok = True
    elif _HIGH_VALUE_ODDS_MIN <= h1_odds <= _HIGH_VALUE_ODDS_MAX:
        ev_ok = h1_ev >= _HIGH_VALUE_EV_THRESHOLD
    else:
        ev_ok = h1_ev >= EV_THRESHOLD
    if not ev_ok:
        return False

    if month in _SUMMER_DIRT_SKIP_MONTHS and course_type == "ダート":
        return False
    if month in _BAD_SEASON_MONTHS and jyo_code in _BAD_VENUE_SKIP_CODES:
        return False
    if month in _BAD_SEASON_MONTHS and distance >= _BAD_SEASON_MAX_DISTANCE:
        return False

    return True


def _process_race(race_id, entries, meta_row, fe, act, bad_trainer) -> dict | None:
    """1レースを処理してレコードを返す（買い対象でなければ None）。"""
    race_date   = meta_row["race_date"]
    course_type = str(meta_row.get("course_type", ""))
    race_name   = str(meta_row.get("race_name", ""))
    distance    = int(meta_row.get("distance", 0) or 0)
    gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
    wx_code     = int(meta_row.get("weather_code", -1) or -1)
    jyo_code    = str(race_id)[4:6]
    month       = race_date.month

    if not course_type or distance == 0 or distance >= 2750:
        return None

    entry_df = entries[["horse_id", "horse_name", "horse_number", "frame_number",
                         "jockey_id"]].copy()
    # sex/age
    if "sex_age" in entries.columns:
        entry_df["sex"] = entries["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entries["sex_age"].str[1:], errors="coerce")
    else:
        entry_df["sex"] = ""
        entry_df["age"] = np.nan
    entry_df["weight_carried"] = pd.to_numeric(entries.get("weight_carried", pd.Series()), errors="coerce")
    entry_df["father"] = ""
    entry_df["mother_father"] = ""

    # オッズ（inference モード用 / lookback モードでは build_entry_features 内で HHI に使用）
    if "odds" in entries.columns:
        odds_raw = pd.to_numeric(entries["odds"], errors="coerce")
        # 正常オッズ判定（最小値 < 15 なら実際の単勝オッズ）
        if odds_raw.dropna().min() < 15.0:
            entry_df["odds"] = odds_raw.values
        elif "last_3f" in entries.columns:
            entry_df["odds"] = pd.to_numeric(entries["last_3f"].values, errors="coerce")
    elif "last_3f" in entries.columns:
        entry_df["odds"] = pd.to_numeric(entries["last_3f"].values, errors="coerce")

    rcc = FeatureEngineer._race_name_to_class_code(race_name)
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
            race_class_code=rcc,
            venue_code=venue_code,
        )
    except Exception as e:
        logger.debug(f"skip {race_id}: {e}")
        return None

    using_bad = (bad_trainer is not None and month in _BAD_SEASON_MONTHS)
    model = bad_trainer if using_bad else act

    X = feat_df[FeatureEngineer.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    win_probs_arr, blended_probs_arr, place_probs_arr = _apply_model(model, X)

    pred_df = feat_df[["horse_id", "horse_name", "horse_number"]].copy()
    pred_df["win_prob"]      = blended_probs_arr  # 順位付け・表示にはアンサンブル確率を使用
    pred_df["win_prob_pure"] = win_probs_arr       # Harville 確率計算には純粋勝利確率を使用
    pred_df["place_prob"]    = place_probs_arr     # 複勝確率（3連複パートナー選択用）
    pred_df = pred_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

    if len(pred_df) < 3:
        return None

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
    h1_ev   = h1_prob * CALIBRATION_FACTOR * h1_odds if not np.isnan(h1_odds) else float("nan")

    buy = is_buy_race(h1_prob, gap, month, is_maiden, n_ent,
                      h1_ev, h1_odds, course_type, jyo_code, distance, using_bad)

    ms   = _BAD_MODEL_MARK_STRONG if using_bad else MARK_STRONG_PROB
    mark = "◎" if buy and h1_prob >= ms else ("○" if buy else "△")
    if mark == "○" and month in _MARK_O_SKIP_MONTHS:
        mark = "△"
        buy  = False

    if not buy:
        return None

    # 実際の着順
    actual = entries[["horse_id", "horse_number", "finish_position"]].copy()
    actual["pos"]    = pd.to_numeric(
        actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    actual["odds_f"] = entries["odds"].apply(_parse_odds) if "odds" in entries.columns else np.nan
    actual = actual.dropna(subset=["pos"]).sort_values("pos")
    if len(actual) < 3:
        return None

    top3 = actual.iloc[:3]
    p1id, p2id, p3id = (str(top3.iloc[i]["horse_id"]) for i in range(3))
    p1n, p2n, p3n    = (str(int(float(top3.iloc[i]["horse_number"]))) for i in range(3))
    win_odds = float(top3.iloc[0]["odds_f"])

    honmei_id  = str(pred_df.iloc[0]["horse_id"])
    honmei_num = str(int(float(pred_df.iloc[0]["horse_number"])))

    valid_ids = pred_df["horse_id"].astype(str).tolist()
    odds_list = [float(odds_map.get(hid, float("nan"))) for hid in valid_ids]
    odds_list = [o if not np.isnan(o) and o > 1.0 else 5.0 for o in odds_list]
    mkt_probs = _market_probs(odds_list)
    # Harville には純粋勝利確率を使用（アンサンブル blend では Harville 公式が崩れる）
    probs_raw = pred_df["win_prob_pure"].tolist()
    ps = sum(probs_raw)
    model_probs = [p / ps for p in probs_raw] if ps > 0 else probs_raw
    hi = valid_ids.index(honmei_id) if honmei_id in valid_ids else 0

    # ── pred_df ランク（win_prob 降順）→ Harville index マップ ────────────
    _num_to_pred_rank: dict[str, int] = {
        str(int(float(row["horse_number"]))): idx
        for idx, (_, row) in enumerate(pred_df.iterrows())
    }

    # 馬連プール: Harville quinella確率（◎とpartnerが上位2着に入る理論確率）で選択
    # 理論的に最適な選択法: P(◎1着,partner2着) + P(partner1着,◎2着) を最大化
    # 市場フィルタ: partner odds > 30 倍はほぼ的中期待なしのため除外
    baren_candidates: list[tuple[str, float]] = []
    for _, row in pred_df[pred_df["horse_id"].astype(str) != honmei_id].iterrows():
        pnum = str(int(float(row["horse_number"])))
        phid = str(row["horse_id"])
        vi_p = valid_ids.index(phid) if phid in valid_ids else None
        p_odds_val = float(odds_map.get(phid, float("nan")))
        if not np.isnan(p_odds_val) and p_odds_val > 30.0:
            continue  # ロングショット除外
        if vi_p is not None and hi is not None:
            q_prob = _prob_quinella(model_probs, hi, vi_p)  # Harville quinella確率
        else:
            q_prob = float(row["win_prob"])  # fallback
        baren_candidates.append((pnum, q_prob))
    baren_candidates.sort(key=lambda x: -x[1])
    baren_pool = [
        pnum for pnum, q in baren_candidates[:MAX_BAREN_TICKETS]
        if q >= EV_BAREN_THRESHOLD  # 0.0=フィルタなし
    ]

    # 3連複/3連単プール: place_prob 上位 EV_PARTNER_TOP_N（◎除く、daily_batch.py準拠）
    partner_rows_3f = pred_df[
        pred_df["horse_id"].astype(str) != honmei_id
    ].sort_values("place_prob", ascending=False).head(EV_PARTNER_TOP_N)
    pool = []
    for _, row in partner_rows_3f.iterrows():
        hid = str(row["horse_id"])
        num = str(int(float(row["horse_number"])))
        vi  = valid_ids.index(hid) if hid in valid_ids else None
        pool.append((num, vi))

    # 単勝
    tansho_hit = int(honmei_id == p1id)
    tansho_ret = win_odds * 100 if tansho_hit else 0.0

    # 馬連（win_prob 上位 MAX_BAREN_TICKETS 点）
    baren_combos = [(honmei_num, num) for num in baren_pool]
    actual_baren = {p1n, p2n}
    baren_hit    = int(any(set([h, p]) == actual_baren for h, p in baren_combos))
    baren_ret    = 0.0
    if baren_hit:
        vi1 = valid_ids.index(p1id) if p1id in valid_ids else None
        vi2 = valid_ids.index(p2id) if p2id in valid_ids else None
        if vi1 is not None and vi2 is not None:
            baren_ret = _est_odds(_prob_quinella(mkt_probs, vi1, vi2), "馬連") * 100
    baren_cost = len(baren_combos) * 100

    # ── 3連複: place_prob上位プール × EV（model_p×e_od）降順ソート・5点──────────
    # Harville index: pred_df の win_prob 降順ランクを使用（daily_batch.py と同一ロジック）
    _partner_pred_idxs_3f = {
        num: _num_to_pred_rank.get(num, len(model_probs) - 1)
        for num, _vi in pool
    }
    sf_all = []
    for (na, vi_a), (nb, vi_b) in _comb(pool, 2):
        pa = _partner_pred_idxs_3f.get(na, 1)
        pb = _partner_pred_idxs_3f.get(nb, 2)
        mp = _prob_trio(model_probs, 0, pa, pb)
        eo = (_est_odds(_prob_trio(mkt_probs, hi, vi_a, vi_b), "3連複")
              if hi is not None and vi_a is not None and vi_b is not None
              else _est_odds(mp, "3連複"))
        sf_all.append((na, nb, mp, eo))
    sf_all.sort(key=lambda x: -(x[2] * x[3]))  # EV = model_p × e_od 降順
    sf_sel    = sf_all[:MAX_SANRENFUKU_TICKETS]
    sf_combos = ([(a, b) for a, b, _, _ in sf_sel]
                 if (not sf_sel or _synth_odds([od for _, _, _, od in sf_sel]) >= 1.0)
                 else [])
    actual_trio = {p1n, p2n, p3n}
    sf_hit = int(any(set([honmei_num, a, b]) == actual_trio for a, b in sf_combos))
    sf_ret = 0.0
    if sf_hit:
        vi1 = valid_ids.index(p1id) if p1id in valid_ids else None
        vi2 = valid_ids.index(p2id) if p2id in valid_ids else None
        vi3 = valid_ids.index(p3id) if p3id in valid_ids else None
        if vi1 is not None and vi2 is not None and vi3 is not None:
            sf_ret = _est_odds(_prob_trio(mkt_probs, vi1, vi2, vi3), "3連複") * 100
    sf_cost = len(sf_combos) * 100

    # ── 3連単: place_prob上位プール × EV降順ソート・5点──────────────────────────
    st_all = []
    for (n2, vi_2), (n3, vi_3) in _perm(pool, 2):
        p2 = _partner_pred_idxs_3f.get(n2, 1)
        p3 = _partner_pred_idxs_3f.get(n3, 2)
        mp = _prob_sanrentan(model_probs, 0, p2, p3)
        eo = (_est_odds(_prob_sanrentan(mkt_probs, hi, vi_2, vi_3), "3連単")
              if hi is not None and vi_2 is not None and vi_3 is not None
              else _est_odds(mp, "3連単"))
        st_all.append((n2, n3, mp, eo))
    st_all.sort(key=lambda x: -x[2])  # model_p 降順（EV → 確率重視に変更、hit率改善）
    st_sel    = st_all[:MAX_SANRENTAN_TICKETS]
    st_combos = ([(n2, n3) for n2, n3, _, _ in st_sel]
                 if (not st_sel or _synth_odds([od for _, _, _, od in st_sel]) >= 1.0)
                 else [])
    st_hit = int(honmei_num == p1n and
                 any(n2 == p2n and n3 == p3n for n2, n3 in st_combos))
    st_ret = 0.0
    if st_hit:
        vi2 = valid_ids.index(p2id) if p2id in valid_ids else None
        vi3 = valid_ids.index(p3id) if p3id in valid_ids else None
        if vi2 is not None and vi3 is not None:
            st_ret = _est_odds(_prob_sanrentan(mkt_probs, hi, vi2, vi3), "3連単") * 100
    st_cost = len(st_combos) * 100

    return {
        "race_id":    race_id,
        "race_date":  str(race_date)[:10],
        "race_name":  race_name,
        "year":       race_date.year,
        "race_month": month,
        "course_type": course_type,
        "distance":   distance,
        "mark":       mark,
        "n_entries":  n_ent,
        "h1_prob":    round(h1_prob, 4),
        "h2_prob":    round(h2_prob, 4),
        "gap":        round(gap, 4),
        "h1_odds":    round(h1_odds, 1) if not np.isnan(h1_odds) else None,
        "h1_ev":      round(h1_ev, 3) if not np.isnan(h1_ev) else None,
        "honmei_num": honmei_num,
        "honmei_name": pred_df.iloc[0]["horse_name"],
        "actual_1st": p1n, "actual_2nd": p2n, "actual_3rd": p3n,
        "tansho_hit": tansho_hit, "tansho_ret": round(tansho_ret, 1),
        "baren_hit":  baren_hit,  "baren_cost": baren_cost,  "baren_ret":  round(baren_ret, 1),
        "sf_hit":     sf_hit,     "sf_cost":    sf_cost,     "sf_ret":     round(sf_ret, 1),
        "st_hit":     st_hit,     "st_cost":    st_cost,     "st_ret":     round(st_ret, 1),
        "total_cost": 100 + baren_cost + sf_cost + st_cost,
        "total_ret":  round(tansho_ret + baren_ret + sf_ret + st_ret, 1),
    }


# ── 2024: lookback モード ────────────────────────────────────────────
def run_2024(act: ModelTrainer, bad_trainer) -> pd.DataFrame:
    logger.info("=" * 65)
    logger.info("2024年バックテスト（lookback モード・データリークなし）")
    logger.info("=" * 65)
    t0 = time.time()

    all_res = pd.read_csv(ROOT / "data/raw/train_results.csv", dtype=str)
    meta    = pd.read_csv(ROOT / "data/raw/train_meta.csv", dtype=str)

    # meta の race_date 列を使用（race_id からのパースは会場コードを誤って月扱いするため NG）
    meta["race_date"] = pd.to_datetime(meta["race_date"], errors="coerce")
    meta = meta.dropna(subset=["race_date"]).sort_values("race_date")
    meta24 = meta[meta["race_date"].dt.year == 2024].copy()

    # train_results.csv には race_date がないので meta から結合して付与
    date_map = meta.set_index("race_id")["race_date"].to_dict()
    all_res["race_date"] = all_res["race_id"].map(date_map)

    logger.info(f"対象: {len(meta24):,}R  ({meta24['race_date'].min().date()} 〜 {meta24['race_date'].max().date()})")

    records = []
    unique_dates = sorted(meta24["race_date"].unique())

    for di, rd in enumerate(unique_dates):
        date_races = meta24[meta24["race_date"] == rd]
        hist_before = all_res[all_res["race_date"] < rd]
        try:
            fe = FeatureEngineer(hist_before)
            fe.precompute_aggregations()
        except Exception as e:
            logger.warning(f"skip date {rd}: {e}")
            continue

        for _, meta_row in date_races.iterrows():
            race_id = str(meta_row["race_id"])
            entries = all_res[all_res["race_id"] == race_id].copy()
            if len(entries) < 4:
                continue
            rec = _process_race(race_id, entries, meta_row, fe, act, bad_trainer)
            if rec:
                records.append(rec)

        if (di + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(f"  {di+1}/{len(unique_dates)} 日付済み  買い{len(records)}R  ({elapsed:.1f}分)")

    df = pd.DataFrame(records)
    logger.info(f"2024年完了: {len(df):,}R  ({(time.time()-t0)/60:.1f}分)")
    return df


# ── 2025-2026: inference モード ─────────────────────────────────────
def run_2025_2026(act: ModelTrainer, bad_trainer, year_filter: int | None) -> pd.DataFrame:
    label = f"{year_filter}年" if year_filter else "2025-2026年"
    logger.info("=" * 65)
    logger.info(f"{label}バックテスト（inference モード・feature_stats.pkl）")
    logger.info("=" * 65)
    t0 = time.time()

    results_df = pd.read_csv(ROOT / "data/raw/test_results_new.csv", dtype=str)
    meta_df    = pd.read_csv(ROOT / "data/raw/test_meta_new.csv", dtype=str)

    # meta の race_date 列を使用（race_id からのパースは NG）
    meta_df["race_date"] = pd.to_datetime(meta_df["race_date"], errors="coerce")
    meta_df = meta_df.dropna(subset=["race_date"]).sort_values("race_date")
    if year_filter:
        meta_df    = meta_df[meta_df["race_date"].dt.year == year_filter]
        results_df = results_df[results_df["race_id"].isin(meta_df["race_id"])]

    logger.info(f"対象: {len(meta_df):,}R  ({meta_df['race_date'].min().date()} 〜 {meta_df['race_date'].max().date()})")

    fe = FeatureEngineer.from_stats(settings.stats_path)

    records = []
    for i, (_, meta_row) in enumerate(meta_df.iterrows()):
        race_id = str(meta_row["race_id"])
        entries = results_df[results_df["race_id"] == race_id].copy()
        if len(entries) < 4:
            continue
        rec = _process_race(race_id, entries, meta_row, fe, act, bad_trainer)
        if rec:
            records.append(rec)

        if (i + 1) % 500 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(f"  {i+1}/{len(meta_df)} R処理済み  買い{len(records)}R  ({elapsed:.1f}分)")

    df = pd.DataFrame(records)
    logger.info(f"{label}完了: {len(df):,}R  ({(time.time()-t0)/60:.1f}分)")
    return df


# ── サマリー出力 ─────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, title: str) -> None:
    B = "=" * 72
    print(f"\n{B}\n  {title}")
    print(f"  期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  買い対象: {len(df):,} R")
    print(B)

    def roi(c, r): return r / c * 100 if c > 0 else 0.0

    tc = df["total_cost"].sum(); tr_total = df["total_ret"].sum()
    print(f"\n  【総合】 投票:{int(tc):,}円  回収:{int(tr_total):,}円  ROI:{roi(tc,tr_total):.1f}%")

    print(f"\n  【馬券種別 ROI】")
    for bet, cost_col, ret_col, hit_col in [
        ("単勝",  None,         "tansho_ret", "tansho_hit"),
        ("馬連",  "baren_cost", "baren_ret",  "baren_hit"),
        ("3連複", "sf_cost",    "sf_ret",     "sf_hit"),
        ("3連単", "st_cost",    "st_ret",     "st_hit"),
    ]:
        c = len(df) * 100 if cost_col is None else df[cost_col].sum()
        r = df[ret_col].sum()
        h = df[hit_col].sum()
        print(f"    {bet:<5}: 投票{int(c):>8,}円  回収{int(r):>9,}円  "
              f"ROI {roi(c,r):>6.1f}%  的中{int(h):>3}/{len(df)}R ({h/len(df):.1%})")

    print(f"\n  【月別 ROI（総合）】")
    print(f"  {'月':>3}  {'R数':>4}  {'単勝':>7}  {'馬連':>7}  {'3連複':>7}  {'3連単':>7}  {'総合':>7}")
    for m, g in df.groupby("race_month"):
        print(f"  {m:>3}月  {len(g):>4}R  "
              f"{roi(len(g)*100, g['tansho_ret'].sum()):>6.0f}%  "
              f"{roi(g['baren_cost'].sum(), g['baren_ret'].sum()):>6.0f}%  "
              f"{roi(g['sf_cost'].sum(), g['sf_ret'].sum()):>6.0f}%  "
              f"{roi(g['st_cost'].sum(), g['st_ret'].sum()):>6.0f}%  "
              f"{roi(g['total_cost'].sum(), g['total_ret'].sum()):>6.0f}%")

    print(f"\n  【芝/ダート別】")
    for ct, g in df.groupby("course_type"):
        print(f"    {ct}: {len(g)}R  ROI {roi(g['total_cost'].sum(),g['total_ret'].sum()):.1f}%  "
              f"単勝的中{int(g['tansho_hit'].sum())}  3連複的中{int(g['sf_hit'].sum())}")

    print(f"\n  【確率差(gap)別 ROI】")
    df2 = df.copy()
    df2["gap_bin"] = pd.cut(df2["gap"], bins=[0, .05, .10, .15, .25, 1.],
                            labels=["0-5%","5-10%","10-15%","15-25%","25%+"])
    for gb, g in df2.groupby("gap_bin", observed=True):
        if len(g) == 0: continue
        print(f"    {str(gb):>7}  {len(g):>4}R  "
              f"単勝:{roi(len(g)*100,g['tansho_ret'].sum()):>6.1f}%  "
              f"3連複:{roi(g['sf_cost'].sum(),g['sf_ret'].sum()):>6.1f}%  "
              f"総合:{roi(g['total_cost'].sum(),g['total_ret'].sum()):>6.1f}%")

    print(f"\n  【◎確率分布（Platt scaling 後）】")
    prob_bins = pd.cut(df["h1_prob"], bins=[0,.20,.25,.30,.35,.40,.50,1.],
                       labels=["<20%","20-25%","25-30%","30-35%","35-40%","40-50%","50%+"])
    for pb, g in df.groupby(prob_bins, observed=True):
        if len(g) == 0: continue
        actual_rate = g["tansho_hit"].sum() / len(g)
        print(f"    {str(pb):>8}  {len(g):>4}R  実際的中率:{actual_rate:.1%}  "
              f"単勝ROI:{roi(len(g)*100,g['tansho_ret'].sum()):.0f}%")

    for label_bet, hit_col, ret_col in [("3連複", "sf_hit", "sf_ret"),
                                         ("3連単", "st_hit", "st_ret")]:
        hits = df[df[hit_col] == 1].sort_values(ret_col, ascending=False)
        if len(hits) > 0:
            print(f"\n  【{label_bet} 的中上位10件（推定配当）】")
            for _, row in hits.head(10).iterrows():
                print(f"    {row['race_date']} {row['race_name']}  "
                      f"◎{row['honmei_num']}番{row['honmei_name']}  "
                      f"推定{row[ret_col]:.0f}円  h1_prob={row['h1_prob']:.1%}")
    print(f"\n{B}\n")


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="2024-2026 統合バックテスト（新モデル）")
    parser.add_argument("--year", type=int, default=None,
                        choices=[2024, 2025, 2026],
                        help="絞り込む年（省略時は 2024-2026 全期間）")
    parser.add_argument("--model-path", default=None,
                        help="モデルパス（省略時は settings.model_path）")
    args = parser.parse_args()

    t_all = time.time()

    # モデル読み込み
    model_path = Path(args.model_path) if args.model_path else settings.model_path
    act = ModelTrainer.load(model_path)
    calibrated = "あり" if act.calibrator is not None else "なし（旧モデル）"
    logger.info(f"メインモデル読み込み完了（Platt scaling: {calibrated}）")

    bad_trainer = None
    bad_path = ROOT / "data/models/lgbm_model_bad_season.pkl"
    if bad_path.exists():
        _bt = ModelTrainer.load(bad_path)
        # LightGBM Booster の実際の特徴量数で互換性チェック
        _bt_nfeat = _bt.model.num_feature() if _bt.model is not None else 0
        _cur_nfeat = len(FeatureEngineer.FEATURE_COLUMNS)
        if _bt_nfeat == _cur_nfeat:
            bad_trainer = _bt
            logger.info(f"不調期専用モデル読み込み完了（{_bt_nfeat}特徴量）")
        else:
            logger.warning(
                f"不調期モデルの特徴量数({_bt_nfeat}) が現行({_cur_nfeat}) と不一致 "
                "→ メインモデルで代替（不調期も含めて処理）"
            )
    else:
        logger.info("不調期専用モデルなし → メインモデルで代替")

    frames = []

    if args.year is None or args.year == 2024:
        df24 = run_2024(act, bad_trainer)
        if len(df24) > 0:
            frames.append(df24)

    if args.year is None or args.year in (2025, 2026):
        year_f = args.year if args.year in (2025, 2026) else None
        df2526 = run_2025_2026(act, bad_trainer, year_f)
        if len(df2526) > 0:
            frames.append(df2526)

    if not frames:
        print("買い対象レースなし")
        sys.exit(0)

    df_all = pd.concat(frames, ignore_index=True)
    total_elapsed = (time.time() - t_all) / 60

    # 全体サマリー
    suffix = f"_{args.year}" if args.year else ""
    title  = f"バックテスト結果（新モデル・Platt scaling）: " \
             f"{args.year}年" if args.year else \
             f"バックテスト結果（新モデル・Platt scaling）: 2024-2026年"
    print_summary(df_all, title)

    # 年別サマリー（全期間時）
    if args.year is None and df_all["year"].nunique() > 1:
        for yr, g in df_all.groupby("year"):
            if len(g) > 0:
                print_summary(g, f"バックテスト結果: {yr}年")

    # CSV 保存
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out = ROOT / f"data/processed/backtest_full{suffix}.csv"
    df_all.to_csv(out, index=False)
    logger.info(f"CSV保存: {out}  ({len(df_all):,}R)  総実行時間: {total_elapsed:.1f}分")


if __name__ == "__main__":
    main()
