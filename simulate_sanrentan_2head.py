"""
simulate_sanrentan_2head.py
===========================
3連単: ◎1着固定 vs 2頭軸（◎or△1着）の ROI シミュレーション。

  Strategy A (honmei):  ◎1着固定、pool7からEV上位N点
  Strategy B (2head):   Strategy A + △1着固定でも追加購入
                        △→◎→B, △→B→◎  (B: poolから◎・△を除いた馬)

実行:
    .venv/bin/python simulate_sanrentan_2head.py [--year 2025|2024]
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import permutations as _perm
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from config.settings import settings

# ── 設定（daily_batch.py と同一） ────────────────────────────────
MIN_HONMEI_PROB    = 0.30
MARK_STRONG_PROB   = 0.35
MIN_CONFIDENCE_GAP = 0.05
EV_THRESHOLD       = 1.05
MAIDEN_KEYWORDS    = ("新馬", "未勝利")
ENTRIES_ADJ        = 0.003
ENTRIES_BASE       = 10
ENTRIES_CAP        = 0.05

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

JRA_TAKE = {"3連単": 0.275}
POOL_N     = 7   # パートナー候補プールサイズ
TICKET_N   = 7   # ◎1着で購入する最大点数

# ── ロガー ────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO",  format=_fmt, colorize=True)
logger.add("logs/simulate_sanrentan_2head.log", level="DEBUG", format=_fmt, rotation="10 MB")


# ── ヘルパー関数 ─────────────────────────────────────────────────
def parse_odds(val) -> float:
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

def _prob_st(probs, i, j, k):
    return _harville(probs, [i, j, k])

def _est_odds(prob, bet_type="3連単"):
    take = JRA_TAKE.get(bet_type, 0.275)
    return (1.0 - take) / max(prob, 0.001)

def _synth_odds(odds_list):
    denom = sum(1.0 / max(o, 1e-9) for o in odds_list)
    return 1.0 / denom if denom > 0 else 0.0

def is_buy_race(h1_prob, h2_prob, gap, race_month, is_maiden, n_entries,
                h1_ev, h1_odds, course_type, jyo_code, distance, using_bad_model):
    if using_bad_model:
        prob_threshold = _BAD_MODEL_PROB_THRESHOLD
    else:
        season_base    = _SEASON_PROB_THRESHOLD.get(race_month, MIN_HONMEI_PROB)
        entries_adj    = min(max(0, n_entries - ENTRIES_BASE) * ENTRIES_ADJ, ENTRIES_CAP)
        prob_threshold = season_base + entries_adj

    if np.isnan(h1_ev):
        ev_ok = True
    elif _HIGH_VALUE_ODDS_MIN <= h1_odds <= _HIGH_VALUE_ODDS_MAX:
        ev_ok = h1_ev >= _HIGH_VALUE_EV_THRESHOLD
    else:
        ev_ok = h1_ev >= EV_THRESHOLD

    is_summer_dirt  = (race_month in _SUMMER_DIRT_SKIP_MONTHS and course_type == "ダート")
    is_bad_venue    = (race_month in _BAD_SEASON_MONTHS and jyo_code in _BAD_VENUE_SKIP_CODES)
    is_bad_distance = (race_month in _BAD_SEASON_MONTHS and distance >= _BAD_SEASON_MAX_DISTANCE)

    return (
        not is_maiden
        and gap >= MIN_CONFIDENCE_GAP
        and h1_prob >= prob_threshold
        and ev_ok
        and not is_summer_dirt
        and not is_bad_venue
        and not is_bad_distance
    )


def _select_tickets(model_probs, mkt_probs, valid_ids,
                    first_idx: int, pool: list[tuple[str, int | None]],
                    max_tickets: int) -> list[tuple[str, str, float]]:
    """
    first_idx を1着固定として pool から EV 上位 max_tickets 点を選択する。
    Returns: list of (num_2nd, num_3rd, est_odds)
    """
    candidates = []
    for (num_2, vi_2), (num_3, vi_3) in _perm(pool, 2):
        # モデル確率でのEV計算
        pool_idx_map = {num: i for i, (num, _) in enumerate(pool)}
        p2_idx = pool_idx_map.get(num_2, 1)
        p3_idx = pool_idx_map.get(num_3, 2)

        # モデル確率: first_idx, p2_idx+1 (pool内のインデックスをpred_df全体にずらす)
        # ここでは簡略化: model_probs はpred_df全体のインデックス
        model_p = _prob_st(model_probs, first_idx, vi_2 if vi_2 is not None else p2_idx + 1,
                           vi_3 if vi_3 is not None else p3_idx + 1)

        if vi_2 is not None and vi_3 is not None:
            mkt_p = _prob_st(mkt_probs, first_idx, vi_2, vi_3)
            e_od  = _est_odds(mkt_p, "3連単")
        else:
            e_od  = _est_odds(model_p, "3連単")

        ev = model_p * e_od
        candidates.append((num_2, num_3, ev, e_od))

    candidates.sort(key=lambda x: -x[2])
    sel = candidates[:max_tickets]

    # トリガミ除外
    est_odds_list = [od for _, _, _, od in sel]
    if est_odds_list and _synth_odds(est_odds_list) < 1.0:
        return []

    return [(n2, n3, od) for n2, n3, _, od in sel]


def simulate(year: int) -> None:
    t0 = time.time()

    if year == 2025:
        test_results_csv = ROOT / "data/raw/test_results_new.csv"
        test_meta_csv    = ROOT / "data/raw/test_meta_new.csv"
        mode_label = "inference (feature_stats.pkl)"
    else:
        test_results_csv = ROOT / "data/raw/train_results.csv"
        test_meta_csv    = ROOT / "data/raw/train_meta.csv"
        mode_label = f"lookback (train data, {year}年のみ)"

    logger.info("=" * 70)
    logger.info(f"3連単 2頭軸シミュレーション開始 ({year}年, {mode_label})")
    logger.info("=" * 70)

    test_results = pd.read_csv(test_results_csv, dtype=str)
    test_meta    = pd.read_csv(test_meta_csv,    dtype=str)
    test_meta["race_date"] = pd.to_datetime(
        test_meta["race_id"].str[:8].apply(
            lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
        ), errors="coerce"
    )
    test_meta = test_meta.sort_values("race_date").dropna(subset=["race_date"])

    if year != 2025:
        test_meta = test_meta[test_meta["race_date"].dt.year == year]
        test_results = test_results[test_results["race_id"].isin(test_meta["race_id"])]

    logger.info(f"対象: {len(test_results):,} 行, {test_meta['race_id'].nunique():,} レース")

    trainer = ModelTrainer.load(settings.model_path)
    bad_trainer = None
    bad_model_path = ROOT / "data/models/lgbm_model_bad_season.pkl"
    if bad_model_path.exists():
        bad_trainer = ModelTrainer.load(bad_model_path)

    if year == 2025:
        fe = FeatureEngineer.from_stats(settings.stats_path)
        fe_lookback = None
    else:
        # 2024年: lookbackモード（各レース時点より前の履歴のみ使用）
        fe = None
        all_history = pd.read_csv(ROOT / "data/raw/train_results.csv", dtype=str)
        all_history["race_date"] = pd.to_datetime(
            all_history["race_id"].str[:8].apply(
                lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
            ), errors="coerce"
        )
        fe_lookback = all_history

    records = []
    target_ids = test_meta["race_id"].tolist()
    logger.info(f"推論開始: {len(target_ids):,} レース")

    for i, race_id in enumerate(target_ids):
        race_entries = test_results[test_results["race_id"] == race_id].copy()
        if len(race_entries) < 3:
            continue

        meta_row    = test_meta[test_meta["race_id"] == race_id].iloc[0]
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

        entry_df = race_entries[["horse_id","horse_name","horse_number","frame_number",
                                  "jockey_id","sex_age","weight_carried"]].copy()
        entry_df["sex"] = entry_df["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entry_df["sex_age"].str[1:], errors="coerce")
        entry_df["weight_carried"] = pd.to_numeric(entry_df["weight_carried"], errors="coerce")
        entry_df["father"] = ""
        entry_df["mother_father"] = ""
        if "odds" in race_entries.columns:
            entry_df["odds"] = pd.to_numeric(race_entries["odds"].values, errors="coerce")

        race_class_code = FeatureEngineer._race_name_to_class_code(race_name)
        try:
            venue_code = int(race_id[4:6])
        except Exception:
            venue_code = -1

        try:
            if year == 2025:
                feat_df = fe.build_entry_features(
                    entry_df=entry_df,
                    course_type=course_type,
                    distance=distance,
                    ground_condition_code=gc_code,
                    weather_code=wx_code,
                    race_class_code=race_class_code,
                    venue_code=venue_code,
                )
            else:
                # lookback: race_date より前の履歴でFeatureEngineerを構築
                hist_before = fe_lookback[fe_lookback["race_date"] < race_date]
                _fe = FeatureEngineer(hist_before)
                feat_df = _fe.build_entry_features(
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
        gap = h1_prob - h2_prob
        n_entries = len(race_entries)
        is_maiden = any(k in race_name for k in MAIDEN_KEYWORDS)

        odds_map = {}
        if "odds" in feat_df.columns:
            odds_map = feat_df.set_index("horse_id")["odds"].dropna().to_dict()

        h1_id   = str(pred_df.iloc[0]["horse_id"])
        h1_odds = float(odds_map.get(h1_id, float("nan")))
        h1_ev   = h1_prob * h1_odds if not np.isnan(h1_odds) else float("nan")

        buy = is_buy_race(
            h1_prob, h2_prob, gap, race_month, is_maiden, n_entries,
            h1_ev, h1_odds, course_type, jyo_code, distance, using_bad
        )
        if using_bad:
            mark_strong = _BAD_MODEL_MARK_STRONG
        else:
            mark_strong = MARK_STRONG_PROB
        mark = "◎" if buy and h1_prob >= mark_strong else ("○" if buy else "△")
        if mark == "○" and race_month in _MARK_O_SKIP_MONTHS:
            mark = "△"
            buy = False

        if not buy:
            continue

        # ── 実際の着順取得 ──────────────────────────────────────
        actual = race_entries[["horse_id","horse_number","finish_position","odds"]].copy()
        actual["pos"] = pd.to_numeric(
            actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        actual["odds_f"] = actual["odds"].apply(parse_odds)
        actual = actual.dropna(subset=["pos"]).sort_values("pos")

        if len(actual) < 3:
            continue

        top3 = actual.iloc[:3]
        pos1_num = str(int(float(top3.iloc[0]["horse_number"])))
        pos2_num = str(int(float(top3.iloc[1]["horse_number"])))
        pos3_num = str(int(float(top3.iloc[2]["horse_number"])))

        # ── 市場オッズから確率を推定（Harville用）──────────────
        valid_ids = pred_df["horse_id"].astype(str).tolist()
        horse_num_map = {
            str(int(float(r["horse_number"]))): str(r["horse_id"])
            for _, r in feat_df.iterrows()
            if pd.notna(r.get("horse_number"))
        }

        odds_list = []
        for hid in valid_ids:
            o = float(odds_map.get(hid, float("nan")))
            odds_list.append(o if not np.isnan(o) and o > 1.0 else 5.0)
        mkt_probs_list = _market_probs(odds_list)

        # 正規化モデル確率
        probs_raw = pred_df["win_prob"].tolist()
        probs_sum = sum(probs_raw)
        model_probs = [p / probs_sum for p in probs_raw] if probs_sum > 0 else probs_raw

        # ◎・△のhorse_number と valid_ids インデックス
        honmei_num  = str(int(float(pred_df.iloc[0]["horse_number"])))
        honmei_id   = str(pred_df.iloc[0]["horse_id"])
        honmei_idx  = 0  # pred_df上位
        nibante_num = str(int(float(pred_df.iloc[1]["horse_number"])))
        nibante_id  = str(pred_df.iloc[1]["horse_id"])
        nibante_vi  = valid_ids.index(nibante_id) if nibante_id in valid_ids else 1

        # ── Strategy A: ◎1着固定（current pool7）────────────────
        # pool: ◎除く上位POOL_N頭
        pool_a = []
        for _, row in pred_df[pred_df["horse_id"] != honmei_id].head(POOL_N).iterrows():
            hid = str(row["horse_id"])
            num = str(int(float(row["horse_number"])))
            vi  = valid_ids.index(hid) if hid in valid_ids else None
            pool_a.append((num, vi))

        tickets_a = _select_tickets(model_probs, mkt_probs_list, valid_ids,
                                    honmei_idx, pool_a, TICKET_N)

        # トリガミ除外済みなので tickets_a は空か有効なリスト
        cost_a = len(tickets_a) * 100
        hit_a  = 0
        ret_a  = 0.0
        if honmei_num == pos1_num:
            for n2, n3, e_od in tickets_a:
                if n2 == pos2_num and n3 == pos3_num:
                    hit_a = 1
                    ret_a = e_od * 100
                    break

        # ── Strategy B: 2頭軸（◎ or △1着）──────────────────────
        # B1: ◎1着（Strategy Aと同じ）
        tickets_b1 = tickets_a  # 同じ

        # B2: △1着固定 → ◎・残りpool から2・3着選択
        # pool: △除く上位POOL_N頭（◎を含む）
        # △を1着に固定した際の pool: pred_df先頭から POOL_N+1頭（△除く）を取る
        pool_b2_candidates = pred_df[pred_df["horse_id"] != nibante_id].head(POOL_N + 1)
        pool_b2 = []
        for _, row in pool_b2_candidates.iterrows():
            hid = str(row["horse_id"])
            num = str(int(float(row["horse_number"])))
            vi  = valid_ids.index(hid) if hid in valid_ids else None
            pool_b2.append((num, vi))
        pool_b2 = pool_b2[:POOL_N]  # 最大POOL_N頭

        tickets_b2 = _select_tickets(model_probs, mkt_probs_list, valid_ids,
                                     nibante_vi, pool_b2, TICKET_N)

        # B2の的中チェック: △1が1着の場合
        cost_b2 = len(tickets_b2) * 100
        hit_b2  = 0
        ret_b2  = 0.0
        if nibante_num == pos1_num:
            for n2, n3, e_od in tickets_b2:
                if n2 == pos2_num and n3 == pos3_num:
                    hit_b2 = 1
                    ret_b2 = e_od * 100
                    break

        # Strategy B = B1 + B2 (コスト合計)
        cost_b = cost_a + cost_b2
        hit_b  = hit_a or hit_b2
        ret_b  = ret_a + ret_b2

        result_row = {
            "race_id":     race_id,
            "race_date":   str(race_date)[:10],
            "race_name":   race_name,
            "race_month":  race_month,
            "n_entries":   n_entries,
            "h1_prob":     round(h1_prob, 4),
            "h2_prob":     round(h2_prob, 4),
            "gap":         round(gap, 4),
            "honmei_num":  honmei_num,
            "nibante_num": nibante_num,
            "actual_1st":  pos1_num,
            "actual_2nd":  pos2_num,
            "actual_3rd":  pos3_num,
            # Strategy A
            "a_tickets":   len(tickets_a),
            "a_cost":      cost_a,
            "a_hit":       int(hit_a),
            "a_ret":       round(ret_a, 1),
            # Strategy B (2頭軸)
            "b2_tickets":  len(tickets_b2),
            "b_cost":      cost_b,
            "b_hit":       int(hit_b),
            "b_ret":       round(ret_b, 1),
            "b2_only_hit": int(hit_b2),   # △1着でのみ的中
        }
        records.append(result_row)

        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{len(target_ids)} レース処理済み")

    df = pd.DataFrame(records)
    elapsed = (time.time() - t0) / 60
    logger.info(f"買い対象 {len(df):,} レース処理完了  ({elapsed:.1f}分)")

    # ── 結果出力 ─────────────────────────────────────────────────
    border = "=" * 70
    print(f"\n{border}")
    print(f"  3連単 2頭軸シミュレーション結果 ({year}年)")
    print(f"  期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  買い対象: {len(df):,} R")
    print(border)

    def _roi(df, cost_col, ret_col):
        c = df[cost_col].sum()
        r = df[ret_col].sum()
        return r / c * 100 if c > 0 else 0.0

    roi_a = _roi(df, "a_cost", "a_ret")
    roi_b = _roi(df, "b_cost", "b_ret")
    hit_a = df["a_hit"].sum()
    hit_b = df["b_hit"].sum()

    print(f"\n  【Strategy A: ◎1着固定 (pool{POOL_N})】")
    print(f"    投票総額: {int(df['a_cost'].sum()):,}円  推定回収: {int(df['a_ret'].sum()):,}円  ROI: {roi_a:.1f}%")
    print(f"    的中: {int(hit_a)}/{len(df)}R  平均点数: {df['a_tickets'].mean():.1f}点/R")

    print(f"\n  【Strategy B: 2頭軸 (◎1着 + △1着, pool{POOL_N})】")
    print(f"    投票総額: {int(df['b_cost'].sum()):,}円  推定回収: {int(df['b_ret'].sum()):,}円  ROI: {roi_b:.1f}%")
    print(f"    的中: {int(hit_b)}/{len(df)}R")
    print(f"    うち △1着での的中: {int(df['b2_only_hit'].sum())}R")
    print(f"    追加コスト(△1着分): {int(df['b2_tickets'].sum() * 100):,}円")
    print(f"    △1着分のROI: {df['b_ret'].sum()-df['a_ret'].sum():.0f}円回収 / {int(df['b2_cost'].sum() if 'b2_cost' in df else df['b_cost'].sum()-df['a_cost'].sum()):,}円投票")

    # 追加分
    add_cost = df["b_cost"].sum() - df["a_cost"].sum()
    add_ret  = df["b_ret"].sum() - df["a_ret"].sum()
    print(f"    追加分ROI: {add_ret/add_cost*100 if add_cost > 0 else 0:.1f}%")

    # ◎的中率 vs △的中率
    honmei_1st = (df["actual_1st"] == df["honmei_num"]).sum()
    nibante_1st = (df["actual_1st"] == df["nibante_num"]).sum()
    print(f"\n  [◎・△の1着的中率]")
    print(f"    ◎が1着: {int(honmei_1st)}/{len(df)}R ({honmei_1st/len(df):.1%})")
    print(f"    △が1着: {int(nibante_1st)}/{len(df)}R ({nibante_1st/len(df):.1%})")

    # gap別
    print(f"\n  [確率差(gap)別 ROI]")
    print(f"  {'gap帯':>10}  {'R数':>4}  {'A ROI':>8}  {'B ROI':>8}  {'差':>7}  {'△1着':>6}")
    df["gap_bin"] = pd.cut(
        df["gap"],
        bins=[0, 0.05, 0.10, 0.15, 0.25, 1.0],
        labels=["0-5%", "5-10%", "10-15%", "15-25%", "25%+"]
    )
    for gb, grp in df.groupby("gap_bin", observed=True):
        if len(grp) == 0:
            continue
        ra = _roi(grp, "a_cost", "a_ret")
        rb = _roi(grp, "b_cost", "b_ret")
        n2nd = (grp["actual_1st"] == grp["nibante_num"]).sum()
        print(f"  {str(gb):>10}  {len(grp):>4}R  {ra:>8.1f}%  {rb:>8.1f}%  {rb-ra:>+7.1f}pt  {n2nd:>4}R")

    # 月別
    print(f"\n  [月別 ROI]")
    print(f"  {'月':>4}  {'R数':>4}  {'A ROI':>8}  {'B ROI':>8}  {'差':>7}")
    for month, grp in df.groupby("race_month"):
        ra = _roi(grp, "a_cost", "a_ret")
        rb = _roi(grp, "b_cost", "b_ret")
        print(f"  {month:>4}月  {len(grp):>4}R  {ra:>8.1f}%  {rb:>8.1f}%  {rb-ra:>+7.1f}pt")

    # △1着で的中したレースの詳細
    b2_hits = df[df["b2_only_hit"] == 1]
    if len(b2_hits) > 0:
        print(f"\n  [△1着でのみ的中したレース（計{len(b2_hits)}R）]")
        for _, row in b2_hits.iterrows():
            print(f"    {row['race_date']} {row['race_name']}")
            print(f"      ◎={row['honmei_num']}番({row['h1_prob']:.1%}), △={row['nibante_num']}番({row['h2_prob']:.1%})")
            print(f"      実際: {row['actual_1st']}-{row['actual_2nd']}-{row['actual_3rd']}")

    print(f"\n{border}\n")

    out_path = ROOT / f"data/processed/simulate_{year}_sanrentan_2head.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"詳細結果保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025, choices=[2024, 2025])
    args = parser.parse_args()
    simulate(args.year)
