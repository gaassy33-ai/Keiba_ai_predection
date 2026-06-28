"""
simulate_trio_by_year.py
========================
3連複・3連単 プールサイズ5頭 vs 7頭 シミュレーション（年度指定版）。

- 2025年: test_results_new.csv / test_meta_new.csv を使用（推論モード）
- 2024年: train_results.csv / train_meta.csv を使用
          ルックバック用履歴は対象レース日より前のデータのみ使用（リーク防止）

実行例:
    .venv/bin/python simulate_trio_by_year.py --year 2024
    .venv/bin/python simulate_trio_by_year.py --year 2025
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

# ── 設定（daily_batch.py と同一） ────────────────────────────────
MIN_HONMEI_PROB     = 0.30
MARK_STRONG_PROB    = 0.35
MIN_CONFIDENCE_GAP  = 0.05
EV_THRESHOLD        = 1.05
MAIDEN_KEYWORDS     = ("新馬", "未勝利")
ENTRIES_ADJ         = 0.003
ENTRIES_BASE        = 10
ENTRIES_CAP         = 0.05

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

JRA_TAKE = {"3連複": 0.225, "3連単": 0.275}
POOL_SIZES    = [5, 7]
TICKET_COUNTS = [5, 7]

# ── ロガー ────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/simulate_trio_by_year.log", level="DEBUG", format=_fmt, rotation="10 MB")


# ── ヘルパー ─────────────────────────────────────────────────────
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

def _prob_trio(probs, i, j, k):
    return sum(_harville(probs, list(o)) for o in _perm([i, j, k]))

def _prob_sanrentan(probs, i, j, k):
    return _harville(probs, [i, j, k])

def _est_odds(prob, bet_type):
    take = JRA_TAKE.get(bet_type, 0.225)
    return (1.0 - take) / max(prob, 0.001)

def _synth_odds(odds_list):
    denom = sum(1.0 / max(o, 1e-9) for o in odds_list)
    return 1.0 / denom if denom > 0 else 0.0

def is_buy_race(h1_prob, gap, race_month, is_maiden, n_entries,
                h1_ev, h1_odds, course_type, jyo_code, distance, using_bad):
    if using_bad:
        prob_thr = _BAD_MODEL_PROB_THRESHOLD
    else:
        base = _SEASON_PROB_THRESHOLD.get(race_month, MIN_HONMEI_PROB)
        adj  = min(max(0, n_entries - ENTRIES_BASE) * ENTRIES_ADJ, ENTRIES_CAP)
        prob_thr = base + adj

    if np.isnan(h1_ev):
        ev_ok = True
    elif _HIGH_VALUE_ODDS_MIN <= h1_odds <= _HIGH_VALUE_ODDS_MAX:
        ev_ok = h1_ev >= _HIGH_VALUE_EV_THRESHOLD
    else:
        ev_ok = h1_ev >= EV_THRESHOLD

    return (
        not is_maiden
        and gap >= MIN_CONFIDENCE_GAP
        and h1_prob >= prob_thr
        and ev_ok
        and not (race_month in _SUMMER_DIRT_SKIP_MONTHS and course_type == "ダート")
        and not (race_month in _BAD_SEASON_MONTHS and jyo_code in _BAD_VENUE_SKIP_CODES)
        and not (race_month in _BAD_SEASON_MONTHS and distance >= _BAD_SEASON_MAX_DISTANCE)
    )

def build_combos(pool, ppidx, model_probs, mkt_probs, honmei_idx, ticket_n, bet_type):
    all_items = []
    if bet_type == "3連複":
        for (na, via), (nb, vib) in _comb(pool, 2):
            pa = ppidx.get(na, 1)
            pb = ppidx.get(nb, 2)
            mp = _prob_trio(model_probs, honmei_idx, pa, pb)
            if via is not None and vib is not None:
                e_od = _est_odds(_prob_trio(mkt_probs, honmei_idx, via, vib), "3連複")
            else:
                e_od = _est_odds(mp, "3連複")
            all_items.append((na, nb, mp * e_od, e_od))
    else:
        for (n2, vi2), (n3, vi3) in _perm(pool, 2):
            p2 = ppidx.get(n2, 1)
            p3 = ppidx.get(n3, 2)
            mp = _prob_sanrentan(model_probs, honmei_idx, p2, p3)
            if vi2 is not None and vi3 is not None:
                e_od = _est_odds(_prob_sanrentan(mkt_probs, honmei_idx, vi2, vi3), "3連単")
            else:
                e_od = _est_odds(mp, "3連単")
            all_items.append((n2, n3, mp * e_od, e_od))

    all_items.sort(key=lambda x: -x[2])
    sel = all_items[:ticket_n]
    if sel and _synth_odds([od for *_, od in sel]) < 1.0:
        return []
    return [(a, b) for a, b, *_ in sel]


def simulate(year: int) -> None:
    t0 = time.time()
    logger.info("=" * 65)
    logger.info(f"{year}年 3連複・3連単 プール5 vs 7 シミュレーション開始")
    logger.info("=" * 65)

    # ── データソース切り替え ──────────────────────────────────────
    if year == 2025:
        results_path = ROOT / "data/raw/test_results_new.csv"
        meta_path    = ROOT / "data/raw/test_meta_new.csv"
        use_inference_mode = True   # feature_stats.pkl 使用
    else:
        results_path = ROOT / "data/raw/train_results.csv"
        meta_path    = ROOT / "data/raw/train_meta.csv"
        use_inference_mode = False  # 時系列ルックバック

    all_results = pd.read_csv(results_path, dtype=str)
    all_meta    = pd.read_csv(meta_path,    dtype=str)

    # race_date 付与・年フィルタ
    if "race_date" not in all_meta.columns:
        all_meta["race_date"] = pd.to_datetime(
            all_meta["race_id"].str[:8].apply(
                lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
            ), errors="coerce"
        )
    else:
        all_meta["race_date"] = pd.to_datetime(all_meta["race_date"], errors="coerce")

    all_meta = all_meta.dropna(subset=["race_date"])
    target_meta = all_meta[all_meta["race_date"].dt.year == year].sort_values("race_date")
    logger.info(f"{year}年レース数: {len(target_meta):,}  "
                f"({target_meta['race_date'].min().date()} 〜 {target_meta['race_date'].max().date()})")

    # ── モデル読み込み ────────────────────────────────────────────
    trainer = ModelTrainer.load(settings.model_path)
    bad_trainer = None
    bad_path = ROOT / "data/models/lgbm_model_bad_season.pkl"
    if bad_path.exists():
        bad_trainer = ModelTrainer.load(bad_path)
        logger.info("  不調期専用モデル読み込み完了")

    # 推論モード（2025年）: feature_stats.pkl から FeatureEngineer を構築
    if use_inference_mode:
        fe_base = FeatureEngineer.from_stats(settings.stats_path)
        logger.info("  推論モード: feature_stats.pkl 使用")
    else:
        # 時系列ルックバックモード（2024年）
        # ルックバック用: 全train_results を使うが、各レース評価時は race_date でフィルタ
        fe_base = FeatureEngineer(all_results)
        # race_date を history に付与
        date_map = all_meta.set_index("race_id")["race_date"].to_dict()
        fe_base.history["race_date"] = fe_base.history["race_id"].map(date_map)
        logger.info("  時系列ルックバックモード: train_results.csv 使用")

    # ── 推論ループ ────────────────────────────────────────────────
    records = []
    target_ids = target_meta["race_id"].tolist()
    logger.info(f"推論開始: {len(target_ids):,} レース")

    for i, race_id in enumerate(target_ids):
        race_entries = all_results[all_results["race_id"] == race_id].copy()
        if len(race_entries) < 3:
            continue

        meta_row    = target_meta[target_meta["race_id"] == race_id].iloc[0]
        race_date   = meta_row["race_date"]
        course_type = str(meta_row.get("course_type", ""))
        race_name   = str(meta_row.get("race_name", ""))
        distance    = int(meta_row.get("distance", 0) or 0)
        gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
        wx_code     = int(meta_row.get("weather_code", -1) or -1)
        jyo_code    = str(race_id)[4:6]
        race_month  = race_date.month
        n_entries   = len(race_entries)

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

        # 特徴量生成
        try:
            if use_inference_mode:
                feat_df = fe_base.build_entry_features(
                    entry_df=entry_df, course_type=course_type, distance=distance,
                    ground_condition_code=gc_code, weather_code=wx_code,
                    race_class_code=race_class_code, venue_code=venue_code,
                )
            else:
                # 時系列ルックバック: このレース日より前のデータのみ
                history_before = fe_base.history[fe_base.history["race_date"] < race_date]
                tmp_fe = FeatureEngineer(history_before)
                tmp_fe.precompute_aggregations()
                feat_df = tmp_fe.build_entry_features(
                    entry_df=entry_df, course_type=course_type, distance=distance,
                    ground_condition_code=gc_code, weather_code=wx_code,
                    race_class_code=race_class_code, venue_code=venue_code,
                )
        except Exception as e:
            logger.debug(f"skip {race_id}: {e}")
            continue

        using_bad = (bad_trainer is not None and race_month in _BAD_SEASON_MONTHS)
        act = bad_trainer if using_bad else trainer
        X = feat_df[FeatureEngineer.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
        win_probs = act.model.predict(X, num_threads=1)
        probs = (0.7 * win_probs + 0.3 * act.place_model.predict(X, num_threads=1)
                 if act.place_model is not None else win_probs)

        pred_df = feat_df[["horse_id","horse_name","horse_number"]].copy()
        pred_df["win_prob"] = probs
        pred_df = pred_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
        if len(pred_df) < 3:
            continue

        h1_prob = float(pred_df.iloc[0]["win_prob"])
        h2_prob = float(pred_df.iloc[1]["win_prob"])
        gap     = h1_prob - h2_prob
        is_maiden = any(k in race_name for k in MAIDEN_KEYWORDS)

        odds_map = {}
        if "odds" in feat_df.columns:
            odds_map = feat_df.set_index("horse_id")["odds"].dropna().to_dict()

        h1_id   = str(pred_df.iloc[0]["horse_id"])
        h1_odds = float(odds_map.get(h1_id, float("nan")))
        h1_ev   = h1_prob * h1_odds if not np.isnan(h1_odds) else float("nan")

        buy = is_buy_race(h1_prob, gap, race_month, is_maiden, n_entries,
                          h1_ev, h1_odds, course_type, jyo_code, distance, using_bad)
        mark_strong = _BAD_MODEL_MARK_STRONG if using_bad else MARK_STRONG_PROB
        mark = "◎" if buy and h1_prob >= mark_strong else ("○" if buy else "△")
        if mark == "○" and race_month in _MARK_O_SKIP_MONTHS:
            mark = "△"; buy = False

        if not buy:
            continue

        # 実際の着順
        actual = race_entries[["horse_id","horse_number","finish_position","odds"]].copy()
        actual["pos"] = pd.to_numeric(
            actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        actual["odds_f"] = actual["odds"].apply(parse_odds)
        actual = actual.dropna(subset=["pos"]).sort_values("pos")
        if len(actual) < 3:
            continue

        pos1_num = str(int(float(actual.iloc[0]["horse_number"])))
        pos2_num = str(int(float(actual.iloc[1]["horse_number"])))
        pos3_num = str(int(float(actual.iloc[2]["horse_number"])))
        actual_trio = frozenset([pos1_num, pos2_num, pos3_num])
        actual_st   = (pos1_num, pos2_num, pos3_num)

        valid_ids = pred_df["horse_id"].astype(str).tolist()
        horse_num_to_id = {}
        for _, row in feat_df.iterrows():
            num = str(int(float(row["horse_number"]))) if pd.notna(row.get("horse_number")) else None
            if num:
                horse_num_to_id[num] = str(row["horse_id"])

        odds_list  = [float(odds_map.get(hid, float("nan"))) for hid in valid_ids]
        odds_list  = [o if not np.isnan(o) and o > 1.0 else 5.0 for o in odds_list]
        mkt_probs  = _market_probs(odds_list)
        probs_raw  = pred_df["win_prob"].tolist()
        probs_sum  = sum(probs_raw)
        model_probs = [p / probs_sum for p in probs_raw] if probs_sum > 0 else probs_raw
        honmei_idx  = 0
        honmei_num  = str(int(float(pred_df.iloc[0]["horse_number"])))

        row = {
            "race_id":     race_id,
            "race_date":   str(race_date)[:10],
            "race_month":  race_month,
            "race_name":   race_name,
            "mark":        mark,
            "n_entries":   n_entries,
            "h1_prob":     round(h1_prob, 4),
            "gap":         round(gap, 4),
            "actual_trio": "-".join(sorted(actual_trio, key=int)),
            "actual_st":   "-".join(actual_st),
            "honmei_1st":  int(honmei_num == pos1_num),
        }

        for pool_n, ticket_n in zip(POOL_SIZES, TICKET_COUNTS):
            partners = pred_df[pred_df["horse_id"] != pred_df.iloc[0]["horse_id"]].head(pool_n)
            pool = []
            for _, r in partners.iterrows():
                hid = str(r["horse_id"])
                num = str(int(float(r["horse_number"])))
                vi  = valid_ids.index(hid) if hid in valid_ids else None
                pool.append((num, vi))
            ppidx = {num: i + 1 for i, (num, _) in enumerate(pool)}

            for bet_type in ["3連複", "3連単"]:
                combos = build_combos(pool, ppidx, model_probs, mkt_probs,
                                      honmei_idx, ticket_n, bet_type)
                cost = len(combos) * 100
                hit, hit_od = 0, 0.0

                if bet_type == "3連複":
                    for na, nb in combos:
                        if frozenset([honmei_num, na, nb]) == actual_trio:
                            hit = 1
                            via = valid_ids.index(horse_num_to_id[na]) if na in horse_num_to_id and horse_num_to_id[na] in valid_ids else None
                            vib = valid_ids.index(horse_num_to_id[nb]) if nb in horse_num_to_id and horse_num_to_id[nb] in valid_ids else None
                            if via is not None and vib is not None:
                                hit_od = _est_odds(_prob_trio(mkt_probs, honmei_idx, via, vib), "3連複")
                            break
                else:
                    if honmei_num == pos1_num:
                        for n2, n3 in combos:
                            if n2 == pos2_num and n3 == pos3_num:
                                hit = 1
                                vi2 = valid_ids.index(horse_num_to_id[n2]) if n2 in horse_num_to_id and horse_num_to_id[n2] in valid_ids else None
                                vi3 = valid_ids.index(horse_num_to_id[n3]) if n3 in horse_num_to_id and horse_num_to_id[n3] in valid_ids else None
                                if vi2 is not None and vi3 is not None:
                                    hit_od = _est_odds(_prob_sanrentan(mkt_probs, honmei_idx, vi2, vi3), "3連単")
                                break

                key = f"p{pool_n}_{bet_type}"
                row[f"{key}_combos"] = len(combos)
                row[f"{key}_hit"]    = hit
                row[f"{key}_cost"]   = cost
                row[f"{key}_ret"]    = round(hit_od * 100 if hit else 0.0, 1)
                row[f"{key}_hit_od"] = round(hit_od, 1)

        records.append(row)

        if (i + 1) % 300 == 0:
            logger.info(f"  {i+1}/{len(target_ids)} レース処理済み（買い対象: {len(records)}R）")

    df = pd.DataFrame(records)
    logger.info(f"買い対象 {len(df):,} レース処理完了  ({(time.time()-t0)/60:.1f}分)")

    # ── サマリー表示 ─────────────────────────────────────────────
    border = "=" * 68
    print(f"\n{border}")
    print(f"  {year}年 3連複・3連単 プール5頭 vs 7頭 シミュレーション結果")
    print(f"  テスト期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  買い対象 : {len(df):,} R")
    mode_note = "推論モード(feature_stats.pkl)" if use_inference_mode else "時系列ルックバック(train_results.csv)"
    print(f"  データモード: {mode_note}")
    print(f"  ※オッズは市場単勝から Harville 法で推定（控除率: 3連複22.5%, 3連単27.5%）")
    print(border)

    # 全体サマリー
    for bet_type in ["3連複", "3連単"]:
        print(f"\n  ▼ {bet_type}")
        print(f"  {'':12}  {'投票額':>9}  {'推定回収':>9}  {'ROI':>7}  {'的中':>7}  {'的中率':>7}  {'平均点':>6}")
        for pool_n, ticket_n in zip(POOL_SIZES, TICKET_COUNTS):
            key = f"p{pool_n}_{bet_type}"
            cost = df[f"{key}_cost"].sum()
            ret  = df[f"{key}_ret"].sum()
            hits = df[f"{key}_hit"].sum()
            roi  = ret / cost * 100 if cost > 0 else 0
            avg  = df[f"{key}_combos"].mean()
            label = f"プール{pool_n}頭/{ticket_n}点"
            print(f"  {label:12}  {int(cost):>8,}円  {int(ret):>8,}円  {roi:>6.1f}%"
                  f"  {int(hits):>3}/{len(df)}R  {hits/len(df):>6.2%}  {avg:>5.1f}点")

    # 拡張効果
    print(f"\n  ▼ 拡張効果（プール5→7）")
    print(f"  {'':10}  {'新規捕捉':>6}  {'取りこぼし':>8}  {'追加コスト':>10}  {'追加回収':>9}  {'追加ROI':>8}")
    for bet_type in ["3連複", "3連単"]:
        key5 = f"p5_{bet_type}"
        key7 = f"p7_{bet_type}"
        newly = len(df[(df[f"{key7}_hit"] == 1) & (df[f"{key5}_hit"] == 0)])
        lost  = len(df[(df[f"{key5}_hit"] == 1) & (df[f"{key7}_hit"] == 0)])
        extra_cost = df[f"{key7}_cost"].sum() - df[f"{key5}_cost"].sum()
        extra_ret  = df[f"{key7}_ret"].sum()  - df[f"{key5}_ret"].sum()
        extra_roi  = extra_ret / extra_cost * 100 if extra_cost > 0 else 0
        print(f"  {bet_type:10}  {newly:>5}R  {lost:>7}R  {int(extra_cost):>9,}円  {int(extra_ret):>8,}円  {extra_roi:>7.1f}%")

    # 月別
    for bet_type in ["3連複", "3連単"]:
        key5 = f"p5_{bet_type}"
        key7 = f"p7_{bet_type}"
        print(f"\n  ▼ 月別 ROI [{bet_type}]")
        print(f"  {'月':>4}  {'R数':>4}  {'1着率':>7}  {'pool5 ROI':>10}  {'pool7 ROI':>10}  {'差':>8}  {'p5的中':>6}  {'p7的中':>6}")
        for month, grp in df.groupby("race_month"):
            c5 = grp[f"{key5}_cost"].sum(); r5 = grp[f"{key5}_ret"].sum()
            c7 = grp[f"{key7}_cost"].sum(); r7 = grp[f"{key7}_ret"].sum()
            roi5 = r5/c5*100 if c5 > 0 else 0
            roi7 = r7/c7*100 if c7 > 0 else 0
            h5 = int(grp[f"{key5}_hit"].sum())
            h7 = int(grp[f"{key7}_hit"].sum())
            hon = grp["honmei_1st"].mean()
            print(f"  {month:>4}月  {len(grp):>4}R  {hon:>6.1%}  "
                  f"{roi5:>9.1f}%  {roi7:>9.1f}%  {roi7-roi5:>+8.1f}pt  "
                  f"{h5}/{len(grp)}R  {h7}/{len(grp)}R")

    # gap別
    df["gap_bin"] = pd.cut(df["gap"],
        bins=[0, 0.05, 0.10, 0.15, 0.25, 1.0],
        labels=["0-5%","5-10%","10-15%","15-25%","25%+"])
    for bet_type in ["3連複", "3連単"]:
        key5 = f"p5_{bet_type}"
        key7 = f"p7_{bet_type}"
        print(f"\n  ▼ 確率差(gap)別 ROI [{bet_type}]")
        print(f"  {'gap帯':>10}  {'R数':>4}  {'1着率':>7}  {'pool5 ROI':>10}  {'pool7 ROI':>10}  {'差':>8}")
        for gb, grp in df.groupby("gap_bin", observed=True):
            if len(grp) == 0:
                continue
            c5 = grp[f"{key5}_cost"].sum(); r5 = grp[f"{key5}_ret"].sum()
            c7 = grp[f"{key7}_cost"].sum(); r7 = grp[f"{key7}_ret"].sum()
            roi5 = r5/c5*100 if c5 > 0 else 0
            roi7 = r7/c7*100 if c7 > 0 else 0
            hon = grp["honmei_1st"].mean()
            print(f"  {str(gb):>10}  {len(grp):>4}R  {hon:>6.1%}  "
                  f"{roi5:>9.1f}%  {roi7:>9.1f}%  {roi7-roi5:>+8.1f}pt")

    # プール7のみ新規捕捉
    for bet_type in ["3連複", "3連単"]:
        key5 = f"p5_{bet_type}"
        key7 = f"p7_{bet_type}"
        newly = df[(df[f"{key7}_hit"] == 1) & (df[f"{key5}_hit"] == 0)]
        if len(newly) > 0:
            print(f"\n  ▼ プール7のみ新規捕捉 [{bet_type}]（全{len(newly)}件）")
            for _, r in newly.iterrows():
                disp = r["actual_trio"] if bet_type == "3連複" else r["actual_st"]
                print(f"    {r['race_date']} {r['race_name'][:22]:22s}  "
                      f"実際: {disp}  推定配当: {r[f'{key7}_hit_od']:.0f}倍  "
                      f"◎1着: {'○' if r['honmei_1st'] else '×'}")

    print(f"\n{border}\n")

    out = ROOT / f"data/processed/simulate_{year}_trio.csv"
    df.to_csv(out, index=False)
    logger.info(f"詳細結果保存: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025, help="対象年 (2024 or 2025)")
    args = parser.parse_args()
    simulate(args.year)
