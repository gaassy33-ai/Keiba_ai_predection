"""
simulate_sanrentan.py
=====================
3連単パートナープール 5頭 vs 7頭 の ROI シミュレーション。

現行の予測モデルで買い対象と判定されたレースについて、
EV上位5点 (現行) と EV上位7点 (拡張案) で的中率・ROIを比較する。

3連単オッズ: test_results_new.csv の単勝オッズから Harville 法で推定
             (実際の3連単配当データは未収集のため推定値)

実行:
    .venv/bin/python simulate_sanrentan.py
"""
from __future__ import annotations

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
TEST_RESULTS_CSV = ROOT / "data/raw/test_results_new.csv"
TEST_META_CSV    = ROOT / "data/raw/test_meta_new.csv"

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

JRA_TAKE = {"馬連": 0.225, "馬単": 0.25, "3連複": 0.225, "3連単": 0.275}

POOL_SIZES    = [5, 7]          # 比較するプールサイズ
TICKET_COUNTS = [5, 7]          # 各プールから選ぶ最大点数（プールサイズと対応）

# ── ロガー ────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO",  format=_fmt, colorize=True)
logger.add("logs/simulate_sanrentan.log", level="DEBUG", format=_fmt, rotation="10 MB")


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

def _prob_sanrentan(probs, i, j, k):
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


def simulate() -> None:
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("3連単 プールサイズ 5 vs 7 シミュレーション開始")
    logger.info("=" * 65)

    # ── データ読み込み ────────────────────────────────────────────
    test_results = pd.read_csv(TEST_RESULTS_CSV, dtype=str)
    test_meta    = pd.read_csv(TEST_META_CSV,    dtype=str)
    test_meta["race_date"] = pd.to_datetime(
        test_meta["race_id"].str[:8].apply(
            lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
        ), errors="coerce"
    )
    test_meta = test_meta.sort_values("race_date").dropna(subset=["race_date"])
    logger.info(f"テストデータ: {len(test_results):,} 行, {test_meta['race_id'].nunique():,} レース")

    # ── モデル読み込み ────────────────────────────────────────────
    trainer = ModelTrainer.load(settings.model_path)
    fe = FeatureEngineer.from_stats(settings.stats_path)
    bad_trainer = None
    bad_model_path = ROOT / "data/models/lgbm_model_bad_season.pkl"
    if bad_model_path.exists():
        bad_trainer = ModelTrainer.load(bad_model_path)
        logger.info("  不調期専用モデル読み込み完了")

    # ── 全レース処理 ──────────────────────────────────────────────
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
        actual_st = (pos1_num, pos2_num, pos3_num)  # 3連単の正解

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
        mkt_probs = _market_probs(odds_list)

        # ◎の市場インデックス
        honmei_idx = 0  # pred_df 上位なので常に0

        # ── プールサイズ別に3連単を計算 ─────────────────────────
        result_row = {
            "race_id":    race_id,
            "race_date":  str(race_date)[:10],
            "race_name":  race_name,
            "race_month": race_month,
            "mark":       mark,
            "n_entries":  n_entries,
            "h1_prob":    round(h1_prob, 4),
            "gap":        round(gap, 4),
            "actual_st":  "-".join(actual_st),  # 実際の3連単
        }

        # 正規化モデル確率
        probs_raw = pred_df["win_prob"].tolist()
        probs_sum = sum(probs_raw)
        model_probs = [p / probs_sum for p in probs_raw] if probs_sum > 0 else probs_raw

        for pool_n, ticket_n in zip(POOL_SIZES, TICKET_COUNTS):
            # プール: ◎除く上位 pool_n 頭
            partner_rows = pred_df[pred_df["horse_id"] != pred_df.iloc[0]["horse_id"]].head(pool_n)
            pool = []
            for _, row in partner_rows.iterrows():
                hid = str(row["horse_id"])
                num = str(int(float(row["horse_number"])))
                vi  = valid_ids.index(hid) if hid in valid_ids else None
                pool.append((num, vi))

            partner_pred_idxs = {num: i + 1 for i, (num, _) in enumerate(pool)}

            # 3連単: ◎1着固定 (num_2, num_3) を permutation で全探索
            st_all = []
            for (num_2, vi_2), (num_3, vi_3) in _perm(pool, 2):
                p2 = partner_pred_idxs.get(num_2, 1)
                p3 = partner_pred_idxs.get(num_3, 2)
                model_p = _prob_sanrentan(model_probs, honmei_idx, p2, p3)

                if vi_2 is not None and vi_3 is not None:
                    mkt_p = _prob_sanrentan(mkt_probs, honmei_idx, vi_2, vi_3)
                    e_od  = _est_odds(mkt_p, "3連単")
                else:
                    e_od  = _est_odds(model_p, "3連単")

                ev = model_p * e_od
                st_all.append((num_2, num_3, ev, e_od))

            st_all.sort(key=lambda x: -x[2])
            st_sel = st_all[:ticket_n]
            st_est = [od for _, _, _, od in st_sel]

            # トリガミ除外
            if st_est and _synth_odds(st_est) < 1.0:
                combos = []
            else:
                combos = [(n2, n3) for n2, n3, _, e in st_sel]

            # 的中チェック: 実際の着順 (pos1, pos2, pos3) の中で
            # ◎(pred_df 1位) = pos1 かつ (pos2, pos3) が combo に含まれるか
            honmei_num = str(int(float(pred_df.iloc[0]["horse_number"])))
            hit = 0
            hit_od = 0.0
            if honmei_num == pos1_num:  # ◎が1着
                for n2, n3 in combos:
                    if n2 == pos2_num and n3 == pos3_num:
                        hit = 1
                        # 推定3連単オッズ
                        vi_2 = valid_ids.index(horse_num_map[n2]) if n2 in horse_num_map and horse_num_map[n2] in valid_ids else None
                        vi_3 = valid_ids.index(horse_num_map[n3]) if n3 in horse_num_map and horse_num_map[n3] in valid_ids else None
                        if vi_2 is not None and vi_3 is not None:
                            mkt_p = _prob_sanrentan(mkt_probs, honmei_idx, vi_2, vi_3)
                            hit_od = _est_odds(mkt_p, "3連単")
                        break

            # 投票コスト（的中の有無に関わらず全 combo 購入）
            cost = len(combos) * 100
            ret  = hit_od * 100 if hit else 0.0

            result_row[f"pool{pool_n}_combos"]  = len(combos)
            result_row[f"pool{pool_n}_hit"]     = hit
            result_row[f"pool{pool_n}_cost"]    = cost
            result_row[f"pool{pool_n}_ret"]     = round(ret, 1)
            result_row[f"pool{pool_n}_hit_od"]  = round(hit_od, 1)
            result_row[f"pool{pool_n}_combo_str"] = " / ".join(f"◎→{a}→{b}" for a, b in combos[:3]) + ("…" if len(combos) > 3 else "")

        records.append(result_row)

        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(target_ids)} レース処理済み")

    df = pd.DataFrame(records)
    logger.info(f"買い対象 {len(df):,} レース処理完了  ({(time.time()-t0)/60:.1f}分)")

    # ── 結果サマリー ─────────────────────────────────────────────
    border = "=" * 65
    print(f"\n{border}")
    print("  3連単 プールサイズ 5 vs 7 シミュレーション結果")
    print(f"  テスト期間 : {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  買い対象   : {len(df):,} R")
    print(f"  ※3連単オッズは市場単勝オッズから Harville 法で推定")
    print(border)

    for pool_n in POOL_SIZES:
        total_cost = df[f"pool{pool_n}_cost"].sum()
        total_ret  = df[f"pool{pool_n}_ret"].sum()
        hits       = df[f"pool{pool_n}_hit"].sum()
        roi        = total_ret / total_cost * 100 if total_cost > 0 else 0
        avg_combo  = df[f"pool{pool_n}_combos"].mean()

        print(f"\n  【プール{pool_n}頭 / 最大{TICKET_COUNTS[POOL_SIZES.index(pool_n)]}点】")
        print(f"    投票総額: {int(total_cost):,}円  推定回収: {int(total_ret):,}円  ROI: {roi:.1f}%")
        print(f"    的中数 : {int(hits)}/{len(df)}R ({hits/len(df):.2%})")
        print(f"    平均点数: {avg_combo:.1f}点/R")

    # ── 差分サマリー ──────────────────────────────────────────────
    print(f"\n  [拡張効果: プール5→7の差分]")
    newly_hit = df[(df["pool7_hit"] == 1) & (df["pool5_hit"] == 0)]
    lost_hit  = df[(df["pool5_hit"] == 1) & (df["pool7_hit"] == 0)]
    print(f"    プール7のみ的中（新規捕捉）: {len(newly_hit)}R")
    print(f"    プール5のみ的中（プール7で失われた的中）: {len(lost_hit)}R")
    extra_cost = df["pool7_cost"].sum() - df["pool5_cost"].sum()
    extra_ret  = df["pool7_ret"].sum() - df["pool5_ret"].sum()
    print(f"    追加コスト: {int(extra_cost):,}円  追加回収: {int(extra_ret):,}円")
    print(f"    追加分の ROI: {extra_ret/extra_cost*100 if extra_cost > 0 else 0:.1f}%")

    # ── 月別比較 ─────────────────────────────────────────────────
    print(f"\n  [月別 ROI 比較]")
    print(f"  {'月':>4}  {'R数':>4}  {'pool5 ROI':>10}  {'pool7 ROI':>10}  {'差':>7}")
    for month, grp in df.groupby("race_month"):
        c5 = grp["pool5_cost"].sum()
        r5 = grp["pool5_ret"].sum()
        c7 = grp["pool7_cost"].sum()
        r7 = grp["pool7_ret"].sum()
        roi5 = r5 / c5 * 100 if c5 > 0 else 0
        roi7 = r7 / c7 * 100 if c7 > 0 else 0
        print(f"  {month:>4}月  {len(grp):>4}R  {roi5:>10.1f}%  {roi7:>10.1f}%  {roi7-roi5:>+7.1f}pt")

    # ── gap別比較 ────────────────────────────────────────────────
    print(f"\n  [確率差(gap)別 ROI 比較]")
    print(f"  {'gap帯':>10}  {'R数':>4}  {'pool5 ROI':>10}  {'pool7 ROI':>10}  {'差':>7}")
    df["gap_bin"] = pd.cut(
        df["gap"],
        bins=[0, 0.05, 0.10, 0.15, 0.25, 1.0],
        labels=["0-5%", "5-10%", "10-15%", "15-25%", "25%+"]
    )
    for gb, grp in df.groupby("gap_bin", observed=True):
        if len(grp) == 0:
            continue
        c5 = grp["pool5_cost"].sum()
        r5 = grp["pool5_ret"].sum()
        c7 = grp["pool7_cost"].sum()
        r7 = grp["pool7_ret"].sum()
        roi5 = r5 / c5 * 100 if c5 > 0 else 0
        roi7 = r7 / c7 * 100 if c7 > 0 else 0
        print(f"  {str(gb):>10}  {len(grp):>4}R  {roi5:>10.1f}%  {roi7:>10.1f}%  {roi7-roi5:>+7.1f}pt")

    # ── 的中例（プール7のみ捕捉）──────────────────────────────────
    if len(newly_hit) > 0:
        print(f"\n  [プール7のみ捕捉した的中例（最大5件）]")
        for _, row in newly_hit.head(5).iterrows():
            print(f"    {row['race_date']} {row['race_name']} ({row['race_id']})")
            print(f"      実際: {row['actual_st']}  推定配当: {row['pool7_hit_od']:.0f}倍")
            print(f"      pool7 候補: {row['pool7_combo_str']}")

    print(f"\n{border}\n")

    out_path = ROOT / "data/processed/simulate_sanrentan.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"詳細結果保存: {out_path}")


if __name__ == "__main__":
    simulate()
