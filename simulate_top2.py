"""
simulate_top2.py
================
「◎を2頭まで選ぶ」戦略のROIシミュレーション。

現行の予測モデル（lgbm_model.pkl + feature_stats.pkl）を使い、
各レースで上位2頭を予測して以下3戦略を比較する。

  戦略1 (current): 上位1頭のみに単勝100円
  戦略2 (top2_both): 上位1・2頭に各100円（合計200円/レース）
  戦略3 (top2_swap): ◎が確率閾値未満なら2頭目に単勝100円（合計100円）
  戦略4 (top2_conditional): EVが高い方1頭のみ（100円、EV最大化）

テストデータ: test_results_new.csv（2025-01〜2026-04、収集済み）
推論モード: feature_stats.pkl を使用（高速）

実行:
    .venv/bin/python simulate_top2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from config.settings import settings

# ── 設定 ─────────────────────────────────────────────────────────
TEST_RESULTS_CSV = ROOT / "data/raw/test_results_new.csv"
TEST_META_CSV    = ROOT / "data/raw/test_meta_new.csv"

# daily_batch.py と同じ買い条件
MIN_HONMEI_PROB    = 0.30
MARK_STRONG_PROB   = 0.35
MIN_CONFIDENCE_GAP = 0.05
EV_THRESHOLD       = 1.05
MAIDEN_KEYWORDS    = ("新馬", "未勝利")
ENTRIES_ADJ        = 0.003
ENTRIES_BASE       = 10
ENTRIES_CAP        = 0.05

_SEASON_PROB_THRESHOLD: dict[int, float] = {
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

# ── ロガー ────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO",  format=_fmt, colorize=True)
logger.add("logs/simulate_top2.log", level="DEBUG", format=_fmt, rotation="10 MB")


def parse_odds(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


def is_buy_race(
    honmei_prob: float, taikou_prob: float, gap: float,
    race_month: int, is_maiden: bool, n_entries: int,
    honmei_ev: float, honmei_odds: float,
    course_type: str, jyo_code: str, distance: int,
    using_bad_model: bool,
) -> bool:
    """daily_batch.py と同一の買い条件判定"""
    if using_bad_model:
        prob_threshold = _BAD_MODEL_PROB_THRESHOLD
    else:
        season_base    = _SEASON_PROB_THRESHOLD.get(race_month, MIN_HONMEI_PROB)
        entries_adj    = min(max(0, n_entries - ENTRIES_BASE) * ENTRIES_ADJ, ENTRIES_CAP)
        prob_threshold = season_base + entries_adj

    if np.isnan(honmei_ev):
        ev_ok = True
    elif (_HIGH_VALUE_ODDS_MIN <= honmei_odds <= _HIGH_VALUE_ODDS_MAX):
        ev_ok = honmei_ev >= _HIGH_VALUE_EV_THRESHOLD
    else:
        ev_ok = honmei_ev >= EV_THRESHOLD

    is_summer_dirt  = (race_month in _SUMMER_DIRT_SKIP_MONTHS and course_type == "ダート")
    is_bad_venue    = (race_month in _BAD_SEASON_MONTHS and jyo_code in _BAD_VENUE_SKIP_CODES)
    is_bad_distance = (race_month in _BAD_SEASON_MONTHS and distance >= _BAD_SEASON_MAX_DISTANCE)
    gap_ok          = gap >= MIN_CONFIDENCE_GAP

    return (
        not is_maiden
        and gap_ok
        and honmei_prob >= prob_threshold
        and ev_ok
        and not is_summer_dirt
        and not is_bad_venue
        and not is_bad_distance
    )


def simulate() -> None:
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("◎2頭選択 シミュレーション開始")
    logger.info("=" * 65)

    # ── データ読み込み ────────────────────────────────────────────
    test_results = pd.read_csv(TEST_RESULTS_CSV, dtype=str)
    test_meta    = pd.read_csv(TEST_META_CSV,    dtype=str)
    logger.info(f"テストデータ: {len(test_results):,} 行, {test_meta['race_id'].nunique():,} レース")

    # race_date を取得
    test_meta["race_date"] = pd.to_datetime(
        test_meta["race_id"].str[:8].apply(
            lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
        ), errors="coerce"
    )
    test_meta = test_meta.sort_values("race_date").dropna(subset=["race_date"])

    # ── モデル読み込み ────────────────────────────────────────────
    logger.info("モデル・FeatureEngineer 読み込み中...")
    trainer = ModelTrainer.load(settings.model_path)
    fe = FeatureEngineer.from_stats(settings.stats_path)

    bad_model_path = ROOT / "data/models/lgbm_model_bad_season.pkl"
    bad_trainer: ModelTrainer | None = None
    if bad_model_path.exists():
        bad_trainer = ModelTrainer.load(bad_model_path)
        logger.info("  不調期専用モデル読み込み完了")

    # ── 全レース処理 ──────────────────────────────────────────────
    logger.info(f"推論開始: {len(test_meta):,} レース（推論モード・高速）")
    records = []
    target_ids = test_meta["race_id"].tolist()

    for i, race_id in enumerate(target_ids):
        race_entries = test_results[test_results["race_id"] == race_id].copy()
        if len(race_entries) < 3:
            continue

        meta_row     = test_meta[test_meta["race_id"] == race_id].iloc[0]
        race_date    = meta_row["race_date"]
        course_type  = str(meta_row.get("course_type", ""))
        race_name    = str(meta_row.get("race_name", ""))
        distance     = int(meta_row.get("distance", 0) or 0)
        gc_code      = int(meta_row.get("ground_condition_code", -1) or -1)
        wx_code      = int(meta_row.get("weather_code", -1) or -1)
        jyo_code     = str(race_id)[4:6]
        race_month   = race_date.month
        n_entries    = len(race_entries)

        if not course_type or distance == 0:
            continue

        # entry_df 組み立て
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

        # 特徴量生成（推論モード: feature_stats.pkl 使用）
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

        # 予測
        using_bad_model = (bad_trainer is not None and race_month in _BAD_SEASON_MONTHS)
        active_trainer  = bad_trainer if using_bad_model else trainer

        X = (feat_df[FeatureEngineer.FEATURE_COLUMNS]
             .apply(pd.to_numeric, errors="coerce")
             .fillna(0))
        win_probs = active_trainer.model.predict(X, num_threads=1)
        if active_trainer.place_model is not None:
            place_probs = active_trainer.place_model.predict(X, num_threads=1)
            probs = 0.7 * win_probs + 0.3 * place_probs
        else:
            probs = win_probs

        pred_df = feat_df[["horse_id","horse_name","horse_number"]].copy()
        pred_df["win_prob"] = probs
        pred_df = pred_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

        if len(pred_df) < 2:
            continue

        # 上位2頭の情報
        h1 = pred_df.iloc[0]
        h2 = pred_df.iloc[1]

        h1_prob = float(h1["win_prob"])
        h2_prob = float(h2["win_prob"])
        gap = h1_prob - h2_prob

        # オッズ・EV
        odds_map = {}
        if "odds" in feat_df.columns:
            odds_map = feat_df.set_index("horse_id")["odds"].to_dict()

        h1_odds_raw = odds_map.get(str(h1["horse_id"]), float("nan"))
        h2_odds_raw = odds_map.get(str(h2["horse_id"]), float("nan"))
        h1_odds = float(h1_odds_raw) if not pd.isna(h1_odds_raw) else float("nan")
        h2_odds = float(h2_odds_raw) if not pd.isna(h2_odds_raw) else float("nan")
        h1_ev = h1_prob * h1_odds if not np.isnan(h1_odds) else float("nan")
        h2_ev = h2_prob * h2_odds if not np.isnan(h2_odds) else float("nan")

        # 買い条件判定（1頭目基準）
        is_maiden = any(k in race_name for k in MAIDEN_KEYWORDS)
        buy = is_buy_race(
            h1_prob, h2_prob, gap, race_month, is_maiden, n_entries,
            h1_ev, h1_odds, course_type, jyo_code, distance, using_bad_model
        )
        if using_bad_model:
            mark_strong = _BAD_MODEL_MARK_STRONG
        else:
            mark_strong = MARK_STRONG_PROB
        mark = "◎" if buy and h1_prob >= mark_strong else ("○" if buy else "△")
        if mark == "○" and race_month in _MARK_O_SKIP_MONTHS:
            mark = "△"
            buy = False

        # 実際の着順・オッズ取得
        actual = race_entries[["horse_id","finish_position","odds"]].copy()
        actual["pos"] = pd.to_numeric(
            actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        actual["odds_f"] = actual["odds"].apply(parse_odds)
        actual_map = actual.set_index("horse_id")[["pos","odds_f"]].to_dict("index")

        def get_result(hid):
            r = actual_map.get(str(hid), {})
            pos  = r.get("pos", float("nan"))
            odds = r.get("odds_f", float("nan"))
            win  = int(pos == 1) if not np.isnan(pos) else 0
            ret  = odds * 100 if win and not np.isnan(odds) else 0.0
            return pos, odds, win, ret

        h1_pos, h1_odds_a, h1_win, h1_ret = get_result(h1["horse_id"])
        h2_pos, h2_odds_a, h2_win, h2_ret = get_result(h2["horse_id"])

        records.append({
            "race_id":      race_id,
            "race_date":    str(race_date)[:10],
            "race_name":    race_name,
            "race_month":   race_month,
            "course_type":  course_type,
            "distance":     distance,
            "n_entries":    n_entries,
            "is_maiden":    is_maiden,
            "mark":         mark,
            "is_buy":       buy,
            # 1頭目
            "h1_name":      str(h1["horse_name"]),
            "h1_prob":      round(h1_prob, 4),
            "h1_ev":        round(h1_ev, 3) if not np.isnan(h1_ev) else None,
            "h1_odds_pre":  h1_odds,
            "h1_pos":       h1_pos,
            "h1_odds_a":    h1_odds_a,
            "h1_win":       h1_win,
            "h1_ret":       h1_ret,
            # 2頭目
            "h2_name":      str(h2["horse_name"]),
            "h2_prob":      round(h2_prob, 4),
            "h2_ev":        round(h2_ev, 3) if not np.isnan(h2_ev) else None,
            "h2_odds_pre":  h2_odds,
            "h2_pos":       h2_pos,
            "h2_odds_a":    h2_odds_a,
            "h2_win":       h2_win,
            "h2_ret":       h2_ret,
            "gap":          round(gap, 4),
        })

        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(target_ids)} レース処理済み")

    df = pd.DataFrame(records)
    logger.info(f"処理完了: {len(df):,} レース  ({(time.time()-t0)/60:.1f}分)")

    # ── シミュレーション比較 ─────────────────────────────────────
    bought = df[df["is_buy"] == True].copy()
    border = "=" * 65

    print(f"\n{border}")
    print("  ◎2頭選択 シミュレーション結果")
    print(f"  テスト期間 : {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  総レース数 : {len(df):,} R（買い対象: {len(bought):,} R）")
    print(border)

    # ── 戦略1: 現行（上位1頭のみ 100円）──────────────────────────
    s1_bet = len(bought) * 100
    s1_ret = bought["h1_ret"].sum()
    s1_hits = bought["h1_win"].sum()
    s1_roi  = s1_ret / s1_bet * 100

    # ── 戦略2: 上位2頭に各100円（合計200円/レース）──────────────
    s2_bet = len(bought) * 200
    s2_ret = bought["h1_ret"].sum() + bought["h2_ret"].sum()
    s2_hits = int((bought["h1_win"] | bought["h2_win"]).sum())  # いずれか1頭当たり
    s2_roi  = s2_ret / s2_bet * 100

    # ── 戦略3: EV最大の1頭のみ（100円）──────────────────────────
    # h1_ev と h2_ev を比較し、より高いEVの馬に投票
    bought = bought.copy()
    bought["h1_ev_num"] = pd.to_numeric(bought["h1_ev"], errors="coerce").fillna(0)
    bought["h2_ev_num"] = pd.to_numeric(bought["h2_ev"], errors="coerce").fillna(0)
    bought["bet_ev_win"] = np.where(
        bought["h1_ev_num"] >= bought["h2_ev_num"],
        bought["h1_win"], bought["h2_win"]
    )
    bought["bet_ev_ret"] = np.where(
        bought["h1_ev_num"] >= bought["h2_ev_num"],
        bought["h1_ret"], bought["h2_ret"]
    )
    s3_bet  = len(bought) * 100
    s3_ret  = bought["bet_ev_ret"].sum()
    s3_hits = bought["bet_ev_win"].sum()
    s3_roi  = s3_ret / s3_bet * 100

    # ── 戦略4: h2のEVがEV_THRESHOLD以上のとき2頭目も追加（条件付き）──
    bought["h2_ev_ok"] = (
        bought["h2_ev_num"] >= EV_THRESHOLD
    )
    s4_h1_bet = len(bought) * 100
    s4_h2_bet = bought["h2_ev_ok"].sum() * 100
    s4_bet    = s4_h1_bet + s4_h2_bet
    s4_ret    = bought["h1_ret"].sum() + bought.loc[bought["h2_ev_ok"], "h2_ret"].sum()
    s4_hits   = bought["h1_win"].sum() + bought.loc[bought["h2_ev_ok"], "h2_win"].sum()
    s4_roi    = s4_ret / s4_bet * 100

    print(f"\n  【戦略1】現行: 上位1頭のみ 100円/R")
    print(f"    投票額: {s1_bet:,}円  回収: {int(s1_ret):,}円  ROI: {s1_roi:.1f}%")
    print(f"    的中 : {s1_hits}/{len(bought)}R ({s1_hits/len(bought):.2%})")

    print(f"\n  【戦略2】上位2頭に各100円（合計200円/R）")
    print(f"    投票額: {s2_bet:,}円  回収: {int(s2_ret):,}円  ROI: {s2_roi:.1f}%")
    print(f"    的中 : {s2_hits}/{len(bought)}R（いずれかが1着 {s2_hits/len(bought):.2%}）")
    print(f"    ROI差: {s2_roi - s1_roi:+.1f}pt（vs 戦略1）")

    print(f"\n  【戦略3】EV最大の1頭のみ 100円/R（1か2頭目を切り替え）")
    print(f"    投票額: {s3_bet:,}円  回収: {int(s3_ret):,}円  ROI: {s3_roi:.1f}%")
    print(f"    的中 : {s3_hits}/{len(bought)}R ({s3_hits/len(bought):.2%})")
    print(f"    ROI差: {s3_roi - s1_roi:+.1f}pt（vs 戦略1）")

    print(f"\n  【戦略4】1頭目に100円 + 2頭目EV≥1.05なら追加100円")
    print(f"    投票額: {s4_bet:,}円  回収: {int(s4_ret):,}円  ROI: {s4_roi:.1f}%")
    print(f"    2頭目追加: {bought['h2_ev_ok'].sum()}/{len(bought)}R")
    print(f"    ROI差: {s4_roi - s1_roi:+.1f}pt（vs 戦略1）")

    # ── 月別詳細（戦略1 vs 戦略2）──────────────────────────────
    print(f"\n  [月別 ROI 比較: 戦略1 vs 戦略2]")
    print(f"  {'月':>4}  {'R数':>4}  {'S1 ROI':>8}  {'S2 ROI':>8}  {'差':>6}")
    for month, grp in bought.groupby("race_month"):
        s1r = grp["h1_ret"].sum() / (len(grp) * 100) * 100
        s2r = (grp["h1_ret"].sum() + grp["h2_ret"].sum()) / (len(grp) * 200) * 100
        print(f"  {month:>4}月  {len(grp):>4}R  {s1r:>8.1f}%  {s2r:>8.1f}%  {s2r-s1r:>+6.1f}pt")

    # ── 上位2頭の確率差による層別（戦略2の効果分析）──────────────
    print(f"\n  [確率差(gap)別 ROI 比較: 戦略1 vs 戦略2]")
    print(f"  {'gap帯':>10}  {'R数':>4}  {'S1 ROI':>8}  {'S2 ROI':>8}  {'差':>6}")
    bought["gap_bin"] = pd.cut(
        bought["gap"],
        bins=[0, 0.05, 0.10, 0.15, 0.25, 1.0],
        labels=["0-5%", "5-10%", "10-15%", "15-25%", "25%+"]
    )
    for gb, grp in bought.groupby("gap_bin", observed=True):
        if len(grp) == 0:
            continue
        s1r = grp["h1_ret"].sum() / (len(grp) * 100) * 100
        s2r = (grp["h1_ret"].sum() + grp["h2_ret"].sum()) / (len(grp) * 200) * 100
        print(f"  {str(gb):>10}  {len(grp):>4}R  {s1r:>8.1f}%  {s2r:>8.1f}%  {s2r-s1r:>+6.1f}pt")

    # ── h2の単独成績（参考）─────────────────────────────────────
    print(f"\n  [参考] 2番手(対抗)馬の単独成績（買い対象レースのみ）")
    h2_buy = len(bought)
    h2_win = bought["h2_win"].sum()
    h2_ret = bought["h2_ret"].sum()
    h2_roi = h2_ret / (h2_buy * 100) * 100
    print(f"    投票額: {h2_buy*100:,}円  回収: {int(h2_ret):,}円  ROI: {h2_roi:.1f}%")
    print(f"    的中 : {h2_win}/{h2_buy}R ({h2_win/h2_buy:.2%})")

    # 平均確率・オッズ
    print(f"    平均確率: {bought['h2_prob'].mean():.3f}  平均EV: {bought['h2_ev_num'].mean():.3f}")
    avg_h2_odds = pd.to_numeric(bought['h2_odds_a'], errors='coerce').mean()
    print(f"    平均オッズ(結果): {avg_h2_odds:.1f}倍")

    print(f"\n{border}\n")

    # CSV保存
    out_path = ROOT / "data/processed/simulate_top2.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"詳細結果保存: {out_path}")


if __name__ == "__main__":
    simulate()
