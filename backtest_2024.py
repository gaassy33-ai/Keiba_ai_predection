"""
backtest_2024.py
====================
2024年データを使ったバックテスト（季節傾向確認用・高速版）。

- テストデータ: train_results.csv / train_meta.csv の 2024年分
- 統計: feature_stats.pkl（学習済み）をそのまま利用 → 数分で完了
  ※ジョッキー勝率等の集計に2024年データも含まれるため
    わずかなデータリーク（過楽観）が生じる点に注意。
- モデル: data/models/lgbm_model.pkl（本番モデル）
- 買い条件: daily_batch.py と同一

実行:
    python backtest_2024.py
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

# ── 買い条件（daily_batch.py と同一） ──────────────────────────
MIN_HONMEI_PROB    = 0.30
MARK_STRONG_PROB   = 0.35
MIN_CONFIDENCE_GAP = 0.05
EV_THRESHOLD       = 1.05
MAIDEN_KEYWORDS    = ("新馬", "未勝利")
ENTRIES_ADJ        = 0.003
ENTRIES_BASE       = 10
ENTRIES_CAP        = 0.05

# 季節フィルター（daily_batch.py と同一設定）
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

# 案D: 不調期 開催場フィルタ（不調期ROI 0%会場を除外）
_BAD_SEASON_MONTHS    = {1, 7, 8, 9, 11, 12}
_BAD_VENUE_SKIP_CODES = {"02", "04", "09"}   # 函館・新潟・阪神（不調期ROI 0%）

# 案E: 不調期 距離フィルタ（1800m以上は見送り）
_BAD_SEASON_MAX_DISTANCE = 1800

# 案B: 不調期専用モデルの閾値（スケールが全期間モデルと異なるため専用設定）
# 不調期モデルのprob: mean=0.229, max=0.377（全期間モデルのmean=0.343と大きく異なる）
# ≥0.25: 勝率27.1%, 複勝率70.1% → この水準を基準閾値とする
_BAD_MODEL_PROB_THRESHOLD = 0.25
_BAD_MODEL_MARK_STRONG    = 0.28   # ◎の基準（≥0.28: 勝率27%, 複勝率74%）

BACKTEST_OUT = ROOT / "data" / "processed" / "backtest_2024.csv"

Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/backtest_2024.log", level="DEBUG", format=_fmt, rotation="20 MB")


def parse_odds(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


def evaluate(test_results: pd.DataFrame, test_meta: pd.DataFrame,
             fe: FeatureEngineer, trainer: ModelTrainer,
             bad_season_trainer: ModelTrainer | None = None) -> pd.DataFrame:
    """保存済み統計を使ってレースごとに予測・評価（高速版）"""

    test_meta = test_meta.copy()
    if "race_date" not in test_meta.columns:
        test_meta["race_date"] = pd.to_datetime(
            test_meta["race_id"].str[:8].apply(
                lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
            ), errors="coerce"
        )
    else:
        test_meta["race_date"] = pd.to_datetime(test_meta["race_date"], errors="coerce")
    test_meta = test_meta.sort_values("race_date").dropna(subset=["race_date"])

    results = []
    target_ids = test_meta["race_id"].tolist()
    logger.info(f"バックテスト対象: {len(target_ids):,} レース")

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

        if not course_type or distance == 0 or distance >= 2750:
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

        # 特徴量生成（保存済み統計を使用）
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

        race_month = race_date.month  # race_dateから月を取得（race_idの4:6は場コード）

        # 不調期専用モデルに切り替え（予測前に決定）
        active_trainer = (
            bad_season_trainer
            if bad_season_trainer is not None and race_month in _BAD_SEASON_MONTHS
            else trainer
        )

        # 予測
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

        honmei_prob = float(pred_df.iloc[0]["win_prob"])
        taikou_prob = float(pred_df.iloc[1]["win_prob"]) if len(pred_df) > 1 else 0.0
        gap         = honmei_prob - taikou_prob
        n_entries   = len(race_entries)

        # 不調期専用モデル使用時は専用閾値・◎基準を適用
        using_bad_model = (bad_season_trainer is not None and race_month in _BAD_SEASON_MONTHS)
        if using_bad_model:
            prob_threshold = _BAD_MODEL_PROB_THRESHOLD
            mark_strong    = _BAD_MODEL_MARK_STRONG
        else:
            season_base    = _SEASON_PROB_THRESHOLD.get(race_month, MIN_HONMEI_PROB)
            entries_adj    = min(max(0, n_entries - ENTRIES_BASE) * ENTRIES_ADJ, ENTRIES_CAP)
            prob_threshold = season_base + entries_adj
            mark_strong    = MARK_STRONG_PROB
        is_maiden      = any(k in race_name for k in MAIDEN_KEYWORDS)

        if "odds" in feat_df.columns:
            honmei_id       = str(pred_df.iloc[0]["horse_id"])
            odds_col        = feat_df.set_index("horse_id")["odds"]
            raw_odds        = odds_col.get(honmei_id)
            honmei_odds_val = float(raw_odds) if raw_odds is not None else float("nan")
            honmei_ev       = honmei_prob * honmei_odds_val if not np.isnan(honmei_odds_val) else float("nan")
            if (not np.isnan(honmei_odds_val)
                    and _HIGH_VALUE_ODDS_MIN <= honmei_odds_val <= _HIGH_VALUE_ODDS_MAX):
                ev_thr = _HIGH_VALUE_EV_THRESHOLD
            else:
                ev_thr = EV_THRESHOLD
            ev_ok = np.isnan(honmei_ev) or honmei_ev >= ev_thr
        else:
            honmei_ev = float("nan")
            ev_ok     = True

        is_summer_dirt = (race_month in _SUMMER_DIRT_SKIP_MONTHS and course_type == "ダート")

        # 案D: 不調期の開催場フィルタ
        jyo_code = str(race_id)[4:6]
        is_bad_venue = (race_month in _BAD_SEASON_MONTHS and jyo_code in _BAD_VENUE_SKIP_CODES)

        # 案E: 不調期の距離フィルタ
        is_bad_distance = (race_month in _BAD_SEASON_MONTHS and distance >= _BAD_SEASON_MAX_DISTANCE)

        # 案F: 不調期は複勝メイン
        bet_mode = "複勝" if race_month in _BAD_SEASON_MONTHS else "単勝"

        gap_ok  = gap >= MIN_CONFIDENCE_GAP
        is_buy  = (not is_maiden) and gap_ok and (honmei_prob >= prob_threshold) and ev_ok and (not is_summer_dirt) and (not is_bad_venue) and (not is_bad_distance)

        if is_buy and honmei_prob >= mark_strong:
            mark = "◎"
        elif is_buy:
            mark = "○"
        else:
            mark = "△"

        if mark == "○" and race_month in _MARK_O_SKIP_MONTHS:
            mark = "△"
            is_buy = False

        actual = race_entries[["horse_id","finish_position","odds"]].copy()
        actual["finish_pos_num"] = pd.to_numeric(
            actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        actual["odds_f"] = actual["odds"].apply(parse_odds)

        honmei_actual = actual[actual["horse_id"] == str(pred_df.iloc[0]["horse_id"])]
        honmei_pos    = honmei_actual["finish_pos_num"].iloc[0] if not honmei_actual.empty else np.nan
        honmei_odds_a = honmei_actual["odds_f"].iloc[0]         if not honmei_actual.empty else np.nan

        honmei_win   = int(honmei_pos == 1) if not np.isnan(honmei_pos) else 0
        honmei_place = int(honmei_pos <= 3) if not np.isnan(honmei_pos) else 0
        honmei_ret   = honmei_odds_a * 100  if honmei_win and not np.isnan(honmei_odds_a) else 0.0

        results.append({
            "race_id":           race_id,
            "race_date":         str(race_date)[:10],
            "race_name":         race_name,
            "course_type":       course_type,
            "distance":          distance,
            "n_entries":         n_entries,
            "is_maiden":         is_maiden,
            "mark":              mark,
            "is_buy":            is_buy,
            "honmei_name":       pred_df.iloc[0]["horse_name"],
            "honmei_prob":       round(honmei_prob, 4),
            "taikou_prob":       round(taikou_prob, 4),
            "gap":               round(gap, 4),
            "honmei_ev":         round(honmei_ev, 3) if not np.isnan(honmei_ev) else None,
            "honmei_odds":       honmei_odds_a,
            "honmei_actual_pos": honmei_pos,
            "honmei_win":        honmei_win,
            "honmei_place":      honmei_place,
            "honmei_return":     honmei_ret,
            "bet_mode":          bet_mode,
        })

        if (i + 1) % 200 == 0:
            buys = [r for r in results if r["is_buy"]]
            wins = sum(r["honmei_win"] for r in buys)
            roi  = (sum(r["honmei_return"] for r in buys) / (len(buys) * 100) * 100) if buys else 0
            win_rate_str = f"{wins/len(buys):.1%}" if buys else "-"
            logger.info(f"  {i+1}/{len(target_ids)} | 買い: {len(buys)}R | "
                        f"的中: {wins}R ({win_rate_str}) | "
                        f"ROI: {roi:.1f}%")

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame) -> None:
    bought = df[df["is_buy"] == True]
    n_buy  = len(bought)
    border = "=" * 62

    print(f"\n{border}")
    print("  2024年 バックテスト結果（高速版・統計流用）")
    print(f"  テスト期間  : {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  総レース数  : {len(df):,} R")
    print(f"  買い対象    : {n_buy:,} R  （見送り: {len(df)-n_buy:,} R）")
    print(border)

    if n_buy == 0:
        print("  ⚠️  買い対象レースなし")
        return

    hits      = bought["honmei_win"].sum()
    places    = bought["honmei_place"].sum()
    total_bet = n_buy * 100
    total_ret = bought["honmei_return"].sum()
    roi       = total_ret / total_bet * 100

    print(f"  {'単勝的中率（買い対象）':<20}  {hits/n_buy:>8.2%}  ({hits}/{n_buy})")
    print(f"  {'複勝的中率（買い対象）':<20}  {places/n_buy:>8.2%}  ({places}/{n_buy})")
    print(f"  {'単勝ROI（買い対象）':<21}  {roi:>8.1f}%  (¥{int(total_ret):,} / ¥{total_bet:,})")
    avg_hit_odds = bought.loc[bought["honmei_win"]==1, "honmei_odds"].mean()
    print(f"  {'的中時平均オッズ':<23}  {avg_hit_odds:>8.1f} 倍")

    print(f"\n  [マーク別 内訳]")
    for mark in ["◎", "○"]:
        mg = bought[bought["mark"] == mark]
        if len(mg) == 0:
            continue
        mh   = mg["honmei_win"].sum()
        mroi = mg["honmei_return"].sum() / (len(mg) * 100) * 100
        print(f"    {mark}  {len(mg):>4}R  的中率 {mh/len(mg):.2%}  ROI {mroi:.1f}%")

    print(f"\n  [月別 的中率・ROI]")
    df2 = bought.copy()
    df2["ym"] = df2["race_date"].str[:7]
    for ym, grp in df2.groupby("ym"):
        h = grp["honmei_win"].sum()
        r = grp["honmei_return"].sum() / (len(grp) * 100) * 100
        print(f"    {ym}  {len(grp):>3}R  的中率 {h/len(grp):.2%}  ROI {r:.1f}%")

    print(f"\n  [コース別 的中率]")
    for ct, grp in bought.groupby("course_type"):
        h = grp["honmei_win"].sum()
        print(f"    {ct}  {h/len(grp):.2%}  ({h}/{len(grp)})")

    print(f"\n  [距離帯別 的中率]")
    bought2 = bought.copy()
    bought2["dist_bin"] = pd.cut(
        pd.to_numeric(bought2["distance"], errors="coerce"),
        bins=[0,1400,1800,2200,9999], labels=["短距離","マイル","中距離","長距離"]
    )
    for db, grp in bought2.groupby("dist_bin", observed=True):
        h = grp["honmei_win"].sum()
        print(f"    {db}  {h/len(grp):.2%}  ({h}/{len(grp)})")

    n_all_ok = len(df[~df["is_maiden"]])
    h_all    = df[~df["is_maiden"]]["honmei_win"].sum()
    print(f"\n  [参考] 全レース（新馬除外・フィルタなし）: {h_all/n_all_ok:.2%} ({h_all}/{n_all_ok})")
    print(f"{border}\n")


def main():
    t0 = time.time()
    logger.info("=" * 62)
    logger.info("2024年 バックテスト開始（高速版）")
    logger.info("  ※ feature_stats.pkl の保存済み統計を使用")
    logger.info("=" * 62)

    # 2024年分のみ抽出
    logger.info("データ読み込み中...")
    all_results = pd.read_csv("data/raw/train_results.csv", dtype=str)
    all_meta    = pd.read_csv("data/raw/train_meta.csv",    dtype=str)

    test_results = all_results[all_results["race_id"].str.startswith("2024")].copy()
    test_meta    = all_meta[all_meta["race_id"].str.startswith("2024")].copy()
    logger.info(f"  2024年データ: {len(test_results):,} 行 / {test_meta['race_id'].nunique():,} レース")

    # 保存済み統計から FeatureEngineer を生成（高速）
    fe = FeatureEngineer.from_stats(settings.stats_path)

    # モデル読み込み
    trainer = ModelTrainer.load(settings.model_path)
    logger.info(f"  モデル特徴量数: {trainer.model.num_feature()}")

    # 不調期専用モデル（存在すれば読み込む）
    _bad_model_path = ROOT / "data" / "models" / "lgbm_model_bad_season.pkl"
    bad_season_trainer: ModelTrainer | None = None
    if _bad_model_path.exists():
        bad_season_trainer = ModelTrainer.load(_bad_model_path)
        logger.info(f"  不調期専用モデル読み込み完了")

    # バックテスト評価
    result_df = evaluate(test_results, test_meta, fe, trainer, bad_season_trainer)

    # 保存・表示
    BACKTEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(BACKTEST_OUT, index=False)
    logger.info(f"結果保存: {BACKTEST_OUT}")

    print_summary(result_df)
    logger.info(f"総所要時間: {(time.time()-t0)/60:.1f} 分")


if __name__ == "__main__":
    main()
