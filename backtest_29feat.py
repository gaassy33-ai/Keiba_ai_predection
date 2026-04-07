"""
backtest_29feat.py
====================
29特徴量モデル（2026-04再学習）のバックテスト。

- テストデータ: 2025-01-01 〜 2026-04-05（全10会場）
- 旧テストデータが旧スクレイパーで汚染されているため、
  collect_weekly.py の仕組みを使って再収集してから評価する。
- 本番モデル (data/models/lgbm_model.pkl) をそのまま評価。
- daily_batch.py の買い条件（MIN_HONMEI_PROB / EV等）を適用した
  「実際の投票対象レース」の ROI も計算する。

実行:
    python backtest_29feat.py
    python backtest_29feat.py --skip-collect   # 収集済みなら再収集スキップ
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from src.scraper.netkeiba_scraper import NetkeibaScraper
from config.settings import settings

# ── バックテスト設定 ──────────────────────────────────────────
TEST_START  = date(2025, 1, 1)
TEST_END    = date(2026, 4, 5)
JYO_CODES   = ["01","02","03","04","05","06","07","08","09","10"]

# daily_batch.py と同じ買い条件
MIN_HONMEI_PROB    = 0.30
MARK_STRONG_PROB   = 0.35
MIN_CONFIDENCE_GAP = 0.05
EV_THRESHOLD       = 1.05
MAIDEN_KEYWORDS    = ("新馬", "未勝利")
ENTRIES_ADJ        = 0.003
ENTRIES_BASE       = 10
ENTRIES_CAP        = 0.05

TEST_RESULTS_CSV = ROOT / "data" / "raw" / "test_results_new.csv"
TEST_META_CSV    = ROOT / "data" / "raw" / "test_meta_new.csv"
BACKTEST_OUT     = ROOT / "data" / "processed" / "backtest_29feat.csv"

Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/backtest_29feat.log", level="DEBUG", format=_fmt, rotation="20 MB")


# ─────────────────────────────────────────────────────────────
# Step 1: テストデータ収集
# ─────────────────────────────────────────────────────────────

def collect_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2025-2026年のテストデータを収集（新スクレイパー）"""
    # 既存収集済みチェック
    if TEST_RESULTS_CSV.exists() and TEST_META_CSV.exists():
        done_ids = set(pd.read_csv(TEST_RESULTS_CSV, dtype=str)["race_id"].unique())
        meta_ids = set(pd.read_csv(TEST_META_CSV,    dtype=str)["race_id"].unique())
        logger.info(f"既存テストデータ: {len(done_ids):,} race_ids")
        if len(done_ids) > 100:  # 十分な量があれば再利用
            df_r = pd.read_csv(TEST_RESULTS_CSV, dtype=str)
            df_m = pd.read_csv(TEST_META_CSV,    dtype=str)
            return df_r, df_m

    logger.info(f"テストデータ収集: {TEST_START} 〜 {TEST_END}")
    with NetkeibaScraper() as scraper:
        race_ids = scraper.collect_race_ids_for_period(
            start_date=TEST_START,
            end_date=TEST_END,
            jyo_codes=JYO_CODES,
            save_path="data/raw/test_race_ids_new.csv",
        )
        logger.info(f"対象 race_ids: {len(race_ids):,}")

        # 既収集分を除外
        done_ids: set = set()
        if TEST_RESULTS_CSV.exists():
            done_ids = set(pd.read_csv(TEST_RESULTS_CSV, dtype=str)["race_id"].unique())
        new_ids = [r for r in race_ids if r not in done_ids]
        logger.info(f"新規取得対象: {len(new_ids):,} レース（推定 {len(new_ids)*3//60} 分）")

        results_df, meta_df = scraper.fetch_bulk_results_and_meta(
            new_ids,
            checkpoint_path="data/raw/test_checkpoint_new",
        )

    # 既存分とマージ
    if TEST_RESULTS_CSV.exists() and done_ids:
        existing_r = pd.read_csv(TEST_RESULTS_CSV, dtype=str)
        existing_m = pd.read_csv(TEST_META_CSV,    dtype=str)
        results_df = pd.concat([existing_r, results_df], ignore_index=True).drop_duplicates(subset=["race_id","horse_id"])
        meta_df    = pd.concat([existing_m, meta_df],    ignore_index=True).drop_duplicates(subset=["race_id"])

    TEST_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TEST_RESULTS_CSV, index=False)
    meta_df.to_csv(TEST_META_CSV, index=False)
    logger.info(f"テストデータ保存: {len(results_df):,} 行, {len(meta_df):,} レース")
    return results_df, meta_df


# ─────────────────────────────────────────────────────────────
# Step 2: バックテスト評価
# ─────────────────────────────────────────────────────────────

def parse_odds(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


def evaluate(test_results: pd.DataFrame, test_meta: pd.DataFrame,
             train_results: pd.DataFrame, trainer: ModelTrainer) -> pd.DataFrame:
    """時系列ルックバックでレースごとに予測・評価"""

    # 学習データと統合（ルックバック用）
    full_history = pd.concat([train_results, test_results], ignore_index=True)
    fe_base = FeatureEngineer(full_history)

    # race_date でソート
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

    # full_history にも race_date を付与
    date_map = test_meta.set_index("race_id")["race_date"].to_dict()
    train_date_map = {
        rid: pd.Timestamp(f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}")
        for rid in train_results["race_id"].unique()
        if len(rid) >= 8
    }
    date_map.update(train_date_map)
    fe_base.history["race_date"] = fe_base.history["race_id"].map(date_map)

    results = []
    target_ids = test_meta["race_id"].tolist()
    logger.info(f"バックテスト対象: {len(target_ids):,} レース")

    for i, race_id in enumerate(target_ids):
        race_entries = test_results[test_results["race_id"] == race_id].copy()
        if len(race_entries) < 3:
            continue

        meta_row = test_meta[test_meta["race_id"] == race_id].iloc[0]
        race_date    = meta_row["race_date"]
        course_type  = str(meta_row.get("course_type", ""))
        race_name    = str(meta_row.get("race_name", ""))
        distance     = int(meta_row.get("distance", 0) or 0)
        gc_code      = int(meta_row.get("ground_condition_code", -1) or -1)
        wx_code      = int(meta_row.get("weather_code", -1) or -1)

        if not course_type or distance == 0:
            continue

        # このレースより前のデータのみ使う
        history_before = fe_base.history[fe_base.history["race_date"] < race_date]

        # entry_df 組み立て
        wt_col = "weight_carried_num" if "weight_carried_num" in race_entries.columns else "weight_carried"
        entry_df = race_entries[["horse_id","horse_name","horse_number","frame_number",
                                  "jockey_id","sex_age","weight_carried"]].copy()
        entry_df["sex"] = entry_df["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entry_df["sex_age"].str[1:], errors="coerce")
        entry_df["weight_carried"] = pd.to_numeric(entry_df["weight_carried"], errors="coerce")
        entry_df["father"] = ""
        entry_df["mother_father"] = ""

        # オッズ（EV計算用）: odds列が正しく取得されているか確認
        if "odds" in race_entries.columns:
            entry_df["odds"] = pd.to_numeric(race_entries["odds"].values, errors="coerce")

        # レースクラスコード・会場コード
        race_class_code = FeatureEngineer._race_name_to_class_code(race_name)
        try:
            venue_code = int(race_id[4:6])
        except Exception:
            venue_code = -1

        # 特徴量生成
        tmp_fe = FeatureEngineer(history_before)
        tmp_fe.precompute_aggregations()
        try:
            feat_df = tmp_fe.build_entry_features(
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
        X = (feat_df[FeatureEngineer.FEATURE_COLUMNS]
             .apply(pd.to_numeric, errors="coerce")
             .fillna(0))
        win_probs = trainer.model.predict(X, num_threads=1)
        if trainer.place_model is not None:
            place_probs = trainer.place_model.predict(X, num_threads=1)
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

        # 買い条件判定（daily_batch.py と同一ロジック）
        entries_adj    = min(max(0, n_entries - ENTRIES_BASE) * ENTRIES_ADJ, ENTRIES_CAP)
        prob_threshold = MIN_HONMEI_PROB + entries_adj
        is_maiden      = any(k in race_name for k in MAIDEN_KEYWORDS)

        # EVフィルタ
        if "odds" in feat_df.columns:
            honmei_id  = str(pred_df.iloc[0]["horse_id"])
            odds_col   = feat_df.set_index("horse_id")["odds"]
            raw_odds   = odds_col.get(honmei_id)
            honmei_odds_val = float(raw_odds) if raw_odds is not None else float("nan")
            honmei_ev  = honmei_prob * honmei_odds_val if not np.isnan(honmei_odds_val) else float("nan")
            ev_ok      = np.isnan(honmei_ev) or honmei_ev >= EV_THRESHOLD
        else:
            honmei_ev  = float("nan")
            ev_ok      = True

        gap_ok  = gap >= MIN_CONFIDENCE_GAP
        is_buy  = (not is_maiden) and gap_ok and (honmei_prob >= prob_threshold) and ev_ok

        if is_buy and honmei_prob >= MARK_STRONG_PROB:
            mark = "◎"
        elif is_buy:
            mark = "○"
        else:
            mark = "△"

        # 実際の着順・オッズ
        actual = race_entries[["horse_id","finish_position","odds"]].copy()
        actual["finish_pos_num"] = pd.to_numeric(
            actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        actual["odds_f"] = actual["odds"].apply(parse_odds)

        honmei_actual = actual[actual["horse_id"] == str(pred_df.iloc[0]["horse_id"])]
        honmei_pos    = honmei_actual["finish_pos_num"].iloc[0] if not honmei_actual.empty else np.nan
        honmei_odds_a = honmei_actual["odds_f"].iloc[0] if not honmei_actual.empty else np.nan

        honmei_win   = int(honmei_pos == 1) if not np.isnan(honmei_pos) else 0
        honmei_place = int(honmei_pos <= 3) if not np.isnan(honmei_pos) else 0
        honmei_ret   = honmei_odds_a * 100 if honmei_win and not np.isnan(honmei_odds_a) else 0.0

        results.append({
            "race_id":        race_id,
            "race_date":      str(race_date)[:10],
            "race_name":      race_name,
            "course_type":    course_type,
            "distance":       distance,
            "n_entries":      n_entries,
            "is_maiden":      is_maiden,
            "mark":           mark,
            "is_buy":         is_buy,
            "honmei_name":    pred_df.iloc[0]["horse_name"],
            "honmei_prob":    round(honmei_prob, 4),
            "taikou_prob":    round(taikou_prob, 4),
            "gap":            round(gap, 4),
            "honmei_ev":      round(honmei_ev, 3) if not np.isnan(honmei_ev) else None,
            "honmei_odds":    honmei_odds_a,
            "honmei_actual_pos": honmei_pos,
            "honmei_win":     honmei_win,
            "honmei_place":   honmei_place,
            "honmei_return":  honmei_ret,
        })

        if (i + 1) % 200 == 0:
            done = len(results)
            buys = [r for r in results if r["is_buy"]]
            wins = sum(r["honmei_win"] for r in buys)
            roi  = (sum(r["honmei_return"] for r in buys) / (len(buys) * 100) * 100) if buys else 0
            logger.info(f"  {i+1}/{len(target_ids)} | 買い: {len(buys)}R | "
                        f"的中: {wins}R ({wins/len(buys):.1%} if buys else '-') | "
                        f"ROI: {roi:.1f}%")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Step 3: サマリー表示
# ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    n_all  = len(df)
    bought = df[df["is_buy"] == True]
    n_buy  = len(bought)
    border = "=" * 62

    print(f"\n{border}")
    print("  29特徴量モデル バックテスト結果")
    print(f"  テスト期間  : {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  総レース数  : {n_all:,} R")
    print(f"  買い対象    : {n_buy:,} R  （見送り: {n_all - n_buy:,} R）")
    print(border)

    if n_buy == 0:
        print("  ⚠️  買い対象レースなし")
        print(border)
        return

    hits   = bought["honmei_win"].sum()
    places = bought["honmei_place"].sum()
    total_bet = n_buy * 100
    total_ret = bought["honmei_return"].sum()
    roi   = total_ret / total_bet * 100

    print(f"  {'単勝的中率（買い対象）':<20}  {hits/n_buy:>8.2%}  ({hits}/{n_buy})")
    print(f"  {'複勝的中率（買い対象）':<20}  {places/n_buy:>8.2%}  ({places}/{n_buy})")
    print(f"  {'単勝ROI（買い対象）':<21}  {roi:>8.1f}%  (¥{int(total_ret):,} / ¥{total_bet:,})")
    avg_hit_odds = bought.loc[bought["honmei_win"]==1, "honmei_odds"].mean()
    print(f"  {'的中時平均オッズ':<23}  {avg_hit_odds:>8.1f} 倍")

    # マーク別
    print(f"\n  [マーク別 内訳]")
    for mark in ["◎", "○"]:
        mg = bought[bought["mark"] == mark]
        if len(mg) == 0:
            continue
        mh  = mg["honmei_win"].sum()
        mroi = mg["honmei_return"].sum() / (len(mg) * 100) * 100
        print(f"    {mark}  {len(mg):>4}R  的中率 {mh/len(mg):.2%}  ROI {mroi:.1f}%")

    # 月別推移
    print(f"\n  [月別 的中率・ROI]")
    df2 = bought.copy()
    df2["ym"] = df2["race_date"].str[:7]
    for ym, grp in df2.groupby("ym"):
        h = grp["honmei_win"].sum()
        r = grp["honmei_return"].sum() / (len(grp) * 100) * 100
        print(f"    {ym}  {len(grp):>3}R  的中率 {h/len(grp):.2%}  ROI {r:.1f}%")

    # コース別
    print(f"\n  [コース別 的中率]")
    for ct, grp in bought.groupby("course_type"):
        h = grp["honmei_win"].sum()
        print(f"    {ct}  {h/len(grp):.2%}  ({h}/{len(grp)})")

    # 距離帯別
    print(f"\n  [距離帯別 的中率]")
    bought2 = bought.copy()
    bought2["dist_bin"] = pd.cut(
        pd.to_numeric(bought2["distance"], errors="coerce"),
        bins=[0,1400,1800,2200,9999], labels=["短距離","マイル","中距離","長距離"]
    )
    for db, grp in bought2.groupby("dist_bin", observed=True):
        h = grp["honmei_win"].sum()
        print(f"    {db}  {h/len(grp):.2%}  ({h}/{len(grp)})")

    # 全レース（閾値フィルタなし）参考値
    n_all_ok = len(df[~df["is_maiden"]])
    h_all = df[~df["is_maiden"]]["honmei_win"].sum()
    print(f"\n  [参考] 全レース（新馬除外・フィルタなし）: {h_all/n_all_ok:.2%} ({h_all}/{n_all_ok})")

    print(f"{border}\n")


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collect", action="store_true",
                        help="テストデータ再収集をスキップ（既存 test_results_new.csv を使用）")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 62)
    logger.info("29特徴量モデル バックテスト開始")
    logger.info(f"  テスト期間: {TEST_START} 〜 {TEST_END}")
    logger.info("=" * 62)

    # Step 1: テストデータ収集
    if args.skip_collect and TEST_RESULTS_CSV.exists():
        logger.info("[1/3] テストデータ: 既存ファイルを使用")
        test_results = pd.read_csv(TEST_RESULTS_CSV, dtype=str)
        test_meta    = pd.read_csv(TEST_META_CSV,    dtype=str)
        logger.info(f"  {len(test_results):,} 行, {test_meta['race_id'].nunique():,} レース")
    else:
        logger.info("[1/3] テストデータ収集（新スクレイパー）")
        test_results, test_meta = collect_test_data()

    # Step 2: 学習データ読み込み（ルックバック用）
    logger.info("[2/3] 学習データ読み込み（ルックバック用）")
    train_results = pd.read_csv("data/raw/train_results.csv", dtype=str)
    logger.info(f"  train_results: {len(train_results):,} 行")

    # Step 3: 本番モデル読み込み
    trainer = ModelTrainer.load(settings.model_path)
    logger.info(f"  モデル特徴量数: {trainer.model.num_feature()}")

    # Step 4: バックテスト評価
    logger.info("[3/3] バックテスト評価")
    result_df = evaluate(test_results, test_meta, train_results, trainer)

    # 保存
    BACKTEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(BACKTEST_OUT, index=False)
    logger.info(f"結果保存: {BACKTEST_OUT}")

    # サマリー表示
    print_summary(result_df)

    elapsed = (time.time() - t0) / 60
    logger.info(f"総所要時間: {elapsed:.1f} 分")


if __name__ == "__main__":
    main()
