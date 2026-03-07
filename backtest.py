"""
バックテスト: 学習データ(2024年) → テストデータ(2025〜2026年)

評価指標:
- 単勝的中率  : 予測1位（本命）が実際に1着になった割合
- 複勝的中率  : 予測1位が実際に3着以内に入った割合
- 3連複的中率 : 予測上位3頭が実際の3着以内3頭に完全合致した割合
- 単勝回収率  : ¥100 ずつ本命に単勝馬券を購入したときの回収率
- 対抗回収率  : ¥100 ずつ対抗に単勝馬券を購入したときの回収率
"""

import sys
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from src.scraper.netkeiba_scraper import NetkeibaScraper
from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer


# ============================================================
# 設定
# ============================================================
TRAIN_START = date(2024, 1, 1)
TRAIN_END   = date(2024, 12, 31)
TEST_START  = date(2025, 1, 1)
TEST_END    = date(2026, 3, 7)   # 直近まで

JYO_CODES = ["05", "06", "08", "09"]   # 東京・中山・京都・阪神

DATA_DIR  = Path("data/raw")
MODEL_DIR = Path("data/models")
LOG_DIR   = Path("logs")

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ロガー設定
# ============================================================
def setup_logger():
    logger.remove()
    fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
    logger.add(sys.stdout, level="INFO", format=fmt, colorize=True)
    logger.add(str(LOG_DIR / "backtest.log"), level="INFO", format=fmt,
               rotation="20 MB", retention=3)

setup_logger()


# ============================================================
# Phase 1: データ収集
# ============================================================

def collect_phase(label: str, start: date, end: date, ids_file: str, cp: str):
    """race_id 収集 → result+meta 取得 を返す。チェックポイントから再開可。"""
    ids_path = DATA_DIR / ids_file
    result_path = DATA_DIR / (cp + "_results.csv")
    meta_path   = DATA_DIR / (cp + "_meta.csv")
    cp_full     = str(DATA_DIR / cp)

    with NetkeibaScraper() as scraper:
        # race_id 収集（既にある場合はスキップ）
        if ids_path.exists():
            race_ids = pd.read_csv(ids_path, dtype=str)["race_id"].tolist()
            logger.info(f"[{label}] race_id 既存読み込み: {len(race_ids):,} races")
        else:
            logger.info(f"[{label}] race_id 収集開始...")
            race_ids = scraper.collect_race_ids_for_period(
                start_date=start, end_date=end,
                jyo_codes=JYO_CODES, save_path=str(ids_path)
            )

        # result+meta 取得（チェックポイントから再開）
        if result_path.exists() and meta_path.exists():
            done = pd.read_csv(result_path, dtype={"race_id": str})["race_id"].nunique()
            total = len(race_ids)
            if done >= total * 0.98:  # 98%以上完了なら再利用
                logger.info(f"[{label}] 既存データ再利用 ({done}/{total} races)")
                history_df = pd.read_csv(result_path, dtype={"race_id": str})
                meta_df    = pd.read_csv(meta_path,   dtype={"race_id": str})
                return history_df, meta_df

        logger.info(f"[{label}] result+meta 取得: {len(race_ids):,} races (推定 {len(race_ids)*3//60}分)")
        history_df, meta_df = scraper.fetch_bulk_results_and_meta(
            race_ids, checkpoint_path=cp_full
        )

    logger.info(f"[{label}] 完了: {len(history_df):,}行 / {len(meta_df):,}レース")
    return history_df, meta_df


# ============================================================
# Phase 2: オッズを float に変換するヘルパー
# ============================================================

def parse_odds(val) -> float:
    """'3.5' や '---' などを float に変換"""
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


# ============================================================
# Phase 3: バックテスト評価
# ============================================================

def evaluate_backtest(
    test_history_df: pd.DataFrame,
    test_meta_df: pd.DataFrame,
    full_history_df: pd.DataFrame,   # train+test 全データ（特徴量計算のルックバック用）
    trainer: ModelTrainer,
) -> pd.DataFrame:
    """
    各レースを時系列順に処理し、予測 → 実際の結果を照合する。

    Returns
    -------
    pd.DataFrame
        1行 = 1レース の予測結果テーブル
    """
    from src.features.engineer import FeatureEngineer

    # 日付でソート（chronological order 保証）
    if "race_date" in test_meta_df.columns:
        test_meta_df["race_date"] = pd.to_datetime(test_meta_df["race_date"], errors="coerce")
        test_meta_df = test_meta_df.sort_values("race_date")

    h_full = FeatureEngineer(full_history_df)
    h_full._preprocess_history = h_full._preprocess_history  # 既に前処理済みの df を差し替え
    h_full.history = h_full._preprocess_history(full_history_df)

    results = []
    target_ids = test_meta_df["race_id"].tolist()

    logger.info(f"バックテスト対象: {len(target_ids):,} races")

    for i, race_id in enumerate(target_ids):
        race_entries = test_history_df[test_history_df["race_id"] == race_id].copy()
        if race_entries.empty:
            continue

        meta_row = test_meta_df[test_meta_df["race_id"] == race_id].iloc[0]
        race_date = meta_row.get("race_date", None)
        course_type = str(meta_row.get("course_type", ""))
        distance = int(meta_row.get("distance", 0) or 0)
        gc_code = int(meta_row.get("ground_condition_code", -1) or -1)
        wx_code = int(meta_row.get("weather_code", -1) or -1)

        if not course_type or distance == 0:
            continue

        # このレースより前のデータのみをルックバックに使う
        if pd.notna(race_date):
            history_before = h_full.history[h_full.history.get("race_date", pd.NaT) < race_date] \
                if "race_date" in h_full.history.columns else \
                h_full.history[h_full.history["race_id"] < race_id]
        else:
            history_before = h_full.history[h_full.history["race_id"] < race_id]

        # 出走表 entry_df を組み立て
        entry_df = race_entries[[
            "horse_id", "horse_name", "horse_number", "frame_number",
            "sex_age", "weight_carried", "jockey_id",
        ]].copy()
        entry_df["sex"] = entry_df["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entry_df["sex_age"].str[1:], errors="coerce")
        entry_df["weight_carried"] = pd.to_numeric(entry_df["weight_carried"], errors="coerce")
        entry_df["father"] = ""
        entry_df["mother_father"] = ""

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
            )
        except Exception as e:
            logger.debug(f"skip {race_id}: {e}")
            continue

        # 予測確率
        X = feat_df[FeatureEngineer.FEATURE_COLUMNS].fillna(0)
        probs = trainer.model.predict(X)

        pred_df = feat_df[["horse_id", "horse_name", "horse_number"]].copy()
        pred_df["win_prob"] = probs
        pred_df = pred_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

        honmei_id  = pred_df.iloc[0]["horse_id"]
        taikou_id  = pred_df.iloc[1]["horse_id"] if len(pred_df) > 1 else ""
        top3_ids   = set(pred_df.head(3)["horse_id"].tolist())

        # 実際の着順
        actual = race_entries[["horse_id", "finish_position", "odds"]].copy()
        actual["finish_pos_num"] = pd.to_numeric(
            actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        actual["odds_f"] = actual["odds"].apply(parse_odds)
        actual_winner = actual[actual["finish_pos_num"] == 1]
        actual_top3_ids = set(
            actual[actual["finish_pos_num"].between(1, 3)]["horse_id"].tolist()
        )

        # 本命の実際の成績
        honmei_actual = actual[actual["horse_id"] == honmei_id]
        honmei_pos = honmei_actual["finish_pos_num"].iloc[0] if not honmei_actual.empty else np.nan
        honmei_odds = honmei_actual["odds_f"].iloc[0] if not honmei_actual.empty else np.nan

        # 対抗の実際の成績
        taikou_actual = actual[actual["horse_id"] == taikou_id] if taikou_id else pd.DataFrame()
        taikou_pos    = taikou_actual["finish_pos_num"].iloc[0] if not taikou_actual.empty else np.nan
        taikou_odds   = taikou_actual["odds_f"].iloc[0] if not taikou_actual.empty else np.nan

        # 実際の1着馬オッズ（単勝配当計算用）
        winner_odds = actual_winner["odds_f"].iloc[0] if not actual_winner.empty else np.nan

        results.append({
            "race_id":          race_id,
            "race_date":        str(race_date)[:10] if pd.notna(race_date) else "",
            "race_name":        meta_row.get("race_name", ""),
            "course_type":      course_type,
            "distance":         distance,
            # 本命
            "honmei_name":      pred_df.iloc[0]["horse_name"],
            "honmei_prob":      round(float(probs[0]), 4),
            "honmei_actual_pos": honmei_pos,
            "honmei_odds":      honmei_odds,
            "honmei_win":       int(honmei_pos == 1) if not np.isnan(honmei_pos) else 0,
            "honmei_place":     int(honmei_pos <= 3) if not np.isnan(honmei_pos) else 0,
            # 単勝払戻（¥100 ベット）
            "honmei_return":    honmei_odds * 100 if honmei_pos == 1 and not np.isnan(honmei_odds) else 0.0,
            # 対抗
            "taikou_name":      pred_df.iloc[1]["horse_name"] if len(pred_df) > 1 else "",
            "taikou_actual_pos": taikou_pos,
            "taikou_odds":      taikou_odds,
            "taikou_win":       int(taikou_pos == 1) if not np.isnan(taikou_pos) else 0,
            "taikou_return":    taikou_odds * 100 if taikou_pos == 1 and not np.isnan(taikou_odds) else 0.0,
            # 3連複
            "sanrenfuku_hit":   int(top3_ids == actual_top3_ids),
            # 実際の1着
            "actual_winner":    actual_winner["horse_id"].iloc[0] if not actual_winner.empty else "",
            "actual_winner_odds": winner_odds,
        })

        if (i + 1) % 100 == 0:
            done = len(results)
            wins = sum(r["honmei_win"] for r in results)
            logger.info(f"  {i+1}/{len(target_ids)} | 本命的中: {wins}/{done} ({wins/done:.1%})")

    return pd.DataFrame(results)


# ============================================================
# Phase 4: サマリー表示
# ============================================================

def print_summary(result_df: pd.DataFrame):
    n = len(result_df)
    if n == 0:
        logger.error("バックテスト結果が空です。")
        return

    honmei_wins    = result_df["honmei_win"].sum()
    honmei_places  = result_df["honmei_place"].sum()
    sanrenfuku     = result_df["sanrenfuku_hit"].sum()
    taikou_wins    = result_df["taikou_win"].sum()

    # 回収率
    total_bet_h    = n * 100
    total_ret_h    = result_df["honmei_return"].sum()
    recovery_h     = total_ret_h / total_bet_h * 100

    total_bet_t    = n * 100
    total_ret_t    = result_df["taikou_return"].sum()
    recovery_t     = total_ret_t / total_bet_t * 100

    # 平均オッズ
    avg_odds       = result_df["actual_winner_odds"].mean()
    honmei_avg_odds= result_df.loc[result_df["honmei_win"]==1, "honmei_odds"].mean()

    border = "=" * 58
    print(f"\n{border}")
    print("  バックテスト結果サマリー")
    print(f"  テスト期間  : {result_df['race_date'].min()} 〜 {result_df['race_date'].max()}")
    print(f"  対象レース数: {n:,} レース")
    print(border)
    print(f"  {'指標':<20}  {'値':>10}  {'詳細'}")
    print("-" * 58)
    print(f"  {'単勝的中率（本命）':<18}  {honmei_wins/n:>9.2%}  ({honmei_wins}/{n})")
    print(f"  {'複勝的中率（本命）':<18}  {honmei_places/n:>9.2%}  ({honmei_places}/{n})")
    print(f"  {'単勝的中率（対抗）':<18}  {taikou_wins/n:>9.2%}  ({taikou_wins}/{n})")
    print(f"  {'3連複的中率':<20}  {sanrenfuku/n:>9.2%}  ({sanrenfuku}/{n})")
    print("-" * 58)
    print(f"  {'単勝回収率（本命）':<18}  {recovery_h:>9.1f}%  (¥{int(total_ret_h):,} / ¥{total_bet_h:,})")
    print(f"  {'単勝回収率（対抗）':<18}  {recovery_t:>9.1f}%  (¥{int(total_ret_t):,} / ¥{total_bet_t:,})")
    print("-" * 58)
    print(f"  {'平均単勝オッズ（全馬）':<17}  {avg_odds:>9.1f}倍")
    print(f"  {'平均単勝オッズ（本命的中時）':<14}  {honmei_avg_odds:>9.1f}倍")
    print(border)

    # コース別内訳
    print("\n  [コース別 単勝的中率]")
    for ct, grp in result_df.groupby("course_type"):
        hits = grp["honmei_win"].sum()
        print(f"    {ct}  {hits/len(grp):.2%}  ({hits}/{len(grp)})")

    # 距離帯別内訳
    print("\n  [距離帯別 単勝的中率]")
    result_df["dist_bin"] = pd.cut(result_df["distance"],
        bins=[0,1400,1800,2200,9999], labels=["短距離","マイル","中距離","長距離"])
    for db, grp in result_df.groupby("dist_bin", observed=True):
        hits = grp["honmei_win"].sum()
        print(f"    {db}  {hits/len(grp):.2%}  ({hits}/{len(grp)})")

    print()


# ============================================================
# メイン
# ============================================================

def main():
    t_start = time.time()
    logger.info("=" * 58)
    logger.info("バックテスト開始")
    logger.info(f"  学習: {TRAIN_START} → {TRAIN_END}")
    logger.info(f"  テスト: {TEST_START} → {TEST_END}")
    logger.info("=" * 58)

    # ---------- Phase 1: 学習データ収集 ----------
    logger.info("[1/5] 学習データ収集 (2024年)")
    train_history, train_meta = collect_phase(
        "TRAIN", TRAIN_START, TRAIN_END,
        ids_file="train_race_ids.csv", cp="train"
    )

    # ---------- Phase 2: テストデータ収集 ----------
    logger.info("[2/5] テストデータ収集 (2025〜2026年)")
    test_history, test_meta = collect_phase(
        "TEST", TEST_START, TEST_END,
        ids_file="test_race_ids.csv", cp="test"
    )

    # ---------- Phase 3: 学習 ----------
    logger.info("[3/5] 特徴量エンジニアリング & 学習")
    fe_train = FeatureEngineer(train_history)
    train_df = fe_train.build_training_dataset(
        train_meta,
        output_path="data/processed/train_2024.csv"
    )

    logger.info(f"  学習データ: {len(train_df):,}行 / 勝率={train_df['is_win'].mean():.2%}")

    trainer = ModelTrainer()
    trainer.fit(train_df)
    trainer.save(MODEL_DIR / "lgbm_backtest.pkl")
    logger.info("  モデル保存完了")

    # ---------- Phase 4: バックテスト ----------
    logger.info("[4/5] バックテスト実行 (2025〜2026年)")

    # 全データを結合（ルックバック用）
    full_history = pd.concat([train_history, test_history], ignore_index=True)

    result_df = evaluate_backtest(
        test_history_df=test_history,
        test_meta_df=test_meta,
        full_history_df=full_history,
        trainer=trainer,
    )

    result_df.to_csv("data/processed/backtest_results.csv", index=False)
    logger.info(f"  バックテスト結果保存: data/processed/backtest_results.csv")

    # ---------- Phase 5: サマリー ----------
    logger.info("[5/5] 結果表示")
    elapsed = (time.time() - t_start) / 3600
    logger.info(f"  総実行時間: {elapsed:.1f}時間")
    print_summary(result_df)


if __name__ == "__main__":
    main()
