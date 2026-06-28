"""
retrain_pipeline.py
===================
最新データ（2022-2026）を使ったLTRモデル再学習＋feature_stats再構築の一括パイプライン。

【実行内容】
  STEP 1: train_checkpoint_new + test_new を結合・重複除去
           → data/raw/all_results_2022_2026.csv
           → data/raw/all_meta_2022_2026.csv
  STEP 2: LTRTrainer で LambdaRank モデルを再学習
           → data/models/lgbm_ltr_model.pkl  (上書き。旧版は .bak で保存)
  STEP 3: FeatureEngineer で feature_stats を再構築
           → data/models/feature_stats.pkl   (上書き。旧版は .bak で保存)

実行例:
    nohup .venv/bin/python retrain_pipeline.py > logs/retrain.log 2>&1 &
    .venv/bin/python retrain_pipeline.py --dry-run   # 手順確認のみ
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import LTRTrainer
from config.settings import settings

# ── ロガー ───────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout,           level="INFO",  format=_fmt, colorize=True)
logger.add("logs/retrain.log",   level="DEBUG", format=_fmt, rotation="30 MB")

# ── パス定義 ─────────────────────────────────────────────────────────
TRAIN_RESULTS = Path("data/raw/train_checkpoint_new_results.csv")
TRAIN_META    = Path("data/raw/train_checkpoint_new_meta.csv")
TEST_RESULTS  = Path("data/raw/test_results_new.csv")
TEST_META     = Path("data/raw/test_meta_new.csv")

ALL_RESULTS   = Path("data/raw/all_results_2022_2026.csv")
ALL_META      = Path("data/raw/all_meta_2022_2026.csv")

LTR_MODEL        = Path("data/models/lgbm_ltr_model.pkl")
GATEKEEPER_MODEL = Path("data/models/gatekeeper_model.pkl")
STATS_PATH       = settings.stats_path   # data/models/feature_stats.pkl


def _backup(path: Path) -> None:
    """既存ファイルを .bak にバックアップする。"""
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy(path, bak)
        logger.info(f"  バックアップ: {bak}")


def step1_merge_data(dry_run: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    STEP 1: train + test データを結合・重複除去して保存する。

    Returns
    -------
    (res_df, meta_df) : 結合済みデータフレーム
    """
    logger.info("=" * 60)
    logger.info("STEP 1: データ結合（2022-2026 全期間）")
    logger.info("=" * 60)

    # ── results ──────────────────────────────────────────────────────
    logger.info(f"  [{TRAIN_RESULTS.name}] 読み込み中…")
    train_res = pd.read_csv(TRAIN_RESULTS, dtype=str)
    logger.info(f"  [{TEST_RESULTS.name}]  読み込み中…")
    test_res  = pd.read_csv(TEST_RESULTS,  dtype=str)

    res_df = (
        pd.concat([train_res, test_res], ignore_index=True)
        .drop_duplicates(subset=["race_id", "horse_id"], keep="last")
    )
    res_df = res_df.sort_values(["race_id", "horse_id"]).reset_index(drop=True)

    # ── meta ─────────────────────────────────────────────────────────
    logger.info(f"  [{TRAIN_META.name}] 読み込み中…")
    train_meta = pd.read_csv(TRAIN_META, dtype=str)
    logger.info(f"  [{TEST_META.name}]  読み込み中…")
    test_meta  = pd.read_csv(TEST_META,  dtype=str)

    meta_df = (
        pd.concat([train_meta, test_meta], ignore_index=True)
        .drop_duplicates(subset=["race_id"], keep="last")
    )
    meta_df["race_date"] = pd.to_datetime(meta_df["race_date"], errors="coerce")
    meta_df = meta_df.sort_values("race_date").reset_index(drop=True)

    # ── サマリー出力 ──────────────────────────────────────────────────
    year_counts = meta_df.groupby(meta_df["race_id"].str[:4])["race_id"].count()
    logger.info(f"  結合後: {len(res_df):,} 行  {meta_df['race_id'].nunique():,} レース")
    for yr, cnt in year_counts.items():
        logger.info(f"    {yr}年: {cnt:,}R")
    logger.info(
        f"  期間: {meta_df['race_date'].min().date()} 〜 {meta_df['race_date'].max().date()}"
    )

    if not dry_run:
        ALL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(ALL_RESULTS,  index=False)
        meta_df.to_csv(ALL_META,    index=False)
        logger.info(f"  保存: {ALL_RESULTS}")
        logger.info(f"  保存: {ALL_META}")
    else:
        logger.info("  [dry-run] CSV 保存スキップ")

    return res_df, meta_df


def build_training_df(res_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    特徴量データセット生成（lookback方式）+ 着順ラベル付与。
    LTRTrainer / GatekeeperTrainer の両方で共有する（Two-Brain: 同じ特徴量、別モデル）。
    """
    t0 = time.time()

    # ── 特徴量生成 ──────────────────────────────────────────────────
    logger.info("[特徴量生成] FeatureEngineer 初期化・集計計算")
    fe = FeatureEngineer(res_df)
    fe.precompute_aggregations()
    logger.info(f"  集計完了: {(time.time()-t0)/60:.1f}分")

    logger.info("[特徴量生成] 学習用特徴量データセット生成（lookback 方式）")
    training_df = fe.build_training_dataset(meta_df)
    if training_df.empty:
        logger.error("特徴量データセットが空です。データを確認してください。")
        sys.exit(1)
    logger.info(f"  生成完了: {len(training_df):,} 行  {training_df['race_id'].nunique():,}R  {(time.time()-t0)/60:.1f}分")

    # ── 着順ラベル付与 ────────────────────────────────────────────
    logger.info("[特徴量生成] 着順付与（finish_pos_num）")
    fp_df = res_df[["race_id", "horse_id", "finish_position"]].copy()
    fp_df["finish_pos_num"] = pd.to_numeric(
        fp_df["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    training_df = training_df.merge(
        fp_df[["race_id", "horse_id", "finish_pos_num"]],
        on=["race_id", "horse_id"], how="left",
    )
    training_df["finish_pos_num"] = training_df["finish_pos_num"].fillna(99)
    training_df = training_df[training_df["finish_pos_num"] < 99].copy()
    labels = LTRTrainer.make_rank_labels(training_df)
    logger.info(
        f"  ランクラベル(1着=3,2着=2,3着=1,他=0): 0={( labels==0).sum():,}  "
        f"1={(labels==1).sum():,}  2={(labels==2).sum():,}  3={(labels==3).sum():,}"
    )
    return training_df


def step2_train_ltr(training_df: pd.DataFrame, dry_run: bool = False) -> None:
    """
    STEP 2: LTR モデルの再学習。

    【Two-Brain方針（2026-06-21承認）】
    LTRは「市場情報を見ない純粋な能力評価器」として維持する。
    odds_log / popularity_rank_norm は除外したまま変更しない
    （Gatekeeper側に市場情報の活用を集約する）。
    """
    logger.info("=" * 60)
    logger.info("STEP 2: LTR モデル再学習（LambdaRank / NDCG@3）")
    logger.info("=" * 60)
    t0 = time.time()

    if dry_run:
        logger.info("  [dry-run] モデル学習スキップ")
        return

    logger.info("[2-1] LTRTrainer.fit()（5-fold GroupKFold + Temperature Calibration）")
    trainer = LTRTrainer()

    # odds_log / popularity_rank_norm を除外（市場依存を排除し P_model を独立化）
    # ※ Two-Brain設計の核：この除外は維持する。市場情報は Gatekeeper 側で使う。
    EXCLUDE_ODDS = {"odds_log", "popularity_rank_norm"}
    trainer.feature_columns = [c for c in trainer.feature_columns if c not in EXCLUDE_ODDS]
    logger.info(f"  特徴量: {len(trainer.feature_columns)}個（odds除外後）")

    trainer.fit(training_df, group_col="race_id")
    logger.info(f"  学習完了: OOF NDCG@3={trainer.oof_ndcg3:.4f}  τ*={trainer.temperature:.4f}  {(time.time()-t0)/60:.1f}分")

    logger.info("[2-2] モデル保存")
    _backup(LTR_MODEL)
    trainer.save(LTR_MODEL)
    logger.info(f"  保存完了: {LTR_MODEL}")


def step2b_train_gatekeeper(training_df: pd.DataFrame, dry_run: bool = False) -> None:
    """
    STEP 2b: Gatekeeper モデルの学習（Two-Brain System の市場側の脳）。
    LTR と同じ training_df を再利用する（特徴量生成の二重計算を避ける）。
    odds_log / popularity_rank_norm は除外しない（LTRと逆方針）。
    """
    logger.info("=" * 60)
    logger.info("STEP 2b: Gatekeeper モデル学習（is_top3 二値分類・市場情報込み）")
    logger.info("=" * 60)
    t0 = time.time()

    if dry_run:
        logger.info("  [dry-run] モデル学習スキップ")
        return

    from src.model.gatekeeper import GatekeeperTrainer

    gk = GatekeeperTrainer()
    logger.info(f"  特徴量: {len(gk.feature_columns)}個（odds_log/popularity_rank_norm 含む）")
    gk.fit(training_df, group_col="race_id")
    logger.info(
        f"  学習完了: OOF AUC={gk.oof_auc:.4f}  logloss={gk.oof_logloss:.4f}  "
        f"brier={gk.oof_brier:.4f}  {(time.time()-t0)/60:.1f}分"
    )

    logger.info("  モデル保存")
    _backup(GATEKEEPER_MODEL)
    gk.save(GATEKEEPER_MODEL)
    logger.info(f"  保存完了: {GATEKEEPER_MODEL}")


def step3_rebuild_stats(res_df: pd.DataFrame, dry_run: bool = False) -> None:
    """
    STEP 3: feature_stats.pkl の再構築。
    全期間データで騎手・調教師・馬別統計を計算して保存する。
    """
    logger.info("=" * 60)
    logger.info("STEP 3: feature_stats.pkl 再構築")
    logger.info("=" * 60)
    t0 = time.time()

    logger.info("[3-1] FeatureEngineer 初期化・集計計算（全期間）")
    fe = FeatureEngineer(res_df)
    fe.precompute_aggregations()
    logger.info(f"  集計完了: {(time.time()-t0)/60:.1f}分")

    if dry_run:
        logger.info("  [dry-run] stats 保存スキップ")
        return

    logger.info("[3-2] feature_stats.pkl 保存")
    _backup(STATS_PATH)
    # extra_history_df=None → 学習データのみで horse_recent_form を計算（将来データ混入なし）
    fe.save_stats(STATS_PATH)
    logger.info(f"  保存完了: {STATS_PATH}  {(time.time()-t0)/60:.1f}分")


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR/Gatekeeper 再学習パイプライン（2022-2026・Two-Brain System）")
    parser.add_argument("--dry-run", action="store_true",
                        help="データ確認のみ（モデル学習・ファイル保存をスキップ）")
    parser.add_argument("--skip-merge", action="store_true",
                        help="STEP1スキップ（all_results_2022_2026.csv が既存の場合）")
    parser.add_argument("--retrain-ltr", action="store_true",
                        help="LTRモデルも再学習する（既定はスキップ。特徴量に変更が無いため）")
    parser.add_argument("--skip-gatekeeper", action="store_true",
                        help="Gatekeeperモデルの学習をスキップする")
    parser.add_argument("--skip-stats", action="store_true",
                        help="feature_stats.pkl の再構築をスキップする")
    args = parser.parse_args()

    t_global = time.time()

    logger.info("=" * 60)
    logger.info("Two-Brain 再学習パイプライン（LTR=純粋能力評価 / Gatekeeper=市場込みリスク管理）")
    logger.info(f"  dry-run: {args.dry_run}  retrain-ltr: {args.retrain_ltr}  "
                f"skip-gatekeeper: {args.skip_gatekeeper}")
    logger.info("=" * 60)

    # ── STEP 1: データ結合 ──────────────────────────────────────────
    if args.skip_merge and ALL_RESULTS.exists() and ALL_META.exists():
        logger.info("STEP 1: スキップ（既存の結合ファイルを使用）")
        res_df  = pd.read_csv(ALL_RESULTS, dtype=str)
        meta_df = pd.read_csv(ALL_META,    dtype=str)
        meta_df["race_date"] = pd.to_datetime(meta_df["race_date"], errors="coerce")
    else:
        res_df, meta_df = step1_merge_data(dry_run=args.dry_run)

    # ── STEP 2: 特徴量生成（LTR・Gatekeeper共用）────────────────────
    training_df = build_training_df(res_df, meta_df)

    # ── STEP 2-LTR: LTR 再学習（既定スキップ。特徴量変更なしのため不要）──
    if args.retrain_ltr:
        step2_train_ltr(training_df, dry_run=args.dry_run)
    else:
        logger.info("STEP 2-LTR: スキップ（--retrain-ltr 未指定。LTRの特徴量に変更なし）")

    # ── STEP 2b: Gatekeeper 学習 ────────────────────────────────────
    if not args.skip_gatekeeper:
        step2b_train_gatekeeper(training_df, dry_run=args.dry_run)
    else:
        logger.info("STEP 2b: スキップ（--skip-gatekeeper 指定）")

    # ── STEP 3: feature_stats 再構築 ────────────────────────────────
    if not args.skip_stats:
        step3_rebuild_stats(res_df, dry_run=args.dry_run)
    else:
        logger.info("STEP 3: スキップ（--skip-stats 指定）")

    total = (time.time() - t_global) / 60
    logger.info("=" * 60)
    logger.info(f"パイプライン完了: {total:.1f}分")
    if not args.dry_run:
        logger.info(f"  LTR モデル       : {LTR_MODEL}{'（再学習済み）' if args.retrain_ltr else '（変更なし）'}")
        logger.info(f"  Gatekeeper モデル: {GATEKEEPER_MODEL}{'（スキップ）' if args.skip_gatekeeper else '（再学習済み）'}")
        logger.info(f"  feature_stats    : {STATS_PATH}")
        logger.info("  daily_batch.py はそのまま新モデル・新statsで動作します。")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
