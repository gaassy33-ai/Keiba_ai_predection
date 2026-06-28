"""
train_ltr_model.py
==================
LightGBM LambdaRank (Learning-to-Rank) モデルの学習スクリプト。

【現行モデル (train_jra_model.py) との違い】
- 目的関数: binary → lambdarank
- ラベル: is_win(0/1) → rank_score(0/1/2/3)
- 評価指標: logloss/AUC → NDCG@3
- 確率変換: Platt scaling → softmax (Plackett-Luce)
- bad_season モデル: なし（単一モデルで全月カバー）

【出力】
- data/models/lgbm_ltr_model.pkl  (LTRTrainer オブジェクト)
- data/models/ltr_feature_importance.json

実行例:
    .venv/bin/python train_ltr_model.py
    .venv/bin/python train_ltr_model.py --results-csv data/raw/train_results.csv
    nohup .venv/bin/python train_ltr_model.py > logs/train_ltr.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import LTRTrainer
from config.settings import settings

# ── ロガー設定 ──────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/train_ltr.log", level="DEBUG", format=_fmt, rotation="20 MB")

# ── 出力パス ────────────────────────────────────────────────────────
LTR_MODEL_PATH = Path("data/models/lgbm_ltr_model.pkl")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LTR モデル学習")
    parser.add_argument("--results-csv", default="data/raw/train_results.csv")
    parser.add_argument("--meta-csv",    default="data/raw/train_meta.csv")
    parser.add_argument("--model-output", default=str(LTR_MODEL_PATH))
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    meta_path    = Path(args.meta_csv)
    model_path   = Path(args.model_output)

    for p in [results_path, meta_path]:
        if not p.exists():
            logger.error(f"ファイルが見つかりません: {p}")
            sys.exit(1)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("LTR (LambdaRank) モデル学習開始")
    logger.info(f"  results : {results_path}")
    logger.info(f"  meta    : {meta_path}")
    logger.info(f"  output  : {model_path}")
    logger.info("=" * 60)

    # ── STEP 1: データ読み込み ──────────────────────────────────────
    logger.info("[STEP 1] データ読み込み")
    res_df  = pd.read_csv(results_path, dtype=str)
    meta_df = pd.read_csv(meta_path,    dtype=str)
    logger.info(f"  results: {len(res_df):,} 行  meta: {len(meta_df):,} レース")
    logger.info(f"  対象年: {sorted(res_df['race_id'].str[:4].unique())}")

    # ── STEP 2: 特徴量生成（ルックバック方式）──────────────────────
    logger.info("[STEP 2] 特徴量生成（per-race ルックバック方式・時系列リークなし）")
    fe = FeatureEngineer(res_df)
    fe.precompute_aggregations()

    training_df = fe.build_training_dataset(meta_df)
    if training_df.empty:
        logger.error("特徴量データセットが空です")
        sys.exit(1)

    elapsed1 = (time.time() - t0) / 60
    logger.info(f"[STEP 2] 完了: {len(training_df):,} 行  {elapsed1:.1f}分")
    logger.info(f"  レース数: {training_df['race_id'].nunique():,}")

    # ── STEP 3: ランクラベル付与 ────────────────────────────────────
    logger.info("[STEP 3] ランクラベル付与（1着=3, 2着=2, 3着=1, それ以外=0）")

    # finish_pos_num を res_df から結合
    fp_df = res_df[["race_id", "horse_id", "finish_position"]].copy()
    fp_df["finish_pos_num"] = pd.to_numeric(
        fp_df["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    training_df = training_df.merge(
        fp_df[["race_id", "horse_id", "finish_pos_num"]],
        on=["race_id", "horse_id"],
        how="left",
    )

    # ラベル確認
    labels = LTRTrainer.make_rank_labels(training_df)
    n_total = len(labels)
    logger.info(
        f"  ラベル分布: "
        f"0(4着以下)={( labels==0).sum():,}({(labels==0).mean()*100:.1f}%)  "
        f"1(3着)={(labels==1).sum():,}({(labels==1).mean()*100:.1f}%)  "
        f"2(2着)={(labels==2).sum():,}({(labels==2).mean()*100:.1f}%)  "
        f"3(1着)={(labels==3).sum():,}({(labels==3).mean()*100:.1f}%)"
    )
    training_df["finish_pos_num"] = training_df["finish_pos_num"].fillna(99)
    logger.info(f"  欠損着順: {(training_df['finish_pos_num'] == 99).sum():,} 行（0スコア扱い）")

    # finish_pos_num が取れなかった行を除外（信頼性のないデータ）
    before = len(training_df)
    training_df = training_df[training_df["finish_pos_num"] < 99].copy()
    if before != len(training_df):
        logger.warning(f"  着順不明行を除外: {before - len(training_df):,} 行")

    elapsed2 = (time.time() - t0) / 60
    logger.info(f"[STEP 3] 完了: {len(training_df):,} 行  {elapsed2:.1f}分")

    # ── STEP 4: LTR モデル学習 ─────────────────────────────────────
    logger.info("[STEP 4] LTR (LambdaRank) 学習開始（5-fold GroupKFold）")
    logger.info(f"  最適化指標: NDCG@3（top3 に入る馬の順位精度）")
    logger.info(f"  確率変換: softmax (Plackett-Luce モデル)")

    trainer = LTRTrainer()

    # ── odds 特徴量を除外（市場依存を排除し P_model を独立化）──────
    EXCLUDE_ODDS_FEATURES = {"odds_log", "popularity_rank_norm"}
    trainer.feature_columns = [
        c for c in trainer.feature_columns if c not in EXCLUDE_ODDS_FEATURES
    ]
    logger.info(
        f"  odds除外後の特徴量数: {len(trainer.feature_columns)}個 "
        f"(除外: {sorted(EXCLUDE_ODDS_FEATURES)})"
    )

    trainer.fit(training_df, group_col="race_id")

    elapsed3 = (time.time() - t0) / 60
    logger.info(f"[STEP 4] 完了: {elapsed3:.1f}分  OOF NDCG@3={trainer.oof_ndcg3:.4f}")

    # ── STEP 5: モデル保存 ─────────────────────────────────────────
    logger.info("[STEP 5] モデル保存")
    trainer.save(model_path)

    # feature_stats.pkl はバイナリモデルと共用可能（特徴量は同一）
    # train_jra_model.py で生成した feature_stats.pkl をそのまま使用

    total = (time.time() - t0) / 60
    logger.info("=" * 60)
    logger.info(f"LTR モデル学習完了: {total:.1f}分")
    logger.info(f"  モデル  : {model_path}")
    logger.info(f"  NDCG@3  : {trainer.oof_ndcg3:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
