"""
train_ltr_lite.py
=================
【軽量テスト版】odds_log / popularity_rank_norm を除外した LTR モデル学習。

目的:
  フルデータ (4-5h) 再学習前に、以下の2点を迅速に検証する:
  1. odds 除外後の特徴量重要度 (何が代わりに効くか)
  2. OOS データに対する EV 分布 (EV > 1.0 が出現するか)

設計:
  - 学習データ: 2022-2023 年のみ (特徴エンジニアリング ~18分)
  - 特徴量: FEATURE_COLUMNS から odds_log / popularity_rank_norm を除外
  - CV: 3-fold (5-fold より高速)
  - num_boost_round: 500 (1000 より高速)
  - 出力: data/models/lgbm_ltr_lite.pkl

実行:
    .venv/bin/python train_ltr_lite.py
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

# ── ロガー設定 ──────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/train_ltr_lite.log", level="DEBUG", format=_fmt, rotation="10 MB")

# ── 除外する特徴量 ──────────────────────────────────────────────────
EXCLUDE_FEATURES = {"odds_log", "popularity_rank_norm"}

# ── 学習年 ─────────────────────────────────────────────────────────
TRAIN_YEARS = {"2022", "2023"}   # feature engineering が 18-20分に収まる範囲

LITE_MODEL_PATH = Path("data/models/lgbm_ltr_lite.pkl")


def main() -> None:
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("LTR Lite モデル学習 (odds除外テスト)")
    logger.info(f"  学習年    : {sorted(TRAIN_YEARS)}")
    logger.info(f"  除外特徴量: {sorted(EXCLUDE_FEATURES)}")
    logger.info(f"  出力      : {LITE_MODEL_PATH}")
    logger.info("=" * 60)

    # ── STEP 1: データ読み込み ──────────────────────────────────────
    logger.info("[STEP 1] データ読み込み")
    res_df  = pd.read_csv("data/raw/train_results.csv", dtype=str)
    meta_df = pd.read_csv("data/raw/train_meta.csv",    dtype=str)

    # 学習対象年のメタのみ (feature engineering 時間を制御)
    meta_lite = meta_df[meta_df["race_id"].str[:4].isin(TRAIN_YEARS)].copy()
    logger.info(f"  results : {len(res_df):,}行 (全期間、ルックバック用)")
    logger.info(f"  meta    : {len(meta_lite):,}レース ({sorted(TRAIN_YEARS)}年のみ)")

    # ── STEP 2: 特徴量生成 ─────────────────────────────────────────
    logger.info("[STEP 2] 特徴量生成 (per-race ルックバック)")
    fe = FeatureEngineer(res_df)
    fe.precompute_aggregations()

    training_df = fe.build_training_dataset(meta_lite)
    if training_df.empty:
        logger.error("特徴量データセットが空です")
        sys.exit(1)

    elapsed1 = (time.time() - t0) / 60
    logger.info(f"[STEP 2] 完了: {len(training_df):,}行  {elapsed1:.1f}分")
    logger.info(f"  レース数: {training_df['race_id'].nunique():,}")

    # ── STEP 3: ランクラベル付与 ────────────────────────────────────
    logger.info("[STEP 3] ランクラベル付与")
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
    logger.info(f"  学習行数: {len(training_df):,}")

    # ── STEP 4: LTR 学習 (odds 除外 + 高速設定) ────────────────────
    logger.info("[STEP 4] LTR 学習 (odds除外, 3-fold, 500rounds)")

    trainer = LTRTrainer()

    # odds 特徴量を除外
    original_cols = len(trainer.feature_columns)
    trainer.feature_columns = [
        c for c in trainer.feature_columns if c not in EXCLUDE_FEATURES
    ]
    removed = original_cols - len(trainer.feature_columns)
    logger.info(f"  特徴量: {original_cols} → {len(trainer.feature_columns)} ({removed}個除外: {sorted(EXCLUDE_FEATURES)})")

    # 高速化設定（テスト用）
    trainer.N_CV_FOLDS       = 3
    trainer.NUM_BOOST_ROUND  = 500

    trainer.fit(training_df, group_col="race_id")

    elapsed2 = (time.time() - t0) / 60
    logger.info(f"[STEP 4] 完了: {elapsed2:.1f}分  OOF NDCG@3={trainer.oof_ndcg3:.4f}")

    # ── STEP 5: 保存 ───────────────────────────────────────────────
    # ltr_feature_importance.json の上書きを避けるため、lite 専用パスに保存
    LITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        "model":           trainer.model,
        "feature_columns": trainer.feature_columns,
        "oof_ndcg3":       trainer.oof_ndcg3,
    }, LITE_MODEL_PATH)
    logger.info(f"Lite モデル保存: {LITE_MODEL_PATH}  (NDCG@3={trainer.oof_ndcg3:.4f})")

    # lite 専用の特徴量重要度 JSON
    lite_imp_path = LITE_MODEL_PATH.parent / "ltr_lite_feature_importance.json"
    trainer.save_feature_importance(lite_imp_path)

    # ── STEP 6: 特徴量重要度レポート (モデルから直接) ──────────────
    gain  = trainer.model.feature_importance("gain")
    names = trainer.model.feature_name()
    total_gain = gain.sum() or 1.0
    pairs = sorted(zip(names, gain), key=lambda x: -x[1])[:10]

    logger.info("")
    logger.info("=" * 55)
    logger.info("【特徴量重要度 Top 10 (odds除外モデル)】")
    logger.info("=" * 55)
    for rank, (feat, g) in enumerate(pairs, 1):
        pct = g / total_gain * 100
        bar = "#" * int(pct / 1.5)
        logger.info(f"  {rank:2d}. {feat:<35} {pct:5.1f}%  {bar}")

    total = (time.time() - t0) / 60
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Lite モデル学習完了: {total:.1f}分")
    logger.info(f"  NDCG@3  : {trainer.oof_ndcg3:.4f}")
    logger.info(f"  モデル  : {LITE_MODEL_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
