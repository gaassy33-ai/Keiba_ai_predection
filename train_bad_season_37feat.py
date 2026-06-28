"""
不調期専用モデル 再学習スクリプト（34特徴量対応版）

train_all.csv（expand_and_train.py が生成した34特徴量キャッシュ）をベースに
不調期月（1・7・8・9・11・12月）のレースだけで ModelTrainer を再学習し、
lgbm_model_bad_season.pkl を上書き保存する。

特徴量生成を省略できるため高速（数分）で完了する。
※ 2026-04-26: 37 → 34 特徴量に削減（sire_win_rate/bms_win_rate/jockey_venue_win_rate 削除）

実行:
    .venv/bin/python train_bad_season_37feat.py
    nohup .venv/bin/python train_bad_season_37feat.py > logs/train_bad_season_37feat.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model.trainer import ModelTrainer
from src.features.engineer import FeatureEngineer

Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout,  level="INFO", format=_fmt, colorize=True)
logger.add("logs/train_bad_season_37feat.log", level="DEBUG", format=_fmt, rotation="10 MB")

BAD_SEASON_MONTHS   = {1, 7, 8, 9, 11, 12}
BAD_SEASON_MODEL    = ROOT / "data/models/lgbm_model_bad_season.pkl"
TRAIN_ALL_CSV       = ROOT / "data/processed/train_all.csv"
BACKUP_PATH         = ROOT / "data/models/lgbm_model_bad_season_29feat_backup.pkl"

FEATURE_COLUMNS = FeatureEngineer.FEATURE_COLUMNS   # 37列


def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("不調期専用モデル 再学習（37特徴量）")
    logger.info(f"  対象月: {sorted(BAD_SEASON_MONTHS)}")
    logger.info(f"  特徴量: {len(FEATURE_COLUMNS)}個")
    logger.info("=" * 60)

    # ── 1. 特徴量CSVを読み込む ─────────────────────────────────────
    if not TRAIN_ALL_CSV.exists():
        logger.error(f"train_all.csv が見つかりません: {TRAIN_ALL_CSV}")
        logger.error("先に expand_and_train.py を実行してください。")
        sys.exit(1)

    logger.info(f"[1/4] 特徴量CSV読み込み: {TRAIN_ALL_CSV}")
    training_df = pd.read_csv(TRAIN_ALL_CSV)
    logger.info(f"  全データ: {len(training_df):,} 行  {training_df['race_id'].nunique():,} レース")

    # ── 2. meta から race_date → month を付与 ──────────────────────
    logger.info("[2/4] メタデータから月情報を付与")
    meta_frames = []
    for p in ["data/raw/train_meta.csv", "data/raw/test_meta.csv"]:
        mp = ROOT / p
        if mp.exists():
            meta_frames.append(pd.read_csv(mp, dtype=str))
    if not meta_frames:
        logger.error("meta CSV が見つかりません。")
        sys.exit(1)

    all_meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates(subset="race_id")
    all_meta["race_date"] = pd.to_datetime(all_meta["race_date"], errors="coerce")
    all_meta["_month"] = all_meta["race_date"].dt.month
    # race_id を文字列に統一（train_all は int64, meta は str）
    all_meta["race_id"] = all_meta["race_id"].astype(str)
    date_map = all_meta.set_index("race_id")["_month"].to_dict()

    training_df["race_id"] = training_df["race_id"].astype(str)
    training_df["_month"] = training_df["race_id"].map(date_map)
    unmapped = training_df["_month"].isna().sum()
    if unmapped > 0:
        logger.warning(f"  月情報不明: {unmapped} 行（race_id が meta に存在しない）")

    # ── 3. 不調期月フィルタ ────────────────────────────────────────
    logger.info("[3/4] 不調期月でフィルタ")
    bad_df = training_df[training_df["_month"].isin(BAD_SEASON_MONTHS)].copy()
    logger.info(f"  不調期データ: {len(bad_df):,} 行  {bad_df['race_id'].nunique():,} レース")

    months_dist = bad_df["_month"].value_counts().sort_index()
    for m, cnt in months_dist.items():
        n_races = bad_df[bad_df["_month"] == m]["race_id"].nunique()
        logger.info(f"    {int(m):2d}月: {cnt:,} 行  {n_races} レース")

    if bad_df.empty:
        logger.error("不調期データが空です。meta との race_id が一致しているか確認してください。")
        sys.exit(1)

    # 特徴量カラムの存在チェック
    missing_cols = [c for c in FEATURE_COLUMNS if c not in bad_df.columns]
    if missing_cols:
        logger.error(f"特徴量が不足しています: {missing_cols}")
        sys.exit(1)

    logger.info(f"  特徴量確認: {len(FEATURE_COLUMNS)}列 OK")
    logger.info(f"  勝ち馬率: {bad_df['is_win'].mean():.2%}")

    # ── 4. 旧モデルをバックアップ → 新モデル学習・保存 ───────────
    logger.info("[4/4] ModelTrainer 学習")

    if BAD_SEASON_MODEL.exists():
        import shutil
        shutil.copy(BAD_SEASON_MODEL, BACKUP_PATH)
        logger.info(f"  旧モデルをバックアップ: {BACKUP_PATH}")

    trainer = ModelTrainer()
    trainer.fit(bad_df)

    elapsed = (time.time() - t0) / 60
    logger.info(f"  学習完了: {elapsed:.1f}分")

    trainer.save(BAD_SEASON_MODEL, org="jra")
    logger.info(f"  保存完了: {BAD_SEASON_MODEL}")

    # 確認
    loaded = ModelTrainer.load(BAD_SEASON_MODEL)
    nfeat  = loaded.model.num_feature() if loaded.model else "?"
    has_wc = loaded.calibrator is not None
    has_pc = getattr(loaded, "place_calibrator", None) is not None
    logger.info("  ── 保存済みモデル確認 ──")
    logger.info(f"    特徴量数    : {nfeat}")
    logger.info(f"    win_calib   : {'あり' if has_wc else 'なし'}")
    logger.info(f"    place_calib : {'あり' if has_pc else 'なし'}")

    total = (time.time() - t0) / 60
    logger.info("=" * 60)
    logger.info(f"完了: {total:.1f}分")
    logger.info(f"  モデル: {BAD_SEASON_MODEL}")
    logger.info("  次のステップ: backtest_full.py で効果確認")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
