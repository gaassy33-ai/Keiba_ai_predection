"""
NAR 専用モデル学習スクリプト。

data/raw/nar_results.csv / nar_meta.csv から特徴量を高速生成し
LightGBM モデルを学習する。

JRA モデルとの主な違い:
- 訓練データ: NAR 過去1年分（~55K行）→ 地方競馬固有の騎手・馬統計を学習
- ハイパーパラメータ: JRA より小さい num_leaves, 強め正則化（少データ対応）
- 特徴量生成: vectorized rolling window（per-race ループ不要 → 高速）
  ※ 血統情報（father/mother_father）は NAR データに含まれないため NaN 扱い

実行:
    conda run -n lgb311 python train_nar_model.py
    conda run -n lgb311 python train_nar_model.py --no-place-model  # 単勝のみ
"""

from __future__ import annotations

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from loguru import logger
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from config.settings import settings

# ── ロガー設定 ─────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/train_nar_model.log", level="DEBUG", format=_fmt, rotation="20 MB")

# ── NAR 専用ハイパーパラメータ ─────────────────────────────────
# JRA(num_leaves=127, lr=0.02) より小さいモデル（~55K行 vs JRA ~200K行）
LGBM_PARAMS_NAR = {
    "objective":       "binary",
    "metric":          "binary_logloss",
    "learning_rate":   0.03,    # JRA=0.02 より速い収束
    "num_leaves":      63,      # JRA=127 より小さく過学習防止
    "max_depth":       -1,
    "min_child_samples": 10,   # JRA=20 より小さく（少データ対応）
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "lambda_l1":       0.3,     # JRA=0.1 より強め正則化
    "lambda_l2":       0.3,
    "verbose":         -1,
    "n_jobs":          -1,
    "random_state":    42,
}
NUM_BOOST_ROUND     = 1500
EARLY_STOPPING_ROUNDS = 80
RECENT_N = FeatureEngineer.RECENT_N


def _norm_jid(s: pd.Series) -> pd.Series:
    """jockey_id の先頭ゼロを除去して統一。"""
    return s.astype(str).apply(
        lambda x: str(int(x)) if x.strip().isdigit() else x.strip()
    )


def build_features_vectorized(
    res_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureEngineer]:
    """
    nar_results + nar_meta から学習用特徴量を vectorized 方式で生成する。

    JRA の build_training_dataset（per-race ループ）と異なり、
    全レースを一括処理するため大幅に高速。
    time-series lookback は horse_id × race_id ソート上の
    shift(1).rolling(N) により再現する。

    Returns
    -------
    df : pd.DataFrame
        FEATURE_COLUMNS + [is_win, is_placed, race_id, horse_id] を含む DataFrame
    fe : FeatureEngineer
        save_stats() 用（推論時に再利用する統計を保持）
    """
    logger.info(f"  nar_results: {len(res_df):,} rows  nar_meta: {len(meta_df)} races")

    # nar_results.csv にはスクレイパーがインライン付加した
    # course_type / distance / ground_condition_code / weather_code が既に含まれる。
    # meta のみに存在する race_name / race_date 等は必要なら別途マージ。
    # meta から補完する列を特定（results に既存の列は上書きしない）
    meta_extra_cols = [c for c in ["race_id", "course_type", "distance",
                                    "ground_condition_code", "weather_code"]
                       if c in meta_df.columns and c not in res_df.columns]
    if meta_extra_cols:
        res_with_meta = res_df.merge(
            meta_df[meta_extra_cols], on="race_id", how="left"
        )
        logger.info(f"  meta から補完したカラム: {meta_extra_cols}")
    else:
        res_with_meta = res_df.copy()
        logger.info("  meta 結合不要（必要カラムは results に含まれている）")

    # ── FeatureEngineer で前処理 + 集計統計 (jockey/sire) を計算 ───
    fe = FeatureEngineer(res_with_meta)
    fe.precompute_aggregations()
    h = fe.history  # _preprocess_history 済み

    # ── race_id 昇順ソート（time-series rolling lookback の基準）────
    h = h.sort_values("race_id").reset_index(drop=True)

    # ── 直近 N 走 rolling 平均（shift(1) でデータリーク防止）────────
    for hid_grp, grp in [
        ("recent_avg_pos",    "finish_pos_num"),
        ("recent_avg_last3f", "last_3f_num"),
    ]:
        h[hid_grp] = (
            h.groupby("horse_id")[grp]
            .transform(lambda x: x.shift(1).rolling(RECENT_N, min_periods=1).mean())
        )

    logger.info("  recent_avg_pos / recent_avg_last3f 計算完了")

    # ── レース環境特徴量 ──────────────────────────────────────────
    h["course_type"] = h["course_type"].fillna("")
    h["course_type_code"] = (
        h["course_type"].map({"芝": 0, "ダート": 1}).fillna(1).astype(int)
    )
    h["distance"] = pd.to_numeric(h["distance"], errors="coerce")
    h["distance_bin_code"] = (
        pd.cut(h["distance"], bins=[0, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3])
        .astype(float)
    )
    h["ground_condition_code"] = pd.to_numeric(h["ground_condition_code"], errors="coerce")
    h["weather_code"]           = pd.to_numeric(h["weather_code"],           errors="coerce")

    # ── 産駒勝率のマージ ─────────────────────────────────────────
    if fe._sire_win_rate is not None and "father" in h.columns:
        swr = fe._sire_win_rate.reset_index()
        h = h.merge(swr, on="father", how="left")
    else:
        h["sire_win_rate"] = np.nan

    if fe._bms_win_rate is not None and "mother_father" in h.columns:
        bms = fe._bms_win_rate.reset_index()
        h = h.merge(bms, on="mother_father", how="left")
    else:
        h["bms_win_rate"] = np.nan

    # ── ジョッキー適性のマージ ────────────────────────────────────
    if fe._jockey_course_stats is not None:
        h["distance_bin"] = pd.cut(
            h["distance"], bins=[0, 1400, 1800, 2200, 9999],
            labels=["短距離", "マイル", "中距離", "長距離"]
        ).astype(str)
        h["_jid_norm"] = _norm_jid(h["jockey_id"])

        jcs = fe._jockey_course_stats.copy()
        jcs["_jid_norm"] = _norm_jid(jcs["jockey_id"])
        jcs["distance_bin"] = jcs["distance_bin"].astype(str)

        h = h.merge(
            jcs[["_jid_norm", "course_type", "distance_bin",
                 "jockey_win_rate", "jockey_place_rate", "jockey_runs"]],
            on=["_jid_norm", "course_type", "distance_bin"],
            how="left",
        )
        h = h.drop(columns=["_jid_norm", "distance_bin"], errors="ignore")
    else:
        h["jockey_win_rate"]  = np.nan
        h["jockey_place_rate"] = np.nan
        h["jockey_runs"]      = np.nan

    # ── 数値変換 ─────────────────────────────────────────────────
    h["weight_carried"] = pd.to_numeric(h["weight_carried"], errors="coerce")
    h["age"]            = pd.to_numeric(h.get("age"), errors="coerce")
    h["horse_number"]   = pd.to_numeric(h.get("horse_number"), errors="coerce")
    h["frame_number"]   = pd.to_numeric(h.get("frame_number"), errors="coerce")

    # ── course_type / distance が空のレースを除外 ─────────────────
    h = h[h["course_type"].ne("") & h["distance"].notna() & (h["distance"] > 0)]

    logger.info(f"  特徴量生成完了: {len(h):,} 行  is_win={int(h['is_win'].sum())}")
    return h, fe


def _train_booster(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    label: str,
) -> lgb.Booster:
    """GroupKFold CV → 全データ再学習でブースターを返す。"""
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(X))
    best_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx],      y[va_idx]

        tr_set = lgb.Dataset(X_tr, label=y_tr)
        va_set = lgb.Dataset(X_va, label=y_va, reference=tr_set)

        bst = lgb.train(
            LGBM_PARAMS_NAR,
            tr_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[va_set],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        oof[va_idx] = bst.predict(X_va)
        best_iters.append(bst.best_iteration)
        logger.info(
            f"  [{label}] Fold {fold+1} | iter={bst.best_iteration}"
            f"  logloss={log_loss(y_va, oof[va_idx]):.4f}"
            f"  auc={roc_auc_score(y_va, oof[va_idx]):.4f}"
        )

    logger.info(
        f"  [{label}] OOF  logloss={log_loss(y, oof):.4f}"
        f"  auc={roc_auc_score(y, oof):.4f}"
    )

    best_round = max(best_iters)
    full_bst = lgb.train(
        LGBM_PARAMS_NAR,
        lgb.Dataset(X, label=y),
        num_boost_round=best_round,
    )
    logger.info(f"  [{label}] 全データ再学習: {best_round} rounds")
    return full_bst


def _log_feature_importance(bst: lgb.Booster, label: str) -> None:
    features = bst.feature_name()
    gain  = bst.feature_importance("gain")
    split = bst.feature_importance("split")
    total = gain.sum() or 1
    pairs = sorted(zip(features, gain, split), key=lambda x: -x[1])
    logger.info(f"--- 特徴量重要度 [{label}] ---")
    for f, g, s in pairs:
        logger.info(f"  {f:<24} gain={g:>8,} ({g/total*100:>5.1f}%)  split={s:>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NAR 専用モデル学習")
    parser.add_argument("--no-place-model", action="store_true",
                        help="place モデルを学習しない（単勝モデルのみ）")
    parser.add_argument("--output-dir", default=None,
                        help="モデル保存先ディレクトリ（省略時は data/models）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "nar_lgbm_model.pkl"
    stats_path = out_dir / "nar_feature_stats.pkl"

    # ── データ読み込み ────────────────────────────────────────────
    res_path  = ROOT / "data" / "raw" / "nar_results.csv"
    meta_path = ROOT / "data" / "raw" / "nar_meta.csv"

    if not res_path.exists() or not meta_path.exists():
        logger.error("NAR データが見つかりません。先に collect_nar_history.py を実行してください。")
        sys.exit(1)

    res_df  = pd.read_csv(res_path,  dtype=str)
    meta_df = pd.read_csv(meta_path, dtype=str)
    logger.info(f"データ読み込み完了: results={len(res_df):,} rows  meta={len(meta_df)} races")

    # ── 特徴量生成（vectorized）───────────────────────────────────
    logger.info("[STEP 1] 特徴量生成（vectorized rolling window）")
    df, fe = build_features_vectorized(res_df, meta_df)

    feat_cols = FeatureEngineer.FEATURE_COLUMNS
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        logger.warning(f"  不足特徴量を NaN で補完: {missing}")
        for c in missing:
            df[c] = np.nan

    # ── 学習データ整備 ─────────────────────────────────────────
    df = df.dropna(subset=["is_win"])
    df["is_win"]     = df["is_win"].astype(int)
    if "is_placed" in df.columns:
        df["is_placed"] = df["is_placed"].fillna(0).astype(int)

    X = (df[feat_cols]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0))
    y_win     = df["is_win"].values
    y_place   = df["is_placed"].values if "is_placed" in df.columns else None
    groups    = df["race_id"].values

    logger.info(
        f"  学習データ: {len(X):,} 行  "
        f"is_win={y_win.sum()} ({y_win.mean()*100:.1f}%)  "
        f"races={df['race_id'].nunique()}"
    )

    # ── 勝ちモデル学習 ────────────────────────────────────────────
    logger.info("[STEP 2] 勝ちモデル学習 (win_model)")
    win_bst = _train_booster(X, y_win, groups, label="win_model")
    _log_feature_importance(win_bst, "win_model")

    # ── 複勝モデル学習 ─────────────────────────────────────────
    place_bst: lgb.Booster | None = None
    if not args.no_place_model and y_place is not None:
        logger.info("[STEP 3] 複勝モデル学習 (place_model, is_placed=3着以内)")
        place_bst = _train_booster(X, y_place, groups, label="place_model")
        _log_feature_importance(place_bst, "place_model")
    else:
        logger.info("[STEP 3] 複勝モデル学習 スキップ")

    # ── モデル保存 ─────────────────────────────────────────────
    logger.info("[STEP 4] モデル保存")
    payload = {"model": win_bst, "place_model": place_bst}
    joblib.dump(payload, model_path)
    logger.info(f"  モデル保存: {model_path}")

    # ── 推論用統計保存 ──────────────────────────────────────────
    fe.save_stats(stats_path)
    logger.info(f"  特徴量統計保存: {stats_path}")

    logger.info("=" * 60)
    logger.info("NAR モデル学習完了")
    logger.info(f"  モデル: {model_path}")
    logger.info(f"  統計  : {stats_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
