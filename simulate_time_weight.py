"""
時系列重み付けシミュレーション

学習データに日付ベースのサンプルウェイトを適用し、
モデル性能への影響を時系列ホールドアウトで評価する。

評価方法:
  - 学習: 2022-2024年データ（重み付き）
  - 評価: 2025-2026年データ（重みなし）
  - 指標: AUC, LogLoss, Calibration, 上位予測精度

実行:
  python simulate_time_weight.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer

FEATURE_COLS = FeatureEngineer.FEATURE_COLUMNS

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100


# ======================================================================
# 重み付けスキーム
# ======================================================================

def compute_weights(race_ids: pd.Series, scheme: str) -> np.ndarray:
    """
    各行に対してサンプルウェイトを計算する。

    Parameters
    ----------
    race_ids : pd.Series
        race_id (str, 先頭8文字が YYYYMMDD)
    scheme : str
        重み付けスキーム名

    Returns
    -------
    np.ndarray
        各行のウェイト（正規化: mean=1.0）
    """
    dates = pd.to_datetime(
        race_ids.str[:8], format="%Y%m%d", errors="coerce"
    )
    t_min = dates.min()
    t_max = dates.max()
    span_days = (t_max - t_min).days or 1
    days_from_start = (dates - t_min).dt.days.fillna(0).values
    days_from_end   = (t_max - dates).dt.days.fillna(0).values
    year = race_ids.str[:4].astype(int)

    if scheme == "uniform":
        w = np.ones(len(race_ids))

    elif scheme == "linear":
        # 0.1 (最古) → 1.0 (最新) の線形増加
        w = 0.1 + 0.9 * (days_from_start / span_days)

    elif scheme == "exp_2y":
        # 半減期 = 2年（730日）
        half_life = 730
        w = np.exp(np.log(2) / half_life * -days_from_end)

    elif scheme == "exp_1y":
        # 半減期 = 1年（365日）
        half_life = 365
        w = np.exp(np.log(2) / half_life * -days_from_end)

    elif scheme == "exp_6m":
        # 半減期 = 6ヶ月（182日）
        half_life = 182
        w = np.exp(np.log(2) / half_life * -days_from_end)

    elif scheme == "step":
        # 年ごとに段階的に重みを上げる
        w = np.where(year <= 2022, 0.5,
            np.where(year == 2023, 0.75,
            np.where(year == 2024, 1.5,
            np.where(year == 2025, 3.0, 5.0))))

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # mean=1.0 に正規化
    w = w / w.mean()
    return w.astype(np.float32)


# ======================================================================
# 学習・評価
# ======================================================================

def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scheme: str,
) -> dict:
    """
    重み付きで学習し、テストセットで評価する。

    Returns
    -------
    dict
        {scheme, auc, logloss, top1_win_rate, calibration_slope}
    """
    X_tr = train_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_tr = train_df["is_win"].values
    g_tr = train_df["race_id"].values
    w_tr = compute_weights(train_df["race_id"], scheme)

    X_te = test_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_te = test_df["is_win"].values

    # GroupKFold で early stopping ラウンド数を決定
    gkf = GroupKFold(n_splits=5)
    models = []
    for train_idx, val_idx in gkf.split(X_tr, y_tr, g_tr):
        Xt, Xv = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        yt, yv = y_tr[train_idx], y_tr[val_idx]
        wt = w_tr[train_idx]
        ds_tr = lgb.Dataset(Xt, label=yt, weight=wt)
        ds_vl = lgb.Dataset(Xv, label=yv, reference=ds_tr)
        booster = lgb.train(
            LGBM_PARAMS, ds_tr,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[ds_vl],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        models.append(booster)

    best_rounds = max(m.best_iteration for m in models)

    # 全学習データで最終モデル（重み付き）
    full_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    final_model = lgb.train(
        LGBM_PARAMS, full_ds,
        num_boost_round=best_rounds,
    )

    # テストセット評価
    pred_te = final_model.predict(X_te)
    auc     = roc_auc_score(y_te, pred_te)
    ll      = log_loss(y_te, pred_te)

    # 上位予測精度: 各レースで最高確率の馬が実際に勝つ割合
    test_df2 = test_df.copy()
    test_df2["pred"] = pred_te
    top1 = test_df2.sort_values("pred", ascending=False).groupby("race_id").first()
    top1_win_rate = top1["is_win"].mean()

    # 確率帯別キャリブレーション（10分位）
    bins = np.percentile(pred_te, np.arange(0, 101, 10))
    bins = np.unique(bins)
    pred_bin = np.digitize(pred_te, bins) - 1
    calib_rows = []
    for b in range(len(bins)):
        mask = pred_bin == b
        if mask.sum() < 10:
            continue
        mean_pred = pred_te[mask].mean()
        mean_act  = y_te[mask].mean()
        calib_rows.append((mean_pred, mean_act))

    if len(calib_rows) >= 2:
        preds_c, acts_c = zip(*calib_rows)
        slope = np.polyfit(preds_c, acts_c, 1)[0]
    else:
        slope = float("nan")

    # 重みのサマリー（年別平均ウェイト）
    w_by_year = {}
    for yr in ["2022", "2023", "2024", "2025", "2026"]:
        mask = train_df["race_id"].str[:4] == yr
        if mask.any():
            w_by_year[yr] = float(w_tr[mask.values].mean())

    return {
        "scheme":        scheme,
        "best_rounds":   best_rounds,
        "auc":           round(auc, 5),
        "logloss":       round(ll, 5),
        "top1_win_rate": round(top1_win_rate, 4),
        "calib_slope":   round(slope, 3) if not np.isnan(slope) else None,
        "weight_by_year": w_by_year,
    }


# ======================================================================
# メイン
# ======================================================================

def main():
    print("=" * 65)
    print("時系列重み付けシミュレーション")
    print("  学習: 2022-2024  評価: 2025-2026")
    print("=" * 65)

    df = pd.read_csv(
        ROOT / "data/processed/train_all.csv",
        dtype={"race_id": str}
    )
    print(f"全データ: {len(df):,}行 / {df['race_id'].nunique():,}レース")

    year = df["race_id"].str[:4]
    train_df = df[year.isin(["2022", "2023", "2024"])].copy()
    test_df  = df[year.isin(["2025", "2026"])].copy()
    print(f"  学習: {len(train_df):,}行 / {train_df['race_id'].nunique():,}レース (2022-2024)")
    print(f"  評価: {len(test_df):,}行  / {test_df['race_id'].nunique():,}レース (2025-2026)")
    print()

    schemes = [
        ("uniform",  "均一（ベースライン）"),
        ("linear",   "線形増加（0.1→1.0）"),
        ("exp_2y",   "指数減衰 半減期2年"),
        ("exp_1y",   "指数減衰 半減期1年"),
        ("exp_6m",   "指数減衰 半減期6ヶ月"),
        ("step",     "段階（2022:0.5x/2023:0.75x/2024:1.5x/2025:3x）"),
    ]

    results = []
    for scheme, label in schemes:
        print(f"[{label}]")
        # 重みプレビュー
        w = compute_weights(train_df["race_id"], scheme)
        yr_w = {}
        for yr in ["2022", "2023", "2024"]:
            mask = train_df["race_id"].str[:4] == yr
            yr_w[yr] = w[mask.values].mean()
        print(f"  年別平均ウェイト: 2022={yr_w['2022']:.3f}  2023={yr_w['2023']:.3f}  2024={yr_w['2024']:.3f}")

        res = train_and_eval(train_df, test_df, scheme)
        res["label"] = label
        results.append(res)
        print(f"  best_rounds={res['best_rounds']}  AUC={res['auc']:.5f}  LogLoss={res['logloss']:.5f}")
        print(f"  top1的中率={res['top1_win_rate']:.2%}  calibration_slope={res['calib_slope']}")
        print()

    # サマリー表
    print("=" * 65)
    print("結果サマリー（評価期間: 2025-2026）")
    print("=" * 65)
    print(f"{'スキーム':<28} {'AUC':>8} {'LogLoss':>9} {'top1的中%':>10} {'calib':>7}")
    print("-" * 65)
    base = results[0]
    for r in results:
        dauc = r['auc'] - base['auc']
        dll  = r['logloss'] - base['logloss']
        dwr  = r['top1_win_rate'] - base['top1_win_rate']
        dauc_str = f"({dauc:+.5f})" if dauc != 0 else ""
        print(
            f"{r['label']:<28} "
            f"{r['auc']:>8.5f} "
            f"{r['logloss']:>9.5f} "
            f"{r['top1_win_rate']:>9.2%} "
            f"{r['calib_slope']:>7}"
        )
    print("=" * 65)
    print("※ calib_slope≈1.0 が理想（1未満=過信、1超=過小推定）")
    print("※ top1的中率 = 各レースでモデルが最も高確率と予測した馬の実際の勝率")


if __name__ == "__main__":
    main()
