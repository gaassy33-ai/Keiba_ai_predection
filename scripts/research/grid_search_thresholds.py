"""
grid_search_thresholds.py
==========================
backtest_two_brain_june.py が出力した候補ペア全件CSV
(data/processed/backtest_two_brain_june_candidates.csv) を使い、
min_ev_threshold × gatekeeper_threshold のグリッドサーチで
購入点数・的中数を再集計する（モデル再推論なし・閾値の再適用のみ）。

各レースの先頭候補行（pairs列挙順により必ず axis1×axis2 ペア）から
axis1_id = horse_id_i, axis2_id = horse_id_j を特定し、
各候補行の i/j がどちらの軸か（あるいはパートナーか）を判定して
新しい gatekeeper_threshold での合否を再計算する。

実行:
    .venv/bin/python grid_search_thresholds.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/processed/backtest_two_brain_june_candidates.csv")

EV_THRESHOLDS = [1.05, 1.10, 1.15]
GK_THRESHOLDS = [0.35, 0.40, 0.45]
N_DAYS = 4  # 6/6, 6/7, 6/13, 6/14


def main() -> None:
    df = pd.read_csv(CSV_PATH, dtype={"race_id": str, "horse_id_i": str, "horse_id_j": str})

    # ── 各レースの axis1_id / axis2_id を特定（先頭行 = axis1×axis2 ペア）──
    axis_map = (
        df.groupby("race_id", sort=False)
        .first()[["horse_id_i", "horse_id_j"]]
        .rename(columns={"horse_id_i": "axis1_id", "horse_id_j": "axis2_id"})
    )
    df = df.join(axis_map, on="race_id")

    hid_i = df["horse_id_i"]
    hid_j = df["horse_id_j"]
    i_is_axis1 = hid_i == df["axis1_id"]
    i_is_axis2 = hid_i == df["axis2_id"]
    j_is_axis1 = hid_j == df["axis1_id"]
    j_is_axis2 = hid_j == df["axis2_id"]

    p_safe_i = np.where(i_is_axis1, df["p_axis1_safe"], np.where(i_is_axis2, df["p_axis2_safe"], np.nan))
    p_safe_j = np.where(j_is_axis1, df["p_axis1_safe"], np.where(j_is_axis2, df["p_axis2_safe"], np.nan))
    df["_p_safe_i"] = p_safe_i
    df["_p_safe_j"] = p_safe_j

    base_pass = (
        df["passed_longshot"].astype(bool)
        & df["passed_est_odds"].astype(bool)
        & df["passed_p_model"].astype(bool)
        & (~df["race_axis_reject"].astype(bool))
    )

    print("=" * 78)
    print(f"グリッドサーチ: min_ev_threshold × gatekeeper_threshold（{N_DAYS}日間・候補{len(df)}件）")
    print("固定条件: est_odds<=150, axis_max_odds<=10, p_model>=0.01")
    print("=" * 78)

    header = f"{'gatekeeper_threshold':<22}" + "".join(f"EV>={t:<6}" for t in EV_THRESHOLDS)
    print(header)
    print("-" * 78)

    results = []
    for gk_thr in GK_THRESHOLDS:
        gk_pass_i = pd.isna(df["_p_safe_i"]) | (df["_p_safe_i"] >= gk_thr)
        gk_pass_j = pd.isna(df["_p_safe_j"]) | (df["_p_safe_j"] >= gk_thr)
        gk_pass = gk_pass_i & gk_pass_j

        row_label = f"{gk_thr:<22}"
        for ev_thr in EV_THRESHOLDS:
            would_buy = base_pass & gk_pass & (df["ev"] >= ev_thr)
            buy = df[would_buy]
            n_bets = len(buy)
            n_races = buy["race_id"].nunique()
            n_hits = int(buy["hit"].sum())
            cost = n_bets * 100
            payout = float(buy["payout"].sum())
            roi = payout / cost * 100 if cost else 0.0
            results.append({
                "gatekeeper_threshold": gk_thr, "min_ev_threshold": ev_thr,
                "n_bets": n_bets, "n_races": n_races, "n_hits": n_hits,
                "cost": cost, "payout": payout, "roi": roi,
            })
            row_label += f"{n_bets}点/{n_races}R/{n_hits}的中".ljust(18)
        print(row_label)

    print()
    res_df = pd.DataFrame(results)
    res_df["per_day"] = res_df["n_bets"] / N_DAYS
    print(res_df.to_string(index=False))

    # ── 目標レンジ（4日間合計 15〜30点）に最も近い組み合わせ ──────
    target_lo, target_hi = 15, 30
    in_range = res_df[(res_df["n_bets"] >= target_lo) & (res_df["n_bets"] <= target_hi)]
    print()
    print("=" * 78)
    if not in_range.empty:
        print(f"目標レンジ（{target_lo}〜{target_hi}点/4日間）に合致する組み合わせ:")
        print(in_range.sort_values("n_bets").to_string(index=False))
    else:
        print(f"目標レンジ（{target_lo}〜{target_hi}点/4日間）に厳密合致する組み合わせはありません。")
        res_df["dist"] = res_df["n_bets"].apply(
            lambda x: 0 if target_lo <= x <= target_hi else min(abs(x - target_lo), abs(x - target_hi))
        )
        closest = res_df.sort_values("dist").head(5)
        print("目標に最も近い上位5組:")
        print(closest.drop(columns=["dist"]).to_string(index=False))
    print("=" * 78)

    out_path = Path("data/processed/grid_search_thresholds.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\nCSV保存: {out_path}")


if __name__ == "__main__":
    main()
