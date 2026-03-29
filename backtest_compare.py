"""
新旧モデル比較バックテスト（高速版 v2）
  - from_stats() で horse_recent_form を事前計算済み統計から参照（高速推論パス）
  - 特徴量を全レース分キャッシュしてから両モデルで予測
  - 旧モデル: lgbm_model_old.pkl（23特徴量、MIN_HONMEI_PROB=0.15、EV≥1.05）
  - 新モデル: lgbm_model.pkl    （26特徴量、MIN_HONMEI_PROB=0.25、EV≥1.15）
  - テストデータ: data/raw/test_results.csv（2025-01 〜 2026-03）

実行: python backtest_compare.py
"""
from __future__ import annotations

import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pickle

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer

# ── 旧モデルの特徴量カラム（23個）────────────────────────────
OLD_FEATURE_COLUMNS = [
    "frame_number", "horse_number", "age", "weight_carried",
    "course_type_code", "distance", "ground_condition_code", "weather_code",
    "trainer_win_rate", "trainer_place_rate",
    "jockey_win_rate", "jockey_place_rate", "jockey_runs",
    "recent_avg_pos", "recent_avg_last3f", "recent_top3_rate",
    "race_class_code", "venue_code", "venue_ground_code",
    "n_entries", "market_hhi", "prev_margin", "prev_last3f_rank_norm",
]
NEW_FEATURE_COLUMNS = FeatureEngineer.FEATURE_COLUMNS  # 26個

CONFIGS = [
    {
        "label": "旧モデル (23特徴量 / prob≥0.15 / EV≥1.05)",
        "model_path": ROOT / "data/models/lgbm_model_old.pkl",
        "feature_cols": OLD_FEATURE_COLUMNS,
        "min_prob": 0.15,
        "min_gap": 0.05,
        "ev_threshold": 1.05,
    },
    {
        "label": "新モデル (26特徴量 / prob≥0.25 / EV≥1.15)",
        "model_path": ROOT / "data/models/lgbm_model.pkl",
        "feature_cols": NEW_FEATURE_COLUMNS,
        "min_prob": 0.25,
        "min_gap": 0.05,
        "ev_threshold": 1.15,
    },
    {
        "label": "新モデル調整版 (26特徴量 / prob≥0.30 / EV≥1.15)",
        "model_path": ROOT / "data/models/lgbm_model.pkl",
        "feature_cols": NEW_FEATURE_COLUMNS,
        "min_prob": 0.30,
        "min_gap": 0.05,
        "ev_threshold": 1.15,
    },
]

DATE_FROM = "2025-01-01"
VENUE_MAP = {
    "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神",
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "10": "小倉",
}

# 一時的な全履歴統計ファイル
TMP_STATS = ROOT / "data/models/feature_stats_allhistory.pkl"


def parse_odds(val):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


def main():
    t0 = time.time()

    # ── データ読み込み ─────────────────────────────────────────
    print("データ読み込み中...")
    train_res = pd.read_csv(ROOT / "data/raw/train_results.csv", dtype=str)
    test_res  = pd.read_csv(ROOT / "data/raw/test_results.csv",  dtype=str)
    test_meta = pd.read_csv(ROOT / "data/raw/test_meta.csv",     dtype=str)

    all_history = pd.concat([train_res, test_res], ignore_index=True)
    print(f"全履歴: {len(all_history):,} rows")

    test_meta["race_date"] = pd.to_datetime(test_meta["race_date"], errors="coerce")
    target_meta = test_meta[test_meta["race_date"] >= DATE_FROM].sort_values("race_date")
    print(f"対象レース: {len(target_meta)} races ({DATE_FROM} 〜)")

    # ── 全履歴の統計を事前計算して保存（初回のみ）─────────────
    if not TMP_STATS.exists():
        print("全履歴の特徴量統計を計算中（初回のみ・数分かかります）...")
        fe_full = FeatureEngineer(all_history)
        fe_full.precompute_aggregations()
        fe_full.save_stats(TMP_STATS)
        print(f"統計保存完了: {TMP_STATS} ({time.time()-t0:.0f}s)")
    else:
        print(f"既存統計を使用: {TMP_STATS}")

    # ── from_stats() で高速推論用 FeatureEngineer を生成 ──────
    print("FeatureEngineer (推論モード) 読み込み中...")
    fe = FeatureEngineer.from_stats(TMP_STATS)
    print(f"読み込み完了 ({time.time()-t0:.0f}s)")

    # ── 特徴量を全レース分まとめて生成（キャッシュ）────────────
    print("特徴量生成中...")
    feat_cache: dict[str, pd.DataFrame] = {}
    odds_cache: dict[str, dict] = {}    # race_id → {horse_id: odds}
    pos_cache:  dict[str, dict] = {}    # race_id → {horse_id: finish_pos}

    target_ids = target_meta["race_id"].tolist()
    for i, race_id in enumerate(target_ids):
        meta_row    = target_meta[target_meta["race_id"] == race_id].iloc[0]
        course_type = str(meta_row.get("course_type", "") or "")
        distance    = int(meta_row.get("distance", 0) or 0)
        gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
        wx_code     = int(meta_row.get("weather_code", -1) or -1)
        race_name   = str(meta_row.get("race_name", "") or "")

        if not course_type or distance == 0:
            continue

        race_entries = test_res[test_res["race_id"] == race_id].copy()
        if len(race_entries) < 3:
            continue

        cols = ["horse_id", "horse_name", "horse_number", "frame_number",
                "sex_age", "weight_carried", "jockey_id"]
        if "trainer_name" in race_entries.columns:
            cols.append("trainer_name")
        entry_df = race_entries[cols].copy()
        entry_df["sex"] = entry_df["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entry_df["sex_age"].str[1:], errors="coerce")
        entry_df["weight_carried"] = pd.to_numeric(entry_df["weight_carried"], errors="coerce")
        entry_df["father"]         = ""
        entry_df["mother_father"]  = ""
        if "trainer_name" not in entry_df.columns:
            entry_df["trainer_name"] = ""
        if "last_3f" in race_entries.columns:
            entry_df["odds"] = pd.to_numeric(race_entries["last_3f"].values, errors="coerce")

        race_class_code = FeatureEngineer._race_name_to_class_code(race_name)
        try:
            venue_code = int(race_id[4:6])
        except Exception:
            venue_code = -1

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
            feat_cache[race_id] = feat_df
        except Exception:
            continue

        # 実際の着順・オッズをキャッシュ
        actual = race_entries.copy()
        actual["fp"]     = pd.to_numeric(actual["finish_position"], errors="coerce")
        actual["odds_f"] = actual["last_3f"].apply(parse_odds)
        actual = actual[actual["fp"].notna()]
        pos_cache[race_id]  = dict(zip(actual["horse_id"], actual["fp"]))
        odds_cache[race_id] = dict(zip(actual["horse_id"], actual["odds_f"]))

        if (i + 1) % 500 == 0:
            pct = (i + 1) / len(target_ids) * 100
            print(f"  特徴量: {i+1}/{len(target_ids)} ({pct:.0f}%) {time.time()-t0:.0f}s")

    print(f"特徴量生成完了: {len(feat_cache)} races ({time.time()-t0:.0f}s)")

    # ── 各モデルでバックテスト ─────────────────────────────────
    all_summaries = []

    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']} ===")
        obj = joblib.load(cfg["model_path"])
        win_model   = obj["model"]
        place_model = obj.get("place_model")
        min_prob    = cfg["min_prob"]
        min_gap     = cfg["min_gap"]
        ev_thr      = cfg["ev_threshold"]
        feat_cols   = cfg["feature_cols"]

        results = []

        for race_id, feat_df in feat_cache.items():
            meta_row   = target_meta[target_meta["race_id"] == race_id].iloc[0]
            race_date  = meta_row["race_date"]
            race_name  = str(meta_row.get("race_name", "") or "")
            course_type = str(meta_row.get("course_type", "") or "")
            distance   = int(meta_row.get("distance", 0) or 0)
            venue_name = VENUE_MAP.get(str(race_id)[4:6], str(race_id)[4:6])

            fd = feat_df.copy()
            for c in feat_cols:
                if c not in fd.columns:
                    fd[c] = 0.0

            X = fd[feat_cols].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
            win_probs_arr = win_model.predict(X, num_threads=1)
            if place_model is not None:
                place_probs_arr = place_model.predict(X, num_threads=1)
                blended = 0.7 * win_probs_arr + 0.3 * place_probs_arr
            else:
                blended = win_probs_arr

            pred_df = fd[["horse_id", "horse_name", "horse_number"]].copy()
            pred_df["prob"] = blended
            pred_df = pred_df.sort_values("prob", ascending=False).reset_index(drop=True)

            honmei_prob = float(pred_df.iloc[0]["prob"])
            taikou_prob = float(pred_df.iloc[1]["prob"]) if len(pred_df) > 1 else 0.0
            gap = honmei_prob - taikou_prob

            honmei_id = str(pred_df.iloc[0]["horse_id"])
            odds_col  = fd.set_index("horse_id")["odds"] if "odds" in fd.columns else {}
            raw_odds  = odds_col.get(honmei_id)
            honmei_odds_pre = float(raw_odds) if raw_odds is not None else float("nan")
            honmei_ev = honmei_prob * honmei_odds_pre if not np.isnan(honmei_odds_pre) else 0.0

            is_skip = (
                honmei_prob < min_prob
                or gap < min_gap
                or honmei_ev < ev_thr
            )

            pos_map  = pos_cache.get(race_id, {})
            odds_map = odds_cache.get(race_id, {})
            honmei_pos    = pos_map.get(honmei_id, float("nan"))
            honmei_odds_v = odds_map.get(honmei_id, float("nan"))

            ts_hit = int(honmei_pos == 1) if not np.isnan(honmei_pos) else 0
            fk_hit = int(honmei_pos <= 3) if not np.isnan(honmei_pos) else 0
            ts_ret = honmei_odds_v * 100 if ts_hit and not np.isnan(honmei_odds_v) else 0.0
            fk_ret = max(1.0, honmei_odds_v * 0.35) * 100 if fk_hit and not np.isnan(honmei_odds_v) else 0.0

            results.append({
                "race_id": race_id,
                "race_date": str(race_date)[:10],
                "race_name": race_name,
                "venue": venue_name,
                "course_type": course_type,
                "distance": distance,
                "honmei_name": str(pred_df.iloc[0]["horse_name"]),
                "honmei_prob": round(honmei_prob, 4),
                "gap": round(gap, 4),
                "honmei_ev": round(honmei_ev, 4),
                "is_skip": int(is_skip),
                "honmei_pos": honmei_pos,
                "honmei_odds": honmei_odds_v,
                "tansho_bet": 0 if is_skip else 100,
                "tansho_hit": 0 if is_skip else ts_hit,
                "tansho_ret": 0.0 if is_skip else ts_ret,
                "fukusho_bet": 0 if is_skip else 100,
                "fukusho_hit": 0 if is_skip else fk_hit,
                "fukusho_ret": 0.0 if is_skip else fk_ret,
            })

        tag = "old" if "旧" in cfg["label"] else "new"
        out = ROOT / f"data/processed/backtest_compare_{tag}.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"  CSV保存: {out}")

        s = _print_summary(cfg["label"], results)
        all_summaries.append(s)

    _print_diff(all_summaries[0], all_summaries[1])
    _print_diff(all_summaries[0], all_summaries[2])
    print(f"\n総処理時間: {time.time()-t0:.0f}s")


def _print_summary(label: str, rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    buy = df[df["is_skip"] == 0].copy()
    total = len(df)
    n_buy = len(buy)

    ts_hit = int(buy["tansho_hit"].sum())
    ts_bet = int(buy["tansho_bet"].sum())
    ts_ret = buy["tansho_ret"].sum()
    fk_hit = int(buy["fukusho_hit"].sum())
    fk_ret = buy["fukusho_ret"].sum()

    ts_rate = ts_hit / n_buy if n_buy else 0
    ts_roi  = ts_ret / ts_bet if ts_bet else 0
    fk_rate = fk_hit / n_buy if n_buy else 0
    fk_roi  = fk_ret / (n_buy * 100) if n_buy else 0

    venue_stats = (
        buy.groupby("venue")
        .agg(races=("tansho_bet", "count"), hits=("tansho_hit", "sum"),
             ret=("tansho_ret", "sum"), bet=("tansho_bet", "sum"))
        .assign(hit_rate=lambda d: d["hits"] / d["races"],
                roi=lambda d: d["ret"] / d["bet"])
        .sort_values("hit_rate", ascending=False)
    )

    bins    = [0, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0]
    plabels = ["<20%", "20-25%", "25-30%", "30-35%", "35-40%", "40%+"]
    buy["prob_bin"] = pd.cut(buy["honmei_prob"], bins=bins, labels=plabels)
    prob_stats = (
        buy.groupby("prob_bin", observed=False)
        .agg(races=("tansho_bet", "count"), hits=("tansho_hit", "sum"),
             ret=("tansho_ret", "sum"), bet=("tansho_bet", "sum"))
        .assign(hit_rate=lambda d: d["hits"] / d["races"].replace(0, np.nan),
                roi=lambda d: d["ret"] / d["bet"].replace(0, np.nan))
    )

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  対象: {total}R  買い: {n_buy}R  スキップ: {total-n_buy}R")
    print(f"  単勝: {ts_hit}/{n_buy}R 的中率{ts_rate:.1%}  ROI {ts_roi:.1%}  "
          f"(投資¥{ts_bet:,} 回収¥{int(ts_ret):,})")
    print(f"  複勝: {fk_hit}/{n_buy}R 的中率{fk_rate:.1%}  ROI {fk_roi:.1%}")

    print("\n  【会場別単勝成績（買い対象のみ）】")
    for v, row in venue_stats.iterrows():
        bar = "█" * int(row["hit_rate"] * 20)
        print(f"    {v:4s} {int(row['hits']):>2}/{int(row['races']):>3}R "
              f"的中{row['hit_rate']:.1%} ROI{row['roi']:.1%}  {bar}")

    print("\n  【確率帯別単勝成績】")
    for pb, row in prob_stats.iterrows():
        if pd.isna(row["races"]) or row["races"] == 0:
            continue
        rate = row["hit_rate"] if not pd.isna(row["hit_rate"]) else 0
        roi  = row["roi"]      if not pd.isna(row["roi"])      else 0
        bar  = "█" * int(rate * 20)
        print(f"    {pb:8s} {int(row['hits']):>2}/{int(row['races']):>3}R "
              f"的中{rate:.1%} ROI{roi:.1%}  {bar}")

    return {
        "label": label, "total": total, "buy": n_buy,
        "ts_hit": ts_hit, "ts_rate": ts_rate, "ts_roi": ts_roi,
        "fk_rate": fk_rate, "fk_roi": fk_roi,
    }


def _print_diff(old: dict, new: dict):
    print(f"\n{'='*60}")
    print("  【新旧モデル 差分比較】")
    print(f"{'='*60}")

    def row(name, key, fmt=".1%"):
        o, n = old[key], new[key]
        d = n - o
        sign = "+" if d >= 0 else ""
        arrow = "↑" if d > 0.001 else ("↓" if d < -0.001 else "→")
        return f"  {name:14s}  旧:{o:{fmt}}  新:{n:{fmt}}  {arrow}{sign}{d:{fmt}}"

    print(f"  買いレース      旧:{old['buy']}R  新:{new['buy']}R  ({new['buy']-old['buy']:+d}R)")
    print(row("単勝的中率",  "ts_rate"))
    print(row("単勝回収率",  "ts_roi"))
    print(row("複勝的中率",  "fk_rate"))
    print(row("複勝回収率",  "fk_roi"))


if __name__ == "__main__":
    main()
