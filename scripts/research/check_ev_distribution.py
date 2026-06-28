"""
check_ev_distribution.py
========================
Lite モデル (odds除外) の EV 分布を検証する。

比較:
  - フルモデル (lgbm_ltr_model.pkl):    odds_log 59% → EV ≈ 0.775 集中
  - Liteモデル (lgbm_ltr_lite.pkl):     odds除外   → EV が広がるか検証

対象データ: 2026年1月 (約10日間, ~7分)
  Walk-Forward で feature stats を構築し、EV の分布を比較する。

実行:
    .venv/bin/python check_ev_distribution.py
"""
from __future__ import annotations

import sys
import time
from itertools import combinations as _comb, permutations as _perm
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import LTRTrainer

Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)

JRA_TAKE_WIDE  = 0.225
LONGSHOT_MAX   = 30.0
CHECK_MONTHS   = {1}          # 2026年1月のみ (高速化)
MAX_BETS_CHECK = 20           # 分布確認用に上限を広げる


# ── ユーティリティ ───────────────────────────────────────────────────
def _market_probs(odds_list):
    raw = np.array([1.0 / max(o, 1.01) for o in odds_list])
    s = raw.sum()
    return raw / s if s > 0 else raw

def _harville(probs, order):
    p, rem = 1.0, 1.0
    for idx in order:
        if rem < 1e-9: return 0.0
        p *= probs[idx] / rem
        rem -= probs[idx]
    return p

def _prob_wide(probs, i, j):
    return sum(
        sum(_harville(probs, list(o)) for o in _perm([i, j, k]))
        for k in range(len(probs)) if k != i and k != j
    )

def _compute_ev(model_probs, mkt_probs, i, j):
    pm = _prob_wide(model_probs, i, j)
    pk = _prob_wide(mkt_probs,   i, j)
    est = (1.0 - JRA_TAKE_WIDE) / pk if pk > 1e-9 else 999.0
    return pm * est


def _run_check(ltr: LTRTrainer, res_df: pd.DataFrame, meta_df: pd.DataFrame,
               label: str) -> list[float]:
    """指定モデルで 2026年1月 の全ペア EV を計算し、リストで返す。"""
    meta_2026 = meta_df[
        meta_df["race_id"].str.startswith("2026")
    ].copy()
    meta_2026["race_date"] = pd.to_datetime(meta_2026["race_date"], errors="coerce")
    meta_2026 = meta_2026.dropna(subset=["race_date"])
    meta_2026 = meta_2026[meta_2026["race_date"].dt.month.isin(CHECK_MONTHS)]

    race_dates = sorted(meta_2026["race_date"].unique())
    logger.info(f"  [{label}] 対象日数: {len(race_dates)}日")

    date_map = meta_df.set_index("race_id")["race_date"].to_dict()
    all_evs: list[float] = []

    for di, race_date in enumerate(race_dates):
        history_before = res_df[
            pd.to_datetime(
                res_df["race_id"].map(date_map), errors="coerce"
            ) < race_date
        ].drop_duplicates(subset=["race_id", "horse_id"]).copy()

        if history_before.empty:
            continue

        fe = FeatureEngineer(history_before)
        fe.precompute_aggregations()

        for _, meta_row in meta_2026[meta_2026["race_date"] == race_date].iterrows():
            race_id = meta_row["race_id"]
            entries = res_df[res_df["race_id"] == race_id].drop_duplicates("horse_id")
            if entries.empty or len(entries) < 3:
                continue

            course_type = str(meta_row.get("course_type", ""))
            distance    = int(meta_row.get("distance", 0) or 0)
            if not course_type or distance == 0 or distance >= 2750:
                continue

            entry_df = entries[["horse_id","horse_name","horse_number",
                                 "frame_number","jockey_id"]].copy()
            entry_df["sex"] = ""
            entry_df["age"] = np.nan
            entry_df["weight_carried"] = np.nan
            entry_df["father"] = ""
            entry_df["mother_father"] = ""

            if "odds" in entries.columns:
                o = pd.to_numeric(entries["odds"], errors="coerce")
                entry_df["odds"] = o.values if o.dropna().min() < 15.0 else np.nan

            rcc = FeatureEngineer._race_name_to_class_code(
                str(meta_row.get("race_name", ""))
            )
            try:
                vc = int(race_id[4:6])
            except Exception:
                vc = -1

            try:
                feat_df = fe.build_entry_features(
                    entry_df, course_type, distance,
                    int(meta_row.get("ground_condition_code", -1) or -1),
                    int(meta_row.get("weather_code", -1) or -1),
                    rcc, vc,
                )
            except Exception:
                continue

            if len(feat_df) < 3:
                continue

            X = (feat_df[ltr.feature_columns]
                 .apply(pd.to_numeric, errors="coerce").fillna(0.0))
            raw_scores  = ltr.predict(X)
            model_probs = LTRTrainer.scores_to_probs(raw_scores)

            horse_ids = feat_df["horse_id"].astype(str).tolist()
            odds_map  = {}
            if "odds" in feat_df.columns:
                odds_map = {str(k): v for k, v in
                            feat_df.set_index("horse_id")["odds"].dropna().to_dict().items()}

            odds_list = [float(odds_map.get(hid, float("nan"))) for hid in horse_ids]
            odds_list = [o if not np.isnan(o) and o > 1.0 else 5.0 for o in odds_list]
            mkt_probs = _market_probs(odds_list)

            n = len(horse_ids)
            for i, j in _comb(range(n), 2):
                if horse_ids[i] == horse_ids[j]:
                    continue
                o_i = float(odds_map.get(horse_ids[i], float("nan")))
                o_j = float(odds_map.get(horse_ids[j], float("nan")))
                if (not np.isnan(o_i) and o_i > LONGSHOT_MAX) or \
                   (not np.isnan(o_j) and o_j > LONGSHOT_MAX):
                    continue
                ev = _compute_ev(model_probs, mkt_probs, i, j)
                all_evs.append(ev)

        logger.info(f"  [{label}] {di+1}/{len(race_dates)}日処理  累計EVペア:{len(all_evs):,}")

    return all_evs


def _report(evs: list[float], label: str) -> None:
    a = np.array(evs)
    logger.info(f"\n{'='*55}")
    logger.info(f"【{label}】EV分布 (2026年1月, ロングショット除外)")
    logger.info(f"{'='*55}")
    logger.info(f"  総ペア数  : {len(a):,}")
    logger.info(f"  EV 平均   : {a.mean():.3f}")
    logger.info(f"  EV 中央値 : {np.median(a):.3f}")
    logger.info(f"  EV 標準偏差: {a.std():.3f}")
    logger.info(f"  EV 最大   : {a.max():.3f}")
    logger.info(f"  EV 最小   : {a.min():.3f}")
    logger.info(f"  EV≥0.80  : {(a>=0.80).sum():,}件 ({(a>=0.80).mean()*100:.1f}%)")
    logger.info(f"  EV≥1.00  : {(a>=1.00).sum():,}件 ({(a>=1.00).mean()*100:.1f}%)")
    logger.info(f"  EV≥1.05  : {(a>=1.05).sum():,}件 ({(a>=1.05).mean()*100:.1f}%)")
    logger.info(f"  EV≥1.20  : {(a>=1.20).sum():,}件 ({(a>=1.20).mean()*100:.1f}%)")
    logger.info(f"  EV≥1.50  : {(a>=1.50).sum():,}件 ({(a>=1.50).mean()*100:.1f}%)")

    # ヒストグラム (テキスト)
    logger.info(f"\n  EVヒストグラム:")
    bins = [0.0, 0.5, 0.7, 0.775, 0.85, 1.0, 1.1, 1.2, 1.5, 2.0, 99.0]
    labels_h = ["<0.5","0.5-0.7","0.7-0.775","0.775-0.85",
                 "0.85-1.0","1.0-1.1","1.1-1.2","1.2-1.5","1.5-2.0","≥2.0"]
    for lbl, lo, hi in zip(labels_h, bins[:-1], bins[1:]):
        cnt = ((a >= lo) & (a < hi)).sum()
        pct = cnt / len(a) * 100
        bar = "#" * int(pct / 2)
        logger.info(f"    {lbl:<12}: {cnt:>6,} ({pct:5.1f}%)  {bar}")


def main() -> None:
    t0 = time.time()

    full_model_path = Path("data/models/lgbm_ltr_model.pkl")
    lite_model_path = Path("data/models/lgbm_ltr_lite.pkl")

    for p in [lite_model_path]:
        if not p.exists():
            logger.error(f"モデルが見つかりません: {p}")
            logger.error("先に train_ltr_lite.py を実行してください")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("EV分布 比較チェック (2026年1月)")
    logger.info("=" * 60)

    train_res  = pd.read_csv("data/raw/train_results.csv",    dtype=str)
    test_res   = pd.read_csv("data/raw/test_results_new.csv", dtype=str)
    train_meta = pd.read_csv("data/raw/train_meta.csv",       dtype=str)
    test_meta  = pd.read_csv("data/raw/test_meta_new.csv",    dtype=str)

    res_df  = pd.concat([train_res, test_res],   ignore_index=True)
    meta_df = pd.concat([train_meta, test_meta], ignore_index=True)
    meta_df["race_date"] = pd.to_datetime(meta_df["race_date"], errors="coerce")

    results: dict[str, list[float]] = {}

    # ── フルモデル（odds込み）──────────────────────────────────────
    if full_model_path.exists():
        logger.info("\n[1/2] フルモデル (odds込み) の EV を計算中...")
        ltr_full = LTRTrainer.load(full_model_path)
        evs_full = _run_check(ltr_full, res_df, meta_df, "フルモデル(odds込み)")
        results["フルモデル(odds込み)"] = evs_full
        _report(evs_full, "フルモデル(odds込み)")

    # ── Lite モデル（odds除外）────────────────────────────────────
    logger.info("\n[2/2] Liteモデル (odds除外) の EV を計算中...")
    ltr_lite = LTRTrainer.load(lite_model_path)
    evs_lite = _run_check(ltr_lite, res_df, meta_df, "Liteモデル(odds除外)")
    results["Liteモデル(odds除外)"] = evs_lite
    _report(evs_lite, "Liteモデル(odds除外)")

    # ── 並べて比較 ─────────────────────────────────────────────────
    if len(results) == 2:
        a_full = np.array(results["フルモデル(odds込み)"])
        a_lite = np.array(results["Liteモデル(odds除外)"])
        logger.info(f"\n{'='*55}")
        logger.info("【比較サマリー】")
        logger.info(f"{'='*55}")
        logger.info(f"  {'':25}  {'フル':>10}  {'Lite':>10}")
        logger.info(f"  {'EV 平均':25}  {a_full.mean():>10.3f}  {a_lite.mean():>10.3f}")
        logger.info(f"  {'EV 中央値':25}  {np.median(a_full):>10.3f}  {np.median(a_lite):>10.3f}")
        logger.info(f"  {'EV≥1.0 比率':25}  {(a_full>=1.0).mean()*100:>9.1f}%  {(a_lite>=1.0).mean()*100:>9.1f}%")
        logger.info(f"  {'EV≥1.2 比率':25}  {(a_full>=1.2).mean()*100:>9.1f}%  {(a_lite>=1.2).mean()*100:>9.1f}%")

    total = (time.time() - t0) / 60
    logger.info(f"\n総実行時間: {total:.1f}分")


if __name__ == "__main__":
    main()
