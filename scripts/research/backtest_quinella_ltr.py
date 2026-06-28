"""
backtest_quinella_ltr.py
========================
馬連 + 軸馬流し + Temperature Scaling バックテスト。

【変更点（対 backtest_wide_ltr.py）】
  - 券種    : ワイド（top3） → 馬連（top2）
  - 控除率  : 22.5% → 17.5%
  - 確率式  : Harville 3着展開（O(n)） → Harville 2着版（O(1)）
  - 買い目  : 全ペア C(N,2) → 軸馬流し（axis1×axis2, axis1×P, axis2×P）
  - temperature : CLI 引数 → ltr.temperature（自動適用）

実行例:
    .venv/bin/python backtest_quinella_ltr.py              # 2024-2026 全期間
    .venv/bin/python backtest_quinella_ltr.py --year 2026  # 2026年のみ
    .venv/bin/python backtest_quinella_ltr.py --ev-threshold 1.2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import LTRTrainer
from src.betting.ltr_ev_engine import prob_quinella, est_quinella_odds_fn

# ── ロガー ───────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout,                         level="INFO",  format=_fmt, colorize=True)
logger.add("logs/backtest_quinella_ltr.log",   level="DEBUG", format=_fmt, rotation="30 MB")

# ── デフォルトパラメータ ─────────────────────────────────────────────
DEFAULT_EV_THRESHOLD    = 0.8    # 感度分析用に広め（サマリーで閾値別に集計）
DEFAULT_MAX_BETS        = 20     # 保存上限（サマリー集計で再絞込み）
DEFAULT_PARTNER_TOP_N   = 5      # 軸馬流しのパートナー数
DEFAULT_AXIS_MAX_ODDS   = 0.0    # 0.0 = フィルター無効（感度分析用に全レース保存）
LONGSHOT_ODDS_MAX       = 30.0
EST_QUINELLA_ODDS_MAX   = 100.0  # 感度分析用に広め


# ── ユーティリティ ───────────────────────────────────────────────────

def _parse_odds(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


def _market_probs(odds_list: list[float]) -> np.ndarray:
    raw = np.array([1.0 / max(o, 1.01) for o in odds_list])
    s = raw.sum()
    return raw / s if s > 0 else raw


# ── レース処理 ───────────────────────────────────────────────────────

def _process_race(
    race_id:       str,
    entries:       pd.DataFrame,
    meta_row:      pd.Series,
    fe:            FeatureEngineer,
    ltr:           LTRTrainer,
    ev_threshold:  float,
    max_bets:      int,
    partner_top_n: int,
    axis_max_odds: float = 0.0,   # 0.0 = フィルター無効
) -> list[dict] | None:
    """
    軸馬流し方式で馬連買い目を生成し、EV・ヒット情報を付与して返す。
    """
    race_date   = meta_row["race_date"]
    course_type = str(meta_row.get("course_type", ""))
    race_name   = str(meta_row.get("race_name", ""))
    distance    = int(meta_row.get("distance", 0) or 0)
    gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
    wx_code     = int(meta_row.get("weather_code", -1) or -1)

    if not course_type or distance == 0 or distance >= 2750:
        return None

    # ── entry_df 組み立て ──────────────────────────────────────────
    entry_df = entries[["horse_id", "horse_name", "horse_number",
                         "frame_number", "jockey_id"]].copy()
    if "sex_age" in entries.columns:
        entry_df["sex"] = entries["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entries["sex_age"].str[1:], errors="coerce")
    else:
        entry_df["sex"] = ""
        entry_df["age"] = np.nan
    entry_df["weight_carried"] = pd.to_numeric(
        entries.get("weight_carried", pd.Series(dtype=str)), errors="coerce"
    )
    entry_df["father"]        = ""
    entry_df["mother_father"] = ""

    # オッズ列の取得
    if "odds" in entries.columns:
        odds_raw = pd.to_numeric(entries["odds"], errors="coerce")
        entry_df["odds"] = (
            odds_raw.values if odds_raw.dropna().min() < 15.0
            else pd.to_numeric(entries.get("last_3f", pd.Series(dtype=str)), errors="coerce").values
        )
    elif "last_3f" in entries.columns:
        entry_df["odds"] = pd.to_numeric(entries["last_3f"].values, errors="coerce")

    rcc = FeatureEngineer._race_name_to_class_code(race_name)
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
            race_class_code=rcc,
            venue_code=venue_code,
        )
    except Exception as e:
        logger.debug(f"skip {race_id}: {e}")
        return None

    if len(feat_df) < 2:
        return None

    # ── LTR スコア → temperature scaling → Plackett-Luce 確率 ──────
    X = (
        feat_df[ltr.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    raw_scores   = ltr.predict(X)
    temperature  = getattr(ltr, "temperature", 1.0)
    model_probs  = LTRTrainer.scores_to_probs(raw_scores, temperature=temperature)

    # ── 市場確率（単勝オッズから） ──────────────────────────────────
    horse_ids  = feat_df["horse_id"].astype(str).tolist()
    horse_nums = [str(int(float(n))) for n in feat_df["horse_number"].tolist()]

    odds_map: dict[str, float] = {}
    if "odds" in feat_df.columns:
        odds_map = {
            str(k): v
            for k, v in feat_df.set_index("horse_id")["odds"].dropna().to_dict().items()
        }

    odds_list = []
    for hid in horse_ids:
        o = float(odds_map.get(hid, float("nan")))
        odds_list.append(o if not np.isnan(o) and o > 1.0 else 5.0)
    mkt_probs = _market_probs(odds_list)

    # ── 実際の着順（top2 = 馬連ヒット判定）──────────────────────────
    actual = entries[["horse_id", "horse_number", "finish_position"]].copy()
    actual["pos"] = pd.to_numeric(
        actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    actual = actual.dropna(subset=["pos"]).sort_values("pos")
    if len(actual) < 2:
        return None
    top2_ids   = {str(hid) for hid in actual.iloc[:2]["horse_id"]}
    top3_nums  = [str(int(float(actual.iloc[i]["horse_number"]))) for i in range(min(3, len(actual)))]

    # ── 軸馬・パートナー選出 ─────────────────────────────────────────
    n_total    = len(horse_ids)
    sorted_idx = list(np.argsort(model_probs)[::-1])

    axis1 = sorted_idx[0]
    axis2 = sorted_idx[1] if n_total >= 2 else None
    partners  = sorted_idx[2 : 2 + partner_top_n]  # 3位以下の上位N頭

    # ── 軸馬のオッズを記録（感度分析用・フィルター前に取得）────────
    o_ax1 = float(odds_map.get(horse_ids[axis1], float("nan")))
    o_ax2 = float(odds_map.get(horse_ids[axis2], float("nan"))) if axis2 is not None else float("nan")

    # ── 軸馬の信頼性フィルター ────────────────────────────────────
    if axis_max_odds > 0:
        if (not np.isnan(o_ax1) and o_ax1 > axis_max_odds) or \
           (axis2 is not None and not np.isnan(o_ax2) and o_ax2 > axis_max_odds):
            return None  # 軸馬が低評価 → レース見送り

    # ── ペア候補列挙 ────────────────────────────────────────────────
    pairs: list[tuple[int, int]] = []
    if axis2 is not None:
        pairs.append((axis1, axis2))
    for p_idx in partners:
        pairs.append((axis1, p_idx))
        if axis2 is not None:
            pairs.append((axis2, p_idx))

    # ── EV 計算・フィルタリング ──────────────────────────────────────
    bets: list[dict] = []

    for i, j in pairs:
        if horse_ids[i] == horse_ids[j]:
            continue

        hid_i = horse_ids[i]
        hid_j = horse_ids[j]

        o_i = float(odds_map.get(hid_i, float("nan")))
        o_j = float(odds_map.get(hid_j, float("nan")))

        # ロングショット除外
        if (not np.isnan(o_i) and o_i > LONGSHOT_ODDS_MAX) or \
           (not np.isnan(o_j) and o_j > LONGSHOT_ODDS_MAX):
            continue

        p_model_ij  = prob_quinella(model_probs, i, j)
        p_market_ij = prob_quinella(mkt_probs,   i, j)
        est_odds_ij = est_quinella_odds_fn(p_market_ij)

        if est_odds_ij > EST_QUINELLA_ODDS_MAX:
            continue

        ev = p_model_ij * est_odds_ij
        if ev < ev_threshold:
            continue

        # 馬連ヒット: 1着・2着を両方的中
        hit = (hid_i in top2_ids) and (hid_j in top2_ids)
        ret = round(est_odds_ij * 100.0, 1) if hit else 0.0

        bets.append({
            "race_id":     race_id,
            "race_date":   str(race_date.date()) if hasattr(race_date, "date") else str(race_date),
            "race_name":   race_name,
            "year":        int(str(race_id)[:4]),
            "race_month":  race_date.month if hasattr(race_date, "month") else 0,
            "course_type": course_type,
            "distance":    distance,
            "n_entries":   n_total,
            "horse_num_i": horse_nums[i],
            "horse_num_j": horse_nums[j],
            "horse_id_i":  hid_i,
            "horse_id_j":  hid_j,
            "odds_i":      round(o_i, 1) if not np.isnan(o_i) else None,
            "odds_j":      round(o_j, 1) if not np.isnan(o_j) else None,
            "p_model":     round(p_model_ij,  4),
            "p_market":    round(p_market_ij, 4),
            "ev":          round(ev, 3),
            "est_odds":    round(est_odds_ij, 1),
            "cost":        100,
            "ret":         ret,
            "hit":         int(hit),
            "actual_1st":  top3_nums[0] if top3_nums else "",
            "actual_2nd":  top3_nums[1] if len(top3_nums) > 1 else "",
            "actual_3rd":  top3_nums[2] if len(top3_nums) > 2 else "",
            "is_axis_pair":  int(i == axis1 and j == axis2),
            "axis1_odds":    round(o_ax1, 1) if not np.isnan(o_ax1) else None,
            "axis2_odds":    round(o_ax2, 1) if not np.isnan(o_ax2) else None,
        })

    if not bets:
        return None

    bets.sort(key=lambda x: -x["ev"])
    return bets[:max_bets]


# ── 年別バックテスト ────────────────────────────────────────────────

def _run_year(
    year:          int,
    res_df:        pd.DataFrame,
    meta_df:       pd.DataFrame,
    ltr:           LTRTrainer,
    ev_threshold:  float,
    max_bets:      int,
    partner_top_n: int,
    axis_max_odds: float = 0.0,
) -> list[dict]:
    logger.info("=" * 60)
    logger.info(f"{year}年バックテスト（Walk-Forward / 馬連軸馬流し）")
    logger.info("=" * 60)

    year_meta = meta_df[meta_df["race_id"].str.startswith(str(year))].copy()
    year_meta["race_date"] = pd.to_datetime(year_meta["race_date"], errors="coerce")
    year_meta = year_meta.dropna(subset=["race_date"]).sort_values("race_date")

    race_dates = sorted(year_meta["race_date"].unique())
    logger.info(
        f"対象: {year_meta['race_id'].nunique():,}R  "
        f"({race_dates[0].date()} 〜 {race_dates[-1].date()})"
    )

    # race_date → race_id のマップを事前構築（高速化）
    meta_date_map = meta_df.set_index("race_id")["race_date"].to_dict()

    all_records: list[dict] = []
    t0 = time.time()

    for di, race_date in enumerate(race_dates):
        # Walk-Forward: 当日以前のデータで FeatureEngineer を構築
        history_before = res_df[
            pd.to_datetime(
                res_df["race_id"].map(meta_date_map), errors="coerce"
            ) < race_date
        ].copy()

        if history_before.empty:
            logger.debug(f"  {race_date.date()}: 履歴データなし → スキップ")
            continue

        fe = FeatureEngineer(history_before)
        fe.precompute_aggregations()

        day_meta = year_meta[year_meta["race_date"] == race_date]
        day_buys = 0

        for _, meta_row in day_meta.iterrows():
            race_id = meta_row["race_id"]
            entries = res_df[res_df["race_id"] == race_id]
            if entries.empty:
                continue
            entries = entries.drop_duplicates(subset=["horse_id"])
            bets = _process_race(
                race_id, entries, meta_row, fe, ltr,
                ev_threshold, max_bets, partner_top_n, axis_max_odds,
            )
            if bets:
                all_records.extend(bets)
                day_buys += len(bets)

        if (di + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(
                f"  {di+1}/{len(race_dates)} 日付済み  "
                f"累計{len(all_records)}点  ({elapsed:.1f}分)"
            )

    elapsed_total = (time.time() - t0) / 60
    n_races_with = len({r["race_id"] for r in all_records})
    logger.info(
        f"{year}年完了: {len(all_records)}点  {n_races_with}レース  ({elapsed_total:.1f}分)"
    )
    return all_records


# ── ROI サマリー ────────────────────────────────────────────────────

def _print_axis_sensitivity(df: pd.DataFrame) -> None:
    """
    axis_max_odds 感度分析：フィルター閾値別の「健全性指標」を出力する。

    注：全期間が学習データに含まれるため ROI 絶対値ではなく
    ① 的中率（target: ~10%前後）
    ② 買い目点数の安定性
    ③ 軸馬オッズ分布
    を重視してレポートする。
    """
    axis_thresholds = [0.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    ev_thr  = 1.2
    max_b   = 5

    logger.info("")
    logger.info("=" * 78)
    logger.info("【軸馬フィルター（axis_max_odds）感度分析】")
    logger.info(f"  固定条件: EV≥{ev_thr}  top{max_b}点/R")
    logger.info("=" * 78)
    logger.info(
        f"  {'axis_max_odds':<16} {'R数':>5} {'点数':>6} {'的中率':>7} "
        f"{'ROI':>8}  {'軸馬avg_odds':>12}  {'軸馬>閾値で除外R':>14}"
    )
    logger.info("  " + "-" * 76)

    no_filter = df[df["ev"] >= ev_thr].copy()
    base_races = no_filter["race_id"].nunique()

    for thr in axis_thresholds:
        sub = no_filter.copy()
        if thr > 0:
            ax_ok = (
                (sub["axis1_odds"].fillna(0) <= thr) &
                (sub["axis2_odds"].fillna(0) <= thr)
            )
            sub = sub[ax_ok]

        if sub.empty:
            logger.info(f"  {'無制限' if thr==0 else f'≤{thr:.0f}倍':<16}  -- (買い目なし)")
            continue

        # top5/R に絞る
        sub = (
            sub.sort_values("ev", ascending=False)
            .groupby("race_id", sort=False)
            .head(max_b)
        )
        n_races  = sub["race_id"].nunique()
        n_bets   = len(sub)
        n_hits   = int(sub["hit"].sum())
        cost     = n_bets * 100
        ret      = float(sub["ret"].sum())
        hr       = n_hits / n_bets * 100
        roi      = ret / cost * 100
        excluded = base_races - n_races
        # 軸馬オッズの平均（フィルター後）
        ax_avg   = sub["axis1_odds"].dropna().mean()

        label = "無制限（参考）" if thr == 0 else f"≤{thr:.0f}倍"
        logger.info(
            f"  {label:<16} {n_races:>5} {n_bets:>6} {hr:>6.1f}%  "
            f"{roi:>7.1f}%  {ax_avg:>12.1f}倍  {excluded:>14}R除外"
        )

    # 軸馬オッズ分布
    logger.info("")
    logger.info("  ── 軸馬オッズ分布（axis1 / EV≥1.2 全レース）──")
    ax1 = no_filter.drop_duplicates("race_id")["axis1_odds"].dropna()
    for q_label, q_val in [("中央値", 0.5), ("75%ile", 0.75), ("90%ile", 0.90), ("95%ile", 0.95), ("最大", 1.0)]:
        v = ax1.quantile(q_val) if q_val < 1.0 else ax1.max()
        logger.info(f"    {q_label:<8}: {v:.1f}倍")
    logger.info(
        f"    10倍以下  : {(ax1 <= 10).mean()*100:.1f}%  "
        f"({(ax1 <= 10).sum():,}R / {len(ax1):,}R)"
    )
    logger.info("=" * 78)


def _print_summary(df: pd.DataFrame) -> None:
    """EV閾値 × max_bets のマトリクスでROIを出力する。"""
    ev_thresholds = (1.0, 1.2, 1.5, 2.0)
    max_bets_list = (5, 3, 0)

    def _sim(data: pd.DataFrame, thr: float, max_b: int) -> dict:
        sub = data[data["ev"] >= thr].copy()
        if sub.empty:
            return {"n_races": 0, "n_bets": 0, "n_hits": 0, "cost": 0, "ret": 0.0}
        if max_b > 0:
            sub = (
                sub.sort_values("ev", ascending=False)
                .groupby("race_id", sort=False)
                .head(max_b)
            )
        return {
            "n_races": int(sub["race_id"].nunique()),
            "n_bets":  len(sub),
            "n_hits":  int(sub["hit"].sum()),
            "cost":    len(sub) * 100,
            "ret":     float(sub["ret"].sum()),
        }

    def _row(label: str, s: dict) -> str:
        if s["n_bets"] == 0:
            return f"  {label:<26}  -- (買い目なし)"
        hr  = s["n_hits"] / s["n_bets"] * 100
        roi = s["ret"]    / s["cost"]   * 100
        return (
            f"  {label:<26} {s['n_races']:>5} {s['n_bets']:>6} "
            f"{hr:>6.1f}%  {roi:>7.1f}%  "
            f"¥{s['cost']:>8,}  ¥{int(s['ret']):>9,}"
        )

    years = sorted(df["year"].unique())
    hdr   = f"  {'条件':<26} {'R数':>5} {'点数':>6} {'的中率':>7} {'ROI':>8}  {'投資額':>9}  {'払戻額':>10}"

    sections: list[tuple[str, pd.DataFrame]] = []
    for y in years:
        oos = " ★真OOS" if y == 2026 else ""
        sections.append((f"{y}年{oos}", df[df["year"] == y]))
    sections.append((f"{min(years)}-{max(years)} 全期間", df))

    for sec_label, sec_df in sections:
        logger.info("")
        logger.info("=" * 78)
        logger.info(f"【馬連LTR バックテスト: {sec_label}】")
        logger.info("=" * 78)
        logger.info(hdr)
        logger.info("  " + "-" * 76)
        for thr in ev_thresholds:
            for mb in max_bets_list:
                mb_label = f"top{mb}" if mb > 0 else "全点"
                label = f"EV≥{thr:.1f} {mb_label}"
                s = _sim(sec_df, thr, mb)
                logger.info(_row(label, s))
            logger.info("  " + "·" * 60)

        # 芝/ダート別（EV≥1.5, top5 推奨設定）
        logger.info(f"\n  ── コース別（EV≥1.5, top5）──")
        for ct in ["芝", "ダート"]:
            s = _sim(sec_df[sec_df["course_type"] == ct], 1.5, 5)
            logger.info(_row(f"{ct} EV≥1.5 top5", s))

    logger.info("=" * 78)


# ── メインエントリー ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LTR + EV 馬連 軸馬流し バックテスト")
    parser.add_argument("--year",          type=int, default=None)
    parser.add_argument("--ev-threshold",  type=float, default=DEFAULT_EV_THRESHOLD)
    parser.add_argument("--max-bets",      type=int,   default=DEFAULT_MAX_BETS)
    parser.add_argument("--partner-top-n", type=int,   default=DEFAULT_PARTNER_TOP_N)
    parser.add_argument("--axis-max-odds", type=float, default=DEFAULT_AXIS_MAX_ODDS,
                        help="軸馬オッズ上限（0.0=無効, 10.0=推奨）")
    parser.add_argument("--ltr-model",     default="data/models/lgbm_ltr_model.pkl")
    parser.add_argument("--train-results", default="data/raw/train_checkpoint_new_results.csv")
    parser.add_argument("--train-meta",    default="data/raw/train_checkpoint_new_meta.csv")
    parser.add_argument("--test-results",  default="data/raw/test_results_new.csv")
    parser.add_argument("--test-meta",     default="data/raw/test_meta_new.csv")
    args = parser.parse_args()

    t_global = time.time()

    # ── LTR モデル読み込み ──────────────────────────────────────────
    ltr_path = Path(args.ltr_model)
    if not ltr_path.exists():
        logger.error(f"LTR モデルが見つかりません: {ltr_path}")
        sys.exit(1)
    ltr = LTRTrainer.load(ltr_path)

    temperature = getattr(ltr, "temperature", 1.0)
    logger.info("=" * 60)
    logger.info("LTR + EV 馬連 軸馬流し バックテスト")
    logger.info(f"  EV 閾値     : {args.ev_threshold}")
    logger.info(f"  partner_top_n: {args.partner_top_n}")
    logger.info(f"  temperature : {temperature:.4f} (モデル保存値)")
    logger.info(f"  axis_max_odds: {args.axis_max_odds} ({'有効' if args.axis_max_odds > 0 else '無効（全レース対象）'})")
    logger.info(f"  LTR モデル  : {ltr_path}  NDCG@3={ltr.oof_ndcg3:.4f}")
    logger.info("=" * 60)

    # ── データ読み込み ──────────────────────────────────────────────
    train_res  = pd.read_csv(Path(args.train_results), dtype=str)
    train_meta = pd.read_csv(Path(args.train_meta),    dtype=str)
    train_meta["race_date"] = pd.to_datetime(train_meta["race_date"], errors="coerce")

    test_res  = pd.read_csv(Path(args.test_results),  dtype=str)
    test_meta = pd.read_csv(Path(args.test_meta),     dtype=str)
    test_meta["race_date"] = pd.to_datetime(test_meta["race_date"], errors="coerce")
    logger.info(
        f"訓練: {len(train_res):,}行 {train_meta['race_id'].nunique():,}R  "
        f"テスト: {len(test_res):,}行 {test_meta['race_id'].nunique():,}R"
    )

    target_years = [args.year] if args.year else [2024, 2025, 2026]
    all_records: list[dict] = []

    for year in target_years:
        if year == 2024:
            res_df  = train_res
            meta_df = train_meta
        else:
            res_df  = pd.concat([train_res, test_res],   ignore_index=True)
            meta_df = pd.concat([train_meta, test_meta], ignore_index=True)
            meta_df["race_date"] = pd.to_datetime(meta_df["race_date"], errors="coerce")

        records = _run_year(
            year, res_df, meta_df, ltr,
            args.ev_threshold, args.max_bets, args.partner_top_n, args.axis_max_odds,
        )
        all_records.extend(records)

    if not all_records:
        logger.warning("買い目が一件もありませんでした。EV閾値を下げてください。")
        return

    result_df = pd.DataFrame(all_records)

    # ── CSV 保存 ────────────────────────────────────────────────────
    out_path = Path("data/processed/backtest_quinella_ltr.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.info(f"CSV保存: {out_path}  ({len(result_df):,}点)")

    # ── ROI サマリー出力 ────────────────────────────────────────────
    _print_summary(result_df)

    # ── 軸馬フィルター感度分析 ──────────────────────────────────────
    _print_axis_sensitivity(result_df)

    total = (time.time() - t_global) / 60
    logger.info(f"\n総実行時間: {total:.1f}分")


if __name__ == "__main__":
    main()
