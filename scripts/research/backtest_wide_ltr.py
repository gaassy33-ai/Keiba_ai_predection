"""
backtest_wide_ltr.py
====================
LTR (LambdaRank) モデル + EV ベースのワイド買い目構築バックテスト。

【パラダイムシフト】
  旧: is_win モデル → ◎軸固定 → 確率順に紐選択
  新: LTR モデル → softmax 確率 → EV = P_model × 推定オッズ で全ペア評価
      → EV > 閾値 のペアのみ購入（過剰人気を自動排除）

【時系列リーク対策】
  全期間 Walk-Forward 方式:
  日付 T のレースを予測する際、feature_stats は T 以前のデータのみで計算。
  feature_stats.pkl は使用しない。

【買い目ルール】
  - 馬券種 : ワイド
  - レース内で C(N,2) 全ペアの EV を計算
  - EV > EV_THRESHOLD のペアを購入（上位 MAX_BETS_PER_RACE 点まで）
  - レース選択条件: なし（EV フィルタが自動的に不確実レースを排除）

実行例:
    .venv/bin/python backtest_wide_ltr.py              # 2024-2026 全期間
    .venv/bin/python backtest_wide_ltr.py --year 2025  # 単年
    .venv/bin/python backtest_wide_ltr.py --ev-threshold 1.1
"""
from __future__ import annotations

import argparse
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
from config.settings import settings

# ── ロガー ───────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout,                       level="INFO",  format=_fmt, colorize=True)
logger.add("logs/backtest_wide_ltr.log",     level="DEBUG", format=_fmt, rotation="30 MB")

# ── ワイド JRA 控除率 ────────────────────────────────────────────────
JRA_TAKE_WIDE = 0.225   # 22.5%

# ── デフォルトパラメータ ─────────────────────────────────────────────
# 注: 感度分析のため EV 閾値は低め (0.8) に設定し、全買い目候補を CSV 保存する。
# 実運用時の最適閾値は _print_summary() の出力から決定すること。
DEFAULT_EV_THRESHOLD  = 0.8
DEFAULT_MAX_BETS      = 20      # 感度分析用: 1レース最大20点まで保存
DEFAULT_TEMPERATURE   = 1.0     # softmax 温度（1.0=変換なし）
LONGSHOT_ODDS_MAX     = 30.0    # 単勝オッズ30倍超の馬はペアから除外

# 推定ワイドオッズ上限（これを超えるペアは計算ノイズとして除外）
# P_market が極端に小さい場合 est_odds が爆発する問題への対処。
# 実際の JRA ワイドオッズは通常 100 倍以下で推移する。
EST_WIDE_ODDS_MAX     = 100.0

# ── ユーティリティ ───────────────────────────────────────────────────

def _parse_odds(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")

def _market_probs(odds_list: list[float]) -> np.ndarray:
    """単勝オッズリスト → 市場確率ベクトル（合計 ≈ 1.0）"""
    raw = np.array([1.0 / max(o, 1.01) for o in odds_list])
    s = raw.sum()
    return raw / s if s > 0 else raw

def _harville(probs: np.ndarray | list[float], order: list[int]) -> float:
    """Harville 公式: 指定順序での着順確率"""
    p, rem = 1.0, 1.0
    for idx in order:
        if rem < 1e-9:
            return 0.0
        p *= probs[idx] / rem
        rem -= probs[idx]
    return p

def _prob_trio(probs, i: int, j: int, k: int) -> float:
    """3連複確率: P(i, j, k が top3 に入る) using Harville"""
    return sum(_harville(probs, list(o)) for o in _perm([i, j, k]))

def _prob_wide(probs, i: int, j: int) -> float:
    """
    ワイド確率: P(馬i と 馬j が両方 top3 に入る)
    = Σ_{k≠i,j} P_trio(i, j, k)
    """
    n = len(probs)
    total = 0.0
    for k in range(n):
        if k != i and k != j:
            total += _prob_trio(probs, i, j, k)
    return total

def _est_wide_odds(p_market: float) -> float:
    """市場確率 → 推定ワイドオッズ（JRA控除率 22.5% 反映）"""
    if p_market <= 1e-9:
        return 999.0
    return (1.0 - JRA_TAKE_WIDE) / p_market

def _compute_ev_wide(
    model_probs: np.ndarray,
    mkt_probs: np.ndarray,
    i: int,
    j: int,
) -> tuple[float, float, float]:
    """
    ワイド (i, j) の EV を計算する。

    Returns
    -------
    (p_model, p_market, ev) のタプル
      p_model  : モデルのワイド推定確率
      p_market : 市場のワイド推定確率（オッズベース）
      ev       : 期待値 = p_model × 推定オッズ
    """
    p_model  = _prob_wide(model_probs, i, j)
    p_market = _prob_wide(mkt_probs,   i, j)
    est_odds = _est_wide_odds(p_market)
    ev = p_model * est_odds
    return p_model, p_market, ev


# ── レース処理 ───────────────────────────────────────────────────────

def _process_race(
    race_id: str,
    entries: pd.DataFrame,
    meta_row: pd.Series,
    fe: FeatureEngineer,
    ltr: LTRTrainer,
    ev_threshold: float,
    max_bets: int,
    temperature: float,
) -> list[dict] | None:
    """
    1レースを処理し、EV > ev_threshold のワイド買い目リストを返す。
    買い目なし or 処理不可の場合は None。

    Returns
    -------
    list[dict] or None
      dict keys: race_id, race_date, race_name, year, race_month,
                 course_type, distance, n_entries,
                 horse_num_i, horse_num_j, horse_id_i, horse_id_j,
                 p_model, p_market, ev, est_odds,
                 cost, ret, hit,
                 actual_1st, actual_2nd, actual_3rd
    """
    race_date   = meta_row["race_date"]
    course_type = str(meta_row.get("course_type", ""))
    race_name   = str(meta_row.get("race_name", ""))
    distance    = int(meta_row.get("distance", 0) or 0)
    gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
    wx_code     = int(meta_row.get("weather_code", -1) or -1)

    if not course_type or distance == 0 or distance >= 2750:
        return None

    entry_df = entries[["horse_id", "horse_name", "horse_number",
                         "frame_number", "jockey_id"]].copy()
    if "sex_age" in entries.columns:
        entry_df["sex"] = entries["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entries["sex_age"].str[1:], errors="coerce")
    else:
        entry_df["sex"] = ""
        entry_df["age"] = np.nan
    entry_df["weight_carried"] = pd.to_numeric(
        entries.get("weight_carried", pd.Series()), errors="coerce"
    )
    entry_df["father"]        = ""
    entry_df["mother_father"] = ""

    if "odds" in entries.columns:
        odds_raw = pd.to_numeric(entries["odds"], errors="coerce")
        if odds_raw.dropna().min() < 15.0:
            entry_df["odds"] = odds_raw.values
        elif "last_3f" in entries.columns:
            entry_df["odds"] = pd.to_numeric(entries["last_3f"].values, errors="coerce")
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

    if len(feat_df) < 3:
        return None

    # ── LTR スコア → Plackett-Luce 確率 ────────────────────────────
    # ltr.feature_columns を使用（odds除外モデルでは odds_log 等が含まれない）
    X = (
        feat_df[ltr.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    raw_scores  = ltr.predict(X)
    model_probs = LTRTrainer.scores_to_probs(raw_scores, temperature=temperature)

    # ── 市場確率（単勝オッズから） ──────────────────────────────────
    horse_ids = feat_df["horse_id"].astype(str).tolist()
    horse_nums = [str(int(float(n))) for n in feat_df["horse_number"].tolist()]

    odds_map: dict[str, float] = {}
    if "odds" in feat_df.columns:
        # horse_id を文字列キーに統一（型不一致によるルックアップ失敗を防ぐ）
        odds_map = {
            str(k): v
            for k, v in feat_df.set_index("horse_id")["odds"].dropna().to_dict().items()
        }

    odds_list = []
    for hid in horse_ids:
        o = float(odds_map.get(hid, float("nan")))
        odds_list.append(o if not np.isnan(o) and o > 1.0 else 5.0)

    mkt_probs = _market_probs(odds_list)

    # ── 実際の着順 ──────────────────────────────────────────────────
    actual = entries[["horse_id", "horse_number", "finish_position"]].copy()
    actual["pos"] = pd.to_numeric(
        actual["finish_position"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    actual = actual.dropna(subset=["pos"]).sort_values("pos")
    if len(actual) < 3:
        return None
    top3_ids = set(str(hid) for hid in actual.iloc[:3]["horse_id"])
    top3_nums = [str(int(float(actual.iloc[i]["horse_number"]))) for i in range(3)]

    # ── 全ペア EV 計算 ──────────────────────────────────────────────
    n = len(horse_ids)
    bets: list[dict] = []

    for i, j in _comb(range(n), 2):
        # 自己ペア除外（horse_id 重複エントリーによる誤ペアを防ぐ）
        if horse_ids[i] == horse_ids[j]:
            continue

        # ロングショット除外（どちらか一方でも単勝30倍超なら除外）
        o_i = float(odds_map.get(horse_ids[i], float("nan")))
        o_j = float(odds_map.get(horse_ids[j], float("nan")))
        if (not np.isnan(o_i) and o_i > LONGSHOT_ODDS_MAX) or \
           (not np.isnan(o_j) and o_j > LONGSHOT_ODDS_MAX):
            continue  # 片方でもロングショットなら除外

        p_model, p_market, ev = _compute_ev_wide(model_probs, mkt_probs, i, j)

        # 推定ワイドオッズ上限フィルター（ノイズ除去）
        est_odds_check = _est_wide_odds(p_market)
        if est_odds_check > EST_WIDE_ODDS_MAX:
            continue   # P_market が極端に小さい → 計算ノイズ

        if ev < ev_threshold:
            continue

        est_odds = _est_wide_odds(p_market)
        # ワイドヒット: i も j も top3 に入っていれば的中
        hit = (horse_ids[i] in top3_ids) and (horse_ids[j] in top3_ids)
        ret  = round(est_odds * 100.0, 1) if hit else 0.0

        bets.append({
            "race_id":     race_id,
            "race_date":   str(race_date.date()) if hasattr(race_date, "date") else str(race_date),
            "race_name":   race_name,
            "year":        int(str(race_id)[:4]),
            "race_month":  race_date.month if hasattr(race_date, "month") else 0,
            "course_type": course_type,
            "distance":    distance,
            "n_entries":   n,
            "horse_num_i": horse_nums[i],
            "horse_num_j": horse_nums[j],
            "horse_id_i":  horse_ids[i],
            "horse_id_j":  horse_ids[j],
            "odds_i":      round(o_i, 1) if not np.isnan(o_i) else None,
            "odds_j":      round(o_j, 1) if not np.isnan(o_j) else None,
            "p_model":     round(p_model,  4),
            "p_market":    round(p_market, 4),
            "ev":          round(ev, 3),
            "est_odds":    round(est_odds, 1),
            "cost":        100,
            "ret":         ret,
            "hit":         int(hit),
            "actual_1st":  top3_nums[0],
            "actual_2nd":  top3_nums[1],
            "actual_3rd":  top3_nums[2],
        })

    if not bets:
        return None

    # EV 降順で上位 max_bets 点に絞る
    bets.sort(key=lambda x: -x["ev"])
    return bets[:max_bets]


# ── 年別バックテスト ────────────────────────────────────────────────

def _run_year(
    year: int,
    res_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    ltr: LTRTrainer,
    ev_threshold: float,
    max_bets: int,
    temperature: float,
) -> list[dict]:
    """
    指定年の全レースを Walk-Forward 方式で処理し、全買い目レコードを返す。
    """
    logger.info("=" * 60)
    logger.info(f"{year}年バックテスト（Walk-Forward 方式・時系列リークなし）")
    logger.info("=" * 60)

    year_meta = meta_df[meta_df["race_id"].str.startswith(str(year))].copy()
    year_meta["race_date"] = pd.to_datetime(year_meta["race_date"], errors="coerce")
    year_meta = year_meta.dropna(subset=["race_date"]).sort_values("race_date")

    race_dates = sorted(year_meta["race_date"].unique())
    logger.info(f"対象: {year_meta['race_id'].nunique():,}R  ({race_dates[0].date()} 〜 {race_dates[-1].date()})")

    all_records: list[dict] = []
    t0 = time.time()

    for di, race_date in enumerate(race_dates):
        # Walk-Forward: 当日以前のデータだけで統計を構築
        history_before = res_df[
            pd.to_datetime(
                res_df["race_id"].map(
                    meta_df.set_index("race_id")["race_date"].to_dict()
                ),
                errors="coerce",
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
            race_id  = meta_row["race_id"]
            entries  = res_df[res_df["race_id"] == race_id]
            if entries.empty:
                continue
            # 重複エントリー除外（train + test concat で同一 horse_id が2行入る場合）
            entries = entries.drop_duplicates(subset=["horse_id"])
            bets = _process_race(
                race_id, entries, meta_row, fe, ltr,
                ev_threshold, max_bets, temperature,
            )
            if bets:
                all_records.extend(bets)
                day_buys += len(bets)

        if (di + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            n_races  = len(all_records)
            logger.info(
                f"  {di+1}/{len(race_dates)} 日付済み  "
                f"買い目{n_races}点  ({elapsed:.1f}分)"
            )

    elapsed_total = (time.time() - t0) / 60
    n_races_with_bets = len({r["race_id"] for r in all_records})
    logger.info(
        f"{year}年完了: {len(all_records)}点  {n_races_with_bets}レース  ({elapsed_total:.1f}分)"
    )
    return all_records


# ── ROI サマリー ────────────────────────────────────────────────────

def _print_summary(
    df: pd.DataFrame,
    ev_thresholds: tuple[float, ...] = (1.0, 1.1, 1.2, 1.5),
    max_bets_list: tuple[int, ...] = (5, 0),   # 0 = 上限なし
) -> None:
    """
    EV閾値 × max_bets マトリクス形式のROIサマリーを表示する。

    【重要】df は EV 下限閾値 (0.8) で保存された全候補を含む。
    各閾値・max_bets の組み合わせで後付けシミュレーションを行う。

    【出力形式】
    1. 年別＋全期間: EV閾値×max_bets のサマリーブロック
    2. 2026年 詳細セクション（真のOOS）: R数・点数・的中率・ROI・投資額・払戻額
    3. 芝/ダート内訳（EV≥1.1, top5）
    """
    def _sim(data: pd.DataFrame, thr: float, max_b: int) -> dict:
        """EV >= thr かつ 1レース上位 max_b 点のシミュレーション結果"""
        sub = data[data["ev"] >= thr].copy()
        if sub.empty:
            return {"n_races": 0, "n_bets": 0, "n_hits": 0, "cost": 0, "ret": 0.0}
        if max_b > 0:
            # 同一 race_id 内で EV 降順 top max_b 点に絞る
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

    def _block(data: pd.DataFrame, mb: int) -> None:
        """指定 max_bets のEV閾値別テーブルを出力する"""
        mb_label = f"1R上位{mb}点" if mb > 0 else "上限なし（参考）"
        logger.info(f"\n  ── {mb_label} ──")
        logger.info(f"  {'EV閾値':<10} {'R数':>5} {'点数':>6} {'的中率':>7} {'ROI':>8}")
        logger.info("  " + "-" * 42)
        for thr in ev_thresholds:
            s = _sim(data, thr, mb)
            if s["n_bets"] == 0:
                logger.info(f"  EV≥{thr:<6.1f}  -- (買い目なし)")
                continue
            hr  = s["n_hits"] / s["n_bets"] * 100
            roi = s["ret"] / s["cost"] * 100
            logger.info(
                f"  EV≥{thr:<6.1f} {s['n_races']:>5} {s['n_bets']:>6} "
                f"{hr:>6.1f}%  {roi:>7.1f}%"
            )

    def _detail_row(label: str, s: dict) -> str:
        if s["n_bets"] == 0:
            return f"  {label:<24}  -- (買い目なし)"
        hr  = s["n_hits"] / s["n_bets"] * 100
        roi = s["ret"] / s["cost"] * 100
        return (
            f"  {label:<24} {s['n_races']:>5} {s['n_bets']:>6} {hr:>6.1f}%  "
            f"{roi:>7.1f}%  ¥{s['cost']:>8,}  ¥{int(s['ret']):>9,}"
        )

    years = sorted(df["year"].unique())

    # ── 全期間＋年別サマリーブロック ────────────────────────────────
    logger.info("")
    logger.info("=" * 75)
    logger.info("【ワイド LTR バックテスト サマリー】")
    logger.info("=" * 75)

    sections: list[tuple[str, pd.DataFrame]] = []
    for y in years:
        oos = " ★真OOS" if y == 2026 else ""
        sections.append((f"{y}年{oos}", df[df["year"] == y]))
    y_range = f"{min(years)}-{max(years)}"
    sections.append((f"{y_range} 全期間", df))

    for sec_label, sec_df in sections:
        logger.info(f"\n━━━ {sec_label} ━━━")
        for mb in max_bets_list:
            _block(sec_df, mb)

    # ── 2026年 詳細セクション（真のOOS）────────────────────────────
    if 2026 in years:
        df26 = df[df["year"] == 2026]
        logger.info("")
        logger.info("=" * 75)
        logger.info("【2026年 詳細レポート（真のOOSデータ）】")
        logger.info(
            f"  {'条件':<24} {'R数':>5} {'点数':>6} {'的中率':>7} "
            f"{'ROI':>8}  {'投資額':>9}  {'払戻額':>10}"
        )
        logger.info("=" * 75)

        for thr in ev_thresholds:
            for mb in max_bets_list:
                mb_label = f"top{mb}" if mb > 0 else "無制限"
                label    = f"EV≥{thr:.1f} {mb_label}"
                s = _sim(df26, thr, mb)
                logger.info(_detail_row(label, s))
            logger.info("  " + "-" * 73)

        # 芝/ダート別（EV≥1.1, top5）
        rec_thr = 1.1
        rec_mb  = 5
        logger.info(f"\n  ── 芝/ダート別（EV≥{rec_thr}, 1R上位{rec_mb}点） ──")
        for ct in ["芝", "ダート"]:
            sub_ct = df26[df26["course_type"] == ct]
            s = _sim(sub_ct, rec_thr, rec_mb)
            label = f"{ct} EV≥{rec_thr} top{rec_mb}"
            logger.info(_detail_row(label, s))

    logger.info("=" * 75)


# ── メインエントリー ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LTR + EV ワイド バックテスト")
    parser.add_argument("--year",          type=int, default=None,
                        help="対象年（省略時は 2024-2026 全期間）")
    parser.add_argument("--ev-threshold",  type=float, default=DEFAULT_EV_THRESHOLD,
                        help=f"EV 閾値（デフォルト: {DEFAULT_EV_THRESHOLD}）")
    parser.add_argument("--max-bets",      type=int,   default=DEFAULT_MAX_BETS,
                        help=f"1レース最大購入点数（デフォルト: {DEFAULT_MAX_BETS}）")
    parser.add_argument("--temperature",   type=float, default=DEFAULT_TEMPERATURE,
                        help=f"softmax 温度（デフォルト: {DEFAULT_TEMPERATURE}）")
    parser.add_argument("--ltr-model",     default=str(Path("data/models/lgbm_ltr_model.pkl")))
    parser.add_argument("--train-results", default="data/raw/train_results.csv")
    parser.add_argument("--train-meta",    default="data/raw/train_meta.csv")
    parser.add_argument("--test-results",  default="data/raw/test_results_new.csv")
    parser.add_argument("--test-meta",     default="data/raw/test_meta_new.csv")
    args = parser.parse_args()

    t_global = time.time()

    # ── LTR モデル読み込み ──────────────────────────────────────────
    ltr_path = Path(args.ltr_model)
    if not ltr_path.exists():
        logger.error(f"LTR モデルが見つかりません: {ltr_path}")
        logger.error("先に train_ltr_model.py を実行してください")
        sys.exit(1)
    ltr = LTRTrainer.load(ltr_path)

    logger.info("=" * 60)
    logger.info("LTR + EV ワイド バックテスト")
    logger.info(f"  EV 閾値     : {args.ev_threshold}")
    logger.info(f"  最大購入点数: {args.max_bets}")
    logger.info(f"  softmax 温度: {args.temperature}")
    logger.info(f"  LTR モデル  : {ltr_path}  NDCG@3={ltr.oof_ndcg3:.4f}")
    logger.info("=" * 60)

    # ── データ読み込み ──────────────────────────────────────────────
    train_res  = pd.read_csv(Path(args.train_results), dtype=str)
    train_meta = pd.read_csv(Path(args.train_meta),    dtype=str)
    train_meta["race_date"] = pd.to_datetime(train_meta["race_date"], errors="coerce")

    test_res: pd.DataFrame | None  = None
    test_meta: pd.DataFrame | None = None
    test_res_path  = Path(args.test_results)
    test_meta_path = Path(args.test_meta)
    if test_res_path.exists() and test_meta_path.exists():
        test_res  = pd.read_csv(test_res_path,  dtype=str)
        test_meta = pd.read_csv(test_meta_path, dtype=str)
        test_meta["race_date"] = pd.to_datetime(test_meta["race_date"], errors="coerce")
        logger.info(f"テストデータ: {len(test_res):,} 行  {test_meta['race_id'].nunique():,} レース")

    target_years = [args.year] if args.year else [2024, 2025, 2026]
    all_records: list[dict] = []

    for year in target_years:
        if year == 2024:
            res_df  = train_res
            meta_df = train_meta
        else:
            if test_res is None or test_meta is None:
                logger.warning(f"{year}年: テストデータなし → スキップ")
                continue
            # Walk-Forward: 2025/2026 も全履歴（train + test 当日以前）を使用
            res_df  = pd.concat([train_res, test_res],   ignore_index=True)
            meta_df = pd.concat([train_meta, test_meta], ignore_index=True)
            meta_df["race_date"] = pd.to_datetime(meta_df["race_date"], errors="coerce")

        records = _run_year(
            year, res_df, meta_df, ltr,
            args.ev_threshold, args.max_bets, args.temperature,
        )
        all_records.extend(records)

    if not all_records:
        logger.warning("買い目が一件もありませんでした。EV閾値を下げてください。")
        return

    result_df = pd.DataFrame(all_records)

    # ── CSV 保存 ────────────────────────────────────────────────────
    out_path = Path("data/processed/backtest_wide_ltr.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.info(f"CSV保存: {out_path}  ({len(result_df):,}点)")

    # ── ROI サマリー出力 ────────────────────────────────────────────
    _print_summary(result_df, ev_thresholds=(1.0, 1.1, 1.2, 1.5), max_bets_list=(5, 0))

    total = (time.time() - t_global) / 60
    logger.info(f"\n総実行時間: {total:.1f}分")


if __name__ == "__main__":
    main()
