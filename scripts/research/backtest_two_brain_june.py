"""
backtest_two_brain_june.py
===========================
本番パイプライン（src.betting.ltr_ev_engine.evaluate_race）を直接経由した
Two-Brain System（LTR + PairCalibrator + Gatekeeper）の検証用バックテスト。

config/strategy.yaml の現在値（EV>=1.2, est_quinella_odds_max=150.0,
gatekeeper_threshold=0.50 等）をそのまま読み込み、daily_batch.process_race()
と同じ呼び出し方で evaluate_race() を実行する。

対象: 2026年6月のダートレース（pre-scraped 結果データが揃っている範囲）

実行:
    .venv/bin/python backtest_two_brain_june.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import LTRTrainer
from src.model.gatekeeper import GatekeeperTrainer
from src.model.pair_calibrator import PairCalibrator
from src.betting.ltr_ev_engine import evaluate_race
from config.settings import StrategyConfig

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

START_DATE = pd.Timestamp("2026-06-01")
END_DATE   = pd.Timestamp("2026-06-21")


def load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train_checkpoint_new（〜2025-04）+ test_meta/test_results（2025-01〜直近）を結合。"""
    train_res  = pd.read_csv(ROOT / "data/raw/train_checkpoint_new_results.csv", dtype=str)
    train_meta = pd.read_csv(ROOT / "data/raw/train_checkpoint_new_meta.csv",    dtype=str)
    test_res   = pd.read_csv(ROOT / "data/raw/test_results.csv", dtype=str)
    test_meta  = pd.read_csv(ROOT / "data/raw/test_meta.csv",    dtype=str)

    res  = pd.concat([train_res, test_res], ignore_index=True).drop_duplicates(
        subset=["race_id", "horse_id"], keep="last"
    )
    meta = pd.concat([train_meta, test_meta], ignore_index=True).drop_duplicates(
        subset=["race_id"], keep="last"
    )
    meta["race_date"] = pd.to_datetime(meta["race_date"], errors="coerce")
    return res, meta


def build_entry_df(entries: pd.DataFrame) -> pd.DataFrame:
    entry_df = entries[["horse_id", "horse_name", "horse_number", "frame_number", "jockey_id"]].copy()
    if "sex_age" in entries.columns:
        entry_df["sex"] = entries["sex_age"].str[0]
        entry_df["age"] = pd.to_numeric(entries["sex_age"].str[1:], errors="coerce")
    else:
        entry_df["sex"] = ""
        entry_df["age"] = np.nan
    entry_df["weight_carried"] = pd.to_numeric(
        entries.get("weight_carried", pd.Series(dtype=str)), errors="coerce"
    )
    entry_df["trainer_name"]  = entries.get("trainer_name", "")
    entry_df["father"]        = ""
    entry_df["mother_father"] = ""
    entry_df["odds"] = pd.to_numeric(entries.get("odds", pd.Series(dtype=str)), errors="coerce")
    return entry_df


def process_race(
    race_id: str,
    entries: pd.DataFrame,
    meta_row: pd.Series,
    fe: FeatureEngineer,
    ltr: LTRTrainer,
    cfg,
    pair_calibrator: PairCalibrator,
    gatekeeper: GatekeeperTrainer,
) -> pd.DataFrame | None:
    race_date   = meta_row["race_date"]
    course_type = str(meta_row.get("course_type", ""))
    race_name   = str(meta_row.get("race_name", ""))
    distance    = int(meta_row.get("distance", 0) or 0)
    gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
    wx_code     = int(meta_row.get("weather_code", -1) or -1)

    if course_type not in cfg.target_surface or distance == 0 or distance >= 2750:
        return None
    if len(entries) < 3:
        return None

    entry_df = build_entry_df(entries)
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
            race_date=race_date,
        )
    except Exception as e:
        logger.debug(f"skip {race_id}: {e}")
        return None

    if len(feat_df) < 3:
        return None

    odds_map: dict[str, float] = {}
    horse_names: dict[str, str] = {}
    for _, e in entries.iterrows():
        hid = str(e["horse_id"])
        horse_names[hid] = str(e["horse_name"])
        try:
            o = float(str(e["odds"]).replace(",", "").strip())
            if o > 1.0:
                odds_map[hid] = o
        except (ValueError, TypeError):
            pass

    # ── 実際の着順 ──────────────────────────────────────────────
    actual = entries[["horse_id", "horse_number", "finish_position"]].copy()
    actual["pos"] = pd.to_numeric(
        actual["finish_position"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    actual = actual.dropna(subset=["pos"]).sort_values("pos")
    if len(actual) < 2:
        return None
    pos_map = {str(hid): int(p) for hid, p in zip(actual["horse_id"], actual["pos"])}
    top2_ids = {str(hid) for hid in actual.iloc[:2]["horse_id"]}

    bets, candidates_df = evaluate_race(
        feat_df, odds_map, horse_names, ltr, cfg,
        return_candidates=True,
        pair_calibrator=pair_calibrator,
        gatekeeper=gatekeeper,
        gatekeeper_threshold=getattr(cfg, "gatekeeper_threshold", 0.50),
    )
    if candidates_df.empty:
        return None

    candidates_df.insert(0, "race_id", race_id)
    candidates_df.insert(1, "race_name", race_name)
    candidates_df.insert(2, "race_date", str(race_date.date()))
    candidates_df.insert(3, "course_type", course_type)
    candidates_df["pos_i"] = candidates_df["horse_id_i"].map(pos_map)
    candidates_df["pos_j"] = candidates_df["horse_id_j"].map(pos_map)
    candidates_df["hit"]   = (
        candidates_df["horse_id_i"].isin(top2_ids) & candidates_df["horse_id_j"].isin(top2_ids)
    ).astype(int)
    candidates_df["payout"] = np.where(
        candidates_df["hit"] == 1, candidates_df["est_quinella_odds"] * 100.0, 0.0
    )
    return candidates_df


def main() -> None:
    cfg = StrategyConfig.load()
    ltr = LTRTrainer.load(cfg.ltr_model_path)
    gatekeeper = GatekeeperTrainer.load(cfg.gatekeeper_model_path)
    pair_calibrator = PairCalibrator.load(cfg.pair_calibrator_path)

    logger.info("=" * 70)
    logger.info("Two-Brain System 本番パイプライン経由バックテスト（2026年6月）")
    logger.info(f"  min_ev_threshold     = {cfg.betting.min_ev_threshold}")
    logger.info(f"  est_quinella_odds_max= {cfg.betting.est_quinella_odds_max}")
    logger.info(f"  gatekeeper_threshold = {cfg.betting.gatekeeper_threshold}")
    logger.info(f"  min_p_model_threshold= {cfg.betting.min_p_model_threshold}")
    logger.info(f"  axis_max_odds        = {cfg.betting.axis_max_odds}")
    logger.info("=" * 70)

    res, meta = load_history()

    target_meta = meta[
        (meta["race_date"] >= START_DATE) & (meta["race_date"] <= END_DATE)
        & (meta["course_type"].isin(cfg.betting.target_surface))
    ].copy().sort_values("race_date")

    if target_meta.empty:
        logger.error("対象期間のダートレースが見つかりません。")
        return

    avail_dates = sorted(target_meta["race_date"].dt.date.unique())
    logger.info(f"対象ダートレース: {target_meta['race_id'].nunique()}R  開催日: {avail_dates}")
    if avail_dates and avail_dates[-1] < END_DATE.date():
        logger.warning(
            f"⚠ pre-scraped 結果データは {avail_dates[-1]} までしかありません"
            f"（{END_DATE.date()}までの完全な期間データは未取得）。取得済み範囲で実行します。"
        )

    meta_date_map = meta.set_index("race_id")["race_date"].to_dict()
    all_candidates: list[pd.DataFrame] = []

    for race_date in sorted(target_meta["race_date"].unique()):
        history_before = res[
            pd.to_datetime(res["race_id"].map(meta_date_map), errors="coerce") < race_date
        ]
        if history_before.empty:
            continue
        fe = FeatureEngineer(history_before)
        fe.precompute_aggregations()

        day_meta = target_meta[target_meta["race_date"] == race_date]
        for _, meta_row in day_meta.iterrows():
            race_id = meta_row["race_id"]
            entries = res[res["race_id"] == race_id].drop_duplicates(subset=["horse_id"])
            if entries.empty:
                continue
            cdf = process_race(race_id, entries, meta_row, fe, ltr, cfg.betting,
                                pair_calibrator, gatekeeper)
            if cdf is not None:
                all_candidates.append(cdf)

    if not all_candidates:
        logger.warning("候補ペアが一件も生成されませんでした。")
        return

    cand = pd.concat(all_candidates, ignore_index=True)
    out_path = ROOT / "data/processed/backtest_two_brain_june_candidates.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out_path, index=False)
    logger.info(f"候補ペア全件 CSV保存: {out_path}  ({len(cand):,}件)")

    _print_report(cand, target_meta, cfg.betting.gatekeeper_threshold)


def _print_report(cand: pd.DataFrame, target_meta: pd.DataFrame, gk_threshold: float = 0.50) -> None:
    n_target_races = target_meta["race_id"].nunique()
    n_days = target_meta["race_date"].dt.date.nunique()

    buy = cand[cand["would_buy"] == True].copy()  # noqa: E712

    logger.info("")
    logger.info("=" * 70)
    logger.info("【1. 全体サマリー】")
    logger.info("=" * 70)
    logger.info(f"  対象ダートレース数        : {n_target_races}R（{n_days}日間）")
    if buy.empty:
        logger.info("  買い目が出力されたレース数: 0R")
        logger.info("  総購入点数               : 0点")
    else:
        n_buy_races = buy["race_id"].nunique()
        n_bets = len(buy)
        n_hit_bets = int(buy["hit"].sum())
        hit_race = buy.groupby("race_id")["hit"].max()
        n_hit_races = int(hit_race.sum())
        cost = n_bets * 100
        payout = float(buy["payout"].sum())
        roi = payout / cost * 100 if cost else 0.0
        logger.info(f"  買い目が出力されたレース数: {n_buy_races}R / {n_target_races}R "
                    f"({n_buy_races/n_target_races*100:.1f}%)")
        logger.info(f"  総購入点数               : {n_bets}点（1日あたり平均 {n_bets/n_days:.1f}点）")
        logger.info(f"  的中点数 / 的中率（点）   : {n_hit_bets}点 / {n_hit_bets/n_bets*100:.1f}%")
        logger.info(f"  的中レース数 / 的中率（R）: {n_hit_races}R / {n_hit_races/n_buy_races*100:.1f}%")
        logger.info(f"  投資額 / 払戻額          : ¥{cost:,} / ¥{int(payout):,}")
        logger.info(f"  想定ROI（均等買い）       : {roi:.1f}%")

    logger.info("")
    logger.info("=" * 70)
    logger.info("【2-1. Gatekeeperの働き（軸馬棄却の検証）】")
    logger.info("=" * 70)
    # evaluate_race() のペア列挙順は必ず (axis1, axis2) が先頭行になるため、
    # 各レースの先頭候補行から axis1=horse_id_i / axis2=horse_id_j の
    # finish_pos・p_axis_safe を直接取得できる。
    rej_records = []
    for race_id, g in cand.groupby("race_id", sort=False):
        r0 = g.iloc[0]
        p1, p2 = r0["p_axis1_safe"], r0["p_axis2_safe"]
        if pd.notna(p1):
            rej_records.append({
                "race_id": race_id, "axis": "axis1", "p_safe": p1,
                "rejected": p1 < gk_threshold, "finish_pos": r0["pos_i"],
            })
        if pd.notna(p2):
            rej_records.append({
                "race_id": race_id, "axis": "axis2", "p_safe": p2,
                "rejected": p2 < gk_threshold, "finish_pos": r0["pos_j"],
            })
    rej_df = pd.DataFrame(rej_records)
    if not rej_df.empty:
        rejected_df = rej_df[rej_df["rejected"] == True]  # noqa: E712
        n_rejected = len(rejected_df)
        n_total_axes = len(rej_df)
        avoided = rejected_df[rejected_df["finish_pos"] > 3]
        n_avoided_correct = len(avoided)
        logger.info(f"  LTR選出軸馬（axis1/axis2）総数     : {n_total_axes}件")
        logger.info(f"  Gatekeeperにより棄却された軸馬数   : {n_rejected}件 "
                    f"({n_rejected/n_total_axes*100:.1f}%)")
        if n_rejected:
            logger.info(f"  棄却馬のうち実際に4着以下だった数  : {n_avoided_correct}件 "
                        f"（回避成功率 {n_avoided_correct/n_rejected*100:.1f}%）")
            missed = rejected_df[rejected_df["finish_pos"] <= 3]
            logger.info(f"  棄却馬のうち実際は3着以内だった数  : {len(missed)}件 "
                        f"（誤棄却 = 取りこぼし）")
        else:
            logger.info("  棄却された軸馬はありませんでした。")
    else:
        logger.info("  Gatekeeperの判定データがありません（モデル未読込 or 候補0件）。")

    logger.info("")
    logger.info("=" * 70)
    logger.info("【2-2. バリューゾーンの捕捉（推定オッズ 50〜150倍）】")
    logger.info("=" * 70)
    if buy.empty:
        logger.info("  買い目がないため集計不可。")
    else:
        vz = buy[(buy["est_quinella_odds"] > 50.0) & (buy["est_quinella_odds"] <= 150.0)]
        logger.info(f"  50〜150倍レンジで抽出された買い目数: {len(vz)}点")
        if len(vz):
            vz_hits = vz[vz["hit"] == 1]
            logger.info(f"  うち的中点数                      : {len(vz_hits)}点")
            if len(vz_hits):
                for _, row in vz_hits.iterrows():
                    logger.info(
                        f"    - {row['race_date']} {row['race_name']} "
                        f"({row['race_id']}): {row['horse_name_i']}-{row['horse_name_j']} "
                        f"推定オッズ{row['est_quinella_odds']:.1f}倍 → payout ¥{int(row['payout']):,}"
                    )
            else:
                logger.info("  的中レースはありませんでした。")
            logger.info(f"  50〜150倍レンジの投資額/払戻額     : "
                        f"¥{len(vz)*100:,} / ¥{int(vz['payout'].sum()):,}")
        else:
            logger.info("  このレンジで購入した買い目はありませんでした。")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
