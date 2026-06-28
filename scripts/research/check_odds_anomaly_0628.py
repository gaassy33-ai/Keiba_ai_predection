"""
check_odds_anomaly_0628.py
===========================
2026-06-28 の本番異常（43点／10R購入・0的中）のRoot Cause調査。

data/logs/predictions/2026-06-28.csv は odds_i/odds_j がほぼ全件欠損しており、
本番のオッズスクレイピングが失敗していた疑いがある。本スクリプトは
netkeibaの結果ページ（確定オッズ・全頭分）を再取得し、

  1. 本番ログ（QuinellaBet）には無い p_model_raw（補正前確率）を、
     当時と同じ FeatureEngineer / LTR / PairCalibrator パイプラインに通して
     復元し、PairCalibratorの圧縮効果を検証する
  2. 当日と同じパイプラインに、本番と同じ「オッズ欠損」状態を再現して通し、
     EV・Gatekeeperスコアが「実オッズが取れていた場合」とどう変化するかを比較する
  3. 軸馬（LTRのtop2）のGatekeeperスコア分布を、的中/不的中レース別に出す

実行:
    .venv/bin/python check_odds_anomaly_0628.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}
PREDICTIONS_CSV = ROOT / "data/logs/predictions/2026-06-28.csv"


def fetch_full_result(race_id: str) -> pd.DataFrame | None:
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.select_one("table.RaceTable01")
    if not table:
        return None

    rows = []
    for tr in table.select("tr.HorseList"):
        tds = tr.select("td")
        if len(tds) < 15:
            continue
        rank_text = tds[0].get_text(strip=True)
        try:
            rank = int(rank_text)
        except ValueError:
            rank = 99  # 中止・除外

        horse_link = tr.select_one("td a[href*='/horse/']")
        jockey_link = tr.select_one("td a[href*='/jockey/']")
        if not horse_link:
            continue

        horse_id = re.search(r"/horse/(\d+)", horse_link.get("href", "")).group(1)
        jockey_m = re.search(r"/jockey/(?:result/recent/)?(\d+)", jockey_link.get("href", "")) if jockey_link else None
        jockey_id = jockey_m.group(1) if jockey_m else ""

        sex_age = tds[4].get_text(strip=True)
        odds_text = tds[10].get_text(strip=True).replace(",", "")
        try:
            odds_val = float(odds_text)
        except ValueError:
            odds_val = float("nan")

        rows.append({
            "finish_position": rank,
            "frame_number": tds[1].get_text(strip=True),
            "horse_number": tds[2].get_text(strip=True),
            "horse_id": horse_id,
            "horse_name": horse_link.get_text(strip=True),
            "sex_age": sex_age,
            "weight_carried": tds[5].get_text(strip=True),
            "jockey_id": jockey_id,
            "trainer_name": tds[13].get_text(strip=True),
            "odds_final": odds_val,
        })
    return pd.DataFrame(rows)


def load_history() -> pd.DataFrame:
    train_res = pd.read_csv(ROOT / "data/raw/train_checkpoint_new_results.csv", dtype=str)
    test_res = pd.read_csv(ROOT / "data/raw/test_results.csv", dtype=str)
    return pd.concat([train_res, test_res], ignore_index=True).drop_duplicates(
        subset=["race_id", "horse_id"], keep="last"
    )


def build_entry_df(field: pd.DataFrame, use_real_odds: bool) -> pd.DataFrame:
    entry_df = field[["horse_id", "horse_name", "horse_number", "frame_number", "jockey_id"]].copy()
    entry_df["sex"] = field["sex_age"].str[0]
    entry_df["age"] = pd.to_numeric(field["sex_age"].str[1:], errors="coerce")
    entry_df["weight_carried"] = pd.to_numeric(field["weight_carried"], errors="coerce")
    entry_df["trainer_name"] = field["trainer_name"]
    entry_df["father"] = ""
    entry_df["mother_father"] = ""
    entry_df["odds"] = field["odds_final"] if use_real_odds else np.nan
    return entry_df


def main() -> None:
    cfg = StrategyConfig.load()
    ltr = LTRTrainer.load(cfg.ltr_model_path)
    gatekeeper = GatekeeperTrainer.load(cfg.gatekeeper_model_path)
    pair_calibrator = PairCalibrator.load(cfg.pair_calibrator_path)

    pred = pd.read_csv(PREDICTIONS_CSV, dtype={"race_id": str})
    race_meta = pred.groupby("race_id").agg(
        race_name=("race_name", "first"), course_type=("course_type", "first"),
        distance=("distance", "first"),
    ).reset_index()

    history = load_history()

    logger.info("=" * 78)
    logger.info("2026-06-28 本番異常調査: オッズ欠損 再現テスト")
    logger.info("=" * 78)

    rows = []
    bet_level_rows = []
    for _, meta_row in race_meta.iterrows():
        race_id = meta_row["race_id"]
        logger.info(f"処理中: {race_id} {meta_row['race_name']}")
        field = fetch_full_result(race_id)
        time.sleep(0.8)
        if field is None or len(field) < 3:
            logger.warning(f"  結果取得失敗: {race_id}")
            continue

        valid_field = field[field["finish_position"] < 90].copy()
        fe = FeatureEngineer(history[
            pd.to_numeric(history["race_id"], errors="coerce") < int(race_id)
        ])
        fe.precompute_aggregations()

        rcc = FeatureEngineer._race_name_to_class_code(meta_row["race_name"])
        try:
            venue_code = int(race_id[4:6])
        except Exception:
            venue_code = -1

        for cond_label, use_real_odds in [("実際の本番(オッズ欠損)", False), ("本来あるべき姿(実オッズ)", True)]:
            entry_df = build_entry_df(valid_field, use_real_odds)
            try:
                feat_df = fe.build_entry_features(
                    entry_df=entry_df, course_type=meta_row["course_type"],
                    distance=int(meta_row["distance"]), ground_condition_code=-1,
                    weather_code=-1, race_class_code=rcc, venue_code=venue_code,
                )
            except Exception as e:
                logger.warning(f"  特徴量生成失敗 {race_id} ({cond_label}): {e}")
                continue

            odds_map = {
                str(r["horse_id"]): float(r["odds_final"])
                for _, r in valid_field.iterrows() if use_real_odds and not np.isnan(r["odds_final"])
            }
            horse_names = {str(r["horse_id"]): str(r["horse_name"]) for _, r in valid_field.iterrows()}

            bets, candidates_df = evaluate_race(
                feat_df, odds_map, horse_names, ltr, cfg.betting,
                return_candidates=True, pair_calibrator=pair_calibrator,
                gatekeeper=gatekeeper, gatekeeper_threshold=cfg.betting.gatekeeper_threshold,
            )
            if candidates_df.empty:
                continue

            if cond_label == "実際の本番(オッズ欠損)":
                bought = (
                    candidates_df[candidates_df["would_buy"] == True]  # noqa: E712
                    .sort_values("ev", ascending=False)
                    .head(cfg.betting.max_bets_per_race)
                )
                for _, b in bought.iterrows():
                    bet_level_rows.append({
                        "race_id": race_id, "race_name": meta_row["race_name"],
                        "p_model_raw": b["p_model_raw"], "p_model_calibrated": b["p_model"],
                        "ev": b["ev"], "est_quinella_odds": b["est_quinella_odds"],
                    })

            r0 = candidates_df.iloc[0]
            axis1_id, axis2_id = r0["horse_id_i"], r0["horse_id_j"]
            pos_map = dict(zip(valid_field["horse_id"].astype(str), valid_field["finish_position"]))

            n_would_buy = int(candidates_df["would_buy"].sum())
            rows.append({
                "race_id": race_id, "race_name": meta_row["race_name"],
                "condition": cond_label, "n_horses": len(valid_field),
                "n_candidates_ev_pass": int(candidates_df["passed_ev"].sum()),
                "n_would_buy": n_would_buy,
                "axis1_name": horse_names.get(axis1_id), "axis1_pos": pos_map.get(axis1_id),
                "axis1_p_safe": r0["p_axis1_safe"],
                "axis2_name": horse_names.get(axis2_id), "axis2_pos": pos_map.get(axis2_id),
                "axis2_p_safe": r0["p_axis2_safe"],
                "p_model_raw_mean": candidates_df["p_model_raw"].mean(),
                "p_model_calibrated_mean": candidates_df["p_model"].mean(),
                "est_odds_mean": candidates_df["est_quinella_odds"].mean(),
            })

    result = pd.DataFrame(rows)
    out_path = ROOT / "data/processed/check_odds_anomaly_0628.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info(f"CSV保存: {out_path}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    logger.info("\n" + result.to_string(index=False))

    bet_df = pd.DataFrame(bet_level_rows)
    bet_out = ROOT / "data/processed/check_odds_anomaly_0628_bets.csv"
    bet_df.to_csv(bet_out, index=False)
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"実際の本番条件で再現された買い目（{len(bet_df)}点・cf.本番ログ43点）の p_model 検証")
    logger.info("=" * 78)
    logger.info(f"  p_model_raw   平均: {bet_df['p_model_raw'].mean():.4f}")
    logger.info(f"  p_model(補正後) 平均: {bet_df['p_model_calibrated'].mean():.4f}")
    logger.info(f"  圧縮比（raw/calibrated）: {bet_df['p_model_raw'].mean() / bet_df['p_model_calibrated'].mean():.2f}倍")
    logger.info(f"CSV保存: {bet_out}")


if __name__ == "__main__":
    main()
