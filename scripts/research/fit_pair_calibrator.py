"""
fit_pair_calibrator.py
=======================
蓄積された実運用ログ（data/logs/predictions/ + data/logs/shadow_candidates/）と
実際のレース結果を突合し、PairCalibrator（Platt Scaling）を学習する。

データソース:
  - data/logs/predictions/*.csv      : 4/25〜運用してきた「購入した点」のログ
  - data/logs/shadow_candidates/*.csv: 6/21〜のシャドー稼働で記録した「全候補ペア」
    （フィルター落ち含む。purchasedより遥かにデータ量が多く、生存バイアスが無い）
  - data/analysis/result_cache.json  : analyze_production_performance.py が
    キャッシュした各レースの着順・払戻（無ければ自動取得）

実行方法:
    .venv/bin/python fit_pair_calibrator.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.pair_calibrator import PairCalibrator
from analyze_production_performance import fetch_result, CACHE_PATH

PRED_DIR = ROOT / "data" / "logs" / "predictions"
SHADOW_DIR = ROOT / "data" / "logs" / "shadow_candidates"
OUT_PATH = ROOT / "data" / "models" / "pair_calibrator.pkl"

Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_results(race_ids: list[str], cache: dict) -> dict:
    new_count = 0
    for rid in race_ids:
        if rid in cache and cache[rid] is not None:
            continue
        try:
            res = fetch_result(rid)
        except Exception as e:
            logger.warning(f"  ⚠ {rid} 取得エラー: {e}")
            res = None
        cache[rid] = res
        new_count += 1
        time.sleep(0.8)
    if new_count:
        save_cache(cache)
        logger.info(f"  新規取得: {new_count}件")
    return cache


def main() -> None:
    rows = []

    # ── ① predictions（購入済み点・生存バイアスあり）──────────────────
    for f in sorted(PRED_DIR.glob("*.csv")):
        df = pd.read_csv(f, dtype=str)
        df["source"] = "predictions"
        rows.append(df)

    # ── ② shadow_candidates（全候補・フィルター落ち含む）────────────
    for f in sorted(SHADOW_DIR.glob("*.csv")):
        df = pd.read_csv(f, dtype=str)
        df["source"] = "shadow"
        rows.append(df)

    if not rows:
        logger.error("ログが見つかりません")
        sys.exit(1)

    full = pd.concat(rows, ignore_index=True)
    full["p_model"] = pd.to_numeric(full["p_model"], errors="coerce")
    full = full.dropna(subset=["p_model", "race_id", "horse_num_i", "horse_num_j"])

    race_ids = full["race_id"].unique().tolist()
    logger.info(f"対象race_id数: {len(race_ids)}  対象ペア数: {len(full)}")

    cache = load_cache()
    cache = ensure_results(race_ids, cache)

    hit_flags = []
    keep_mask = []
    for _, row in full.iterrows():
        res = cache.get(row["race_id"])
        if res is None:
            keep_mask.append(False)
            hit_flags.append(None)
            continue
        top2 = {h["num"] for h in res["top5"] if h["rank"] <= 2}
        ni, nj = str(row["horse_num_i"]), str(row["horse_num_j"])
        hit_flags.append(1 if (ni in top2 and nj in top2) else 0)
        keep_mask.append(True)

    full["hit"] = hit_flags
    full = full[keep_mask].copy()
    full = full.drop_duplicates(subset=["race_id", "horse_num_i", "horse_num_j"])

    logger.info(f"結果確定済みペア数: {len(full)}  的中: {int(full['hit'].sum())}件")

    if len(full) < 30:
        logger.warning("サンプル数が少なすぎます（<30）。キャリブレーション結果は参考値にとどめてください。")

    calibrator = PairCalibrator()
    calibrator.fit(full["p_model"].values, full["hit"].values)
    calibrator.save(OUT_PATH)

    # ── 検証: 補正前後でのΣp_model vs 実的中数 ──────────────────────
    raw_sum = full["p_model"].sum()
    calibrated = calibrator.transform(full["p_model"].values)
    cal_sum = calibrated.sum()
    actual = full["hit"].sum()

    logger.info("=" * 60)
    logger.info(f"学習サンプル数   : {calibrator.n_samples}  (的中 {calibrator.n_positive}件)")
    logger.info(f"補正前 Σp_model  : {raw_sum:.1f}")
    logger.info(f"補正後 Σp_model  : {cal_sum:.1f}")
    logger.info(f"実際の的中数     : {actual}")
    logger.info(f"保存先           : {OUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
