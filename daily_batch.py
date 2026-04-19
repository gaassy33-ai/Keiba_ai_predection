"""
daily_batch.py
当日出走表取得 → LightGBM 予測 → LINE Flex Message 送信

実行方法:
    python daily_batch.py                    # 当日
    python daily_batch.py --date 2026-03-08  # 日付指定

買い目（回収率100%超券種に絞り込み）:
    - レース絞り込み: honmei_prob ≥ 0.30 かつ 信頼度差 ≥ 0.05 かつ 未勝利・新馬除外
    - 単勝: ◎ 1点
    - 複勝: ◎ 1点（参考表示）
    - 馬連: ◎ - モデル上位5頭から EV 上位 最大3点（トリガミ除外）
    - 3連複: ◎軸 × モデル上位5頭 → Harville降順 最大5点（合成オッズ<1.0はケン）
    - 3連単: ◎1着固定 × モデル上位5頭 → Harville降順 最大5点（合成オッズ<1.0はケン）

環境変数 (.env):
    LINE_CHANNEL_ACCESS_TOKEN
    LINE_CHANNEL_SECRET
    LINE_TARGET_USER_ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from itertools import combinations as _comb, permutations as _perm
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from src.scraper.base_scraper import RaceInfo
from src.scraper.netkeiba_scraper import NetkeibaScraper
from config.settings import settings

# ======================================================================
# 定数
# ======================================================================
MIN_HONMEI_PROB      = 0.30   # ○最低勝率（バックテスト: 0.30未満はROI<60%のため除外）
MARK_STRONG_PROB     = 0.40   # ◎最低勝率（○との境界） ※0.35→0.40 2026-04-19改訂
MIN_CONFIDENCE_GAP   = 0.07   # 勝率差フィルター ※0.05→0.07 2026-04-19改訂
# ※ 的中レースの70%は差7〜10%に集中しているため0.10への引き上げは損失が大きすぎる
# EV フィルタ（model_prob × CALIBRATION_FACTOR × odds ≥ 閾値 のみ買い）
# 2026-04-19 実績分析: モデル確率が実績の約1.6倍過信 → 補正係数0.65を導入
# (旧: EV_THRESHOLD=1.05, model_prob×odds で計算)
CALIBRATION_FACTOR   = 0.65   # モデル確率の過信補正係数（実績単勝率/モデル平均確率 ≈ 0.62）
EV_THRESHOLD         = 1.05   # キャリブレーション補正後EV基準値（calibrated_EV ≥ 1.05）
# 改善②: 出走頭数ペナルティ（1頭増えるごとに+0.3%、上限+5%）
ENTRIES_THRESHOLD_ADJ = 0.003   # per horse over 10
ENTRIES_THRESHOLD_BASE = 10
ENTRIES_THRESHOLD_CAP  = 0.05
# 改善③: レースクラス別ブースト（未勝利・新馬は+5%上乗せ）
MAIDEN_PROB_BOOST    = 0.05
MAIDEN_KEYWORDS      = ("新馬", "未勝利")

# ======================================================================
# 季節フィルター設定
# バックテスト(2024-2025年)分析に基づく季節別買い条件の動的変更
# ======================================================================
# 案①: 季節別 確率閾値
#   好調期(4-6月)  : ROI160%  → 現状維持 (0.30)
#   中間期(3,10,11月): ROI81%  → やや絞り込み (0.33)
#   不調期(その他)  : ROI68%  → 厳格化 (0.38)
_SEASON_PROB_THRESHOLD: dict[int, float] = {
    1: 0.38, 2: 0.38,                    # 不調期
    3: 0.33,                              # 中間期
    4: 0.30, 5: 0.30, 6: 0.30,           # 好調期
    7: 0.38, 8: 0.38, 9: 0.38,           # 不調期
    10: 0.33, 11: 0.33,                  # 中間期
    12: 0.38,                             # 不調期
}

# 案②: 夏ダート除外 (7-9月のダートはROI12-61%と極端に低い)
_SUMMER_DIRT_SKIP_MONTHS = {7, 8, 9}    # この月のダートレースは見送り

# 案③: 月別マーク制限 (◎のみ or ○も買う)
#   ○を見送る月: 10月(○ROI 0%)、2月(○ROI 38%)、1月(○ROI 39%)
_MARK_O_SKIP_MONTHS = {1, 2, 10}        # この月は○(低確率◎)を見送り、◎のみ

# 案④: オッズ帯別 EV 閾値緩和
#   8〜15倍のとき EV閾値を緩和して拾いやすくする（ROI164%）
_HIGH_VALUE_ODDS_MIN = 8.0
_HIGH_VALUE_ODDS_MAX = 15.0
_HIGH_VALUE_EV_THRESHOLD = 1.00         # 8-15倍のとき1.00に緩和 ※0.95→1.00 2026-04-19改訂

# 大穴馬フィルタ: 単勝20倍超は市場確率5%未満 → モデルの過信が大きくスキップ
_MAX_LONGSHOT_ODDS = 20.0

# 案D: 不調期 開催場フィルタ（不調期ROI 0%会場を除外）
_BAD_SEASON_MONTHS    = {1, 7, 8, 9, 11, 12}
_BAD_VENUE_SKIP_CODES = {"02", "04", "09"}   # 函館・新潟・阪神（不調期ROI 0%）

# 案E: 不調期 距離フィルタ（1800m以上は見送り）
_BAD_SEASON_MAX_DISTANCE = 1800

# 案B: 不調期専用モデルの閾値（スケールが全期間モデルと異なるため専用設定）
_BAD_MODEL_PROB_THRESHOLD = 0.25
_BAD_MODEL_MARK_STRONG    = 0.28
EV_PARTNER_TOP_N     = 7      # EV計算の候補プール（上位7頭）※5→7に拡張（3連複カバレッジ改善）
MAX_BAREN_TICKETS    = 3      # 馬連最大点数
MAX_UMATAN_TICKETS   = 3      # 馬単最大点数
MAX_SANRENFUKU_TICKETS = 7    # 3連複最大点数 ※5→7に拡張
MAX_SANRENTAN_TICKETS  = 7    # 3連単最大点数 ※5→7に拡張（ROI +24.3pt、追加分ROI 266%）
TORIKAMI_THRESHOLD   = 1.05   # トリガミ判定閾値（推定オッズがこれ未満は除外）

# JRA 控除率
JRA_TAKE = {"馬連": 0.225, "馬単": 0.25, "3連複": 0.225, "3連単": 0.275}

VENUE_COLORS: dict[str, str] = {
    "札幌": "#1a5276", "函館": "#154360", "福島": "#4a7c4e",
    "新潟": "#8b4513", "東京": "#1a472a", "中山": "#8b0000",
    "中京": "#b8860b", "京都": "#6b4c9a", "阪神": "#1b3a6b",
    "小倉": "#2f6b9a",
}
DEFAULT_COLOR = "#333333"

# ======================================================================
# ロガー初期化
# ======================================================================
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/daily_batch.log", level="DEBUG", format=_fmt, rotation="20 MB")


# ======================================================================
# ユーティリティ
# ======================================================================

def _parse_odds(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")


def _market_probs(odds_list: list[float]) -> np.ndarray:
    raw = np.array([1.0 / max(o, 1.01) for o in odds_list])
    s = raw.sum()
    return raw / s if s > 0 else raw


def _harville(probs: list[float], order: list[int]) -> float:
    p, rem = 1.0, 1.0
    for idx in order:
        if rem < 1e-9:
            return 0.0
        p *= probs[idx] / rem
        rem -= probs[idx]
    return p


def _prob_quinella(probs: list[float], i: int, j: int) -> float:
    return _harville(probs, [i, j]) + _harville(probs, [j, i])


def _prob_trio(probs: list[float], i: int, j: int, k: int) -> float:
    """3連複確率（i,j,k が任意順で1〜3着）"""
    return sum(_harville(probs, list(o)) for o in _perm([i, j, k]))


def _prob_sanrentan(probs: list[float], i: int, j: int, k: int) -> float:
    """3連単確率（i→j→k の順）"""
    return _harville(probs, [i, j, k])


def _synth_odds(odds_list: list[float]) -> float:
    """合成オッズ（全点的中でのトータル期待値）"""
    denom = sum(1.0 / max(o, 1e-9) for o in odds_list)
    return 1.0 / denom if denom > 0 else 0.0


def _est_odds(prob: float, bet_type: str, org: str = "jra") -> float:
    take = JRA_TAKE.get(bet_type, 0.225)
    return (1.0 - take) / max(prob, 0.001)


# ======================================================================
# 履歴データ読み込み（特徴量生成用）
# ======================================================================

def load_history() -> pd.DataFrame:
    raw = ROOT / "data" / "raw"
    dfs = []
    for fname in ("train_results.csv", "test_results.csv"):
        p = raw / fname
        if p.exists():
            dfs.append(pd.read_csv(p, dtype=str))
    if not dfs:
        raise FileNotFoundError("train_results.csv / test_results.csv が見つかりません")
    hist = pd.concat(dfs, ignore_index=True)
    hist["race_date"] = pd.to_datetime(
        hist["race_id"].str[:8].apply(
            lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else None
        ),
        errors="coerce",
    )
    logger.info(f"履歴データ読み込み完了: {len(hist):,} rows")
    return hist


# ======================================================================
# 予測ログ保存（docs/predictions_log.csv）
# ======================================================================

PREDICTIONS_LOG = ROOT / "docs" / "predictions_log.csv"
PREDICTIONS_LOG_COLS = [
    "date", "race_id", "race_name", "honmei_num", "honmei_name",
    "honmei_prob", "taikou_prob", "gap", "mark", "is_buy",
    "skip_reason",        # 見送り理由コード (maiden/prob/gap/ev/multi/空文字=買い)
    "tansho_hit", "tansho_ret",
    "umatan_str", "umatan_hit", "umatan_ret",
]


def _save_prediction_log(
    target_date: date,
    venue_results: dict[str, list[dict | None]],
) -> None:
    """
    予測結果を docs/predictions_log.csv に追記する。
    hit / payout は未確定（週次バッチで更新）のため空欄で保存。
    当日分が既に存在する場合は上書き（重複排除）。
    """
    rows = []
    for results in venue_results.values():
        for r in results:
            if r is None:
                continue
            hon = r["honmei_num"]
            um_str = ",".join(
                f"{hon}\u2192{p}" for p in r.get("umatan_partners", [])
            )
            # 見送り理由コード（is_buy=True なら空文字）
            if r["is_buy"]:
                skip_reason = ""
            else:
                reasons = []
                if r.get("is_maiden"):
                    reasons.append("maiden")
                if r.get("skip_prob"):
                    reasons.append("prob")
                if r.get("skip_gap"):
                    reasons.append("gap")
                if r.get("skip_ev"):
                    reasons.append("ev")
                if r.get("skip_venue"):
                    reasons.append("venue")
                if r.get("skip_distance"):
                    reasons.append("distance")
                skip_reason = "+".join(reasons) if reasons else "unknown"
            rows.append({
                "date":         str(target_date),
                "race_id":      r["race_id"],
                "race_name":    r.get("race_name", ""),
                "honmei_num":   hon,
                "honmei_name":  r["honmei_name"],
                "honmei_prob":  r["honmei_prob"],
                "taikou_prob":  r.get("taikou_prob", ""),
                "gap":          r.get("gap", ""),
                "mark":         r.get("mark", "△"),
                "is_buy":       r["is_buy"],
                "skip_reason":  skip_reason,
                "tansho_hit":   "",
                "tansho_ret":   "",
                "umatan_str":   um_str,
                "umatan_hit":   "",
                "umatan_ret":   "",
            })

    if not rows:
        logger.info("  予測ログ: 対象レースなし（スキップ）")
        return

    new_df = pd.DataFrame(rows, columns=PREDICTIONS_LOG_COLS)

    if PREDICTIONS_LOG.exists():
        existing = pd.read_csv(PREDICTIONS_LOG, dtype=str)
        # 旧バージョン互換: skip_reason 列がなければ空文字で補完
        if "skip_reason" not in existing.columns:
            existing.insert(existing.columns.get_loc("is_buy") + 1, "skip_reason", "")
        # 当日分を削除して新データで上書き
        existing = existing[existing["date"] != str(target_date)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PREDICTIONS_LOG, index=False)
    logger.info(f"  予測ログ保存: {PREDICTIONS_LOG} ({len(rows)} 件追記)")


# ======================================================================
# 1レース予測・買い目生成
# ======================================================================

def predict_and_bet(
    race_info: RaceInfo,
    fe: FeatureEngineer,
    trainer: ModelTrainer,
    org: str = "jra",
    race_date: "date | None" = None,
) -> dict | None:
    """
    RaceInfo を受け取り、予測と買い目を生成する。

    Returns
    -------
    dict | None
        {
          "race_id": str,
          "race_name": str,
          "race_num": str,          # "1R" - "12R"
          "is_buy": bool,
          "honmei_num": str,        # ◎ 馬番
          "honmei_name": str,
          "honmei_prob": float,
          "baren_partners": list[str],  # 馬連 相手馬番リスト
        }
    """
    entries = race_info.entries
    if len(entries) < 3:
        logger.debug(f"skip {race_info.race_id}: 出走頭数不足 ({len(entries)})")
        return None

    if not race_info.course_type or not race_info.distance:
        logger.debug(f"skip {race_info.race_id}: course_type/distance 未取得")
        return None

    # --- entry_df 組み立て ---
    entry_df = pd.DataFrame([{
        "horse_id":       e.horse_id,
        "horse_name":     e.horse_name,
        "horse_number":   e.horse_number,
        "frame_number":   e.frame_number,
        "sex":            e.sex,
        "age":            e.age,
        "weight_carried": e.weight_carried,
        "jockey_id":      e.jockey_id,
        "trainer_name":   e.trainer_name,
        "father":         e.father_name,
        "mother_father":  e.mother_father_name,
        "odds":           e.odds,          # 改善⑥: HHI 計算用
    } for e in entries])

    # --- 特徴量生成 ---
    from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP
    gc_code = GROUND_CONDITION_MAP.get(race_info.ground_condition, -1)
    wx_code = WEATHER_MAP.get(race_info.weather, -1)

    # 改善④⑤: レースクラスコード・会場コード
    race_class_code = FeatureEngineer._race_name_to_class_code(race_info.race_name or "")
    try:
        venue_code = int(race_info.race_id[4:6])
    except Exception:
        venue_code = -1

    try:
        feat_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition_code=gc_code,
            weather_code=wx_code,
            race_class_code=race_class_code,
            venue_code=venue_code,
        )
    except Exception as e:
        logger.warning(f"特徴量生成失敗 {race_info.race_id}: {e}")
        return None

    X = (feat_df[FeatureEngineer.FEATURE_COLUMNS]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0))
    # num_threads=1: macOS で LightGBM マルチスレッドが SIGSEGV を起こすため単スレッド化
    win_probs = trainer.model.predict(X, num_threads=1)
    if trainer.place_model is not None:
        place_probs = trainer.place_model.predict(X, num_threads=1)
        win_blend = 0.7
        probs = win_blend * win_probs + (1.0 - win_blend) * place_probs
    else:
        probs = win_probs

    pred_df = feat_df[["horse_id", "horse_name", "horse_number"]].copy()
    pred_df["win_prob"] = probs
    pred_df = pred_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

    # --- マーク判定 ---
    honmei_prob = float(pred_df.iloc[0]["win_prob"])
    taikou_prob = float(pred_df.iloc[1]["win_prob"]) if len(pred_df) > 1 else 0.0
    gap = honmei_prob - taikou_prob
    n_entries = len(entries)

    # レース日の月（季節フィルター用）
    # race_date が渡された場合はその月を使用（--date 指定時の正確な季節判定）
    race_month = race_date.month if race_date is not None else date.today().month

    # 改善②: 出走頭数に応じた動的閾値（多頭数は確率が薄まるため厳格化）
    # 季節フィルター①: 月別確率閾値を適用（好調期は0.30、中間期0.33、不調期0.38）
    # 案B: 不調期専用モデル使用時は専用閾値（確率スケールが異なるため）
    _using_bad_model = getattr(trainer, "_is_bad_season_model", False)
    if _using_bad_model:
        prob_threshold = _BAD_MODEL_PROB_THRESHOLD
    else:
        season_base = _SEASON_PROB_THRESHOLD.get(race_month, MIN_HONMEI_PROB)
        entries_adj = min(
            max(0, n_entries - ENTRIES_THRESHOLD_BASE) * ENTRIES_THRESHOLD_ADJ,
            ENTRIES_THRESHOLD_CAP,
        )
        prob_threshold = season_base + entries_adj

    # 新馬・未勝利戦は除外（バックテスト: ROI35%と低く再現性が低いため）
    is_maiden = any(k in str(race_info.race_name) for k in MAIDEN_KEYWORDS)

    gap_ok = gap >= MIN_CONFIDENCE_GAP

    # 改善⑨: EV フィルタ — feat_df["odds"] は build_entry_features で付加済み
    honmei_id_str = str(pred_df.iloc[0]["horse_id"])
    odds_col_map  = feat_df.set_index("horse_id")["odds"] if "odds" in feat_df.columns else {}
    _raw_odds = odds_col_map.get(honmei_id_str)
    honmei_odds_pre = float(_raw_odds) if _raw_odds is not None else float("nan")

    # キャリブレーション補正EV:
    #   calibrated_ev = honmei_prob × CALIBRATION_FACTOR × odds
    #   実績分析(2026-04-19): 購入馬券の実単勝率20.4% / モデル平均確率32.7% ≈ 0.62
    #   CALIBRATION_FACTOR=0.65 で補正することで EV過大評価を抑制
    honmei_ev = (honmei_prob * CALIBRATION_FACTOR * honmei_odds_pre
                 if _raw_odds is not None and not np.isnan(honmei_odds_pre) else float("nan"))

    # 季節フィルター④: 8〜15倍のとき EV 閾値を緩和（ROI164%の高バリューゾーン）
    if (not np.isnan(honmei_odds_pre)
            and _HIGH_VALUE_ODDS_MIN <= honmei_odds_pre <= _HIGH_VALUE_ODDS_MAX):
        ev_threshold = _HIGH_VALUE_EV_THRESHOLD
    else:
        ev_threshold = EV_THRESHOLD

    # オッズ未取得時（nan）は EV フィルタをスキップして確率フィルタのみ適用
    ev_ok = (np.isnan(honmei_ev) or honmei_ev >= ev_threshold)

    # 大穴馬フィルタ: 20倍超は市場確率5%未満 → モデルの過信が大きくスキップ
    is_longshot = (not np.isnan(honmei_odds_pre) and honmei_odds_pre > _MAX_LONGSHOT_ODDS)

    # 季節フィルター②: 夏場（7-9月）のダートは見送り（ROI 12-61%と極端に低い）
    is_summer_dirt = (
        race_month in _SUMMER_DIRT_SKIP_MONTHS
        and str(race_info.course_type) == "ダート"
    )

    # 案D: 不調期の開催場フィルタ（函館・新潟・阪神は不調期ROI 0%）
    jyo_code_str = str(race_info.race_id)[4:6]
    is_bad_venue = (race_month in _BAD_SEASON_MONTHS and jyo_code_str in _BAD_VENUE_SKIP_CODES)

    # 案E: 不調期の距離フィルタ（1800m以上は不調期ROI 32%以下）
    is_bad_distance = (
        race_month in _BAD_SEASON_MONTHS
        and int(race_info.distance or 0) >= _BAD_SEASON_MAX_DISTANCE
    )

    # 案F: 不調期は複勝メイン
    bet_mode = "複勝" if race_month in _BAD_SEASON_MONTHS else "単勝"

    _mark_strong = _BAD_MODEL_MARK_STRONG if _using_bad_model else MARK_STRONG_PROB
    if is_maiden or not gap_ok or honmei_prob < prob_threshold or not ev_ok or is_summer_dirt or is_bad_venue or is_bad_distance or is_longshot:
        mark = "△"
    elif honmei_prob >= _mark_strong:
        mark = "◎"
    else:
        mark = "○"

    # 季節フィルター③: 月別マーク制限（○ROIが極端に低い月は◎のみ）
    if mark == "○" and race_month in _MARK_O_SKIP_MONTHS:
        mark = "△"

    is_skip = (mark == "△")

    logger.info(
        f"  probs: {mark}{pred_df.iloc[0]['horse_name']}={honmei_prob:.4f}"
        f"  対抗={taikou_prob:.4f}  差={gap:.4f}"
        f"  EV={honmei_ev:.2f}({'OK' if ev_ok else 'NG'})"
        + (f"  [季節フィルター: 夏ダート見送り]" if is_summer_dirt else "")
        + (f"  [季節フィルター: 不調期会場({jyo_code_str})見送り]" if is_bad_venue else "")
        + (f"  [季節フィルター: 不調期距離({race_info.distance}m≥1800m)見送り]" if is_bad_distance else "")
        + (f"  [季節フィルター: {race_month}月○制限]" if mark == "△" and not is_maiden and not is_summer_dirt and not is_bad_venue and not is_bad_distance and gap_ok and honmei_prob >= prob_threshold and ev_ok else "")
    )

    honmei_row = pred_df.iloc[0]
    honmei_num  = str(int(honmei_row["horse_number"]))
    honmei_name = str(honmei_row["horse_name"])

    # レース番号（race_id 末尾2桁）
    race_num_str = f"{int(race_info.race_id[-2:])}R"

    result = {
        "race_id":            race_info.race_id,
        "race_name":          race_info.race_name,
        "race_num":           race_num_str,
        "is_buy":             not is_skip,
        "mark":               mark,
        "honmei_num":         honmei_num,
        "honmei_name":        honmei_name,
        "honmei_prob":        round(honmei_prob, 3),
        "taikou_prob":        round(taikou_prob, 3),
        "gap":                round(gap, 3),
        "n_entries":          n_entries,
        "is_maiden":          is_maiden,
        # △通知用: スキップ理由フラグ
        "skip_prob":          honmei_prob < prob_threshold,
        "skip_gap":           not gap_ok,
        "skip_ev":            not ev_ok,
        "skip_summer_dirt":   is_summer_dirt,
        "skip_venue":         is_bad_venue,
        "skip_distance":      is_bad_distance,
        "skip_mark_o":        (mark == "△" and not is_maiden and not is_summer_dirt
                               and not is_bad_venue and not is_bad_distance
                               and gap_ok and honmei_prob >= prob_threshold and ev_ok),
        "prob_threshold":     round(prob_threshold, 3),
        "race_month":         race_month,
        "bet_mode":           bet_mode,
        "baren_partners":     [],
        "umatan_partners":    [],
        "sanrenfuku_combos":  [],
        "sanrentan_combos":   [],
    }

    if is_skip:
        return result

    # --- 共通: オッズマップ・市場確率 ---
    odds_map = {e.horse_id: e.odds for e in entries if e.odds}
    all_ids      = pred_df["horse_id"].tolist()
    all_odds_raw = [_parse_odds(odds_map.get(hid, float("nan"))) for hid in all_ids]
    valid_pairs  = [(hid, o) for hid, o in zip(all_ids, all_odds_raw)
                    if not np.isnan(o) and o > 1.0]
    mkt_probs: list[float] = []
    valid_ids: list[str]   = []
    if valid_pairs:
        vids, vodds = zip(*valid_pairs)
        valid_ids   = list(vids)
        mkt_probs   = _market_probs(list(vodds)).tolist()

    def vidx(horse_id: str) -> int | None:
        return valid_ids.index(horse_id) if horse_id in valid_ids else None

    hi = vidx(honmei_row["horse_id"])

    # --- モデル上位 EV_PARTNER_TOP_N 頭（◎除く）---
    partner_rows = pred_df[pred_df["horse_id"] != honmei_row["horse_id"]].head(EV_PARTNER_TOP_N)

    # --- 馬連: EV スコア上位3頭・トリガミ除外 ---
    scored = []
    for _, row in partner_rows.iterrows():
        hid  = str(row["horse_id"])
        prob = float(row["win_prob"])
        odds = _parse_odds(odds_map.get(hid, 5.0))
        if np.isnan(odds) or odds <= 1.0:
            odds = 5.0
        scored.append((hid, prob * odds, str(int(row["horse_number"]))))
    scored.sort(key=lambda x: x[1], reverse=True)

    baren_nums: list[str] = []
    for hid, _ev, num in scored[:MAX_BAREN_TICKETS]:
        vi = vidx(hid)
        if hi is not None and vi is not None and mkt_probs:
            e_odds = _est_odds(_prob_quinella(mkt_probs, hi, vi), "馬連", org=org)
            if e_odds < TORIKAMI_THRESHOLD:
                continue
        baren_nums.append(num)

    # --- 馬単: ◎1着固定 × 上位EV頭 → Harville降順 最大3点 ---
    um_all: list[tuple[str, float]] = []
    for hid, _ev, num in scored:
        vi = vidx(hid)
        if hi is not None and vi is not None and mkt_probs:
            e_od = _est_odds(_harville(mkt_probs, [hi, vi]), "馬単", org=org)
            if e_od < TORIKAMI_THRESHOLD:
                continue
        else:
            e_od = 999.0
        um_all.append((num, e_od))

    um_all.sort(key=lambda x: -x[1])
    um_sel = um_all[:MAX_UMATAN_TICKETS]
    um_est = [e for _, e in um_sel]
    umatan_partners = (
        [num for num, _ in um_sel]
        if (not um_est or _synth_odds(um_est) >= 1.0)
        else []
    )

    pool: list[tuple[str, int | None]] = []
    sanrenfuku_combos: list[tuple[str, str]] = []
    sanrentan_combos:  list[tuple[str, str]] = []

    if True:
        for _, row in partner_rows.iterrows():
            vi = vidx(str(row["horse_id"]))
            pool.append((str(int(row["horse_number"])), vi))

        # オッズ未取得時はモデル確率（正規化済み）でHarvilleを代替
        _probs_raw = pred_df["win_prob"].tolist()
        _probs_sum = sum(_probs_raw)
        _model_probs = [p / _probs_sum for p in _probs_raw] if _probs_sum > 0 else _probs_raw
        # pred_df index→確率 (0=本命, 1〜=相手順)
        _partner_pred_idxs = {
            num: i + 1
            for i, (num, _vi) in enumerate(pool)
        }

        # 3連複: モデル確率(trio)降順でソート（市場オッズ非依存→時刻による買い目ブレを防止）
        # (num_a, num_b, model_p, e_od)
        sf_all: list[tuple[str, str, float, float]] = []
        for (num_a, vi_a), (num_b, vi_b) in _comb(pool, 2):
            pa      = _partner_pred_idxs.get(num_a, 1)
            pb      = _partner_pred_idxs.get(num_b, 2)
            model_p = _prob_trio(_model_probs, 0, pa, pb)

            if hi is not None and vi_a is not None and vi_b is not None and mkt_probs:
                mkt_p = _prob_trio(mkt_probs, hi, vi_a, vi_b)
                e_od  = _est_odds(mkt_p, "3連複", org=org)
            else:
                e_od = _est_odds(model_p, "3連複", org=org)

            sf_all.append((num_a, num_b, model_p, e_od))

        sf_all.sort(key=lambda x: -x[2])   # モデル確率降順（市場オッズ非依存）
        sf_sel = sf_all[:MAX_SANRENFUKU_TICKETS]
        sf_est = [od for _, _, _, od in sf_sel]
        sanrenfuku_combos = (
            [(a, b) for a, b, _, _ in sf_sel]
            if (not sf_est or _synth_odds(sf_est) >= 1.0)
            else []
        )

        # 3連単: モデル確率(◎→A→B)降順でソート（市場オッズ非依存→時刻による買い目ブレを防止）
        # (num_2, num_3, model_p, e_od)
        st_all: list[tuple[str, str, float, float]] = []
        for (num_2, vi_2), (num_3, vi_3) in _perm(pool, 2):
            p2      = _partner_pred_idxs.get(num_2, 1)
            p3      = _partner_pred_idxs.get(num_3, 2)
            model_p = _prob_sanrentan(_model_probs, 0, p2, p3)

            if hi is not None and vi_2 is not None and vi_3 is not None and mkt_probs:
                mkt_p = _prob_sanrentan(mkt_probs, hi, vi_2, vi_3)
                e_od  = _est_odds(mkt_p, "3連単", org=org)
            else:
                e_od = _est_odds(model_p, "3連単", org=org)

            st_all.append((num_2, num_3, model_p, e_od))

        st_all.sort(key=lambda x: -x[2])   # モデル確率降順（市場オッズ非依存）
        st_sel = st_all[:MAX_SANRENTAN_TICKETS]
        st_est = [od for _, _, _, od in st_sel]
        sanrentan_combos = (
            [(n2, n3) for n2, n3, _, _ in st_sel]
            if (not st_est or _synth_odds(st_est) >= 1.0)
            else []
        )

    result["is_buy"]            = True
    result["baren_partners"]    = baren_nums
    result["umatan_partners"]   = umatan_partners
    result["sanrenfuku_combos"] = sanrenfuku_combos
    result["sanrentan_combos"]  = sanrentan_combos
    return result


# ======================================================================
# Flex Message 構築
# ======================================================================

def _race_row_component(race_result: dict | None, race_num: int) -> list[dict]:
    """1レース分の行コンポーネント（separator + box）を返す"""
    label = f"{race_num:>2}R"

    # データ取得失敗レース
    if race_result is None:
        body_content = {
            "type": "box",
            "layout": "horizontal",
            "paddingStart": "14px", "paddingEnd": "14px",
            "paddingTop": "7px", "paddingBottom": "7px",
            "contents": [
                {"type": "text", "text": label,
                 "size": "sm", "color": "#cccccc", "flex": 0},
                {"type": "text", "text": "— データなし",
                 "size": "sm", "color": "#cccccc", "flex": 1, "margin": "md"},
            ],
        }
        return [{"type": "separator"}, body_content]

    mark     = race_result.get("mark", "△")
    hon_num  = race_result["honmei_num"]
    hon_name = race_result["honmei_name"]
    prob     = race_result["honmei_prob"]
    t_prob   = race_result.get("taikou_prob", 0.0)
    gap      = race_result.get("gap", 0.0)

    # 確率サマリー行（全レース共通）
    prob_str = f"{prob:.1%}  差+{gap:.1%}  対抗{t_prob:.1%}"

    if mark == "◎":
        label_color  = "#1a5533"
        label_weight = "bold"
    elif mark == "○":
        label_color  = "#7a5500"
        label_weight = "bold"
    else:  # △
        label_color  = "#999999"
        label_weight = "regular"

    # ── △（見送り）: 馬名 + 確率のみ ──────────────────────────
    if mark == "△":
        body_content = {
            "type": "box",
            "layout": "horizontal",
            "paddingStart": "14px", "paddingEnd": "14px",
            "paddingTop": "7px", "paddingBottom": "7px",
            "contents": [
                {"type": "text", "text": label,
                 "size": "sm", "color": label_color, "flex": 0},
                {
                    "type": "box", "layout": "vertical",
                    "flex": 1, "margin": "md",
                    "contents": [
                        {"type": "text",
                         "text": f"△ {hon_num} {hon_name}",
                         "size": "sm", "color": "#999999", "wrap": True},
                        {"type": "text", "text": prob_str,
                         "size": "xs", "color": "#bbbbbb", "margin": "xs", "wrap": True},
                    ],
                },
            ],
        }
        return [{"type": "separator"}, body_content]

    # ── ◎ / ○: 馬券情報あり ──────────────────────────────────
    partners    = race_result.get("baren_partners", [])
    um_partners = race_result.get("umatan_partners", [])
    sf_combos   = race_result.get("sanrenfuku_combos", [])
    st_combos   = race_result.get("sanrentan_combos", [])

    baren_str = (
        f"馬連: {hon_num}-{' / '.join(partners)}"
        if partners else "馬連: なし"
    )
    umatan_str = (
        "馬単: " + " / ".join(f"{hon_num}→{p}" for p in um_partners)
        if um_partners else "馬単: なし"
    )
    sf_str = (
        "3連複: " + " / ".join(f"{hon_num}-{a}-{b}" for a, b in sf_combos)
        if sf_combos else "3連複: なし"
    )
    st_str = (
        "3連単: " + " / ".join(f"{hon_num}→{n2}→{n3}" for n2, n3 in st_combos)
        if st_combos else "3連単: なし"
    )

    ticket_color  = "#555555"
    umatan_color  = "#7a5500"
    sf_color      = "#4a4a8a"
    st_color      = "#7a3a3a"

    bet_mode  = race_result.get("bet_mode", "単勝")
    bet_label = "複勝メイン" if bet_mode == "複勝" else "単・複"
    detail_items: list[dict] = [
        {"type": "text",
         "text": f"{mark}{hon_num} {hon_name}（{bet_label}）",
         "size": "sm", "weight": "bold", "color": label_color, "wrap": True},
        {"type": "text", "text": prob_str,
         "size": "xs", "color": "#888888", "margin": "xs", "wrap": True},
        {"type": "text", "text": baren_str,
         "size": "xs", "color": ticket_color, "margin": "xs", "wrap": True},
        {"type": "text", "text": umatan_str,
         "size": "xs", "color": umatan_color, "margin": "xs", "wrap": True},
    ]
    if sf_combos:
        detail_items.append(
            {"type": "text", "text": sf_str,
             "size": "xs", "color": sf_color, "margin": "xs", "wrap": True}
        )
    if st_combos:
        detail_items.append(
            {"type": "text", "text": st_str,
             "size": "xs", "color": st_color, "margin": "xs", "wrap": True}
        )

    body_content = {
        "type": "box",
        "layout": "horizontal",
        "paddingStart": "14px", "paddingEnd": "14px",
        "paddingTop": "9px", "paddingBottom": "9px",
        "contents": [
            {"type": "text", "text": label,
             "size": "sm", "weight": label_weight, "color": label_color, "flex": 0},
            {"type": "box", "layout": "vertical",
             "flex": 1, "margin": "md", "contents": detail_items},
        ],
    }

    return [{"type": "separator"}, body_content]


def _build_venue_bubble(
    venue_name: str,
    target_date: date,
    race_results: list[dict | None],
) -> dict:
    """1会場分のバブルを構築する"""
    color = VENUE_COLORS.get(venue_name, DEFAULT_COLOR)

    # レース番号順に整列（1R〜12R）
    by_num: dict[int, dict | None] = {}
    for r in race_results:
        if r is not None:
            try:
                n = int(r["race_id"][-2:])
            except (ValueError, KeyError, TypeError):
                continue
            by_num[n] = r

    strong_count = sum(1 for r in by_num.values() if r and r.get("mark") == "◎")
    ok_count     = sum(1 for r in by_num.values() if r and r.get("mark") == "○")
    buy_count    = strong_count + ok_count
    total_count  = len(by_num)

    # レース行を生成（1R〜12R）
    race_row_components: list[dict] = []
    all_race_nums = sorted(by_num.keys()) if by_num else list(range(1, 13))
    for n in all_race_nums:
        race_row_components.extend(_race_row_component(by_num.get(n), n))
    race_row_components.append({"type": "separator"})

    weekday = ["月", "火", "水", "木", "金", "土", "日"][target_date.weekday()]
    header_text = f"{target_date.year}年{target_date.month}月{target_date.day}日（{weekday}）"

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": color,
            "paddingAll": "16px",
            "contents": [
                {
                    "type": "text",
                    "text": venue_name,
                    "weight": "bold",
                    "color": "#ffffff",
                    "size": "lg",
                },
                {
                    "type": "text",
                    "text": header_text,
                    "color": "#ffffff99",
                    "size": "xs",
                    "margin": "xs",
                },
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "none",
            "paddingAll": "0px",
            "contents": race_row_components,
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#f5f5f5",
            "paddingAll": "10px",
            "contents": [
                {
                    "type": "text",
                    "text": f"◎{strong_count}R  ○{ok_count}R  △{total_count - buy_count}R  計{total_count}R",
                    "size": "xs",
                    "color": "#888888",
                    "align": "center",
                }
            ],
        },
    }


def build_flex_message(
    target_date: date,
    venue_results: dict[str, list[dict | None]],
) -> dict:
    """
    会場別カルーセル Flex Message を組み立てる。

    Parameters
    ----------
    venue_results : dict[str, list[dict | None]]
        venue_name → list of predict_and_bet() 結果
    """
    bubbles = [
        _build_venue_bubble(venue, target_date, races)
        for venue, races in venue_results.items()
        if races  # 空会場はスキップ
    ]

    if not bubbles:
        # 全会場スキップの場合は1枚だけテキストバブルを表示
        bubbles = [{
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [{
                    "type": "text",
                    "text": "本日は対象レースなし",
                    "align": "center",
                    "color": "#888888",
                }],
            },
        }]

    venues_str = "・".join(venue_results.keys())
    alt_text = (
        f"{target_date.strftime('%m/%d')} 競馬予測 | {venues_str}"
    )

    return {
        "type": "flex",
        "altText": alt_text,
        "contents": {
            "type": "carousel",
            "contents": bubbles,
        },
    }


# ======================================================================
# LINE Messaging API 送信
# ======================================================================

def send_line_flex(flex_msg: dict) -> None:
    """LINE Messaging API の Push Message で Flex Message を送信する"""
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.line_channel_access_token}",
    }
    payload = {
        "to": settings.line_target_user_id,
        "messages": [flex_msg],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        logger.error(f"LINE 送信失敗: {resp.status_code} {resp.text}")
        resp.raise_for_status()
    logger.info(f"LINE 送信完了 (status={resp.status_code})")


# ======================================================================
# メイン
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="競馬予測 日次バッチ")
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="予測対象日 (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LINE 送信をスキップしてローカルに JSON を出力",
    )
    args = parser.parse_args()
    target_date = date.fromisoformat(args.date)
    org = "jra"

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"daily_batch 開始: {target_date}  [JRA]")
    logger.info("=" * 60)

    # ── 過去日付ガード: 2日以上前の日付での実行を防止 ──────────────
    _today = date.today()
    _days_diff = (_today - target_date).days
    if _days_diff >= 2 and not args.dry_run:
        logger.error("=" * 60)
        logger.error(f"  ⚠️  過去日付での実行を検出しました！")
        logger.error(f"  指定日: {target_date}  /  今日: {_today}  /  差: {_days_diff}日")
        logger.error(f"  誤ったLINE通知を防ぐため実行を中止します。")
        logger.error(f"  正しい日付で再実行してください: --date {_today}")
        logger.error("=" * 60)
        return

    # 二重実行ガード: 当日の flex が既に docs/ に存在する場合はスキップ
    _docs_flex = ROOT / "docs" / f"flex_{target_date}.json"
    if _docs_flex.exists() and not args.dry_run:
        logger.warning(f"  本日分 ({target_date}) の flex が既に存在します: {_docs_flex}")
        logger.warning(f"  二重実行を防ぐためスキップします。強制実行は docs/flex_{target_date}.json を削除してください。")
        return

    # ── 1. モデル・履歴読み込み ──────────────────────────────────
    logger.info("[1/5] モデル・FeatureEngineer 読み込み")
    trainer = ModelTrainer.load(settings.model_path, org=org)

    # 不調期専用モデル（存在すれば読み込む）
    _bad_season_model_path = ROOT / "data" / "models" / "lgbm_model_bad_season.pkl"
    bad_season_trainer: ModelTrainer | None = None
    if _bad_season_model_path.exists():
        bad_season_trainer = ModelTrainer.load(_bad_season_model_path, org=org)
        bad_season_trainer._is_bad_season_model = True  # 閾値切り替え用フラグ
        logger.info(f"  不調期専用モデル読み込み完了: {_bad_season_model_path.name}")

    # ── 2. FeatureEngineer 構築 ──────────────────────────────────
    logger.info("[2/5] FeatureEngineer 構築（feature_stats.pkl から）")
    fe = FeatureEngineer.from_stats(settings.stats_path)
    logger.info("  FeatureEngineer 準備完了")

    # ── 3. 当日レーススケジュール取得 ────────────────────────────
    logger.info("[3/5] 当日レーススケジュール取得")
    scraper = NetkeibaScraper()
    try:
        schedule = scraper.fetch_race_schedule_by_date(target_date)
    except Exception as e:
        logger.error(f"スケジュール取得失敗: {e}")
        schedule = {}

    # 対象会場のみ絞り込み（settings.target_jyo_codes）
    jyo_to_name = scraper.VENUE_CODE_TO_NAME
    target_venues = {jyo_to_name[c] for c in settings.target_jyo_code_list if c in jyo_to_name}
    schedule = {k: v for k, v in schedule.items() if k in target_venues}
    if not schedule:
        logger.warning(f"  対象会場なし（target_venues={target_venues}）")
        # dry-run でも Flex を作って終了
    logger.info(f"  対象会場: {list(schedule.keys())}")

    # ── 4. レースごとに予測・買い目生成 ─────────────────────────
    logger.info("[4/5] 予測・買い目生成")
    venue_results: dict[str, list[dict | None]] = {}

    for venue, race_ids in schedule.items():
        logger.info(f"  [{venue}] {len(race_ids)} レース処理開始")
        venue_results[venue] = []

        for race_id in race_ids:
            try:
                race_info: RaceInfo = scraper.fetch_today_entries(race_id)
            except Exception as e:
                logger.warning(f"    {race_id} 出走表取得失敗: {e}")
                venue_results[venue].append(None)
                continue

            # 不調期（1,7-9,11-12月）は専用モデルを使用
            _race_month = target_date.month
            _active_trainer = (
                bad_season_trainer
                if bad_season_trainer is not None and _race_month in _BAD_SEASON_MONTHS
                else trainer
            )
            result = predict_and_bet(race_info, fe, _active_trainer, org=org, race_date=target_date)
            venue_results[venue].append(result)

            if result and result["is_buy"]:
                hon = result['honmei_num']
                baren_str = "-".join([hon] + result["baren_partners"]) or "なし"
                sf_str = " / ".join(f"{hon}-{a}-{b}" for a, b in result["sanrenfuku_combos"]) or "なし"
                st_str = " / ".join(f"{hon}→{n2}→{n3}" for n2, n3 in result["sanrentan_combos"]) or "なし"
                logger.info(
                    f"    {result['race_num']:>3}  ◎{hon} {result['honmei_name']}"
                    f"  (prob={result['honmei_prob']:.2%})"
                )
                logger.info(f"          馬連: {baren_str}")
                logger.info(f"          3連複: {sf_str}")
                logger.info(f"          3連単: {st_str}")
            elif result:
                logger.info(f"    {result['race_num']:>3}  見送り")

    scraper.close()

    # ── 4.5 予測ログ保存（結果は後で週次バッチが更新）────────────
    _save_prediction_log(target_date, venue_results)

    # 買い対象0件の場合: 見送り理由サマリーをログに出力
    all_results = [r for results in venue_results.values() for r in results if r is not None]
    buy_count   = sum(1 for r in all_results if r.get("is_buy"))
    if buy_count == 0 and all_results:
        from collections import Counter
        reason_counter: Counter = Counter()
        for r in all_results:
            if r.get("is_maiden"):
                reason_counter["maiden（新馬・未勝利）"] += 1
            if r.get("skip_prob"):
                reason_counter[f"prob（最高確率={r['honmei_prob']:.1%} < 閾値{r.get('prob_threshold', MIN_HONMEI_PROB):.1%}）"] += 1
            if r.get("skip_gap"):
                reason_counter[f"gap（信頼度差={r.get('gap', 0):.3f} < {MIN_CONFIDENCE_GAP}）"] += 1
            if r.get("skip_ev"):
                reason_counter["ev（EV閾値未達）"] += 1
            if r.get("skip_summer_dirt"):
                reason_counter["季節フィルター: 夏ダート（7-9月）"] += 1
            if r.get("skip_mark_o"):
                reason_counter[f"季節フィルター: {r.get('race_month', '')}月は◎のみ（○見送り）"] += 1
        logger.warning(f"  ⚠️  本日の買い対象: 0件（全{len(all_results)}R を見送り）")
        logger.warning("  見送り理由内訳:")
        for reason, cnt in reason_counter.most_common():
            logger.warning(f"    {reason}: {cnt}R")
        # 最高確率レースを1件表示（デバッグ用）
        top_race = max(all_results, key=lambda r: r.get("honmei_prob", 0))
        logger.warning(
            f"  本日の最高予測確率レース: {top_race.get('race_name', '')} "
            f"({top_race.get('honmei_name', '')} {top_race.get('honmei_prob', 0):.1%})"
        )

    # ── 5. Flex Message 構築・送信 ──────────────────────────────
    logger.info("[5/5] Flex Message 構築・送信")
    flex_msg = build_flex_message(target_date, venue_results)

    # Flex JSON を docs/ に保存（git commit → GitHub Pages 経由で Webhook が取得）
    # logs/ にも保存（ローカル確認用）
    flex_json_str = json.dumps(flex_msg, ensure_ascii=False, indent=2)
    flex_cache = ROOT / "logs" / f"flex_{target_date}.json"
    flex_cache.parent.mkdir(parents=True, exist_ok=True)
    flex_cache.write_text(flex_json_str, encoding="utf-8")
    logger.info(f"  Flex キャッシュ保存 (logs): {flex_cache}")

    if args.dry_run:
        logger.info("  [dry-run] LINE 送信・docs/ 保存スキップ")
    else:
        docs_flex = ROOT / "docs" / f"flex_{target_date}.json"
        docs_flex.parent.mkdir(parents=True, exist_ok=True)
        docs_flex.write_text(flex_json_str, encoding="utf-8")
        logger.info(f"  Flex キャッシュ保存 (docs): {docs_flex}")
        send_line_flex(flex_msg)

    # ── 6. GitHub Pages 成績ページ更新 ───────────────────────────
    try:
        from src.line.stats_page import generate_stats_page
        generate_stats_page()
        logger.info("  stats.html 生成完了")
    except Exception as e:
        logger.warning(f"  stats.html 生成失敗（続行）: {e}")

    elapsed = time.time() - t0
    logger.info(f"完了（{elapsed/60:.1f}分）")


if __name__ == "__main__":
    main()
