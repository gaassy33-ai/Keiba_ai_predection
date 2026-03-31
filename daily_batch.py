"""
daily_batch.py
当日出走表取得 → LightGBM 予測 → LINE Flex Message 送信

実行方法:
    python daily_batch.py                    # 当日
    python daily_batch.py --date 2026-03-08  # 日付指定

買い目（回収率100%超券種に絞り込み）:
    - レース絞り込み: honmei_prob ≥ 0.15 かつ 信頼度差 ≥ 0.05
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
MIN_HONMEI_PROB      = 0.25   # ○最低勝率（以下は△）バックテスト2026実測: 0.30は買いレースが激減
MARK_STRONG_PROB     = 0.35   # ◎最低勝率（○との境界）
MIN_CONFIDENCE_GAP   = 0.05   # 勝率差フィルター（以下は△）
# EV フィルタ（model_prob × odds ≥ 閾値 のみ買い）
# バックテスト2026実測: EV≥1.15は買いレース42Rに激減→1.05に戻す
EV_THRESHOLD         = 1.05   # 5% のポジティブエッジを要求
# 改善②: 出走頭数ペナルティ（1頭増えるごとに+0.3%、上限+5%）
ENTRIES_THRESHOLD_ADJ = 0.003   # per horse over 10
ENTRIES_THRESHOLD_BASE = 10
ENTRIES_THRESHOLD_CAP  = 0.05
# 改善③: レースクラス別ブースト（未勝利・新馬は+5%上乗せ）
MAIDEN_PROB_BOOST    = 0.05
MAIDEN_KEYWORDS      = ("新馬", "未勝利")
EV_PARTNER_TOP_N     = 5      # EV計算の候補プール（上位5頭）
MAX_BAREN_TICKETS    = 3      # 馬連最大点数
MAX_UMATAN_TICKETS   = 3      # 馬単最大点数
MAX_SANRENFUKU_TICKETS = 5    # 3連複最大点数
MAX_SANRENTAN_TICKETS  = 5    # 3連単最大点数
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

    # 改善②: 出走頭数に応じた動的閾値（多頭数は確率が薄まるため厳格化）
    base_threshold = MIN_HONMEI_PROB
    entries_adj = min(
        max(0, n_entries - ENTRIES_THRESHOLD_BASE) * ENTRIES_THRESHOLD_ADJ,
        ENTRIES_THRESHOLD_CAP,
    )
    prob_threshold = base_threshold + entries_adj

    # 改善③: 新馬・未勝利戦は閾値を追加ブースト
    is_maiden = any(k in str(race_info.race_name) for k in MAIDEN_KEYWORDS)
    if is_maiden:
        prob_threshold += MAIDEN_PROB_BOOST

    gap_ok = gap >= MIN_CONFIDENCE_GAP

    # 改善⑨: EV フィルタ — feat_df["odds"] は build_entry_features で付加済み
    honmei_id_str = str(pred_df.iloc[0]["horse_id"])
    odds_col_map  = feat_df.set_index("horse_id")["odds"] if "odds" in feat_df.columns else {}
    _raw_odds = odds_col_map.get(honmei_id_str)
    honmei_odds_pre = float(_raw_odds) if _raw_odds is not None else float("nan")
    honmei_ev = (honmei_prob * honmei_odds_pre
                 if _raw_odds is not None and not np.isnan(honmei_odds_pre) else float("nan"))
    # オッズ未取得時（nan）は EV フィルタをスキップして確率フィルタのみ適用
    ev_ok = (np.isnan(honmei_ev) or honmei_ev >= EV_THRESHOLD)

    if not gap_ok or honmei_prob < prob_threshold or not ev_ok:
        mark = "△"
    elif honmei_prob >= MARK_STRONG_PROB:
        mark = "◎"
    else:
        mark = "○"

    is_skip = (mark == "△")

    logger.info(
        f"  probs: {mark}{pred_df.iloc[0]['horse_name']}={honmei_prob:.4f}"
        f"  対抗={taikou_prob:.4f}  差={gap:.4f}"
        f"  EV={honmei_ev:.2f}({'OK' if ev_ok else 'NG'})"
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
        "prob_threshold":     round(prob_threshold, 3),
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

        # 3連複: EV = モデル確率(trio) × 推定市場配当  ← 馬連と同じモデルEVアプローチ
        # (num_a, num_b, ev_score, e_od)
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

            ev = model_p * e_od   # EV: モデル期待確率 × 推定配当
            sf_all.append((num_a, num_b, ev, e_od))

        sf_all.sort(key=lambda x: -x[2])   # EV降順
        sf_sel = sf_all[:MAX_SANRENFUKU_TICKETS]
        sf_est = [od for _, _, _, od in sf_sel]
        sanrenfuku_combos = (
            [(a, b) for a, b, _, _ in sf_sel]
            if (not sf_est or _synth_odds(sf_est) >= 1.0)
            else []
        )

        # 3連単: EV = モデル確率(◎→A→B) × 推定市場配当
        # (num_2, num_3, ev_score, e_od)
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

            ev = model_p * e_od
            st_all.append((num_2, num_3, ev, e_od))

        st_all.sort(key=lambda x: -x[2])   # EV降順
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

    detail_items: list[dict] = [
        {"type": "text",
         "text": f"{mark}{hon_num} {hon_name}（単・複）",
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

    # ── 1. モデル・履歴読み込み ──────────────────────────────────
    logger.info("[1/5] モデル・FeatureEngineer 読み込み")
    trainer = ModelTrainer.load(settings.model_path, org=org)

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

            result = predict_and_bet(race_info, fe, trainer, org=org)
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

    docs_flex = ROOT / "docs" / f"flex_{target_date}.json"
    docs_flex.parent.mkdir(parents=True, exist_ok=True)
    docs_flex.write_text(flex_json_str, encoding="utf-8")
    logger.info(f"  Flex キャッシュ保存 (docs): {docs_flex}")

    if args.dry_run:
        logger.info("  [dry-run] LINE 送信スキップ")
    else:
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
