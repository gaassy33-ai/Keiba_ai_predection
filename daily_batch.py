"""
daily_batch.py
当日出走表取得 → LightGBM 予測 → LINE Flex Message 送信

実行方法:
    python daily_batch.py                    # 当日
    python daily_batch.py --date 2026-03-08  # 日付指定

買い目（単勝 + 馬連のみ・回収率100%超券種に絞り込み）:
    - レース絞り込み: honmei_prob ≥ 0.15 かつ 信頼度差 ≥ 0.05
    - 単勝: ◎ 1点
    - 複勝: ◎ 1点（参考表示）
    - 馬連: ◎ - EV上位3頭 最大3点（トリガミ除外）

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
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from src.scraper.netkeiba_scraper import NetkeibaScraper, RaceInfo
from config.settings import settings

# ======================================================================
# 定数
# ======================================================================
MIN_HONMEI_PROB    = 0.15   # ◎最低勝率（以下はケン）
MIN_CONFIDENCE_GAP = 0.05   # ◎-対抗 信頼度差（以下はケン）
EV_PARTNER_TOP_N   = 5      # EV計算の候補プール（上位5頭から選ぶ）
MAX_BAREN_TICKETS  = 3      # 馬連最大点数
TORIKAMI_THRESHOLD = 1.05   # トリガミ判定閾値（推定オッズがこれ未満は除外）

# JRA 控除率
JRA_TAKE = {"馬連": 0.225}

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


def _est_odds(prob: float, bet_type: str) -> float:
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
# 1レース予測・買い目生成
# ======================================================================

def predict_and_bet(
    race_info: RaceInfo,
    fe: FeatureEngineer,
    trainer: ModelTrainer,
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
        "father":         e.father_name,
        "mother_father":  e.mother_father_name,
    } for e in entries])

    # --- 特徴量生成 ---
    from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP
    gc_code = GROUND_CONDITION_MAP.get(race_info.ground_condition, -1)
    wx_code = WEATHER_MAP.get(race_info.weather, -1)

    try:
        feat_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=race_info.course_type,
            distance=race_info.distance,
            ground_condition_code=gc_code,
            weather_code=wx_code,
        )
    except Exception as e:
        logger.warning(f"特徴量生成失敗 {race_info.race_id}: {e}")
        return None

    X = feat_df[FeatureEngineer.FEATURE_COLUMNS].fillna(0)
    win_probs = trainer.model.predict(X)
    if trainer.place_model is not None:
        place_probs = trainer.place_model.predict(X)
        probs = 0.7 * win_probs + 0.3 * place_probs
    else:
        probs = win_probs

    pred_df = feat_df[["horse_id", "horse_name", "horse_number"]].copy()
    pred_df["win_prob"] = probs
    pred_df = pred_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

    # --- レース絞り込み ---
    honmei_prob = float(pred_df.iloc[0]["win_prob"])
    taikou_prob = float(pred_df.iloc[1]["win_prob"]) if len(pred_df) > 1 else 0.0
    is_skip = (
        honmei_prob < MIN_HONMEI_PROB
        or (honmei_prob - taikou_prob) < MIN_CONFIDENCE_GAP
    )

    honmei_row = pred_df.iloc[0]
    honmei_num  = str(int(honmei_row["horse_number"]))
    honmei_name = str(honmei_row["horse_name"])

    # レース番号（race_id 末尾2桁）
    race_num_str = f"{int(race_info.race_id[-2:])}R"

    result = {
        "race_id":       race_info.race_id,
        "race_name":     race_info.race_name,
        "race_num":      race_num_str,
        "is_buy":        False,
        "honmei_num":    honmei_num,
        "honmei_name":   honmei_name,
        "honmei_prob":   round(honmei_prob, 3),
        "baren_partners": [],
    }

    if is_skip:
        return result

    # --- EV 相手選び（馬連用） ---
    odds_map = {e.horse_id: e.odds for e in entries if e.odds}
    candidates = pred_df[pred_df["horse_id"] != honmei_row["horse_id"]].head(EV_PARTNER_TOP_N)
    scored = []
    for _, row in candidates.iterrows():
        hid  = str(row["horse_id"])
        prob = float(row["win_prob"])
        odds = _parse_odds(odds_map.get(hid, 5.0))
        if np.isnan(odds) or odds <= 1.0:
            odds = 5.0
        scored.append((hid, prob * odds, str(int(row["horse_number"]))))
    scored.sort(key=lambda x: x[1], reverse=True)
    ev_top = scored[:MAX_BAREN_TICKETS]

    # --- 馬連 トリガミフィルタ ---
    all_ids  = pred_df["horse_id"].tolist()
    all_odds_raw = [_parse_odds(odds_map.get(hid, float("nan"))) for hid in all_ids]
    valid_pairs = [(hid, o) for hid, o in zip(all_ids, all_odds_raw)
                   if not np.isnan(o) and o > 1.0]
    mkt_probs: list[float] = []
    valid_ids: list[str]   = []
    if valid_pairs:
        vids, vodds = zip(*valid_pairs)
        valid_ids  = list(vids)
        mkt_probs  = _market_probs(list(vodds)).tolist()

    def vidx(horse_id: str) -> int | None:
        return valid_ids.index(horse_id) if horse_id in valid_ids else None

    hi = vidx(honmei_row["horse_id"])
    baren_nums: list[str] = []
    for hid, _ev, num in ev_top:
        vi = vidx(hid)
        if hi is not None and vi is not None and mkt_probs:
            e_odds = _est_odds(_prob_quinella(mkt_probs, hi, vi), "馬連")
            if e_odds < TORIKAMI_THRESHOLD:
                continue
        baren_nums.append(num)

    result["is_buy"]         = True
    result["baren_partners"] = baren_nums
    return result


# ======================================================================
# Flex Message 構築
# ======================================================================

def _race_row_component(race_result: dict | None, race_num: int) -> list[dict]:
    """1レース分の行コンポーネント（separator + box）を返す"""
    label = f"{race_num:>2}R"  # 右詰めで幅を揃える（1R→" 1R"）

    if race_result is None or not race_result.get("is_buy"):
        # 見送りレース
        body_content = {
            "type": "box",
            "layout": "horizontal",
            "paddingStart": "14px", "paddingEnd": "14px",
            "paddingTop": "7px", "paddingBottom": "7px",
            "contents": [
                {
                    "type": "text", "text": label,
                    "size": "sm", "color": "#cccccc",
                    "flex": 0,
                },
                {
                    "type": "text", "text": "— 見送り",
                    "size": "sm", "color": "#cccccc",
                    "flex": 1, "margin": "md",
                },
            ],
        }
    else:
        # 購入レース
        hon_num  = race_result["honmei_num"]
        hon_name = race_result["honmei_name"]
        partners = race_result["baren_partners"]
        baren_str = (
            f"馬連: {hon_num} - {', '.join(partners)}"
            if partners else "馬連: なし（単複のみ）"
        )
        body_content = {
            "type": "box",
            "layout": "horizontal",
            "paddingStart": "14px", "paddingEnd": "14px",
            "paddingTop": "9px", "paddingBottom": "9px",
            "contents": [
                {
                    "type": "text", "text": label,
                    "size": "sm", "weight": "bold", "color": "#1a5533",
                    "flex": 0,
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "flex": 1, "margin": "md",
                    "contents": [
                        {
                            "type": "text",
                            "text": f"◎{hon_num} {hon_name}（単・複）",
                            "size": "sm", "weight": "bold",
                            "color": "#1a5533", "wrap": True,
                        },
                        {
                            "type": "text", "text": baren_str,
                            "size": "xs", "color": "#555555",
                            "margin": "xs",
                        },
                    ],
                },
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

    buy_count = sum(1 for r in by_num.values() if r and r.get("is_buy"))
    total_count = len(by_num)

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
                    "text": f"参加 {buy_count}R / {total_count}R中",
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

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"daily_batch 開始: {target_date}")
    logger.info("=" * 60)

    # ── 1. モデル・履歴読み込み ──────────────────────────────────
    logger.info("[1/5] モデル・FeatureEngineer 読み込み")
    trainer = ModelTrainer.load(settings.model_path)

    # ── 2. FeatureEngineer 構築 ──────────────────────────────────
    # feature_stats.pkl（リポジトリに含まれる）から推論用統計を読み込む。
    # GitHub Actions のクリーン環境では data/raw/*.csv が存在しないため
    # FeatureEngineer(history_df) は使用しない。
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
    # ※ fetch_race_schedule_by_date が venue_name を返すため、
    #    設定の jyo_code を venue_name に変換して照合する
    JYO_TO_NAME = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    }
    target_venues = {JYO_TO_NAME[c] for c in settings.target_jyo_code_list
                     if c in JYO_TO_NAME}
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

            result = predict_and_bet(race_info, fe, trainer)
            venue_results[venue].append(result)

            if result and result["is_buy"]:
                partners_str = ", ".join(result["baren_partners"])
                logger.info(
                    f"    {result['race_num']:>3}  ◎{result['honmei_num']} "
                    f"{result['honmei_name']} "
                    f"(prob={result['honmei_prob']:.2%})  "
                    f"馬連: {result['honmei_num']}-{partners_str}"
                )
            elif result:
                logger.info(f"    {result['race_num']:>3}  見送り")

    scraper.close()

    # ── 5. Flex Message 構築・送信 ──────────────────────────────
    logger.info("[5/5] Flex Message 構築・送信")
    flex_msg = build_flex_message(target_date, venue_results)

    # Webhook キャッシュとして常に保存（「今日の予想」キーワード応答で使用）
    flex_cache = ROOT / "logs" / f"flex_{target_date}.json"
    flex_cache.write_text(
        json.dumps(flex_msg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  Flex キャッシュ保存: {flex_cache}")

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
