"""
daily_batch.py
==============
当日出走表取得 → LTR + EV 馬連買い目生成 → LINE テキスト配信

【戦略設定】
  config/strategy.yaml で EV 閾値・対象コース・点数上限を管理。
  コードを変更せず YAML を編集するだけで戦略を切り替えられる。

【買い目ロジック】
  - LTR (LambdaRank) モデル → temperature scaling → Harville 馬連確率（軸馬流し）
  - EV = P_model × 推定馬連オッズ（市場オッズ・控除率 17.5% から計算）
  - EV ≥ min_ev_threshold のペアを EV 降順 top max_bets_per_race 点購入
  - target_surface（ダート/芝）でレースを絞り込み

【時系列リーク対策】
  feature_stats.pkl（学習時の統計量）を使用し、当日データは含まない。

実行方法:
    .venv/bin/python daily_batch.py                    # 当日
    .venv/bin/python daily_batch.py --date 2026-04-29  # 日付指定
    .venv/bin/python daily_batch.py --dry-run          # LINE 送信スキップ
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import settings, strategy
from src.betting.ltr_ev_engine import QuinellaBet, evaluate_race
from src.betting.line_formatter import format_daily_message, format_race_section
from src.betting.line_wide_flex import send_wide_flex
from src.features.engineer import FeatureEngineer
from src.model.trainer import LTRTrainer
from src.model.gatekeeper import GatekeeperTrainer
from src.model.pair_calibrator import PairCalibrator
from src.scraper.base_scraper import RaceInfo
from src.scraper.netkeiba_scraper import NetkeibaScraper

# ──────────────────────────────────────────────────────────────────────────────
# ロガー初期化
# ──────────────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout,              level="INFO",  format=_fmt, colorize=True)
logger.add("logs/daily_batch.log", level="DEBUG", format=_fmt, rotation="20 MB")


# ──────────────────────────────────────────────────────────────────────────────
# 予測ログ保存
# ──────────────────────────────────────────────────────────────────────────────

PREDICTIONS_DIR = ROOT / "data" / "logs" / "predictions"

_PRED_LOG_COLUMNS = [
    "date", "race_id", "race_name", "course_type", "distance",
    "horse_num_i", "horse_id_i", "horse_name_i",
    "horse_num_j", "horse_id_j", "horse_name_j",
    "odds_i", "odds_j",
    "est_quinella_odds",  # 推定馬連オッズ（市場・控除率 17.5% 反映）
    "p_model",            # AI 予測馬連確率
    "p_market",           # 市場馬連確率
    "ev",                 # 算出 EV
    "ev_rank",            # レース内 EV 順位（1 = 最高）
]


def save_prediction_log(
    target_date: date,
    race_bets: list[dict],   # [{"race_id", "race_name", "course_type", "distance", "bets": list[QuinellaBet]}, ...]
) -> Path:
    """
    日次予測ログを data/logs/predictions/YYYY-MM-DD.csv に保存する。

    当日分が既に存在する場合は上書きする。
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{target_date}.csv"

    rows: list[dict] = []
    for race in race_bets:
        for rank, bet in enumerate(race["bets"], start=1):
            rows.append({
                "date":         str(target_date),
                "race_id":      race["race_id"],
                "race_name":    race["race_name"],
                "course_type":  race["course_type"],
                "distance":     race["distance"],
                "horse_num_i":  bet.horse_num_i,
                "horse_id_i":   bet.horse_id_i,
                "horse_name_i": bet.horse_name_i,
                "horse_num_j":  bet.horse_num_j,
                "horse_id_j":   bet.horse_id_j,
                "horse_name_j": bet.horse_name_j,
                "odds_i":       bet.odds_i,
                "odds_j":       bet.odds_j,
                "est_quinella_odds": bet.est_quinella_odds,
                "p_model":      bet.p_model,
                "p_market":     bet.p_market,
                "ev":           bet.ev,
                "ev_rank":      rank,
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PRED_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  予測ログ保存: {out_path}  ({len(rows)} 行)")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# シャドー稼働ログ（フィルター落ちも含む全候補ペア）
# ──────────────────────────────────────────────────────────────────────────────

SHADOW_DIR = ROOT / "data" / "logs" / "shadow_candidates"


def _append_shadow_candidates(target_date: date, candidates_df: pd.DataFrame) -> None:
    """
    1レース分の全候補ペア（フィルター落ち含む）を
    data/logs/shadow_candidates/YYYY-MM-DD.csv に追記保存する。
    キャリブレーション検証（p_model vs 実際の的中率）のため、
    フィルターで落ちたペアも would_buy=False として残す。
    """
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SHADOW_DIR / f"{target_date}.csv"
    candidates_df = candidates_df.copy()
    candidates_df.insert(0, "date", str(target_date))
    write_header = not out_path.exists()
    candidates_df.to_csv(out_path, mode="a", index=False, header=write_header)


# ──────────────────────────────────────────────────────────────────────────────
# LINE 送信（テキスト形式）
# ──────────────────────────────────────────────────────────────────────────────

_LINE_MSG_ID_FILE = ROOT / "logs" / "line_sent_messages.json"


def _load_sent_message_ids() -> list[dict]:
    if _LINE_MSG_ID_FILE.exists():
        try:
            return json.loads(_LINE_MSG_ID_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_sent_message_ids(records: list[dict]) -> None:
    _LINE_MSG_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LINE_MSG_ID_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _delete_line_message(message_id: str) -> bool:
    url = f"https://api.line.me/v2/bot/message/{message_id}"
    headers = {"Authorization": f"Bearer {settings.line_channel_access_token}"}
    resp = requests.delete(url, headers=headers, timeout=15)
    if resp.status_code == 200:
        logger.info(f"  LINE メッセージ削除: id={message_id}")
        return True
    logger.warning(f"  LINE メッセージ削除失敗: id={message_id} status={resp.status_code}")
    return False


def _delete_old_line_messages() -> None:
    """保存済み旧メッセージをすべて削除する。"""
    records = _load_sent_message_ids()
    for rec in records:
        _delete_line_message(rec["id"])
    _save_sent_message_ids([])


def send_line_text(text: str, target_date_str: str = "") -> None:
    """
    LINE Messaging API の Push Message でテキストメッセージを送信する。
    送信前に前回メッセージを削除し、送信後にメッセージ ID を保存する。
    """
    _delete_old_line_messages()

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.line_channel_access_token}",
    }
    payload = {
        "to": settings.line_target_user_id,
        "messages": [{"type": "text", "text": text}],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        logger.error(f"LINE 送信失敗: {resp.status_code} {resp.text}")
        resp.raise_for_status()
    logger.info(f"LINE 送信完了 (status={resp.status_code})")

    try:
        from datetime import datetime
        body = resp.json()
        sent_msgs = body.get("sentMessages", [])
        records = _load_sent_message_ids()
        for m in sent_msgs:
            records.append({
                "id":      m["id"],
                "date":    target_date_str,
                "sent_at": datetime.now().isoformat(),
            })
        _save_sent_message_ids(records)
        logger.info(f"  送信メッセージID保存: {[m['id'] for m in sent_msgs]}")
    except Exception as e:
        logger.warning(f"  メッセージID保存失敗（続行）: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 1レース処理
# ──────────────────────────────────────────────────────────────────────────────

def process_race(
    race_info:       RaceInfo,
    fe:              FeatureEngineer,
    ltr:             LTRTrainer,
    race_date:       date,
    pair_calibrator: PairCalibrator | None = None,
    gatekeeper:      GatekeeperTrainer | None = None,
) -> dict | None:
    """
    1レースを処理し、EV フィルタ済みの買い目情報を返す。

    G1特別モード（cfg.always_predict_g1=True かつ race_info.is_g1=True）の場合、
    target_surface / min_ev_threshold / axis_max_odds をバイパスして必ず推論を実行する。

    Returns
    -------
    dict | None
        {"race_id", "race_name", "course_type", "distance", "is_g1", "bets": list[QuinellaBet]}
        または None（処理不可・条件不一致）
    """
    import dataclasses
    cfg = strategy.betting

    # ── G1特別モード判定 ─────────────────────────────────────────
    is_g1_race  = getattr(race_info, "is_g1", False)
    g1_override = cfg.always_predict_g1 and is_g1_race
    if g1_override:
        logger.info(f"  🏆 G1特別モード適用: {race_info.race_id} ({race_info.race_name})")

    # ── コースフィルター（G1はバイパス）────────────────────────────
    course_type = str(race_info.course_type or "")
    if not g1_override and course_type not in cfg.target_surface:
        logger.debug(f"  skip {race_info.race_id}: コース={course_type} 対象外")
        return None

    distance = int(race_info.distance or 0)
    if distance == 0 or distance >= 2750:
        logger.debug(f"  skip {race_info.race_id}: 距離={distance}m 範囲外")
        return None

    entries = race_info.entries
    if len(entries) < 3:
        logger.debug(f"  skip {race_info.race_id}: 出走頭数不足 ({len(entries)}頭)")
        return None

    # ── entry_df 組み立て ──────────────────────────────────────────
    entry_df = pd.DataFrame([{
        "horse_id":       e.horse_id,
        "horse_name":     e.horse_name,
        "horse_number":   e.horse_number,
        "frame_number":   e.frame_number,
        "sex":            getattr(e, "sex", ""),
        "age":            getattr(e, "age", np.nan),
        "weight_carried": getattr(e, "weight_carried", np.nan),
        "jockey_id":      e.jockey_id,
        "trainer_name":   getattr(e, "trainer_name", ""),
        "father":         getattr(e, "father_name", ""),
        "mother_father":  getattr(e, "mother_father_name", ""),
        "odds":           e.odds,
    } for e in entries])

    # ── 特徴量生成 ─────────────────────────────────────────────────
    from src.scraper.weather import GROUND_CONDITION_MAP, WEATHER_MAP
    gc_code = GROUND_CONDITION_MAP.get(race_info.ground_condition, -1)
    wx_code = WEATHER_MAP.get(race_info.weather, -1)
    rcc = FeatureEngineer._race_name_to_class_code(race_info.race_name or "")
    try:
        venue_code = int(race_info.race_id[4:6])
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
        logger.warning(f"  特徴量生成失敗 {race_info.race_id}: {e}")
        return None

    if len(feat_df) < 3:
        return None

    # ── オッズマップ・馬名マップ ────────────────────────────────────
    odds_map: dict[str, float] = {}
    horse_names: dict[str, str] = {}
    for e in entries:
        hid = str(e.horse_id)
        horse_names[hid] = str(e.horse_name)
        if e.odds:
            try:
                o = float(str(e.odds).replace(",", "").strip())
                if o > 1.0:
                    odds_map[hid] = o
            except (ValueError, TypeError):
                pass

    # ── EV 計算・買い目抽出 ─────────────────────────────────────────
    # G1特別モード: min_ev_threshold / axis_max_odds / est_quinella_odds_max を緩和した
    # 一時的な cfg を生成して渡す。evaluate_race() 本体は変更しない。
    effective_cfg = (
        dataclasses.replace(
            cfg,
            min_ev_threshold      = 0.0,    # EV制限なし（全ペアをEV降順で出力）
            axis_max_odds         = 0.0,    # 軸馬オッズ上限撤廃（人気薄の軸も許容）
            est_quinella_odds_max = 100.0,  # G1はここまで緩和（999は非現実的な超穴を拾いすぎる）
            min_p_model_threshold = 0.0,    # G1は18頭前後で個別確率が低くなるため撤廃
            longshot_odds_max     = 50.0,   # G1は50倍まで（通常の30倍より緩め）
        )
        if g1_override else cfg
    )

    # G1特別モード（全フィルターをバイパス）は Gatekeeper も適用しない
    effective_gatekeeper = None if g1_override else gatekeeper
    gatekeeper_threshold = getattr(cfg, "gatekeeper_threshold", 0.50)

    shadow_mode = getattr(cfg, "shadow_mode", False)
    if shadow_mode:
        bets, candidates_df = evaluate_race(
            feat_df, odds_map, horse_names, ltr, effective_cfg,
            return_candidates=True,
            pair_calibrator=pair_calibrator,
            gatekeeper=effective_gatekeeper,
            gatekeeper_threshold=gatekeeper_threshold,
        )
        if len(candidates_df):
            candidates_df.insert(0, "race_id", race_info.race_id)
            candidates_df.insert(1, "race_name", race_info.race_name or "")
            candidates_df.insert(2, "is_g1", is_g1_race)
            _append_shadow_candidates(race_date, candidates_df)
    else:
        bets = evaluate_race(
            feat_df, odds_map, horse_names, ltr, effective_cfg,
            pair_calibrator=pair_calibrator,
            gatekeeper=effective_gatekeeper,
            gatekeeper_threshold=gatekeeper_threshold,
        )

    if not bets:
        logger.debug(f"  {race_info.race_id}: 買い目なし（G1={is_g1_race}）")
        return None

    race_num = f"{int(race_info.race_id[-2:])}R"

    prefix = "🏆 G1 " if is_g1_race else "✅"
    logger.info(f"  {prefix} {race_info.race_id}  {course_type}{distance}m  {len(bets)}点")
    for bet in bets:
        ev_note = " ※G1特例" if g1_override and bet.ev < 1.0 else ""
        logger.info(
            f"       {bet.horse_num_i}番({bet.horse_name_i}) - "
            f"{bet.horse_num_j}番({bet.horse_name_j})"
            f"  EV={bet.ev:.2f}  想定{bet.est_quinella_odds:.1f}倍{ev_note}"
        )

    return {
        "race_id":     race_info.race_id,
        "race_name":   race_info.race_name or race_num,
        "course_type": course_type,
        "distance":    distance,
        "is_g1":       is_g1_race,
        "bets":        bets,
    }


# ──────────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LTR + EV 馬連 日次バッチ")
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="予測対象日 (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LINE 送信をスキップしてローカルに結果を出力",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="当日の重複送信ガードを無視して強制送信（デバッグ用）",
    )
    args = parser.parse_args()
    target_date = date.fromisoformat(args.date)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"daily_batch 開始: {target_date}")
    logger.info(f"  戦略: {strategy.betting.target_surface}  "
                f"EV≥{strategy.betting.min_ev_threshold}  "
                f"max_bets={strategy.betting.max_bets_per_race or '無制限'}")
    logger.info("=" * 60)

    # ── 過去日付ガード ──────────────────────────────────────────────
    _today = date.today()
    if (_today - target_date).days >= 2 and not args.dry_run:
        logger.error(f"過去日付 ({target_date}) での実行を中止します（今日: {_today}）")
        logger.error("正しい日付で再実行してください。--dry-run を使用して下さい。")
        return

    # ── 当日重複送信ガード ──────────────────────────────────────────
    # dry-run / --force / shadow_mode（LINE送信自体を行わない）の場合はスキップ
    if not args.dry_run and not args.force and not strategy.betting.shadow_mode:
        already_sent = [
            r for r in _load_sent_message_ids()
            if r.get("date") == str(target_date)
        ]
        if already_sent:
            logger.error("=" * 60)
            logger.error(f"⛔ 本日 ({target_date}) は既にLINE通知済みです")
            logger.error(f"   送信済みメッセージ: {len(already_sent)}件")
            logger.error(f"   送信日時: {already_sent[0].get('sent_at', '不明')}")
            logger.error("   重複送信を防ぐため処理を中断します。")
            logger.error("   内容確認のみなら --dry-run を使用してください。")
            logger.error("   強制再送信が必要な場合のみ --force を使用してください。")
            logger.error("=" * 60)
            return

    # ── [1] モデル読み込み ─────────────────────────────────────────
    logger.info("[1/5] LTR モデル読み込み")
    ltr_path = strategy.ltr_model_path
    if not ltr_path.exists():
        logger.error(f"LTR モデルが見つかりません: {ltr_path}")
        logger.error("train_ltr_model.py を実行してモデルを作成してください。")
        return
    ltr = LTRTrainer.load(ltr_path)
    logger.info(f"  モデル読み込み完了: NDCG@3={ltr.oof_ndcg3:.4f}  "
                f"特徴量={len(ltr.feature_columns)}個")

    # ── [1b] Two-Brain System: Gatekeeper・PairCalibrator 読み込み ──
    # いずれも存在しない場合は None のまま継続（既存ロジックに完全フォールバック）。
    gatekeeper: GatekeeperTrainer | None = None
    gk_path = strategy.gatekeeper_model_path
    if gk_path.exists():
        gatekeeper = GatekeeperTrainer.load(gk_path)
        logger.info(f"  Gatekeeper 読み込み完了: AUC={gatekeeper.oof_auc:.4f}  "
                    f"閾値={strategy.betting.gatekeeper_threshold}")
    else:
        logger.info(f"  Gatekeeper モデル未検出（{gk_path}）: 軸馬フィルターは未適用")

    pair_calibrator: PairCalibrator | None = None
    pc_path = strategy.pair_calibrator_path
    if pc_path.exists():
        pair_calibrator = PairCalibrator.load(pc_path)
        logger.info(f"  PairCalibrator 読み込み完了: "
                    f"学習サンプル={pair_calibrator.n_samples}件（的中{pair_calibrator.n_positive}件）")
    else:
        logger.info(f"  PairCalibrator 未検出（{pc_path}）: P_model 補正は未適用")

    # ── [2] FeatureEngineer 構築 ───────────────────────────────────
    logger.info("[2/5] FeatureEngineer 構築")
    stats_path = strategy.stats_path
    if not stats_path.exists():
        logger.error(f"feature_stats.pkl が見つかりません: {stats_path}")
        return
    fe = FeatureEngineer.from_stats(stats_path)
    logger.info("  FeatureEngineer 準備完了")

    # ── [3] 当日レーススケジュール取得 ────────────────────────────
    logger.info("[3/5] 当日レーススケジュール取得")
    scraper = NetkeibaScraper()
    try:
        schedule = scraper.fetch_race_schedule_by_date(target_date)
    except Exception as e:
        logger.error(f"スケジュール取得失敗: {e}")
        schedule = {}

    # 対象会場フィルター（settings.target_jyo_codes）
    jyo_to_name = scraper.VENUE_CODE_TO_NAME
    target_venues = {jyo_to_name[c] for c in settings.target_jyo_code_list if c in jyo_to_name}
    schedule = {k: v for k, v in schedule.items() if k in target_venues}
    logger.info(f"  対象会場: {list(schedule.keys())}  "
                f"合計 {sum(len(v) for v in schedule.values())} レース")

    # ── シャドー稼働: 再実行時の重複追記を防ぐため当日分をリセット ──────
    if strategy.betting.shadow_mode:
        logger.info("  🔒 シャドー稼働モード: LINE通知は行わず、全候補ペアをログ出力します")
        shadow_path = SHADOW_DIR / f"{target_date}.csv"
        if shadow_path.exists():
            shadow_path.unlink()

    # ── [4] レースごとに EV 計算・買い目生成 ─────────────────────
    logger.info("[4/5] EV 計算・買い目生成")
    race_bets: list[dict] = []     # process_race() の結果

    for venue, race_ids in schedule.items():
        logger.info(f"  [{venue}] {len(race_ids)} レース")
        for race_id in race_ids:
            try:
                race_info: RaceInfo = scraper.fetch_today_entries(race_id)
            except Exception as e:
                logger.warning(f"    {race_id} 出走表取得失敗: {e}")
                continue

            result = process_race(
                race_info, fe, ltr, target_date,
                pair_calibrator=pair_calibrator,
                gatekeeper=gatekeeper,
            )
            if result:
                # LINE 表示用に会場名を race_name に付与
                race_num_str = f"{int(race_id[-2:])}R"
                result["race_name"] = f"{venue}{race_num_str}"
                race_bets.append(result)

    scraper.close()

    total_bets = sum(len(r["bets"]) for r in race_bets)
    logger.info(f"  結果: {len(race_bets)} レース  合計 {total_bets} 点")

    # ── [5a] 予測ログ CSV 保存 ─────────────────────────────────────
    logger.info("[5/5] 予測ログ保存 & LINE 送信")
    if race_bets:
        save_prediction_log(target_date, race_bets)
    else:
        logger.info("  買い目なし: 予測ログ保存スキップ")

    # ── [5b] LINE テキスト生成・送信 ──────────────────────────────
    race_sections = [
        format_race_section(
            race_name=r["race_name"],
            course_type=r["course_type"],
            distance=r["distance"],
            bets=r["bets"],
            race_index=idx + 1,
            is_g1=r.get("is_g1", False),
        )
        for idx, r in enumerate(race_bets)
    ]
    # 空文字（買い目なしレース）を除外
    race_sections = [s for s in race_sections if s]

    # ── テキスト版（ログ・dry-run 保存用） ────────────────────────────
    message = format_daily_message(
        target_date   = target_date,
        race_sections = race_sections,
        cfg           = strategy.betting,
        total_bets    = total_bets,
    )

    logger.info("\n" + "─" * 50)
    logger.info("【LINE 送信テキスト プレビュー】")
    logger.info("─" * 50)
    for line in message.splitlines():
        logger.info(line)
    logger.info("─" * 50)

    if args.dry_run or strategy.betting.shadow_mode:
        reason = "shadow-mode" if strategy.betting.shadow_mode else "dry-run"
        logger.info(f"[{reason}] LINE 送信スキップ")
        dry_path = ROOT / "logs" / f"message_{target_date}.txt"
        dry_path.write_text(message, encoding="utf-8")
        logger.info(f"  メッセージ保存: {dry_path}")
    else:
        # ── Flex Message カルーセル送信 ─────────────────────────────
        send_wide_flex(
            target_date = target_date,
            race_bets   = race_bets,
            cfg         = strategy.betting,
        )

    elapsed = (time.time() - t0) / 60
    logger.info(f"完了（{elapsed:.1f}分）")


if __name__ == "__main__":
    main()
