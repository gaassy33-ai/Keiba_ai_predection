"""
line_wide_flex.py
=================
LTR+EV ワイド買い目を LINE Flex Message（カルーセル形式）で送信するモジュール。

会場ごとに1バブルを生成し、サマリーバブル（先頭）と組み合わせた
カルーセルを返す。横スクロールで会場を切り替えて確認できる。

【カルーセル構成】
  [📊 サマリー] → [🏇 東京] → [🏇 京都] → …

【各バブルのレイアウト】
  ヘッダー : 会場名 / 日付 / レース数・点数
  ボディ   : レースごとにグループ分けした買い目一覧
             ┌ 1R  ダート1300m  3点
             │  [EV3.54] ⑬サリーアン × ③ジャガーライズ
             │           想定 44.5倍
             │  ─────────
             └ 2R  ダート2100m  1点
                   …

【公開 API】
  build_wide_carousel(target_date, race_bets, cfg) -> dict
      LINE messages リストの1要素として渡せる Flex JSON を返す。

  send_wide_flex(target_date, race_bets, cfg) -> None
      カルーセルを LINE Push Message で送信する。
"""

from __future__ import annotations

import json
import re
import requests
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from src.betting.ltr_ev_engine import QuinellaBet
from config.settings import BettingConfig, settings


# ──────────────────────────────────────────────────────────────────────────────
# 定数・ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────

_WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]

_CIRCLED: dict[int, str] = {
    1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
    6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
    11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮",
    16: "⑯", 17: "⑰", 18: "⑱",
}

# 会場ごとのヘッダー背景色（濃色）
_VENUE_COLORS: dict[str, str] = {
    "東京": "#1565C0",
    "中山": "#B71C1C",
    "阪神": "#880E4F",
    "京都": "#4A148C",
    "中京": "#1B5E20",
    "小倉": "#BF360C",
    "函館": "#006064",
    "札幌": "#004D40",
    "新潟": "#33691E",
    "福島": "#4E342E",
}

# コース種別バッジ色
_COURSE_COLORS: dict[str, str] = {
    "ダート": "#6D4C41",
    "芝":    "#2E7D32",
}


def _lighten(hex_color: str, factor: float = 0.40) -> str:
    """hex カラー → 白方向に factor 分だけ明るくした hex を返す（alpha 不使用）。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f"#{r:02X}{g:02X}{b:02X}"


def _circled(num_str: str) -> str:
    try:
        return _CIRCLED.get(int(num_str), num_str)
    except (ValueError, TypeError):
        return str(num_str)


def _extract_venue(race_name: str) -> str:
    m = re.match(r"^([^\d]+)", race_name)
    return m.group(1).strip() if m else race_name


def _extract_race_num(race_name: str) -> str:
    m = re.search(r"(\d+)R", race_name)
    return m.group(1) if m else "?"


def _ev_color(ev: float) -> str:
    """EV 値 → バッジ背景色。"""
    if ev >= 5.0:
        return "#C62828"   # 赤 — 爆発級
    if ev >= 3.0:
        return "#E65100"   # 橙 — 高
    if ev >= 2.0:
        return "#F57F17"   # 黄 — 中
    return "#2E7D32"       # 緑 — EV 閾値ライン


# ──────────────────────────────────────────────────────────────────────────────
# コンポーネント ビルダー
# ──────────────────────────────────────────────────────────────────────────────

def _bet_row(bet: QuinellaBet) -> dict:
    """1点の馬連買い目を横並びカードで表現する。"""
    ev_color = _ev_color(bet.ev)
    ci = _circled(bet.horse_num_i)
    cj = _circled(bet.horse_num_j)

    return {
        "type": "box",
        "layout": "horizontal",
        "alignItems": "center",
        "paddingTop": "5px",
        "paddingBottom": "5px",
        "contents": [
            # ── EV バッジ ───────────────────────────────
            {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": ev_color,
                "cornerRadius": "sm",
                "width": "46px",
                "paddingTop": "3px",
                "paddingBottom": "3px",
                "contents": [
                    {
                        "type": "text",
                        "text": "EV",
                        "size": "xxs",
                        "color": "#FFFFFF",
                        "align": "center",
                    },
                    {
                        "type": "text",
                        "text": f"{bet.ev:.2f}",
                        "size": "sm",
                        "weight": "bold",
                        "color": "#FFFFFF",
                        "align": "center",
                    },
                ],
            },
            # ── 馬ペア + 想定オッズ ─────────────────────
            {
                "type": "box",
                "layout": "vertical",
                "flex": 1,
                "margin": "sm",
                "contents": [
                    {
                        "type": "text",
                        "text": f"{ci}{bet.horse_name_i} × {cj}{bet.horse_name_j}",
                        "size": "sm",
                        "weight": "bold",
                        "color": "#111111",
                        "wrap": True,
                    },
                    {
                        "type": "text",
                        "text": f"想定 {bet.est_quinella_odds:.1f}倍",
                        "size": "xs",
                        "color": "#888888",
                    },
                ],
            },
        ],
    }


_G1_GOLD   = "#B8860B"   # G1ヘッダー: ダークゴールド
_G1_LIGHT  = "#FFF8DC"   # G1ヘッダー背景: コーンシルク（薄い金）


def _race_block(race: dict, course_bg: str) -> list[dict]:
    """
    1レース分のコンポーネントリストを返す。

    G1レース（race["is_g1"]==True）の場合:
      - レースヘッダー背景をゴールド系に変更
      - ヘッダーに「🏆 G1」バッジを付与

    Returns
    -------
    list[dict]
        [レースヘッダー行, bet_row1, bet_row2, ...]
    """
    race_num    = _extract_race_num(race["race_name"])
    course_type = race["course_type"]
    distance    = race["distance"]
    bets        = race["bets"]
    is_g1       = race.get("is_g1", False)

    # G1の場合はゴールド系配色にオーバーライド
    header_bg  = _G1_LIGHT  if is_g1 else _lighten(course_bg, 0.85)
    badge_bg   = _G1_GOLD   if is_g1 else course_bg
    badge_text = "🏆"       if is_g1 else race_num
    badge_color = "#FFFFFF"

    components: list[dict] = [
        # ── レースヘッダー ────────────────────────────────
        {
            "type": "box",
            "layout": "horizontal",
            "alignItems": "center",
            "backgroundColor": header_bg,
            "cornerRadius": "sm",
            "paddingTop": "5px",
            "paddingBottom": "5px",
            "paddingStart": "8px",
            "paddingEnd": "8px",
            "contents": [
                # 番号バッジ（G1はトロフィー絵文字）
                {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": badge_bg,
                    "cornerRadius": "xs",
                    "width": "26px",
                    "height": "26px",
                    "contents": [
                        {
                            "type": "text",
                            "text": badge_text,
                            "size": "xs",
                            "weight": "bold",
                            "color": badge_color,
                            "align": "center",
                            "gravity": "center",
                        }
                    ],
                },
                # コース種別 + 距離（G1はレース名も表示）
                {
                    "type": "text",
                    "text": (
                        f"{race_num}R　{course_type} {distance}m　【G1】"
                        if is_g1 else
                        f"R　{course_type} {distance}m"
                    ),
                    "size": "sm",
                    "weight": "bold",
                    "color": _G1_GOLD if is_g1 else "#333333",
                    "flex": 1,
                    "margin": "xs",
                },
                # 点数
                {
                    "type": "text",
                    "text": f"{len(bets)}点",
                    "size": "xs",
                    "color": "#666666",
                    "align": "end",
                },
            ],
        },
        # ── 買い目リスト ──────────────────────────────────
        *[_bet_row(bet) for bet in bets],
    ]
    return components


def _venue_bubble(venue: str, races: list[dict], date_str: str) -> dict:
    """
    会場バブルを生成する。

    Parameters
    ----------
    venue    : "東京" / "京都" など
    races    : その会場のレース一覧（race_bets の部分リスト）
    date_str : 表示用日付文字列 "5/9(土)"
    """
    header_color = _VENUE_COLORS.get(venue, "#37474F")
    badge_color  = _lighten(header_color, 0.25)   # 少し明るい同系色
    n_races = len(races)
    n_bets  = sum(len(r["bets"]) for r in races)

    # ボディ（レース間にセパレーター）
    body_contents: list[dict] = []
    for i, race in enumerate(races):
        if i > 0:
            body_contents.append({
                "type": "separator",
                "margin": "md",
                "color": "#E0E0E0",
            })
        body_contents.extend(_race_block(race, header_color))

    return {
        "type": "bubble",
        "size": "mega",
        # ── ヘッダー ──────────────────────────────────────
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": header_color,
            "paddingAll": "14px",
            "contents": [
                {
                    "type": "box",
                    "layout": "horizontal",
                    "alignItems": "center",
                    "contents": [
                        {
                            "type": "text",
                            "text": f"🏇 {venue}",
                            "color": "#FFFFFF",
                            "weight": "bold",
                            "size": "xl",
                            "flex": 1,
                        },
                        # 点数バッジ
                        {
                            "type": "box",
                            "layout": "vertical",
                            "backgroundColor": badge_color,
                            "cornerRadius": "md",
                            "paddingTop": "4px",
                            "paddingBottom": "4px",
                            "paddingStart": "10px",
                            "paddingEnd": "10px",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": f"{n_races}R・{n_bets}点",
                                    "color": "#FFFFFF",
                                    "size": "xs",
                                    "weight": "bold",
                                    "align": "center",
                                }
                            ],
                        },
                    ],
                },
                {
                    "type": "text",
                    "text": date_str,
                    "color": "#CCCCCC",
                    "size": "xs",
                    "margin": "xs",
                },
            ],
        },
        # ── ボディ ────────────────────────────────────────
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "14px",
            "spacing": "none",
            "contents": body_contents,
        },
    }


def _summary_bubble(
    target_date: date,
    race_bets:   list[dict],
    cfg:         BettingConfig,
    n_venues:    int,
    total_bets:  int,
) -> dict:
    """先頭のサマリーバブルを生成する。"""
    weekday  = _WEEKDAY_JP[target_date.weekday()]
    date_str = f"{target_date.month}/{target_date.day}({weekday})"
    n_races  = len(race_bets)
    surface_str = "・".join(cfg.target_surface)

    # 最高 EV 買い目を抽出
    all_bets = [
        (bet, race["race_name"])
        for race in race_bets
        for bet in race["bets"]
    ]
    top_note_components: list[dict] = []
    if all_bets:
        top_bet, top_race = max(all_bets, key=lambda x: x[0].ev)
        ci = _circled(top_bet.horse_num_i)
        cj = _circled(top_bet.horse_num_j)
        top_label = (
            f"{top_race}: "
            f"{ci}{top_bet.horse_name_i} × {cj}{top_bet.horse_name_j}"
        )
        top_ev_str = f"EV {top_bet.ev:.2f}｜想定 {top_bet.est_quinella_odds:.1f}倍"
        top_note_components = [
            {"type": "separator"},
            {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#FFF8E1",
                "cornerRadius": "md",
                "paddingAll": "10px",
                "contents": [
                    {
                        "type": "text",
                        "text": "🔥 本日の注目",
                        "size": "xs",
                        "weight": "bold",
                        "color": "#E65100",
                    },
                    {
                        "type": "text",
                        "text": top_label,
                        "size": "sm",
                        "weight": "bold",
                        "color": "#333333",
                        "wrap": True,
                        "margin": "xs",
                    },
                    {
                        "type": "text",
                        "text": top_ev_str,
                        "size": "xs",
                        "color": "#888888",
                    },
                ],
            },
        ]

    return {
        "type": "bubble",
        "size": "mega",
        # ── ヘッダー ──────────────────────────────────────
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#212121",
            "paddingAll": "16px",
            "contents": [
                {
                    "type": "text",
                    "text": "🏇 馬連 AI予測",
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "xl",
                },
                {
                    "type": "text",
                    "text": date_str,
                    "color": "#AAAAAA",
                    "size": "sm",
                    "margin": "xs",
                },
            ],
        },
        # ── ボディ ────────────────────────────────────────
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "16px",
            "spacing": "md",
            "contents": [
                # 数値サマリー行
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "vertical",
                            "flex": 1,
                            "contents": [
                                {
                                    "type": "text",
                                    "text": str(n_races),
                                    "size": "3xl",
                                    "weight": "bold",
                                    "color": "#1565C0",
                                    "align": "center",
                                },
                                {
                                    "type": "text",
                                    "text": "レース",
                                    "size": "xs",
                                    "color": "#888888",
                                    "align": "center",
                                },
                            ],
                        },
                        {"type": "separator"},
                        {
                            "type": "box",
                            "layout": "vertical",
                            "flex": 1,
                            "contents": [
                                {
                                    "type": "text",
                                    "text": str(total_bets),
                                    "size": "3xl",
                                    "weight": "bold",
                                    "color": "#C62828",
                                    "align": "center",
                                },
                                {
                                    "type": "text",
                                    "text": "点",
                                    "size": "xs",
                                    "color": "#888888",
                                    "align": "center",
                                },
                            ],
                        },
                        {"type": "separator"},
                        {
                            "type": "box",
                            "layout": "vertical",
                            "flex": 1,
                            "contents": [
                                {
                                    "type": "text",
                                    "text": str(n_venues),
                                    "size": "3xl",
                                    "weight": "bold",
                                    "color": "#2E7D32",
                                    "align": "center",
                                },
                                {
                                    "type": "text",
                                    "text": "会場",
                                    "size": "xs",
                                    "color": "#888888",
                                    "align": "center",
                                },
                            ],
                        },
                    ],
                },
                {"type": "separator"},
                # 戦略パラメーター
                {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "xs",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "対象コース",
                                    "size": "xs",
                                    "color": "#888888",
                                    "flex": 1,
                                },
                                {
                                    "type": "text",
                                    "text": surface_str,
                                    "size": "xs",
                                    "weight": "bold",
                                    "color": "#333333",
                                    "align": "end",
                                },
                            ],
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "EV 閾値",
                                    "size": "xs",
                                    "color": "#888888",
                                    "flex": 1,
                                },
                                {
                                    "type": "text",
                                    "text": f"≥ {cfg.min_ev_threshold}",
                                    "size": "xs",
                                    "weight": "bold",
                                    "color": "#333333",
                                    "align": "end",
                                },
                            ],
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "上位点数上限",
                                    "size": "xs",
                                    "color": "#888888",
                                    "flex": 1,
                                },
                                {
                                    "type": "text",
                                    "text": f"{cfg.max_bets_per_race}点/R",
                                    "size": "xs",
                                    "weight": "bold",
                                    "color": "#333333",
                                    "align": "end",
                                },
                            ],
                        },
                    ],
                },
                # 注目買い目
                *top_note_components,
                # 操作ヒント
                {"type": "separator"},
                {
                    "type": "text",
                    "text": "👉 右にスクロールして会場別予想をチェック",
                    "size": "xs",
                    "color": "#888888",
                    "align": "center",
                    "wrap": True,
                },
            ],
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────────────────────────────

def build_wide_carousel(
    target_date: date,
    race_bets:   list[dict],
    cfg:         BettingConfig,
) -> dict:
    """
    会場別カルーセル Flex Message JSON を生成する。

    Parameters
    ----------
    target_date : 対象日
    race_bets   : daily_batch の race_bets リスト
                  各要素 = {"race_name", "course_type", "distance", "bets": [WideBet, ...]}
    cfg         : BettingConfig

    Returns
    -------
    dict
        LINE messages リストの1要素として直接渡せる辞書。
        {"type": "flex", "altText": "...", "contents": {...}}
    """
    # 会場ごとにレースをグループ化（出現順を保持）
    venue_races: dict[str, list[dict]] = {}
    for race in race_bets:
        venue = _extract_venue(race["race_name"])
        if venue not in venue_races:
            venue_races[venue] = []
        venue_races[venue].append(race)

    total_bets = sum(len(r["bets"]) for r in race_bets)
    n_venues   = len(venue_races)

    weekday  = _WEEKDAY_JP[target_date.weekday()]
    date_str = f"{target_date.month}/{target_date.day}({weekday})"

    bubbles: list[dict] = [
        _summary_bubble(target_date, race_bets, cfg, n_venues, total_bets),
        *[
            _venue_bubble(venue, races, date_str)
            for venue, races in venue_races.items()
        ],
    ]

    alt_text = (
        f"{date_str} 馬連予測 "
        f"{len(race_bets)}レース {total_bets}点"
    )

    return {
        "type": "flex",
        "altText": alt_text,
        "contents": {
            "type": "carousel",
            "contents": bubbles,
        },
    }


def send_wide_flex(
    target_date: date,
    race_bets:   list[dict],
    cfg:         BettingConfig,
) -> list[str]:
    """
    会場別カルーセル Flex Message を LINE Push Message で送信する。

    送信前に line_sent_messages.json に保存済みの旧メッセージを削除し、
    送信後に新しいメッセージ ID を同ファイルに追記する。

    Returns
    -------
    list[str]
        送信したメッセージ ID のリスト（通常は 1 件）。

    Raises
    ------
    RuntimeError
        LINE API がエラーを返した場合。
    """
    # ── 送信前: 前回メッセージを削除 ────────────────────────────────
    _sent_id_file = Path(__file__).resolve().parent.parent.parent / "logs" / "line_sent_messages.json"

    def _load_ids() -> list[dict]:
        if _sent_id_file.exists():
            try:
                return json.loads(_sent_id_file.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_ids(records: list[dict]) -> None:
        _sent_id_file.parent.mkdir(parents=True, exist_ok=True)
        _sent_id_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    old_records = _load_ids()
    for rec in old_records:
        msg_id = rec.get("id", "")
        if not msg_id:
            continue
        del_url = f"https://api.line.me/v2/bot/message/{msg_id}"
        try:
            del_resp = requests.delete(
                del_url,
                headers={"Authorization": f"Bearer {settings.line_channel_access_token}"},
                timeout=15,
            )
            if del_resp.status_code == 200:
                logger.info(f"  旧メッセージ削除: id={msg_id}")
            else:
                logger.warning(f"  旧メッセージ削除失敗 (id={msg_id} status={del_resp.status_code})")
        except Exception as e:
            logger.warning(f"  旧メッセージ削除スキップ (id={msg_id} ネットワークエラー: {e})")
    _save_ids([])   # 削除済みとしてリセット

    # ── Flex Message 構築・送信 ─────────────────────────────────────
    flex_msg = build_wide_carousel(target_date, race_bets, cfg)

    _LINE_HOST    = "api.line.me"
    _LINE_API_URL = f"https://{_LINE_HOST}/v2/bot/message/push"
    _MAX_RETRIES  = 5          # 最大リトライ回数
    _BACKOFF_BASE = 2          # 指数バックオフ基底（秒）

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.line_channel_access_token}",
    }
    payload = {
        "to": settings.line_target_user_id,
        "messages": [flex_msg],
    }

    resp = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(_LINE_API_URL, headers=headers, json=payload, timeout=30)
            break  # 成功

        except requests.exceptions.ConnectionError as e:
            # DNS 解決失敗の場合は IP 直指定でリトライ
            import socket as _socket
            import warnings as _warnings
            ip_addr = None
            try:
                ip_addr = _socket.getaddrinfo(_LINE_HOST, 443, 0, _socket.SOCK_STREAM)[0][4][0]
                logger.warning(f"  DNS解決: {_LINE_HOST} -> {ip_addr}（IPで再送）")
            except Exception:
                pass

            if ip_addr:
                try:
                    ip_url = f"https://{ip_addr}/v2/bot/message/push"
                    ip_headers = {**headers, "Host": _LINE_HOST}
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")  # SSL SNI 警告を抑制
                        resp = requests.post(
                            ip_url, headers=ip_headers, json=payload,
                            timeout=30, verify=False,
                        )
                    logger.info(f"  IP直接送信成功 ({ip_addr})")
                    break
                except Exception as ip_err:
                    logger.warning(f"  IP直接送信も失敗: {ip_err}")

            wait = _BACKOFF_BASE ** attempt
            if attempt < _MAX_RETRIES:
                logger.warning(
                    f"  LINE送信失敗 ({attempt}/{_MAX_RETRIES}) ネットワークエラー: {e}\n"
                    f"  {wait}秒後にリトライ..."
                )
                import time as _time
                _time.sleep(wait)
            else:
                logger.error(f"LINE Flex 送信失敗（{_MAX_RETRIES}回リトライ後）: {e}")
                raise RuntimeError(f"LINE Flex 送信失敗: {e}") from e

        except Exception as e:
            wait = _BACKOFF_BASE ** attempt
            if attempt < _MAX_RETRIES:
                logger.warning(
                    f"  LINE送信失敗 ({attempt}/{_MAX_RETRIES}): {e}\n"
                    f"  {wait}秒後にリトライ..."
                )
                import time as _time
                _time.sleep(wait)
            else:
                logger.error(f"LINE Flex 送信失敗（{_MAX_RETRIES}回リトライ後）: {e}")
                raise RuntimeError(f"LINE Flex 送信失敗: {e}") from e

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "N/A"
        body = resp.text[:200] if resp is not None else ""
        logger.error(f"LINE Flex 送信失敗: {code} {body}")
        raise RuntimeError(f"LINE Flex 送信失敗: {code} {body}")

    logger.info(f"LINE Flex カルーセル送信完了 (status={resp.status_code})")

    # ── 送信後: メッセージ ID を保存 ─────────────────────────────────
    sent_ids: list[str] = []
    try:
        body = resp.json()
        for m in body.get("sentMessages", []):
            sent_ids.append(m["id"])
        records = [
            {"id": mid, "date": str(target_date), "sent_at": datetime.now().isoformat()}
            for mid in sent_ids
        ]
        _save_ids(records)
        logger.info(f"  送信メッセージID保存: {sent_ids}")
    except Exception as e:
        logger.warning(f"  メッセージID保存失敗（続行）: {e}")

    return sent_ids
