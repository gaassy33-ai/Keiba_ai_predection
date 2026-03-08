"""
朝一括配信 Flex Message（Carousel）ビルダー。

create_morning_carousel(venue_data_list, updated_at) が
LINE カルーセル JSON を生成するコア関数。

venue_data_list の各要素:
    {
        "jyo_code":    str,            # "06" (中山) など
        "jyo_name":    str,            # "中山"
        "date_label":  str,            # "3/8(土)"
        "race_count":  int,            # 予測成功レース数
        "weather":     str,            # "晴"
        "ground_turf": str,            # "良" (芝)
        "ground_dirt": str,            # "良" (ダート) ← なければ ""
        "kaisai_date": str,            # "20250308"
        "races": [
            {
                "race_number":    int,
                "start_time":     str,   # "15:40"
                "is_main":        bool,
                "honmei": {              # None on error
                    "horse_number":  int,
                    "frame_number":  int | str,
                    "horse_name":    str,
                    "win_prob":      float,
                },
                "best_bet_label": str,   # "単 1.8x" など
                "error":          str | None,
            },
            ...
        ]
    }
"""

from __future__ import annotations

from datetime import date

from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    FlexContainer,
    FlexMessage,
    MessagingApi,
    PushMessageRequest,
    TextMessage,
)
from loguru import logger

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────
# 会場カラーテーマ
# ─────────────────────────────────────────────────────────────────────

_JYO_THEMES: dict[str, dict] = {
    "01": {"header_bg": "#002818", "accent": "#a5d6a7", "name": "札幌"},
    "02": {"header_bg": "#002818", "accent": "#a5d6a7", "name": "函館"},
    "03": {"header_bg": "#00281e", "accent": "#80cbc4", "name": "福島"},
    "04": {"header_bg": "#00281e", "accent": "#80cbc4", "name": "新潟"},
    "05": {"header_bg": "#0d1b3e", "accent": "#90caf9", "name": "東京"},
    "06": {"header_bg": "#0d1b3e", "accent": "#90caf9", "name": "中山"},
    "07": {"header_bg": "#00281e", "accent": "#80cbc4", "name": "中京"},
    "08": {"header_bg": "#1b0032", "accent": "#ce93d8", "name": "京都"},
    "09": {"header_bg": "#1b0032", "accent": "#ce93d8", "name": "阪神"},
    "10": {"header_bg": "#001430", "accent": "#82b1ff", "name": "小倉"},
}
_DEFAULT_THEME: dict = {"header_bg": "#0d1b3e", "accent": "#90caf9", "name": "競馬場"}

# 交互行の背景色
_ROW_ODD  = "#0a0d1a"
_ROW_EVEN = "#0f1224"
# メインレース行
_MAIN_BG     = "#1a0f00"
_MAIN_BORDER = "#ff8f00"


# ─────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────

def _num_badge(text: str, bg: str, size_px: int = 16) -> dict:
    """馬番・枠番用の小さな四角バッジ。"""
    px = f"{size_px}px"
    return {
        "type": "box",
        "layout": "vertical",
        "width": px,
        "height": px,
        "cornerRadius": "3px",
        "backgroundColor": bg,
        "alignItems": "center",
        "justifyContent": "center",
        "contents": [
            {
                "type": "text",
                "text": str(text),
                "size": "xxs",
                "color": "#ffffff",
                "align": "center",
                "gravity": "center",
            }
        ],
    }


def make_best_bet_label(strategies: list) -> str:
    """
    BetLine リストから最も EV の高い買い目の短縮ラベルを返す。
    runner.py から import して使うために public 関数として公開。
    """
    if not strategies:
        return "−"
    best = max(strategies, key=lambda s: s.ev)
    abbr_map = {
        "単勝": "単",  "複勝": "複",  "馬連": "馬連",
        "馬単": "馬単", "ワイド": "W", "3連複": "3複", "3連単": "3単",
    }
    abbr = abbr_map.get(best.bet_type, best.bet_type[:2])
    if best.ev >= 1.3:
        return f"{abbr} {best.ev:.1f}x"
    return abbr


# ─────────────────────────────────────────────────────────────────────
# 列ヘッダー行
# ─────────────────────────────────────────────────────────────────────

def _col_header_row() -> dict:
    return {
        "type": "box",
        "layout": "horizontal",
        "backgroundColor": "#1a237e",
        "paddingTop": "xs",
        "paddingBottom": "xs",
        "paddingStart": "sm",
        "paddingEnd": "sm",
        "contents": [
            {
                "type": "text", "text": "R",
                "size": "xxs", "color": "#90caf9",
                "flex": 1, "weight": "bold",
            },
            {
                "type": "text", "text": "発走",
                "size": "xxs", "color": "#90caf9", "flex": 2,
            },
            {
                "type": "text", "text": "◎ 本命",
                "size": "xxs", "color": "#90caf9", "flex": 5,
            },
            {
                "type": "text", "text": "確率",
                "size": "xxs", "color": "#90caf9",
                "flex": 2, "align": "center",
            },
            {
                "type": "text", "text": "買い目",
                "size": "xxs", "color": "#90caf9",
                "flex": 3, "align": "end",
            },
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# レース行
# ─────────────────────────────────────────────────────────────────────

def _race_row(race: dict, row_index: int) -> dict:
    """
    1レース分の横並び行コンポーネントを生成する。

    列構成（flex 比 = 1:2:5:2:3）:
        R#  |  発走  |  枠番badge 馬番badge 馬名  |  確率  |  買い目
    """
    is_main   = race.get("is_main", False)
    has_error = bool(race.get("error"))

    bg_color   = _MAIN_BG if is_main else (_ROW_ODD if row_index % 2 == 0 else _ROW_EVEN)
    r_color    = "#ff8f00" if is_main else "#777799"
    time_color = "#ffcc80" if is_main else "#888899"
    padding_v  = "sm" if is_main else "xs"

    # ── R番号列 ─────────────────────────────────────────
    rnum_text = f"{race.get('race_number', '?')}R"
    if is_main:
        r_col: dict = {
            "type": "box",
            "layout": "vertical",
            "flex": 1,
            "contents": [
                {"type": "text", "text": rnum_text, "size": "xs",  "color": "#ff8f00", "weight": "bold"},
                {"type": "text", "text": "★",        "size": "xxs", "color": "#ff8f00"},
            ],
        }
    else:
        r_col = {
            "type": "text", "text": rnum_text,
            "size": "xs", "color": r_color, "flex": 1,
        }

    # ── 発走時刻列 ──────────────────────────────────────
    t_col: dict = {
        "type": "text",
        "text": race.get("start_time", "--:--"),
        "size": "xs", "color": time_color, "flex": 2,
    }

    # ── 本命馬列 ──────────────────────────────────────
    if has_error:
        honmei_col: dict = {
            "type": "text", "text": "取得失敗",
            "size": "xs", "color": "#444455", "flex": 5,
        }
        prob_col: dict = {
            "type": "text", "text": "−",
            "size": "xs", "color": "#444455", "flex": 2, "align": "center",
        }
        bet_col: dict = {
            "type": "text", "text": "−",
            "size": "xxs", "color": "#444455", "flex": 3, "align": "end",
        }
    else:
        honmei    = race.get("honmei") or {}
        win_prob  = float(honmei.get("win_prob", 0.0))
        horse_num = honmei.get("horse_number", 0)
        horse_name = honmei.get("horse_name", "---")

        # 枠番バッジ
        try:
            frame_int   = int(honmei.get("frame_number", 0))
            frame_text  = str(frame_int) if frame_int > 0 else "−"
        except (TypeError, ValueError):
            frame_text  = "−"

        badge_px   = 18 if is_main else 16
        frame_bg   = "#6d4c41" if is_main else "#5d4037"
        horse_bg   = "#283593" if is_main else "#1a237e"
        name_color = "#ffffff" if is_main else "#ccccdd"

        # 馬名を 6 文字に制限（LINE xs サイズの表示限界）
        name_short = (horse_name[:6] + "…") if len(horse_name) > 6 else horse_name

        honmei_col = {
            "type": "box",
            "layout": "horizontal",
            "flex": 5,
            "alignItems": "center",
            "spacing": "xs",
            "contents": [
                _num_badge(frame_text,       frame_bg, badge_px),
                _num_badge(str(horse_num),   horse_bg, badge_px),
                {
                    "type": "text",
                    "text": name_short,
                    "size": "xs",
                    "color": name_color,
                    "weight": "bold" if is_main else "regular",
                    "flex": 1,
                    "wrap": False,
                },
            ],
        }

        prob_col = {
            "type": "text",
            "text": f"{win_prob:.0%}",
            "size": "xs",
            "color": "#ffb74d" if is_main else "#64b5f6",
            "weight": "bold" if is_main else "regular",
            "flex": 2,
            "align": "center",
        }
        bet_col = {
            "type": "text",
            "text": race.get("best_bet_label", "−"),
            "size": "xxs",
            "color": "#ffe082" if is_main else "#a5d6a7",
            "weight": "bold" if is_main else "regular",
            "flex": 3,
            "align": "end",
        }

    row: dict = {
        "type": "box",
        "layout": "horizontal",
        "backgroundColor": bg_color,
        "paddingTop": padding_v,
        "paddingBottom": padding_v,
        "paddingStart": "sm",
        "paddingEnd": "sm",
        "contents": [r_col, t_col, honmei_col, prob_col, bet_col],
    }

    # メインレース行にゴールド枠線
    if is_main:
        row["borderWidth"] = "2px"
        row["borderColor"] = _MAIN_BORDER

    return row


# ─────────────────────────────────────────────────────────────────────
# 会場バブル
# ─────────────────────────────────────────────────────────────────────

_WEATHER_ICONS: dict[str, str] = {
    "晴": "☀",
    "曇": "☁",
    "雨": "🌧",
    "小雨": "🌦",
    "雪": "❄",
}


def _venue_bubble(venue: dict, updated_at: str) -> dict:
    """1会場分の Flex bubble コンポーネントを生成する。"""
    jyo_code = venue.get("jyo_code", "06")
    theme    = _JYO_THEMES.get(jyo_code, _DEFAULT_THEME)
    jyo_name = venue.get("jyo_name") or theme["name"]

    header_bg   = theme["header_bg"]
    accent      = theme["accent"]
    date_label  = venue.get("date_label", "")
    race_count  = venue.get("race_count", 0)
    weather     = venue.get("weather", "")
    ground_turf = venue.get("ground_turf", "")
    ground_dirt = venue.get("ground_dirt", "")
    kaisai_date = venue.get("kaisai_date", "")

    # 天候テキスト
    wx_icon = _WEATHER_ICONS.get(weather, "")
    wx_text = f"{wx_icon} {weather}".strip() if weather else "−"

    # 馬場テキスト
    ground_parts = []
    if ground_turf:
        ground_parts.append(f"芝: {ground_turf}")
    if ground_dirt:
        ground_parts.append(f"ダ: {ground_dirt}")
    ground_text = "  ".join(ground_parts) if ground_parts else "馬場取得中"

    # ── ヘッダー ──────────────────────────────────────────────────
    header: dict = {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": header_bg,
        "paddingAll": "md",
        "contents": [
            {
                "type": "box",
                "layout": "horizontal",
                "alignItems": "center",
                "contents": [
                    {
                        "type": "text",
                        "text": f"🏇 {jyo_name}",
                        "color": "#ffffff",
                        "weight": "bold",
                        "size": "xl",
                        "flex": 1,
                    },
                    {
                        "type": "text",
                        "text": f"{date_label}  {race_count}R",
                        "color": accent,
                        "size": "xs",
                        "align": "end",
                    },
                ],
            },
            {
                "type": "box",
                "layout": "horizontal",
                "margin": "xs",
                "spacing": "md",
                "contents": [
                    {"type": "text", "text": wx_text,     "size": "xs", "color": "#ffd54f"},
                    {"type": "text", "text": ground_text, "size": "xs", "color": "#b0bec5"},
                ],
            },
        ],
    }

    # ── ボディ（レース一覧テーブル）────────────────────────────────
    races = sorted(venue.get("races", []), key=lambda r: r.get("race_number", 99))
    body_rows: list[dict] = [_col_header_row()]
    for i, race in enumerate(races):
        body_rows.append(_race_row(race, row_index=i))

    body: dict = {
        "type": "box",
        "layout": "vertical",
        "spacing": "none",
        "paddingAll": "none",
        "contents": body_rows,
    }

    # ── フッター ───────────────────────────────────────────────────
    netkeiba_url = (
        f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
        if kaisai_date
        else "https://race.netkeiba.com/"
    )

    footer: dict = {
        "type": "box",
        "layout": "horizontal",
        "backgroundColor": "#05070f",
        "paddingAll": "sm",
        "contents": [
            {
                "type": "text",
                "text": f"🔄 {updated_at} 更新",
                "size": "xxs",
                "color": "#555577",
                "flex": 1,
            },
            {
                "type": "text",
                "text": "netkeiba →",
                "size": "xxs",
                "color": "#3d5afe",
                "align": "end",
                "action": {
                    "type": "uri",
                    "label": "netkeiba で確認",
                    "uri": netkeiba_url,
                },
            },
        ],
    }

    return {
        "type": "bubble",
        "size": "giga",
        "header": header,
        "body":   body,
        "footer": footer,
    }


# ─────────────────────────────────────────────────────────────────────
# カルーセル組み立て
# ─────────────────────────────────────────────────────────────────────

def create_morning_carousel(
    venue_data_list: list[dict],
    updated_at: str,
) -> dict | None:
    """
    全会場の venue_data リストから Flex カルーセル JSON を生成する。

    Returns
    -------
    dict | None
        FlexContainer.from_dict() に渡せる carousel dict。
        venue_data_list が空なら None。
    """
    if not venue_data_list:
        return None

    bubbles = [_venue_bubble(v, updated_at) for v in venue_data_list]

    # LINE カルーセルは最大 12 バブル
    if len(bubbles) > 12:
        logger.warning(f"会場数 {len(bubbles)} > 12 → 先頭12件に制限")
        bubbles = bubbles[:12]

    return {"type": "carousel", "contents": bubbles}


# ─────────────────────────────────────────────────────────────────────
# LINE 送信クラス
# ─────────────────────────────────────────────────────────────────────

class MorningNotifier:
    """朝一括配信の LINE push 送信を担当する。"""

    def send_carousel(self, carousel: dict, target_date: date) -> None:
        """カルーセルを LINE に push 送信する（1通のメッセージ）。"""
        dow = "土" if target_date.weekday() == 5 else "日"
        alt_text = f"🏇 本日の全レース予想 {target_date.strftime('%-m/%-d')}({dow})"

        self._push(
            FlexMessage(
                alt_text=alt_text,
                contents=FlexContainer.from_dict(carousel),
            )
        )

    def send_text(self, text: str) -> None:
        """テキストメッセージを push 送信する（エラー通知用）。"""
        self._push(TextMessage(text=text))

    def _push(self, *messages) -> None:
        config = Configuration(access_token=settings.line_channel_access_token)
        with ApiClient(config) as client:
            try:
                MessagingApi(client).push_message(
                    PushMessageRequest(
                        to=settings.line_target_user_id,
                        messages=list(messages),
                    )
                )
                logger.info(f"LINE push 送信完了: {len(messages)} message(s)")
            except Exception as e:
                logger.error(f"LINE push 失敗: {e}")
                raise
