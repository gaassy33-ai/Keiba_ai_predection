"""
LINE Messaging API を使って予測結果を Flex Message で送信する。

create_prediction_message(race_data) が Flex JSON を生成するコア関数。
race_data の構造:
    {
        "race_name":       str,   # "東京11R 日本ダービー GⅠ"
        "grade":           str,   # "GⅠ" | "GⅡ" | "GⅢ" | "L" | "OP" | "一般"
        "race_id":         str,   # "202405050811"  (netkeiba リンク用)
        "venue":           str,   # "東京"
        "course_type":     str,   # "芝" | "ダ"
        "distance":        int,   # 2400
        "weather":         str,   # "晴"
        "ground_condition": str,  # "良"
        "deadline":        str,   # "15:20"
        "horses": [
            {
                "mark":         str,       # "◎" | "○" | "▲"
                "horse_number": int,
                "horse_name":   str,
                "jockey_name":  str,
                "win_prob":     float,     # 0.0-1.0
                "tags":         list[str], # ["血統適性：高", "斤量減：有利"]
            },
            ...  # 最大3頭
        ],
    }
"""

from __future__ import annotations

from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    FlexMessage,
    FlexContainer,
    TextMessage,
)
from loguru import logger

from config.settings import settings
from src.model.predictor import PredictionResult
from src.betting.strategy import BetLine, BET_TYPE_COLOR


# ---------------------------------------------------------------------------
# グレード別カラー定義
# ---------------------------------------------------------------------------
_GRADE_COLORS: dict[str, str] = {
    "GⅠ":  "#C62828",  # 赤
    "G1":  "#C62828",
    "GⅡ":  "#1565C0",  # 濃青
    "G2":  "#1565C0",
    "GⅢ":  "#2E7D32",  # 濃緑
    "G3":  "#2E7D32",
    "L":   "#6A1B9A",  # 紫 (Listed)
    "OP":  "#E65100",  # オレンジ
    "一般": "#37474F",  # ダークグレー
}

# 印別スタイル定義
_MARK_STYLES: dict[str, dict] = {
    "◎": {"bg": "#C62828", "card_bg": "#fff8f8", "prob_color": "#C62828"},
    "○": {"bg": "#1565C0", "card_bg": "#f3f8ff", "prob_color": "#1565C0"},
    "▲": {"bg": "#2E7D32", "card_bg": "#f3fff5", "prob_color": "#2E7D32"},
}

# タグ色ローテーション（ポジティブ・ニュートラル・注意を交互に）
_TAG_PALETTES = [
    {"bg": "#E8F5E9", "text": "#2E7D32"},  # 緑
    {"bg": "#E3F2FD", "text": "#1565C0"},  # 青
    {"bg": "#FFF8E1", "text": "#F57F17"},  # 黄
    {"bg": "#F3E5F5", "text": "#6A1B9A"},  # 紫
]


def _detect_grade(race_name: str) -> str:
    """レース名文字列からグレードを推定する。"""
    for grade in ("GⅠ", "G1", "GⅡ", "G2", "GⅢ", "G3", "L"):
        if grade in race_name:
            return grade
    if "オープン" in race_name or "OP" in race_name:
        return "OP"
    return "一般"


def _header_color(grade: str) -> str:
    return _GRADE_COLORS.get(grade, _GRADE_COLORS["一般"])


def _tag_badge(text: str, palette: dict) -> dict:
    return {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": palette["bg"],
        "cornerRadius": "md",
        "paddingTop": "3px",
        "paddingBottom": "3px",
        "paddingStart": "8px",
        "paddingEnd": "8px",
        "contents": [
            {
                "type": "text",
                "text": text,
                "size": "xxs",
                "color": palette["text"],
            }
        ],
    }


def _valid(v) -> bool:
    """NaN / None / 空文字でなければ True。"""
    if v is None or v == "":
        return False
    try:
        import math
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _horse_card(horse: dict) -> dict:
    """1頭分の馬カードコンポーネントを生成する。"""
    mark = horse.get("mark", "◎")
    style = _MARK_STYLES.get(mark, _MARK_STYLES["◎"])
    num = str(horse.get("horse_number", "-"))
    name = horse.get("horse_name", "---")
    jockey = horse.get("jockey_name", "")
    prob = horse.get("win_prob", 0.0)
    tags: list[str] = horse.get("tags", [])

    jockey_place_rate = horse.get("jockey_place_rate")
    recent_avg_pos    = horse.get("recent_avg_pos")
    recent_avg_last3f = horse.get("recent_avg_last3f")

    tag_badges = [
        _tag_badge(t, _TAG_PALETTES[i % len(_TAG_PALETTES)])
        for i, t in enumerate(tags[:4])
    ]

    # --- 騎手行: 名前 + 連対率 ---
    jockey_label = f"騎手：{jockey}" if jockey else "騎手：---"
    place_rate_label = (
        f"連対率 {float(jockey_place_rate):.0%}"
        if _valid(jockey_place_rate) else "連対率 ---"
    )

    # --- 成績行: 直近着順 + 上がり3F ---
    avg_pos_label = (
        f"直近 平均{float(recent_avg_pos):.1f}着"
        if _valid(recent_avg_pos) else "直近 ---"
    )
    last3f_label = (
        f"上がり {float(recent_avg_last3f):.1f}秒"
        if _valid(recent_avg_last3f) else "上がり ---"
    )

    return {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": style["card_bg"],
        "cornerRadius": "lg",
        "paddingAll": "md",
        "contents": [
            # --- 上段: 印 / 馬番 / 馬名 / 勝率 ---
            {
                "type": "box",
                "layout": "horizontal",
                "alignItems": "center",
                "contents": [
                    {
                        "type": "box",
                        "layout": "vertical",
                        "backgroundColor": style["bg"],
                        "cornerRadius": "sm",
                        "width": "30px",
                        "height": "30px",
                        "contents": [{"type": "text", "text": mark, "color": "#ffffff",
                                      "size": "sm", "align": "center", "gravity": "center"}],
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "backgroundColor": "#1a237e",
                        "cornerRadius": "sm",
                        "width": "26px",
                        "height": "26px",
                        "margin": "sm",
                        "contents": [{"type": "text", "text": num, "color": "#ffffff",
                                      "size": "xs", "align": "center", "gravity": "center"}],
                    },
                    {
                        "type": "text",
                        "text": name,
                        "size": "md",
                        "weight": "bold",
                        "color": "#111111",
                        "flex": 1,
                        "margin": "sm",
                        "wrap": True,
                    },
                    {
                        "type": "text",
                        "text": f"{prob:.1%}",
                        "size": "lg",
                        "weight": "bold",
                        "color": style["prob_color"],
                        "align": "end",
                    },
                ],
            },
            # --- 騎手名 + 連対率 ---
            {
                "type": "box",
                "layout": "horizontal",
                "margin": "xs",
                "contents": [
                    {
                        "type": "text",
                        "text": jockey_label,
                        "size": "xs",
                        "color": "#555555",
                        "flex": 1,
                    },
                    {
                        "type": "text",
                        "text": place_rate_label,
                        "size": "xs",
                        "color": "#1565C0",
                        "weight": "bold",
                        "align": "end",
                    },
                ],
            },
            # --- 直近成績 + 上がり3F ---
            {
                "type": "box",
                "layout": "horizontal",
                "margin": "xs",
                "contents": [
                    {
                        "type": "text",
                        "text": avg_pos_label,
                        "size": "xs",
                        "color": "#555555",
                        "flex": 1,
                    },
                    {
                        "type": "text",
                        "text": last3f_label,
                        "size": "xs",
                        "color": "#2E7D32",
                        "weight": "bold",
                        "align": "end",
                    },
                ],
            },
            # --- タグ行 ---
            *(
                [
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "spacing": "sm",
                        "margin": "sm",
                        "flexWrap": "wrap",
                        "contents": tag_badges,
                    }
                ]
                if tag_badges
                else []
            ),
        ],
    }


# ---------------------------------------------------------------------------
# 買い目セクション Flex ビルダー
# ---------------------------------------------------------------------------

def _bet_row(bet: BetLine) -> dict:
    """1券種グループを横並びカードで表示する。"""
    color = BET_TYPE_COLOR.get(bet.bet_type, "#37474F")
    icon = "🔥 " if bet.is_featured else ""
    alloc_color = "#C62828" if bet.is_featured else "#333333"
    card_bg = "#fff5f5" if bet.is_featured else "#fafafa"

    return {
        "type": "box",
        "layout": "horizontal",
        "margin": "sm",
        "paddingAll": "sm",
        "backgroundColor": card_bg,
        "cornerRadius": "md",
        "alignItems": "center",
        "contents": [
            # 券種バッジ
            {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": color,
                "cornerRadius": "sm",
                "width": "44px",
                "paddingTop": "3px",
                "paddingBottom": "3px",
                "paddingStart": "4px",
                "paddingEnd": "4px",
                "contents": [
                    {
                        "type": "text",
                        "text": bet.bet_type,
                        "size": "xxs",
                        "color": "#ffffff",
                        "weight": "bold",
                        "align": "center",
                    }
                ],
            },
            # ラベル / 説明
            {
                "type": "box",
                "layout": "vertical",
                "flex": 1,
                "margin": "sm",
                "contents": [
                    {
                        "type": "text",
                        "text": f"{icon}{bet.label}",
                        "size": "xs",
                        "weight": "bold" if bet.is_featured else "regular",
                        "color": "#111111",
                        "wrap": True,
                    },
                    {
                        "type": "text",
                        "text": f"EV:{bet.ev:.2f}  {bet.combo_count}点",
                        "size": "xxs",
                        "color": "#888888",
                    },
                ],
            },
            # 投資額
            {
                "type": "text",
                "text": f"¥{bet.allocation:,}",
                "size": "sm",
                "weight": "bold",
                "color": alloc_color,
                "align": "end",
            },
        ],
    }


def _betting_section(bets: list[BetLine], budget: int = 10_000) -> list[dict]:
    """買い目セクション全体のコンポーネントリストを返す。bets が空なら空リスト。"""
    if not bets:
        return []

    used = sum(b.allocation for b in bets)
    components: list[dict] = [
        {"type": "separator", "margin": "lg"},
        {
            "type": "box",
            "layout": "horizontal",
            "margin": "lg",
            "contents": [
                {
                    "type": "text",
                    "text": "🎯 推奨買い目",
                    "size": "sm",
                    "weight": "bold",
                    "color": "#333333",
                    "flex": 1,
                },
                {
                    "type": "text",
                    "text": f"予算 ¥{used:,} / ¥{budget:,}",
                    "size": "xs",
                    "color": "#888888",
                    "align": "end",
                },
            ],
        },
        *[_bet_row(b) for b in bets],
    ]
    return components


def create_prediction_message(race_data: dict) -> dict:
    """
    Flex Message の JSON 構造（bubble）を生成する。

    Parameters
    ----------
    race_data : dict
        モジュール docstring を参照。

    Returns
    -------
    dict
        FlexContainer.from_dict() に渡せる JSON 辞書。
    """
    race_name = race_data.get("race_name", "レース情報")
    grade = race_data.get("grade") or _detect_grade(race_name)
    race_id = race_data.get("race_id", "")
    venue = race_data.get("venue", "")
    course_type = race_data.get("course_type", "")
    distance = race_data.get("distance", "")
    weather = race_data.get("weather", "")
    ground_condition = race_data.get("ground_condition", "")
    deadline = race_data.get("deadline", "")
    horses: list[dict] = race_data.get("horses", [])

    header_bg = _header_color(grade)
    course_label = (f"{course_type} {distance}m" if distance else course_type) or "---"
    weather_label = f"{'☀' if weather == '晴' else '☁' if '曇' in weather else '🌧' if '雨' in weather else ''} {weather} / {ground_condition}".strip() or "---"

    netkeiba_url = (
        f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        if race_id
        else "https://race.netkeiba.com/"
    )

    # 馬カードリスト（セパレーター付き）
    horse_cards: list[dict] = []
    for i, horse in enumerate(horses[:3]):
        if i > 0:
            horse_cards.append({"type": "separator", "margin": "sm"})
        horse_cards.append(_horse_card(horse))

    # 買い目セクション
    strategies: list[BetLine] = race_data.get("strategies", [])
    budget: int = race_data.get("budget", 10_000)
    bet_components = _betting_section(strategies, budget)

    return {
        "type": "bubble",
        "size": "mega",
        # ---- ヘッダー ----
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": header_bg,
            "paddingAll": "lg",
            "contents": [
                # タイトル行（ラベル + グレードバッジ）
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {
                            "type": "text",
                            "text": "馬券AI予想",
                            "color": "#ffcccc",
                            "size": "xs",
                            "flex": 1,
                        },
                        {
                            "type": "box",
                            "layout": "vertical",
                            "backgroundColor": "#ffffff",
                            "cornerRadius": "sm",
                            "paddingTop": "2px",
                            "paddingBottom": "2px",
                            "paddingStart": "8px",
                            "paddingEnd": "8px",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": grade,
                                    "size": "xs",
                                    "color": header_bg,
                                    "weight": "bold",
                                }
                            ],
                        },
                    ],
                },
                # レース名
                {
                    "type": "text",
                    "text": race_name,
                    "color": "#ffffff",
                    "size": "lg",
                    "weight": "bold",
                    "wrap": True,
                    "margin": "sm",
                },
                # コース情報 / 天候
                {
                    "type": "box",
                    "layout": "horizontal",
                    "margin": "sm",
                    "contents": [
                        {
                            "type": "text",
                            "text": course_label,
                            "color": "#ffcccc",
                            "size": "xs",
                            "flex": 1,
                        },
                        {
                            "type": "text",
                            "text": weather_label,
                            "color": "#ffcccc",
                            "size": "xs",
                            "align": "end",
                        },
                    ],
                },
            ],
        },
        # ---- ボディ ----
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "paddingAll": "md",
            "contents": [*horse_cards, *bet_components],
        },
        # ---- フッター ----
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "paddingAll": "md",
            "backgroundColor": "#fafafa",
            "contents": [
                # 締め切り時刻
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {
                            "type": "text",
                            "text": "⏰ 締め切り",
                            "size": "sm",
                            "color": "#555555",
                            "flex": 1,
                        },
                        {
                            "type": "text",
                            "text": deadline if deadline else "---",
                            "size": "sm",
                            "weight": "bold",
                            "color": "#C62828",
                            "align": "end",
                        },
                    ],
                },
                # netkeiba リンクボタン
                {
                    "type": "button",
                    "action": {
                        "type": "uri",
                        "label": "netkeiba で詳細を見る",
                        "uri": netkeiba_url,
                    },
                    "style": "primary",
                    "color": "#1a237e",
                    "height": "sm",
                    "margin": "sm",
                },
                # 免責
                {
                    "type": "text",
                    "text": "※ 予想は参考情報です。馬券購入は自己責任でお願いします。",
                    "size": "xxs",
                    "color": "#aaaaaa",
                    "wrap": True,
                    "align": "center",
                    "margin": "sm",
                },
            ],
        },
        "styles": {
            "footer": {"separator": True},
        },
    }


# ---------------------------------------------------------------------------
# PredictionResult → race_data 変換ヘルパー
# ---------------------------------------------------------------------------

# SHAP テキスト中の日本語ラベル → タグ文字列マッピング
# explainer.py の FEATURE_LABELS に対応
_FEATURE_TAG_MAP: dict[str, tuple[str, str, str]] = {
    "父産駒勝率":         ("血統適性", "高", "低"),
    "母父産駒勝率":       ("母父適性", "高", "低"),
    "騎手コース勝率":     ("騎手実績", "高", "低"),
    "騎手コース連対率":   ("騎手連対率", "高", "低"),
    "直近平均着順":       ("直近成績", "良好", "不振"),
    "直近平均上がり3F":   ("上がり3F", "速い", "遅い"),
    "斤量":               ("斤量", "軽", "重"),
    "枠番":               ("枠順", "有利", "不利"),
    "馬場状態":           ("馬場適性", "高", "低"),
    "天候":               ("天候適性", "高", "低"),
    "距離":               ("距離適性", "高", "低"),
    "馬齢":               ("馬齢", "有利", "不利"),
}


def _shap_text_to_tags(shap_text: str, horse_name: str) -> list[str]:
    """
    SHAP テキストから当該馬のタグリストを生成する。
    shap_text の各行の形式: "  ▲ feature_name: +0.05" or "  ▽ feature_name: -0.03"
    """
    tags: list[str] = []
    for line in shap_text.split("\n"):
        if horse_name not in line and not line.startswith("  "):
            continue
        direction = "▲" if "▲" in line else ("▽" if "▽" in line else None)
        if direction is None:
            continue
        for feature, (label, pos_word, neg_word) in _FEATURE_TAG_MAP.items():
            if feature in line:
                word = f"{label}：{pos_word}" if direction == "▲" else f"{label}：{neg_word}"
                tags.append(word)
                break
        if len(tags) >= 3:
            break
    return tags


def _result_to_race_data(
    result: PredictionResult,
    shap_text: str,
    ground_condition: str,
    weather: str,
    course_type: str = "",
    distance: int | str = "",
    deadline: str = "",
) -> dict:
    """PredictionResult を create_prediction_message() 用の race_data に変換する。"""
    grade = _detect_grade(result.race_name)

    horses = []
    for mark, mark_char in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        horse: dict = getattr(result, mark, {})
        if not horse:
            continue
        tags = _shap_text_to_tags(shap_text, horse.get("horse_name", ""))
        horses.append(
            {
                "mark":              mark_char,
                "horse_number":      horse.get("horse_number", "-"),
                "horse_name":        horse.get("horse_name", "---"),
                "jockey_name":       horse.get("jockey_name", ""),
                "jockey_place_rate": horse.get("jockey_place_rate"),
                "recent_avg_pos":    horse.get("recent_avg_pos"),
                "recent_avg_last3f": horse.get("recent_avg_last3f"),
                "win_prob":          horse.get("win_prob", 0.0),
                "tags":              tags,
            }
        )

    return {
        "race_name":        result.race_name,
        "grade":            grade,
        "race_id":          result.race_id,
        "course_type":      course_type,
        "distance":         distance,
        "weather":          weather,
        "ground_condition": ground_condition,
        "deadline":         deadline,
        "horses":           horses,
    }


# ---------------------------------------------------------------------------
# LineNotifier
# ---------------------------------------------------------------------------

class LineNotifier:
    """
    Usage
    -----
    # 高レベル API（PredictionResult を直接渡す）
    >>> notifier = LineNotifier()
    >>> notifier.send_prediction(result, shap_text, "良", "晴")

    # 低レベル API（race_data dict を直接渡す）
    >>> notifier.send_flex(race_data)
    """

    def __init__(self) -> None:
        config = Configuration(access_token=settings.line_channel_access_token)
        self._api_client = ApiClient(config)
        self._messaging_api = MessagingApi(self._api_client)

    def send_flex(self, race_data: dict) -> None:
        """
        race_data dict から Flex Message を組み立てて送信する。

        Parameters
        ----------
        race_data : dict
            create_prediction_message() が受け付ける構造。
        """
        container_dict = create_prediction_message(race_data)
        race_name = race_data.get("race_name", "レース")
        top_horse = race_data.get("horses", [{}])[0].get("horse_name", "")

        flex_msg = FlexMessage(
            alt_text=f"【AI競馬予想】{race_name}  本命: {top_horse}",
            contents=FlexContainer.from_dict(container_dict),
        )
        push_req = PushMessageRequest(
            to=settings.line_target_user_id,
            messages=[flex_msg],
        )
        try:
            self._messaging_api.push_message(push_req)
            logger.info(f"LINE notification sent for {race_name}")
        except Exception as e:
            logger.error(f"Failed to send LINE notification: {e}")
            raise

    def send_prediction(
        self,
        result: PredictionResult,
        shap_text: str = "",
        ground_condition: str = "不明",
        weather: str = "不明",
        course_type: str = "",
        distance: int | str = "",
        deadline: str = "",
    ) -> None:
        """
        PredictionResult から Flex Message を組み立てて送信する。

        Parameters
        ----------
        result : PredictionResult
        shap_text : str
            PredictionExplainer.explain_text() の出力
        ground_condition : str
            馬場状態（例: "良"）
        weather : str
            天候（例: "晴"）
        course_type : str
            コース種別（例: "芝" / "ダート"）
        distance : int | str
            距離（例: 1600）
        deadline : str
            締め切り時刻（例: "15:20"）
        """
        race_data = _result_to_race_data(
            result, shap_text, ground_condition, weather,
            course_type=course_type, distance=distance, deadline=deadline,
        )
        self.send_flex(race_data)

    def send_text(self, text: str) -> None:
        """シンプルなテキストメッセージ送信（エラー通知・テスト用）。"""
        push_req = PushMessageRequest(
            to=settings.line_target_user_id,
            messages=[TextMessage(text=text)],
        )
        try:
            self._messaging_api.push_message(push_req)
        except Exception as e:
            logger.error(f"Failed to send LINE text: {e}")
