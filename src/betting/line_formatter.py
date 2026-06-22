"""
line_formatter.py
=================
LTR + EV 馬連買い目の LINE テキストメッセージ整形モジュール。

【出力形式（改訂版）】
━━━━━━━━━━━━━━━━━━━━
🏇 5/9(土) 馬連予測
ダート | EV≥1.5 | 上位5点
━━━━━━━━━━━━━━━━━━━━

[1] 東京1R　ダート1300m
⑬サリーアン ✕ ③ジャガーライズ
　EV 3.54｜想定 44.5倍
⑧コウソクルリアン ✕ ③ジャガーライズ
　EV 1.95｜想定 23.4倍

[2] 阪神3R　ダート1800m
①タイムトゥパーリィ ✕ ⑧タガノシュープリム
　EV 2.11｜想定 18.3倍

━━━━━━━━━━━━━━━━━━━━
10レース 25点
━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from datetime import date

from src.betting.ltr_ev_engine import QuinellaBet
from config.settings import BettingConfig

# 曜日ラベル
_WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]

# 馬番 → 囲み数字（1〜18）
_CIRCLED = {
    1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
    6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
    11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮",
    16: "⑯", 17: "⑰", 18: "⑱",
}


def _circled(num_str: str) -> str:
    """馬番文字列 → 囲み数字。変換できない場合はそのまま返す。"""
    try:
        return _CIRCLED.get(int(num_str), num_str)
    except (ValueError, TypeError):
        return num_str


def format_race_section(
    race_name:   str,               # "中山11R" など
    course_type: str,               # "ダート" / "芝"
    distance:    int,               # 距離(m)
    bets:        list[QuinellaBet],
    race_index:  int = 0,           # 0 = インデックスなし
    is_g1:       bool = False,      # G1特別モード: True の場合、特別ヘッダーと注記を付与
) -> str:
    """
    1レース分の通知テキストブロックを生成する。

    G1レース（is_g1=True）の場合:
      - ヘッダーに「🏆 【G1特別予想】」を付与
      - EV < 1.0 のペアに「※EV1.0未満・G1特例」を注記

    Returns
    -------
    str
        レースヘッダー + 買い目の2行ペア形式。
        買い目なしの場合は空文字を返す。
    """
    if not bets:
        return ""

    if is_g1:
        # ── G1特別ヘッダー ──────────────────────────────────────────
        header = f"🏆 【G1特別予想】{race_name}（{course_type}{distance}m）"
        lines: list[str] = [header]

        for bet in bets:
            ci = _circled(bet.horse_num_i)
            cj = _circled(bet.horse_num_j)
            pair_line = f"{ci}{bet.horse_name_i} ✕ {cj}{bet.horse_name_j}"
            # EV < 1.0 の場合は特例注記を付与
            ev_note = "　※EV1.0未満・G1特例" if bet.ev < 1.0 else ""
            ev_line = f"　EV {bet.ev:.2f}｜想定 {bet.est_quinella_odds:.1f}倍{ev_note}"
            lines.append(pair_line)
            lines.append(ev_line)

    else:
        # ── 通常フォーマット（変更なし）──────────────────────────────
        prefix = f"[{race_index}] " if race_index else ""
        header = f"{prefix}{race_name}　{course_type}{distance}m"
        lines = [header]

        for bet in bets:
            ci = _circled(bet.horse_num_i)
            cj = _circled(bet.horse_num_j)
            # 1行目: 馬番(囲み) + 馬名 ✕ ペア
            pair_line = f"{ci}{bet.horse_name_i} ✕ {cj}{bet.horse_name_j}"
            # 2行目: EV・想定オッズ
            ev_line   = f"　EV {bet.ev:.2f}｜想定 {bet.est_quinella_odds:.1f}倍"
            lines.append(pair_line)
            lines.append(ev_line)

    return "\n".join(lines)


def format_daily_message(
    target_date:   date,
    race_sections: list[str],   # format_race_section() の結果（空文字は除外済み前提）
    cfg:           BettingConfig,
    total_bets:    int,
) -> str:
    """
    1日分の完全な LINE 通知テキストを生成する。

    Parameters
    ----------
    target_date   : 対象日
    race_sections : 各レースのテキストブロック（空文字除外済み）
    cfg           : BettingConfig（ヘッダー表示用）
    total_bets    : 合計購入点数

    Returns
    -------
    str
        送信用テキスト。全レース該当なしの場合は「本日は対象レースなし」を含むテキスト。
    """
    weekday  = _WEEKDAY_JP[target_date.weekday()]
    date_str = f"{target_date.month}/{target_date.day}({weekday})"

    surface_str = "・".join(cfg.target_surface)
    ev_str      = f"EV≥{cfg.min_ev_threshold}"
    bets_str    = f"上位{cfg.max_bets_per_race}点" if cfg.max_bets_per_race > 0 else "上限なし"

    divider = "━" * 20

    lines: list[str] = [
        divider,
        f"🏇 {date_str} 馬連予測",
        f"{surface_str} | {ev_str} | {bets_str}",
        divider,
    ]

    if race_sections:
        for section in race_sections:
            lines.append("")
            lines.append(section)
    else:
        lines.append("")
        lines.append("本日は対象レースなし（条件を満たすペアなし）")

    n_races = len(race_sections)
    lines.append("")
    lines.append(divider)
    lines.append(f"{n_races}レース {total_bets}点")
    lines.append(divider)

    return "\n".join(lines)
