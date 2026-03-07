"""
成績グラフ生成モジュール。

create_today_results_chart(date) → bytes (PNG)
    当日レース別収支バー + 月間累計折れ線の2段グラフ

create_pnl_chart(records)       → bytes (PNG)
    汎用累計収支折れ線グラフ

予想ログCSV (data/predictions_log.csv) の列:
    date            : YYYY-MM-DD
    race_id         : str
    race_name       : str
    honmei_num      : int
    honmei_name     : str
    honmei_odds     : float   (単勝オッズ)
    actual_winner   : int     (実際の1着馬番、レース後に記録)
    payout          : int     (払い戻し額 / 的中なし=0)
    hit             : bool    (True=的中)
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI 不要モード（スレッドセーフ）
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import pandas as pd


def _setup_japanese_font() -> None:
    """利用可能な日本語フォントを自動検出して設定する。"""
    candidates = [
        "Noto Sans CJK JP",
        "IPAexGothic", "IPAGothic",
        "Hiragino Sans", "ヒラギノ角ゴシック",
        "Yu Gothic", "YuGothic",
        "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return
    # フォールバック: 英語表示になるが動作は維持
    plt.rcParams["font.family"] = "DejaVu Sans"


_setup_japanese_font()

BASE_DIR = Path(__file__).resolve().parents[3]
PREDICTIONS_LOG = BASE_DIR / "data" / "predictions_log.csv"

# ── スタイル定数 ─────────────────────────────────────────────────────────────
_BG_DARK  = "#1a1a2e"
_BG_PANEL = "#16213e"
_WIN_COLOR = "#4CAF50"
_LOSS_COLOR = "#ef5350"
_LINE_COLOR = "#64B5F6"
_TEXT_COLOR = "#cccccc"
_GRID_COLOR = "#2a2a4a"


def _apply_dark_style(ax: plt.Axes) -> None:
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#3a3a5a")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"¥{v:,.0f}"))
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.6, linestyle="--")


def _load_log() -> pd.DataFrame:
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_LOG, parse_dates=["date"])
    return df


# ---------------------------------------------------------------------------
# メイン: 当日成績 + 月間累計 グラフ
# ---------------------------------------------------------------------------

def create_today_results_chart(target_date: date) -> bytes:
    """
    2段グラフを生成して PNG バイト列で返す。

    上段: 当日のレース別収支バーチャート（◎単勝100円想定）
    下段: 当月の累計収支折れ線グラフ
    """
    df = _load_log()

    # --- 当日データ ---
    if df.empty:
        today_df = pd.DataFrame()
    else:
        today_df = df[df["date"].dt.date == target_date].copy()

    # --- 月間データ ---
    if df.empty:
        month_df = pd.DataFrame()
    else:
        month_start = target_date.replace(day=1)
        month_df = df[
            (df["date"].dt.date >= month_start) &
            (df["date"].dt.date <= target_date)
        ].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    fig.patch.set_facecolor(_BG_DARK)
    _apply_dark_style(ax1)
    _apply_dark_style(ax2)
    fig.subplots_adjust(hspace=0.45)

    # ── 上段: 当日レース別収支 ────────────────────────────────────────────
    if today_df.empty:
        ax1.text(0.5, 0.5,
                 f"{target_date}  まだ予想データがありません",
                 ha="center", va="center",
                 color="#888888", fontsize=13, transform=ax1.transAxes)
        ax1.set_title(f"{target_date}  当日成績",
                      color="#ffffff", fontsize=13, pad=12)
    else:
        today_df["net"] = today_df["payout"].fillna(0) - 100
        labels  = today_df["race_name"].str.slice(0, 8).tolist()
        nets    = today_df["net"].tolist()
        colors  = [_WIN_COLOR if n >= 0 else _LOSS_COLOR for n in nets]
        x = range(len(labels))

        bars = ax1.bar(x, nets, color=colors, edgecolor="none", width=0.65, zorder=2)
        ax1.axhline(0, color="#555", linewidth=0.8)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

        # バー上に金額ラベル
        for bar, net in zip(bars, nets):
            va = "bottom" if net >= 0 else "top"
            offset = 10 if net >= 0 else -10
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + offset,
                     f"¥{net:+,}", ha="center", va=va,
                     fontsize=8, color="#ffffff")

        hit_count = sum(1 for n in nets if n > 0)
        total_today = sum(nets)
        hit_rate = hit_count / len(nets) * 100 if nets else 0

        win_patch  = mpatches.Patch(color=_WIN_COLOR,  label=f"的中 {hit_count}R")
        loss_patch = mpatches.Patch(color=_LOSS_COLOR, label=f"外れ {len(nets)-hit_count}R")
        ax1.legend(handles=[win_patch, loss_patch],
                   facecolor="#2a2a4a", edgecolor="none",
                   labelcolor=_TEXT_COLOR, fontsize=9)

        ax1.set_title(
            f"{target_date}  的中 {hit_count}/{len(nets)}R ({hit_rate:.0f}%)  "
            f"収支: {'🟢' if total_today >= 0 else '🔴'} ¥{total_today:+,}",
            color="#ffffff", fontsize=13, pad=12,
        )

    # ── 下段: 月間累計収支 ────────────────────────────────────────────────
    if month_df.empty:
        ax2.text(0.5, 0.5, "月間データがありません",
                 ha="center", va="center",
                 color="#888888", fontsize=13, transform=ax2.transAxes)
    else:
        by_date = (
            month_df.groupby(month_df["date"].dt.date)
            .agg(payout=("payout", "sum"), count=("race_id", "count"))
            .reset_index()
        )
        by_date["net"] = by_date["payout"] - by_date["count"] * 100
        by_date["cumulative"] = by_date["net"].cumsum()

        xs   = list(range(len(by_date)))
        cum  = by_date["cumulative"].tolist()
        xlabels = [d.strftime("%-m/%-d") for d in by_date["date"]]

        ax2.plot(xs, cum, color=_LINE_COLOR, linewidth=2.5,
                 marker="o", markersize=5, markerfacecolor="#ffffff", zorder=3)
        ax2.fill_between(xs, cum, color=_LINE_COLOR, alpha=0.12)
        ax2.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax2.set_xticks(xs)
        ax2.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)

        final = cum[-1] if cum else 0
        month_label = target_date.replace(day=1).strftime("%Y年%-m月")
        ax2.set_title(
            f"{month_label}  累計収支: {'🟢' if final >= 0 else '🔴'} ¥{final:+,}",
            color="#ffffff", fontsize=13, pad=12,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# ---------------------------------------------------------------------------
# 汎用 P&L グラフ
# ---------------------------------------------------------------------------

def create_pnl_chart(records: list[dict]) -> bytes:
    """
    汎用累計収支折れ線グラフ。

    Parameters
    ----------
    records : list[dict]
        [{"date": "1/5", "net": 800}, {"date": "1/6", "net": -100}, ...]
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(_BG_DARK)
    _apply_dark_style(ax)

    dates = [r["date"] for r in records]
    nets  = [r["net"]  for r in records]
    cum   = []
    total = 0
    for n in nets:
        total += n
        cum.append(total)

    xs = range(len(dates))
    ax.plot(xs, cum, color=_LINE_COLOR, linewidth=2.5,
            marker="o", markersize=5, markerfacecolor="#ffffff", zorder=3)
    ax.fill_between(xs, cum, color=_LINE_COLOR, alpha=0.12)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)

    final = cum[-1] if cum else 0
    ax.set_title(
        f"累計収支: {'🟢' if final >= 0 else '🔴'} ¥{final:+,}",
        color="#ffffff", fontsize=13, pad=12,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# ---------------------------------------------------------------------------
# 予想ログ書き込みユーティリティ
# ---------------------------------------------------------------------------

def log_prediction(
    race_id: str,
    race_name: str,
    honmei_num: int,
    honmei_name: str,
    honmei_odds: float = 0.0,
) -> None:
    """予想をCSVに追記する（レース前に呼ぶ）。結果はレース後に update_result() で更新。"""
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date":         date.today().isoformat(),
        "race_id":      race_id,
        "race_name":    race_name,
        "honmei_num":   honmei_num,
        "honmei_name":  honmei_name,
        "honmei_odds":  honmei_odds,
        "actual_winner": "",
        "payout":        0,
        "hit":           False,
    }
    df_new = pd.DataFrame([row])
    if PREDICTIONS_LOG.exists():
        df_new.to_csv(PREDICTIONS_LOG, mode="a", header=False, index=False)
    else:
        df_new.to_csv(PREDICTIONS_LOG, index=False)
