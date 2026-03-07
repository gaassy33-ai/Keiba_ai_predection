"""
GitHub Pages 用 HTML ページ生成モジュール。

生成するページ:
    docs/today.html   - 最新レース予想（毎レース上書き）
    docs/results.html - 当日成績サマリー（毎レース上書き）
    docs/index.html   - ランディングページ（初回のみ生成）
"""

from __future__ import annotations

import html
from datetime import date
from pathlib import Path

import pandas as pd

PAGES_BASE_URL = "https://gaassy33-ai.github.io/Keiba_ai_predection"

LIFF_ID_TODAY   = "2009359724-yHDAAeZg"
LIFF_ID_RESULTS = "2009359724-oGG4JgxI"

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Hiragino Sans', sans-serif;
    background: #0d1117;
    color: #e6edf3;
    max-width: 480px;
    margin: 0 auto;
    padding: 12px;
}
.header {
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    background: #161b22;
    border-left: 4px solid #c62828;
}
.header.g1  { border-left-color: #c62828; }
.header.g2  { border-left-color: #1565c0; }
.header.g3  { border-left-color: #2e7d32; }
.header.op  { border-left-color: #e65100; }
.grade-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
    background: #c62828;
    color: #fff;
    margin-bottom: 6px;
}
.race-name { font-size: 18px; font-weight: bold; margin-bottom: 4px; }
.race-meta { font-size: 12px; color: #8b949e; }
.deadline  { font-size: 13px; color: #ff6b6b; font-weight: bold; margin-top: 4px; }
.card {
    background: #161b22;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}
.horse-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.mark {
    width: 28px; height: 28px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 16px; flex-shrink: 0;
}
.mark.honmei { background: #c62828; }
.mark.taikou { background: #1565c0; }
.mark.ana    { background: #2e7d32; }
.horse-num {
    width: 24px; height: 24px;
    border-radius: 5px;
    background: #1a237e;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: bold; flex-shrink: 0;
}
.horse-name { flex: 1; font-size: 15px; font-weight: bold; }
.win-prob   { font-size: 18px; font-weight: bold; color: #c62828; }
.jockey     { font-size: 12px; color: #8b949e; margin-left: 36px; }
.tags       { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; margin-left: 36px; }
.tag {
    font-size: 11px; padding: 2px 7px;
    border-radius: 10px;
    background: #21262d; color: #8b949e;
}
.section-title {
    font-size: 13px; font-weight: bold; color: #8b949e;
    margin: 14px 0 8px;
    display: flex; align-items: center; justify-content: space-between;
}
.bet-row {
    background: #21262d;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}
.bet-row.featured { background: #2d1515; border: 1px solid #c62828; }
.bet-badge {
    font-size: 11px; font-weight: bold;
    padding: 3px 7px;
    border-radius: 5px;
    background: #1565c0; color: #fff; flex-shrink: 0;
}
.bet-info { flex: 1; }
.bet-label { font-size: 13px; font-weight: bold; }
.bet-meta  { font-size: 11px; color: #8b949e; margin-top: 2px; }
.bet-amount { font-size: 14px; font-weight: bold; color: #e6edf3; }
.bet-amount.featured { color: #ff6b6b; }
.netkeiba-btn {
    display: block;
    background: #1a237e;
    color: #fff;
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: bold;
    margin-top: 14px;
}
.disclaimer { font-size: 11px; color: #484f58; text-align: center; margin-top: 10px; }
.updated    { font-size: 11px; color: #484f58; text-align: center; margin-top: 14px; }
.separator  { border: none; border-top: 1px solid #21262d; margin: 10px 0; }
/* results page */
.result-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.result-mark { font-size: 20px; width: 28px; text-align: center; }
.result-info { flex: 1; }
.result-race { font-size: 13px; font-weight: bold; }
.result-horse { font-size: 12px; color: #8b949e; }
.result-amount { font-size: 14px; font-weight: bold; }
.result-amount.plus  { color: #2ea043; }
.result-amount.minus { color: #f85149; }
.summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
.summary-item { background: #21262d; border-radius: 8px; padding: 12px; text-align: center; }
.summary-value { font-size: 22px; font-weight: bold; }
.summary-label { font-size: 11px; color: #8b949e; margin-top: 2px; }
.nav { display: flex; gap: 8px; margin-bottom: 14px; }
.nav a {
    flex: 1; text-align: center; padding: 10px;
    background: #21262d; border-radius: 8px;
    color: #8b949e; text-decoration: none; font-size: 13px;
}
.nav a.active { background: #1a237e; color: #fff; font-weight: bold; }
"""


def _html_doc(title: str, body: str, active_page: str = "") -> str:
    from datetime import datetime
    updated = datetime.now().strftime("%Y/%m/%d %H:%M 更新")
    nav_today   = "active" if active_page == "today"   else ""
    nav_results = "active" if active_page == "results" else ""

    liff_id = LIFF_ID_TODAY if active_page == "today" else LIFF_ID_RESULTS
    today_url   = f"https://liff.line.me/{LIFF_ID_TODAY}"
    results_url = f"https://liff.line.me/{LIFF_ID_RESULTS}"

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
<script charset="utf-8" src="https://static.line-scdn.net/liff/edge/2/sdk.js"></script>
<script>
  liff.init({{ liffId: "{liff_id}" }}).catch(function(e) {{ console.error(e); }});
</script>
</head>
<body>
<nav class="nav">
  <a href="{today_url}"   class="{nav_today}">AI予想</a>
  <a href="{results_url}" class="{nav_results}">今日の成績</a>
</nav>
{body}
<p class="updated">{updated}</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 予想ページ
# ---------------------------------------------------------------------------

def generate_prediction_page(race_data: dict, path: Path) -> None:
    """
    race_data から docs/today.html を生成する。

    Parameters
    ----------
    race_data : dict
        notifier.create_prediction_message() と同じ構造。
    path : Path
        出力先ファイルパス（例: Path("docs/today.html")）。
    """
    race_name       = race_data.get("race_name", "レース情報")
    grade           = race_data.get("grade", "一般")
    race_id         = race_data.get("race_id", "")
    course_type     = race_data.get("course_type", "")
    distance        = race_data.get("distance", "")
    weather         = race_data.get("weather", "")
    ground_condition = race_data.get("ground_condition", "")
    deadline        = race_data.get("deadline", "")
    horses: list[dict] = race_data.get("horses", [])
    strategies      = race_data.get("strategies", [])
    budget          = race_data.get("budget", 10_000)

    grade_cls_map = {"GⅠ": "g1", "G1": "g1", "GⅡ": "g2", "G2": "g2",
                     "GⅢ": "g3", "G3": "g3", "L": "op", "OP": "op"}
    grade_cls = grade_cls_map.get(grade, "")

    course_label = f"{course_type} {distance}m" if distance else course_type
    meta_parts = [p for p in [course_label, weather, ground_condition] if p]
    meta = " / ".join(meta_parts)

    netkeiba_url = (
        f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        if race_id else "https://race.netkeiba.com/"
    )

    # --- ヘッダー ---
    deadline_html = f'<p class="deadline">⏰ 締め切り {html.escape(deadline)}</p>' if deadline else ""
    header_html = f"""
<div class="header {grade_cls}">
  <span class="grade-badge">{html.escape(grade)}</span>
  <p class="race-name">{html.escape(race_name)}</p>
  <p class="race-meta">{html.escape(meta)}</p>
  {deadline_html}
</div>
"""

    # --- 馬カード ---
    mark_classes = {"◎": "honmei", "○": "taikou", "▲": "ana"}
    horse_cards_html = '<p class="section-title">AI予想印</p>'
    for horse in horses[:3]:
        mark = horse.get("mark", "◎")
        mark_cls = mark_classes.get(mark, "honmei")
        num = html.escape(str(horse.get("horse_number", "-")))
        name = html.escape(horse.get("horse_name", "---"))
        jockey = html.escape(horse.get("jockey_name", ""))
        prob = horse.get("win_prob", 0.0)
        tags = horse.get("tags", [])

        tag_badges = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags[:4])
        tags_html = f'<div class="tags">{tag_badges}</div>' if tag_badges else ""
        jockey_html = f'<p class="jockey">{html.escape(jockey)}</p>' if jockey else ""

        horse_cards_html += f"""
<div class="card">
  <div class="horse-row">
    <span class="mark {mark_cls}">{html.escape(mark)}</span>
    <span class="horse-num">{num}</span>
    <span class="horse-name">{name}</span>
    <span class="win-prob">{prob:.1%}</span>
  </div>
  {jockey_html}
  {tags_html}
</div>
"""

    # --- 買い目セクション ---
    bet_html = ""
    if strategies:
        used = sum(getattr(b, "allocation", 0) for b in strategies)
        bet_html = f"""
<p class="section-title">
  <span>推奨買い目</span>
  <span style="font-weight:normal">予算 ¥{used:,} / ¥{budget:,}</span>
</p>
"""
        for bet in strategies:
            featured_cls = "featured" if getattr(bet, "is_featured", False) else ""
            icon = "🔥 " if getattr(bet, "is_featured", False) else ""
            label = html.escape(getattr(bet, "label", ""))
            bet_type = html.escape(getattr(bet, "bet_type", ""))
            ev = getattr(bet, "ev", 0.0)
            combo = getattr(bet, "combo_count", 1)
            allocation = getattr(bet, "allocation", 0)
            amount_cls = "featured" if getattr(bet, "is_featured", False) else ""
            bet_html += f"""
<div class="bet-row {featured_cls}">
  <span class="bet-badge">{bet_type}</span>
  <div class="bet-info">
    <p class="bet-label">{icon}{label}</p>
    <p class="bet-meta">EV:{ev:.2f}  {combo}点</p>
  </div>
  <span class="bet-amount {amount_cls}">¥{allocation:,}</span>
</div>
"""

    # --- フッター ---
    footer_html = f"""
<a href="{html.escape(netkeiba_url)}" class="netkeiba-btn" target="_blank">
  netkeiba で詳細を見る
</a>
<p class="disclaimer">※ 予想は参考情報です。馬券購入は自己責任でお願いします。</p>
"""

    body = header_html + horse_cards_html + bet_html + footer_html
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html_doc(f"AI予想 - {race_name}", body, active_page="today"), encoding="utf-8")


# ---------------------------------------------------------------------------
# 成績ページ
# ---------------------------------------------------------------------------

def generate_results_page(target_date: date, path: Path) -> None:
    """
    predictions_log.csv から docs/results.html を生成する。

    Parameters
    ----------
    target_date : date
        集計対象日。
    path : Path
        出力先ファイルパス（例: Path("docs/results.html")）。
    """
    from src.line.chart import PREDICTIONS_LOG

    date_str = target_date.strftime("%-m/%-d")
    month_label = target_date.strftime("%-m月")

    if not PREDICTIONS_LOG.exists():
        body = f"""
<div class="card">
  <p style="text-align:center; color:#8b949e; padding:20px 0">
    まだ予想データがありません。<br>
    土日のレース20分前に自動更新されます。
  </p>
</div>
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_html_doc(f"今日の成績 {date_str}", body, active_page="results"), encoding="utf-8")
        return

    df = pd.read_csv(PREDICTIONS_LOG, parse_dates=["date"])
    today_df = df[df["date"].dt.date == target_date].copy()

    rows_html = ""
    if today_df.empty:
        rows_html = '<p style="text-align:center; color:#8b949e; padding:20px 0">本日の予想記録がありません。</p>'
    else:
        for _, row in today_df.iterrows():
            hit = bool(row.get("hit", False))
            payout = int(row.get("payout", 0))
            net = payout - 100
            mark_icon = "✅" if hit else "❌"
            amount = f"¥{net:+,}" if hit else "¥-100"
            amount_cls = "plus" if hit else "minus"
            race_name_short = html.escape(str(row.get("race_name", ""))[:10])
            honmei_num = int(row.get("honmei_num", 0))
            honmei_name = html.escape(str(row.get("honmei_name", "")))
            rows_html += f"""
<div class="result-row">
  <span class="result-mark">{mark_icon}</span>
  <div class="result-info">
    <p class="result-race">{race_name_short}</p>
    <p class="result-horse">◎{honmei_num} {honmei_name}</p>
  </div>
  <span class="result-amount {amount_cls}">{amount}</span>
</div>
<hr class="separator">
"""

    # 集計
    summary_html = ""
    if not today_df.empty:
        hit_count   = int(today_df["hit"].sum())
        total_races = len(today_df)
        hit_rate    = hit_count / total_races * 100
        net_today   = int(today_df["payout"].sum()) - total_races * 100
        today_color = "#2ea043" if net_today >= 0 else "#f85149"

        month_start = target_date.replace(day=1)
        df["date_only"] = df["date"].dt.date
        month_df  = df[df["date_only"] >= month_start]
        net_month = int(month_df["payout"].sum()) - len(month_df) * 100
        month_color = "#2ea043" if net_month >= 0 else "#f85149"

        summary_html = f"""
<div class="summary-grid">
  <div class="summary-item">
    <p class="summary-value">{hit_count}/{total_races}</p>
    <p class="summary-label">本日的中 ({hit_rate:.0f}%)</p>
  </div>
  <div class="summary-item">
    <p class="summary-value" style="color:{today_color}">¥{net_today:+,}</p>
    <p class="summary-label">本日収支</p>
  </div>
  <div class="summary-item" style="grid-column:1/-1">
    <p class="summary-value" style="color:{month_color}">¥{net_month:+,}</p>
    <p class="summary-label">{html.escape(month_label)}累計収支</p>
  </div>
</div>
"""

    body = f"""
<div class="card">
  <p style="font-weight:bold; margin-bottom:12px">今日の成績  {date_str}</p>
  {rows_html}
  {summary_html}
</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html_doc(f"今日の成績 {date_str}", body, active_page="results"), encoding="utf-8")


# ---------------------------------------------------------------------------
# インデックスページ（初回のみ使用）
# ---------------------------------------------------------------------------

def generate_index_page(path: Path) -> None:
    body = """
<div class="card" style="text-align:center; padding:30px 20px">
  <p style="font-size:40px; margin-bottom:12px">🏇</p>
  <p style="font-size:20px; font-weight:bold; margin-bottom:8px">競馬予想 AI</p>
  <p style="color:#8b949e; margin-bottom:20px; font-size:14px">
    土日のレース前に自動更新されます
  </p>
  <a href="today.html"   class="netkeiba-btn" style="margin-bottom:8px">AI予想を見る</a>
  <a href="results.html" class="netkeiba-btn" style="background:#21262d; margin-top:8px">今日の成績を見る</a>
</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html_doc("競馬予想 AI", body), encoding="utf-8")
