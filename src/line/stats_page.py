"""
AI成績・回収率ページを docs/stats.html として生成する。

GitHub Pages で配信: https://gaassy33-ai.github.io/Keiba_ai_predection/stats.html

呼び出し:
    from src.line.stats_page import generate_stats_page
    generate_stats_page()
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

BASE_DIR         = Path(__file__).resolve().parents[2]
PREDICTIONS_LOG  = BASE_DIR / "docs" / "predictions_log.csv"
OUTPUT_PATH      = BASE_DIR / "docs" / "stats.html"


def _load_df() -> pd.DataFrame:
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_LOG, parse_dates=["date"])
    df["date_only"] = df["date"].dt.date
    return df


def generate_stats_page(output_path: Path = OUTPUT_PATH) -> None:
    """predictions_log.csv を読み込んで stats.html を生成する。"""
    df = _load_df()
    today = date.today()

    # ── 全期間集計 ──────────────────────────────────────────────
    if df.empty:
        total_races = hit_count = 0
        total_payout = total_bet = 0
        hit_rate = roi = 0.0
        cum_net = 0
        chart_labels = chart_data = "[]"
    else:
        total_races  = len(df)
        hit_count    = int(df["hit"].sum())
        total_payout = int(df["payout"].sum())
        total_bet    = total_races * 100
        hit_rate     = hit_count / total_races * 100 if total_races else 0
        roi          = total_payout / total_bet * 100 if total_bet else 0
        cum_net      = total_payout - total_bet

        # 累計収支グラフ用（日次）
        by_date = (
            df.groupby("date_only")
            .agg(payout=("payout", "sum"), count=("race_id", "count"))
            .reset_index()
            .sort_values("date_only")
        )
        by_date["net"]        = by_date["payout"] - by_date["count"] * 100
        by_date["cumulative"] = by_date["net"].cumsum()
        chart_labels = str([d.strftime("%-m/%-d") for d in by_date["date_only"]])
        chart_data   = str(by_date["cumulative"].tolist())

    # ── 当月集計 ────────────────────────────────────────────────
    if df.empty:
        month_races = month_hit = 0
        month_net = 0
        month_roi = 0.0
    else:
        month_start = today.replace(day=1)
        month_df    = df[df["date_only"] >= month_start]
        month_races = len(month_df)
        month_hit   = int(month_df["hit"].sum())
        month_net   = int(month_df["payout"].sum()) - month_races * 100
        month_roi   = (int(month_df["payout"].sum()) / (month_races * 100) * 100
                       if month_races else 0)

    # ── 最近の成績（最新20件）────────────────────────────────────
    recent_rows = ""
    if not df.empty:
        recent = df.sort_values("date", ascending=False).head(20)
        for _, row in recent.iterrows():
            hit     = bool(row.get("hit", False))
            payout  = int(row.get("payout", 0))
            net     = payout - 100
            icon    = "✅" if hit else "❌"
            net_str = f'<span class="pos">¥{net:+,}</span>' if hit else f'<span class="neg">¥-100</span>'
            race_name = str(row.get("race_name", ""))[:10]
            hon_num  = int(row.get("honmei_num", 0))
            hon_name = str(row.get("honmei_name", ""))
            d        = row["date"].strftime("%-m/%-d") if hasattr(row["date"], "strftime") else str(row["date"])[:5]
            recent_rows += f"""
            <tr>
              <td>{d}</td>
              <td>{icon}</td>
              <td class="race-name">{race_name}</td>
              <td>◎{hon_num} {hon_name}</td>
              <td class="amount">{net_str}</td>
            </tr>"""

    cum_class   = "pos" if cum_net  >= 0 else "neg"
    month_class = "pos" if month_net >= 0 else "neg"
    updated_at  = today.strftime("%Y/%m/%d")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>AI成績・回収率 | 競馬予想AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Hiragino Sans', 'Noto Sans JP', sans-serif;
  background: #0d1117;
  color: #e6edf3;
  max-width: 480px;
  margin: 0 auto;
  padding: 12px;
  font-size: 14px;
}}
h1 {{
  font-size: 18px;
  padding: 14px 0 4px;
  color: #fff;
  border-bottom: 2px solid #c62828;
  margin-bottom: 14px;
}}
h2 {{
  font-size: 13px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 18px 0 8px;
}}
.card {{
  background: #161b22;
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 10px;
}}
.kpi-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 10px;
}}
.kpi {{
  background: #161b22;
  border-radius: 10px;
  padding: 12px;
  text-align: center;
}}
.kpi-label {{ font-size: 11px; color: #8b949e; margin-bottom: 4px; }}
.kpi-value {{ font-size: 22px; font-weight: bold; }}
.pos {{ color: #4caf50; }}
.neg {{ color: #ef5350; }}
.neutral {{ color: #64b5f6; }}
.chart-wrap {{ position: relative; height: 200px; margin-top: 4px; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}}
th {{
  background: #21262d;
  color: #8b949e;
  font-weight: normal;
  padding: 6px 4px;
  text-align: left;
  font-size: 11px;
}}
td {{
  padding: 6px 4px;
  border-bottom: 1px solid #21262d;
  vertical-align: middle;
}}
td.race-name {{ color: #8b949e; font-size: 11px; }}
td.amount {{ text-align: right; font-weight: bold; }}
.footer {{
  color: #484f58;
  font-size: 11px;
  text-align: center;
  margin-top: 20px;
  padding-bottom: 20px;
}}
</style>
</head>
<body>

<h1>🏇 AI成績・回収率</h1>

<h2>全期間</h2>
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">総レース数</div>
    <div class="kpi-value neutral">{total_races}<span style="font-size:13px">R</span></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">的中率</div>
    <div class="kpi-value neutral">{hit_rate:.1f}<span style="font-size:13px">%</span></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">回収率</div>
    <div class="kpi-value {'pos' if roi >= 100 else 'neg'}">{roi:.1f}<span style="font-size:13px">%</span></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">累計収支</div>
    <div class="kpi-value {cum_class}">¥{cum_net:+,}</div>
  </div>
</div>

<h2>{today.strftime('%-m月')}の成績</h2>
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">的中 / レース数</div>
    <div class="kpi-value neutral">{month_hit}<span style="font-size:14px">/{month_races}R</span></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">月間収支</div>
    <div class="kpi-value {month_class}">¥{month_net:+,}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">月間回収率</div>
    <div class="kpi-value {'pos' if month_roi >= 100 else 'neg'}">{month_roi:.1f}<span style="font-size:13px">%</span></div>
  </div>
  <div class="kpi">
    <div class="kpi-label">月間的中率</div>
    <div class="kpi-value neutral">{(month_hit/month_races*100 if month_races else 0):.1f}<span style="font-size:13px">%</span></div>
  </div>
</div>

<h2>累計収支グラフ</h2>
<div class="card">
  <div class="chart-wrap">
    <canvas id="pnlChart"></canvas>
  </div>
</div>

<h2>最近の予想</h2>
<div class="card" style="padding: 0; overflow: hidden;">
  <table>
    <thead>
      <tr>
        <th>日付</th>
        <th></th>
        <th>レース</th>
        <th>◎本命</th>
        <th style="text-align:right">収支</th>
      </tr>
    </thead>
    <tbody>
      {recent_rows if recent_rows else '<tr><td colspan="5" style="text-align:center;color:#484f58;padding:20px">まだデータがありません</td></tr>'}
    </tbody>
  </table>
</div>

<p class="footer">更新: {updated_at} ／ AIの予想は参考情報です</p>

<script>
const labels = {chart_labels};
const data   = {chart_data};
const ctx    = document.getElementById('pnlChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels,
    datasets: [{{
      data,
      borderColor: data.length && data[data.length-1] >= 0 ? '#4caf50' : '#ef5350',
      backgroundColor: data.length && data[data.length-1] >= 0
        ? 'rgba(76,175,80,0.08)' : 'rgba(239,83,80,0.08)',
      borderWidth: 2,
      pointRadius: data.length <= 20 ? 4 : 2,
      pointBackgroundColor: '#fff',
      tension: 0.3,
      fill: true,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }},
      y: {{
        ticks: {{
          color: '#8b949e', font: {{ size: 10 }},
          callback: v => '¥' + v.toLocaleString()
        }},
        grid: {{ color: '#21262d' }},
      }}
    }}
  }}
}});
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
