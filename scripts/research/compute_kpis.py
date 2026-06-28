"""
compute_kpis.py
===============
analyze_production_performance.py が生成した
data/analysis/full_predictions.pkl + result_cache.json を集計し、
KPI・セグメント別パフォーマンス・失注パターンを算出する。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "analysis" / "result_cache.json"
PRED_PKL = ROOT / "data" / "analysis" / "full_predictions.pkl"

cache: dict = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
df = pd.read_pickle(PRED_PKL)

for col in ["odds_i", "odds_j", "est_quinella_odds", "p_model", "p_market", "ev"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

rows = []
for _, row in df.iterrows():
    rid = row["race_id"]
    res = cache.get(rid)
    if res is None:
        continue  # 未確定/取得失敗

    top5 = res["top5"]
    top2_nums = {h["num"] for h in top5 if h["rank"] <= 2}
    top3_nums = {h["num"] for h in top5 if h["rank"] <= 3}
    is_g1 = res["is_g1"]
    payout = res["quinella_payout"]

    ni = str(row["horse_num_i"])
    nj = str(row["horse_num_j"])

    hit = (ni in top2_nums) and (nj in top2_nums)
    axis_in_top2 = ni in top2_nums          # horse_num_i は常に軸馬(axis1 or axis2)
    partner_in_top2 = nj in top2_nums

    rows.append({
        "src_date": row["src_date"],
        "race_id": rid,
        "race_name": row["race_name"],
        "course_type": row["course_type"],
        "distance": row["distance"],
        "axis_num": ni,
        "partner_num": nj,
        "odds_i": row["odds_i"],
        "odds_j": row["odds_j"],
        "ev": row["ev"],
        "p_model": row["p_model"],
        "est_quinella_odds": row["est_quinella_odds"],
        "is_g1": is_g1,
        "hit": hit,
        "axis_in_top2": axis_in_top2,
        "partner_in_top2": partner_in_top2,
        "payout": payout if hit else 0,
        "cost": 100,
    })

bets = pd.DataFrame(rows)
print(f"集計対象ベット数: {len(bets)} / 全{len(df)}件 (未確定・取得失敗を除く)")
print(f"対象稼働日数: {bets['src_date'].nunique()}")
print(f"対象レース数: {bets['race_id'].nunique()}")

bets.to_pickle(ROOT / "data" / "analysis" / "bets_with_results.pkl")

# ── 1. 全体KPI ──────────────────────────────────────────────────────
total_bets = len(bets)
total_cost = bets["cost"].sum()
total_payout = bets["payout"].sum()
hits = bets["hit"].sum()
hit_rate = hits / total_bets * 100
roi = total_payout / total_cost * 100
balance = total_payout - total_cost

print("\n" + "=" * 60)
print("1. 全体KPI")
print("=" * 60)
print(f"稼働日数        : {bets['src_date'].nunique()}日")
print(f"対象レース数    : {bets['race_id'].nunique()}")
print(f"総購入点数      : {total_bets}点")
print(f"総投資額        : {total_cost:,}円")
print(f"的中点数        : {hits}点")
print(f"的中率          : {hit_rate:.1f}%")
print(f"総払戻金額      : {total_payout:,}円")
print(f"収支            : {balance:+,}円")
print(f"実運用ROI       : {roi:.1f}%")

# 日次バランスで連敗・ドローダウンを算出（レース単位: そのレースの全ベットがhit=0ならレース不的中）
daily = bets.groupby("src_date").agg(
    cost=("cost", "sum"), payout=("payout", "sum")
).reset_index().sort_values("src_date")
daily["balance"] = daily["payout"] - daily["cost"]
daily["cum_balance"] = daily["balance"].cumsum()

# 最大ドローダウン（累積資産の山から谷への最大下落）
running_max = daily["cum_balance"].cummax()
drawdown = daily["cum_balance"] - running_max
max_drawdown = drawdown.min()

# 日次の最大連敗（収支マイナスの日が連続した最大数）
loss_streak = 0
max_loss_streak = 0
for b in daily["balance"]:
    if b < 0:
        loss_streak += 1
        max_loss_streak = max(max_loss_streak, loss_streak)
    else:
        loss_streak = 0

# レース単位の連敗（的中レースが出るまでの不的中レース数の最大値）
race_hit = bets.groupby(["src_date", "race_id"])["hit"].max().reset_index()
race_hit = race_hit.sort_values(["src_date", "race_id"])
race_loss_streak = 0
max_race_loss_streak = 0
for h in race_hit["hit"]:
    if not h:
        race_loss_streak += 1
        max_race_loss_streak = max(max_race_loss_streak, race_loss_streak)
    else:
        race_loss_streak = 0

print(f"最大連敗(日次)  : {max_loss_streak}日")
print(f"最大連敗(レース): {max_race_loss_streak}レース")
print(f"最大ドローダウン: {max_drawdown:,}円")
print("\n[日次収支]")
for _, r in daily.iterrows():
    print(f"  {r['src_date']}: 投資{r['cost']:>5,}円  払戻{r['payout']:>7,.0f}円  収支{r['balance']:>+8,.0f}円  累積{r['cum_balance']:>+9,.0f}円")

# ── 2. モード別比較 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. モード別パフォーマンス比較（通常 vs G1特別モード）")
print("=" * 60)
for mode, sub in [("通常モード", bets[~bets["is_g1"]]), ("G1特別モード", bets[bets["is_g1"]])]:
    if len(sub) == 0:
        print(f"{mode}: 0点")
        continue
    c = sub["cost"].sum()
    p = sub["payout"].sum()
    h = sub["hit"].sum()
    print(f"{mode}: {len(sub)}点  的中{h}点({h/len(sub)*100:.1f}%)  "
          f"投資{c:,}円  払戻{p:,.0f}円  ROI={p/c*100:.1f}%  収支{p-c:+,.0f}円")

# ── 3. 軸馬の精度 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 軸馬（Axis）精度")
print("=" * 60)
axis_unique = bets.drop_duplicates(subset=["race_id", "axis_num"])
axis_top2_rate = axis_unique["axis_in_top2"].mean() * 100
print(f"軸馬指名件数（race×axis_num の重複除去）: {len(axis_unique)}件")
print(f"軸馬の連対率（1着or2着）: {axis_top2_rate:.1f}%")

axis_unique_normal = axis_unique[~axis_unique["is_g1"]]
axis_unique_g1 = axis_unique[axis_unique["is_g1"]]
print(f"  通常モード: {len(axis_unique_normal)}件  連対率{axis_unique_normal['axis_in_top2'].mean()*100:.1f}%")
print(f"  G1モード  : {len(axis_unique_g1)}件  連対率{axis_unique_g1['axis_in_top2'].mean()*100:.1f}%")

# ── 4. 失注パターン分析（レース単位、通常モードのみで分析）──────────────
print("\n" + "=" * 60)
print("4. 失注パターン分析（レース単位）")
print("=" * 60)
race_level = bets.groupby(["src_date", "race_id", "is_g1"]).agg(
    hit=("hit", "max"),
    axis_in_top2=("axis_in_top2", "max"),
).reset_index()
missed = race_level[~race_level["hit"]]
print(f"不的中レース数: {len(missed)} / {len(race_level)}レース")

axis_survived_partner_missed = missed[missed["axis_in_top2"]]
axis_itself_missed = missed[~missed["axis_in_top2"]]
print(f"  軸は来たがヒモが抜けた  : {len(axis_survived_partner_missed)}件 "
      f"({len(axis_survived_partner_missed)/len(missed)*100:.1f}%)")
print(f"  そもそも軸が飛んだ      : {len(axis_itself_missed)}件 "
      f"({len(axis_itself_missed)/len(missed)*100:.1f}%)")

# ヒモ抜けレースで、実際に2着馬(または1着馬)になった「我々が買っていない馬」のオッズ分布
print("\n[ヒモ抜けレースの実際の対抗馬オッズ分布]")
miss_partner_odds = []
for _, r in axis_survived_partner_missed.iterrows():
    res = cache[r["race_id"]]
    top2 = [h for h in res["top5"] if h["rank"] <= 2]
    sub_bets = bets[(bets["race_id"] == r["race_id"]) & (bets["src_date"] == r["src_date"])]
    our_axis_nums = set(sub_bets["axis_num"])
    our_partner_nums = set(sub_bets["partner_num"])
    our_nums = our_axis_nums | our_partner_nums
    for h in top2:
        if h["num"] not in our_nums:
            miss_partner_odds.append(h["win_odds"])

import numpy as np
mp = pd.Series([o for o in miss_partner_odds if o is not None])
print(f"  該当数: {len(mp)}")
if len(mp):
    print(f"  最小/25%/50%/75%/最大: {mp.min():.1f} / {mp.quantile(.25):.1f} / {mp.median():.1f} / {mp.quantile(.75):.1f} / {mp.max():.1f}")
    print(f"  longshot_odds_max(30倍)以内: {(mp<=30).sum()}件 ({(mp<=30).mean()*100:.1f}%)")
    print(f"  est_quinella_odds_max(50倍)換算想定域: 参考として単勝30倍以内の頻度を確認")
    print(f"  オッズ分布(全件): {sorted(mp.round(1).tolist())}")

print("\n[軸が飛んだレースの実際の1-2着馬 人気・オッズ]")
axis_fail_detail = []
for _, r in axis_itself_missed.iterrows():
    res = cache[r["race_id"]]
    top2 = [h for h in res["top5"] if h["rank"] <= 2]
    sub_bets = bets[(bets["race_id"] == r["race_id"]) & (bets["src_date"] == r["src_date"])]
    axis_odds = sub_bets["odds_i"].iloc[0] if len(sub_bets) else None
    axis_fail_detail.append({
        "race_id": r["race_id"],
        "axis_odds": axis_odds,
        "winner_pop": top2[0]["popularity"] if top2 else None,
        "winner_odds": top2[0]["win_odds"] if top2 else None,
        "second_pop": top2[1]["popularity"] if len(top2) > 1 else None,
        "second_odds": top2[1]["win_odds"] if len(top2) > 1 else None,
    })
afd = pd.DataFrame(axis_fail_detail)
print(afd.to_string(index=False))
if len(afd):
    print(f"\n  軸馬の単勝オッズ平均: {afd['axis_odds'].mean():.1f}倍")
    print(f"  実際の1着馬人気の平均: {afd['winner_pop'].mean():.1f}番人気")

# ── 5. EVバンド別パフォーマンス（再確認） ───────────────────────────
print("\n" + "=" * 60)
print("5. EVバンド別パフォーマンス（通常モードのみ）")
print("=" * 60)
normal_bets = bets[~bets["is_g1"]].copy()
bins = [0, 1.0, 1.5, 2.0, 3.0, 5.0, 100]
labels = ["<1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0+"]
normal_bets["ev_band"] = pd.cut(normal_bets["ev"], bins=bins, labels=labels)
for band, sub in normal_bets.groupby("ev_band", observed=True):
    if len(sub) == 0:
        continue
    c, p = sub["cost"].sum(), sub["payout"].sum()
    print(f"  EV{band}: {len(sub)}点  的中{sub['hit'].sum()}点  ROI={p/c*100:.1f}%")
