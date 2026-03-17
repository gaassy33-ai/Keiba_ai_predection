"""
NAR（地方競馬）過去1年バックテスト
- データ: data/raw/nar_results.csv / nar_meta.csv
- モデル: lgbm_model.pkl (JRA モデルを NAR に適用)
  ※ NAR 専用モデルが存在する場合は nar_lgbm_model.pkl を使用
- 買い条件: honmei_prob >= MIN_HONMEI_PROB かつ 信頼度差 >= MIN_GAP
- 馬券: 単勝・複勝・馬連・馬単（Harville 推定払戻）
- NAR控除率 適用（JRA と異なる）

実行: python backtest_nar.py
     python backtest_nar.py --prob 0.20  # 買い条件変更
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
from itertools import permutations, combinations
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer

# ── CLI 引数 ───────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NAR バックテスト")
parser.add_argument("--prob",     type=float, default=0.25,  help="本命最低確率 (default: 0.25)")
parser.add_argument("--gap",      type=float, default=0.05,  help="信頼度差 (default: 0.05)")
parser.add_argument("--date-from",default=None, help="バックテスト開始日 YYYY-MM-DD (default: 全期間)")
parser.add_argument("--ev-pool",  type=int,   default=5,    help="相手プール頭数 (default: 5)")
parser.add_argument("--max-baren",type=int,   default=3,    help="馬連最大点数 (default: 3)")
parser.add_argument("--max-umatan",type=int,  default=3,    help="馬単最大点数 (default: 3)")
args = parser.parse_args()

MIN_HONMEI_PROB = args.prob
MIN_GAP         = args.gap
DATE_FROM       = args.date_from
EV_POOL         = args.ev_pool
MAX_BAREN       = args.max_baren
MAX_UMATAN      = args.max_umatan
TORIKAMI_THR    = 1.05

# NAR 控除率（JRA より若干高め）
NAR_TAKE = {"馬連": 0.25, "馬単": 0.25, "3連複": 0.25, "3連単": 0.295}

OUT_CSV = ROOT / "data/processed/backtest_nar.csv"

NAR_VENUE_CODES = {
    "34": "浦和", "35": "船橋", "36": "大井", "37": "川崎",
    "38": "金沢", "39": "笠松", "40": "名古屋",
    "42": "園田", "43": "姫路", "44": "高知", "45": "佐賀",
}

# ── データパス確認 ─────────────────────────────────────────────
nar_results_path = ROOT / "data" / "raw" / "nar_results.csv"
nar_meta_path    = ROOT / "data" / "raw" / "nar_meta.csv"

if not nar_results_path.exists() or not nar_meta_path.exists():
    print("=" * 60)
    print("ERROR: NAR データファイルが見つかりません")
    print(f"  {nar_results_path}")
    print(f"  {nar_meta_path}")
    print("\nまず以下を実行してください:")
    print("  python collect_nar_history.py")
    print("=" * 60)
    sys.exit(1)

# ── モデルロード ───────────────────────────────────────────────
nar_model_path = ROOT / "data" / "models" / "nar_lgbm_model.pkl"
jra_model_path = ROOT / "data" / "models" / "lgbm_model.pkl"

if nar_model_path.exists():
    obj = joblib.load(nar_model_path)
    model_label = "NAR専用モデル"
elif jra_model_path.exists():
    obj = joblib.load(jra_model_path)
    model_label = "JRAモデル（NAR転用）"
else:
    print("ERROR: モデルファイルが見つかりません")
    sys.exit(1)

win_model   = obj["model"]
place_model = obj.get("place_model")
print(f"モデル: {model_label}  win={type(win_model).__name__}, place={type(place_model).__name__ if place_model else 'None'}")

# ── データ読み込み ─────────────────────────────────────────────
nar_res  = pd.read_csv(nar_results_path, dtype=str)
nar_meta = pd.read_csv(nar_meta_path,    dtype=str)
print(f"NAR results: {len(nar_res):,} rows  meta: {len(nar_meta)} races")

print(f"特徴量ルックバック用 NAR履歴: {len(nar_res):,} rows")

# 対象レース
nar_meta["race_date"] = pd.to_datetime(nar_meta["race_date"], errors="coerce")
if DATE_FROM:
    target_meta = nar_meta[nar_meta["race_date"] >= DATE_FROM].sort_values("race_date")
else:
    target_meta = nar_meta.dropna(subset=["race_date"]).sort_values("race_date")

print(f"対象レース: {len(target_meta)} races  (prob>={MIN_HONMEI_PROB}, gap>={MIN_GAP})")

# ── ユーティリティ ─────────────────────────────────────────────

def parse_odds(val):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return float("nan")

def market_probs(odds_list):
    raw = np.array([1.0 / max(o, 1.01) for o in odds_list])
    s = raw.sum()
    return (raw / s).tolist() if s > 0 else raw.tolist()

def harville(probs, order):
    p, rem = 1.0, 1.0
    for idx in order:
        if rem < 1e-9:
            return 0.0
        p *= probs[idx] / rem
        rem -= probs[idx]
    return p

def prob_quinella(probs, i, j):
    return harville(probs, [i, j]) + harville(probs, [j, i])

def est_odds(prob, bet_type):
    take = NAR_TAKE.get(bet_type, 0.25)
    return (1.0 - take) / max(prob, 0.001)

def synth_odds(elist):
    d = sum(1.0 / max(e, 1e-9) for e in elist)
    return 1.0 / d if d > 0 else 0.0

# ── FeatureEngineer 初期化 ─────────────────────────────────────
# from_stats: JRA の事前計算済み統計を使用（NAR 馬はデフォルト値）
# NAR 馬・騎手の個別統計は nar_res から補完
print("FeatureEngineer 初期化中...")
stats_path = ROOT / "data" / "models" / "feature_stats.pkl"
if stats_path.exists():
    fe = FeatureEngineer.from_stats(stats_path)
    # NAR 馬・騎手の統計で内部集計を上書き（ルックバック精度向上）
    fe.history = fe._preprocess_history(nar_res)
    fe.precompute_aggregations()
else:
    fe = FeatureEngineer(nar_res)
    fe.precompute_aggregations()
print("完了")

# ── バックテスト ──────────────────────────────────────────────
results = []
target_ids = target_meta["race_id"].tolist()

for i, race_id in enumerate(target_ids):
    meta_row    = target_meta[target_meta["race_id"] == race_id].iloc[0]
    race_date   = meta_row["race_date"]
    course_type = str(meta_row.get("course_type", "") or "")
    distance    = int(meta_row.get("distance", 0) or 0)
    gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
    wx_code     = int(meta_row.get("weather_code", -1) or -1)
    race_name   = str(meta_row.get("race_name", "") or "")
    venue_code  = str(race_id)[4:6]
    venue_name  = NAR_VENUE_CODES.get(venue_code, f"地方{venue_code}")

    if not course_type or distance == 0:
        continue

    race_entries = nar_res[nar_res["race_id"] == race_id].copy()
    if len(race_entries) < 3:
        continue

    # entry_df 組み立て
    entry_df = race_entries[[
        "horse_id", "horse_name", "horse_number", "frame_number",
        "sex_age", "weight_carried", "jockey_id",
    ]].copy()
    entry_df["sex"]            = entry_df["sex_age"].str[0]
    entry_df["age"]            = pd.to_numeric(entry_df["sex_age"].str[1:], errors="coerce")
    entry_df["weight_carried"] = pd.to_numeric(entry_df["weight_carried"], errors="coerce")
    entry_df["father"]         = ""
    entry_df["mother_father"]  = ""

    try:
        feat_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=course_type,
            distance=distance,
            ground_condition_code=gc_code,
            weather_code=wx_code,
        )
    except Exception:
        continue

    X = (feat_df[FeatureEngineer.FEATURE_COLUMNS]
         .fillna(0)
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0))
    win_probs_arr = win_model.predict(X)
    if place_model is not None:
        place_probs_arr = place_model.predict(X)
        blended = 0.7 * win_probs_arr + 0.3 * place_probs_arr
    else:
        blended = win_probs_arr

    pred_df = feat_df[["horse_id", "horse_name", "horse_number"]].copy()
    pred_df["prob"] = blended
    pred_df = pred_df.sort_values("prob", ascending=False).reset_index(drop=True)

    honmei_prob = float(pred_df.iloc[0]["prob"])
    taikou_prob = float(pred_df.iloc[1]["prob"]) if len(pred_df) > 1 else 0.0
    gap         = honmei_prob - taikou_prob
    is_skip     = (honmei_prob < MIN_HONMEI_PROB or gap < MIN_GAP)

    # 実際の結果（NAR も同じカラムずれ: odds=last_3f時間, last_3f=単勝オッズ）
    actual = race_entries.copy()
    actual["fp"]     = pd.to_numeric(actual["finish_position"], errors="coerce")
    actual["odds_f"] = actual["last_3f"].apply(parse_odds)  # last_3f列 = 実際の単勝オッズ
    actual = actual[actual["fp"].notna()]

    pos_map  = dict(zip(actual["horse_id"], actual["fp"]))
    odds_map = dict(zip(actual["horse_id"], actual["odds_f"]))

    honmei_id   = str(pred_df.iloc[0]["horse_id"])
    honmei_pos  = pos_map.get(honmei_id, float("nan"))
    honmei_odds = odds_map.get(honmei_id, float("nan"))

    ts_hit = int(honmei_pos == 1) if not np.isnan(honmei_pos) else 0
    fk_hit = int(honmei_pos <= 3) if not np.isnan(honmei_pos) else 0
    ts_ret = honmei_odds * 100 if ts_hit and not np.isnan(honmei_odds) else 0.0
    fk_ret = max(1.0, honmei_odds * 0.35) * 100 if fk_hit and not np.isnan(honmei_odds) else 0.0

    row = {
        "race_id":     race_id,
        "race_date":   str(race_date)[:10],
        "race_name":   race_name,
        "venue":       venue_name,
        "course_type": course_type,
        "distance":    distance,
        "n_horses":    len(race_entries),
        "honmei_name": str(pred_df.iloc[0]["horse_name"]),
        "honmei_num":  str(pred_df.iloc[0]["horse_number"]),
        "honmei_prob": round(honmei_prob, 4),
        "taikou_prob": round(taikou_prob, 4),
        "gap":         round(gap, 4),
        "is_skip":     int(is_skip),
        "honmei_pos":  honmei_pos,
        "honmei_odds": honmei_odds,
        "tansho_bet": 0, "tansho_hit": 0, "tansho_ret": 0.0,
        "fukusho_bet": 0, "fukusho_hit": 0, "fukusho_ret": 0.0,
        "baren_bet":  0, "baren_hit":  0, "baren_ret":  0.0, "baren_pts": 0,
        "umatan_bet": 0, "umatan_hit": 0, "umatan_ret": 0.0, "umatan_pts": 0,
    }

    if not is_skip:
        row["tansho_bet"]  = 100
        row["tansho_hit"]  = ts_hit
        row["tansho_ret"]  = ts_ret
        row["fukusho_bet"] = 100
        row["fukusho_hit"] = fk_hit
        row["fukusho_ret"] = fk_ret

        # 市場確率
        all_ids   = pred_df["horse_id"].tolist()
        all_odds  = [parse_odds(odds_map.get(hid, float("nan"))) for hid in all_ids]
        valid_p   = [(hid, o) for hid, o in zip(all_ids, all_odds) if not np.isnan(o) and o > 1.0]
        if valid_p:
            vids, vodds = zip(*valid_p)
            mkt = market_probs(list(vodds))
        else:
            vids, mkt = [], []

        def vidx(hid):
            return list(vids).index(hid) if hid in list(vids) else None

        hi = vidx(honmei_id)
        partner_rows = pred_df[pred_df["horse_id"] != honmei_id].head(EV_POOL)

        # 馬連
        scored = []
        for _, r in partner_rows.iterrows():
            hid  = str(r["horse_id"])
            prob = float(r["prob"])
            o    = parse_odds(odds_map.get(hid, 5.0))
            if np.isnan(o) or o <= 1.0:
                o = 5.0
            scored.append((hid, prob * o, str(r["horse_number"])))
        scored.sort(key=lambda x: -x[1])

        baren_partners = []
        for hid, _, num in scored[:MAX_BAREN]:
            vi = vidx(hid)
            if hi is not None and vi is not None and mkt:
                eo = est_odds(prob_quinella(mkt, hi, vi), "馬連")
                if eo < TORIKAMI_THR:
                    continue
            baren_partners.append((hid, num))

        win1 = [hid for hid, fp in pos_map.items() if fp == 1]
        win2 = [hid for hid, fp in pos_map.items() if fp == 2]
        br_hit, br_ret = 0, 0.0
        if win1 and win2:
            w1, w2 = win1[0], win2[0]
            for phid, _ in baren_partners:
                if {honmei_id, phid} == {w1, w2}:
                    vi_w1, vi_w2 = vidx(w1), vidx(w2)
                    if vi_w1 is not None and vi_w2 is not None and mkt:
                        br_ret = est_odds(prob_quinella(mkt, vi_w1, vi_w2), "馬連") * 100
                    else:
                        br_ret = 300.0
                    br_hit = 1
                    break

        row["baren_bet"] = len(baren_partners) * 100
        row["baren_pts"] = len(baren_partners)
        row["baren_hit"] = br_hit
        row["baren_ret"] = br_ret

        # 馬単
        um_scored = []
        for hid, _, num in scored:
            vi = vidx(hid)
            if hi is not None and vi is not None and mkt:
                p_um = harville(mkt, [hi, vi])
                eo   = est_odds(p_um, "馬単")
                if eo < TORIKAMI_THR:
                    continue
            else:
                eo = 999.0
            um_scored.append((hid, num, eo))
        um_scored.sort(key=lambda x: -x[2])
        um_partners = um_scored[:MAX_UMATAN]
        if um_partners and synth_odds([e for *_, e in um_partners]) < 1.0:
            um_partners = []

        um_hit, um_ret = 0, 0.0
        if win1 and win2:
            w1, w2 = win1[0], win2[0]
            for phid, _, _ in um_partners:
                if honmei_id == w1 and phid == w2:
                    vi_w2 = vidx(w2)
                    if hi is not None and vi_w2 is not None and mkt:
                        um_ret = est_odds(harville(mkt, [hi, vi_w2]), "馬単") * 100
                    else:
                        um_ret = 500.0
                    um_hit = 1
                    break

        row["umatan_bet"] = len(um_partners) * 100
        row["umatan_pts"] = len(um_partners)
        row["umatan_hit"] = um_hit
        row["umatan_ret"] = um_ret

    results.append(row)

    if (i + 1) % 200 == 0:
        buy_  = [r for r in results if not r["is_skip"]]
        bets  = sum(r["tansho_bet"] for r in buy_)
        rets  = sum(r["tansho_ret"] for r in buy_)
        hits  = sum(r["tansho_hit"] for r in buy_)
        roi   = rets / bets * 100 if bets > 0 else 0
        print(f"  {i+1}/{len(target_ids)} 処理済 | 買い:{len(buy_)}R 単勝{hits}hit ROI:{roi:.1f}%")

# ── 保存 & 集計 ───────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nCSV保存: {OUT_CSV} ({len(df)} rows)")

buy = df[df["is_skip"] == 0]

def roi_str(b, r):
    return f"{r/b*100:.1f}%" if b > 0 else "N/A"

date_min = df["race_date"].min()
date_max = df["race_date"].max()

print(f"\n{'='*65}")
print(f"NAR バックテスト結果  [{date_min} 〜 {date_max}]")
print(f"モデル: {model_label}")
print(f"買い条件: prob>={MIN_HONMEI_PROB}  gap>={MIN_GAP}")
print(f"{'='*65}")
print(f"対象レース: {len(df)}R  買い: {len(buy)}R ({len(buy)/len(df)*100:.1f}%)")

print(f"\n{'馬券種':<8} {'的中':>6} {'賭金':>10} {'払戻(推定)':>12} {'ROI':>8}")
print("-" * 54)
for kind, label in [
    ("tansho", "単勝"), ("fukusho", "複勝"),
    ("baren", "馬連"), ("umatan", "馬単"),
]:
    h = int(buy[f"{kind}_hit"].sum())
    b = int(buy[f"{kind}_bet"].sum())
    r = float(buy[f"{kind}_ret"].sum())
    print(f"{label:<8} {h:>6}回  ¥{b:>8,}  ¥{int(r):>10,}  {roi_str(b, r):>8}")

print(f"\n※ 馬連・馬単の払戻は Harville 推定値（NAR 控除率 25% 適用）")

# 会場別単勝ROI
print(f"\n{'='*65}")
print("会場別 単勝ROI（買いレース）")
print(f"{'='*65}")
venue_stats = []
for vname, sub in buy.groupby("venue"):
    b = sub["tansho_bet"].sum()
    r = sub["tansho_ret"].sum()
    h = sub["tansho_hit"].sum()
    venue_stats.append((vname, len(sub), h, b, r))
venue_stats.sort(key=lambda x: -x[4]/max(x[3],1))
for vname, n, h, b, r in venue_stats:
    print(f"  {vname:<6} {n:>5}R  的中:{h:>4}回  ROI:{r/b*100:>7.1f}%")

# コース別
print(f"\n{'='*65}")
print("コース別 単勝ROI")
print(f"{'='*65}")
for ct in ["ダート", "芝"]:
    sub = buy[buy["course_type"] == ct]
    if len(sub) >= 5:
        b = sub["tansho_bet"].sum()
        r = sub["tansho_ret"].sum()
        h = sub["tansho_hit"].sum()
        print(f"  {ct}: {len(sub)}R  ROI={r/b*100:.1f}%  的中={h/len(sub)*100:.1f}%")

# 本命確率帯別
print(f"\n本命確率帯別 単勝ROI")
print(f"{'='*65}")
for lo, hi_p in [(0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 1.0)]:
    sub = buy[(buy["honmei_prob"] >= lo) & (buy["honmei_prob"] < hi_p)]
    if len(sub) >= 3:
        b = sub["tansho_bet"].sum()
        r = sub["tansho_ret"].sum()
        h = sub["tansho_hit"].sum()
        print(f"  prob {lo:.2f}〜{hi_p:.2f}: {len(sub)}R  ROI={r/b*100:.1f}%  的中={h/len(sub)*100:.1f}%")

# 頭数別
print(f"\n出走頭数別 単勝ROI")
print(f"{'='*65}")
for lo, hi_n in [(3, 7), (7, 10), (10, 13), (13, 20)]:
    sub = buy[(buy["n_horses"] >= lo) & (buy["n_horses"] < hi_n)]
    if len(sub) >= 5:
        b = sub["tansho_bet"].sum()
        r = sub["tansho_ret"].sum()
        h = sub["tansho_hit"].sum()
        print(f"  {lo}〜{hi_n-1}頭: {len(sub)}R  ROI={r/b*100:.1f}%  的中={h/len(sub)*100:.1f}%")

print()
