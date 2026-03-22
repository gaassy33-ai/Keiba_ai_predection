"""
バックテスト (2026-01-04 〜 2026-03-15)
- 学習データ: 2024年 + 2025年 → lgbm_model.pkl
- lgbm_model.pkl の blended model (0.7*win + 0.3*place) で予測
- 買い条件: honmei_prob >= 0.15 かつ 信頼度差 >= 0.05
- 馬券: 単勝・複勝・馬連・3連複・3連単 (Harville推定オッズ)
- 馬連/3連複/3連単 の払戻は実際オッズを Harville 式で推定

実行: python3 backtest_1year.py
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")
from itertools import permutations, combinations
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer

# ── 設定 ──────────────────────────────────────────────────────
DATE_FROM        = "2026-01-01"
MIN_HONMEI_PROB  = 0.15
MIN_GAP          = 0.05
EV_POOL          = 5
MAX_BAREN        = 3
MAX_UMATAN       = 3
MAX_SF           = 5
MAX_ST           = 5
TORIKAMI_THR     = 1.05      # 合成オッズがこれ未満はケン
JRA_TAKE         = {"馬連": 0.225, "馬単": 0.25, "3連複": 0.225, "3連単": 0.275}
OUT_CSV          = ROOT / "data/processed/backtest_2026.csv"

# ── モデルロード ───────────────────────────────────────────────
obj = joblib.load(ROOT / "data/models/lgbm_model.pkl")
win_model   = obj["model"]
place_model = obj.get("place_model")
print(f"model loaded: win={type(win_model).__name__}, place={type(place_model).__name__ if place_model else 'None'}")

# ── データ読み込み ─────────────────────────────────────────────
train_res = pd.read_csv(ROOT / "data/raw/train_results.csv", dtype=str)
test_res  = pd.read_csv(ROOT / "data/raw/test_results.csv",  dtype=str)
test_meta = pd.read_csv(ROOT / "data/raw/test_meta.csv",     dtype=str)

# 全履歴 (特徴量ルックバック用)
all_history = pd.concat([train_res, test_res], ignore_index=True)
print(f"全履歴: {len(all_history):,} rows")

# 対象レース (past 1 year)
test_meta["race_date"] = pd.to_datetime(test_meta["race_date"], errors="coerce")
target_meta = test_meta[test_meta["race_date"] >= DATE_FROM].sort_values("race_date")
print(f"対象レース: {len(target_meta)} races ({DATE_FROM} 〜)")

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

def prob_trio(probs, i, j, k):
    return sum(harville(probs, list(o)) for o in permutations([i, j, k]))

def prob_sanrentan(probs, i, j, k):
    return harville(probs, [i, j, k])

def est_odds(prob, bet_type):
    take = JRA_TAKE.get(bet_type, 0.225)
    return (1.0 - take) / max(prob, 0.001)

def synth_odds(elist):
    d = sum(1.0 / max(e, 1e-9) for e in elist)
    return 1.0 / d if d > 0 else 0.0

# ── FeatureEngineer 初期化 (全履歴) ───────────────────────────
print("FeatureEngineer 初期化中...")
fe = FeatureEngineer(all_history)
fe.precompute_aggregations()
print("完了")

# ── レースごとにバックテスト ───────────────────────────────────
results = []
target_ids = target_meta["race_id"].tolist()

for i, race_id in enumerate(target_ids):
    meta_row   = target_meta[target_meta["race_id"] == race_id].iloc[0]
    race_date  = meta_row["race_date"]
    course_type = str(meta_row.get("course_type", "") or "")
    distance    = int(meta_row.get("distance", 0) or 0)
    gc_code     = int(meta_row.get("ground_condition_code", -1) or -1)
    wx_code     = int(meta_row.get("weather_code", -1) or -1)
    race_name   = str(meta_row.get("race_name", "") or "")

    if not course_type or distance == 0:
        continue

    race_entries = test_res[test_res["race_id"] == race_id].copy()
    if len(race_entries) < 3:
        continue

    # entry_df 組み立て
    cols = ["horse_id", "horse_name", "horse_number", "frame_number",
            "sex_age", "weight_carried", "jockey_id"]
    if "trainer_name" in race_entries.columns:
        cols.append("trainer_name")
    entry_df = race_entries[cols].copy()
    entry_df["sex"] = entry_df["sex_age"].str[0]
    entry_df["age"] = pd.to_numeric(entry_df["sex_age"].str[1:], errors="coerce")
    entry_df["weight_carried"] = pd.to_numeric(entry_df["weight_carried"], errors="coerce")
    entry_df["father"]        = ""
    entry_df["mother_father"] = ""
    if "trainer_name" not in entry_df.columns:
        entry_df["trainer_name"] = ""

    # 特徴量生成
    try:
        feat_df = fe.build_entry_features(
            entry_df=entry_df,
            course_type=course_type,
            distance=distance,
            ground_condition_code=gc_code,
            weather_code=wx_code,
        )
    except Exception as e:
        continue

    X = feat_df[FeatureEngineer.FEATURE_COLUMNS].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    win_probs_arr  = win_model.predict(X)
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
    gap = honmei_prob - taikou_prob

    is_skip = honmei_prob < MIN_HONMEI_PROB or gap < MIN_GAP

    # 実際の結果
    actual = race_entries.copy()
    actual["fp"]   = pd.to_numeric(actual["finish_position"], errors="coerce")
    # NOTE: test_results.csv のカラムずれ:
    #   "odds" 列 = 実際には last_3f タイム
    #   "last_3f" 列 = 実際には単勝オッズ
    actual["odds_f"] = actual["last_3f"].apply(parse_odds)
    actual = actual[actual["fp"].notna()]

    # 着順辞書
    pos_map   = dict(zip(actual["horse_id"], actual["fp"]))
    odds_map  = dict(zip(actual["horse_id"], actual["odds_f"]))
    num_map   = dict(zip(actual["horse_id"], actual["horse_number"].astype(str)))

    honmei_id  = str(pred_df.iloc[0]["horse_id"])
    honmei_pos = pos_map.get(honmei_id, float("nan"))
    honmei_odds_v = odds_map.get(honmei_id, float("nan"))

    # 単勝・複勝
    ts_hit = int(honmei_pos == 1) if not np.isnan(honmei_pos) else 0
    fk_hit = int(honmei_pos <= 3) if not np.isnan(honmei_pos) else 0
    ts_ret = honmei_odds_v * 100 if ts_hit and not np.isnan(honmei_odds_v) else 0.0
    # 複勝払戻: 単勝オッズから近似 (複勝=単勝の約1/3, 最低1.0倍 → ¥100)
    # JRA複勝は別払戻なので正確ではないが近似値として使用
    if fk_hit and not np.isnan(honmei_odds_v):
        fk_approx = max(1.0, honmei_odds_v * 0.35)
        fk_ret = fk_approx * 100
    else:
        fk_ret = 0.0

    row = {
        "race_id": race_id,
        "race_date": str(race_date)[:10],
        "race_name": race_name,
        "course_type": course_type,
        "distance": distance,
        "honmei_name": str(pred_df.iloc[0]["horse_name"]),
        "honmei_num": str(pred_df.iloc[0]["horse_number"]),
        "honmei_prob": round(honmei_prob, 4),
        "taikou_prob": round(taikou_prob, 4),
        "gap": round(gap, 4),
        "is_skip": int(is_skip),
        "honmei_pos": honmei_pos,
        "honmei_odds": honmei_odds_v,
        "tansho_bet": 0, "tansho_hit": 0, "tansho_ret": 0.0,
        "fukusho_bet": 0, "fukusho_hit": 0, "fukusho_ret": 0.0,
        "baren_bet": 0,  "baren_hit": 0,  "baren_ret": 0.0,
        "baren_pts": 0,
        "umatan_bet": 0, "umatan_hit": 0, "umatan_ret": 0.0,
        "umatan_pts": 0,
        "sanrenfuku_bet": 0, "sanrenfuku_hit": 0, "sanrenfuku_ret": 0.0,
        "sanrenfuku_pts": 0,
        "sanrentan_bet": 0,  "sanrentan_hit": 0,  "sanrentan_ret": 0.0,
        "sanrentan_pts": 0,
    }

    if not is_skip:
        row["tansho_bet"] = 100
        row["tansho_hit"] = ts_hit
        row["tansho_ret"] = ts_ret
        row["fukusho_bet"] = 100
        row["fukusho_hit"] = fk_hit
        row["fukusho_ret"] = fk_ret

        # オッズマップ (市場確率計算用)
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

        # 相手プール (◎除く上位EV_POOL頭)
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

        # 馬連的中チェック
        win1 = [hid for hid, fp in pos_map.items() if fp == 1]
        win2 = [hid for hid, fp in pos_map.items() if fp == 2]
        br_hit = 0
        br_ret = 0.0
        if win1 and win2:
            w1, w2 = win1[0], win2[0]
            for phid, _ in baren_partners:
                if {honmei_id, phid} == {w1, w2}:
                    # 払戻推定: Harville
                    vi_w1 = vidx(w1)
                    vi_w2 = vidx(w2)
                    if vi_w1 is not None and vi_w2 is not None and mkt:
                        p_q = prob_quinella(mkt, vi_w1, vi_w2)
                        br_ret = est_odds(p_q, "馬連") * 100
                    else:
                        br_ret = 300.0  # デフォルト推定
                    br_hit = 1
                    break

        row["baren_bet"] = len(baren_partners) * 100
        row["baren_pts"] = len(baren_partners)
        row["baren_hit"] = br_hit
        row["baren_ret"] = br_ret

        # 馬単 (◎1着固定 × 相手 Harville降順 最大MAX_UMATAN点)
        # 馬単確率 = harville([hi, vi]) = P(◎1着) × P(相手2着|◎1着)
        um_scored = []
        for hid, _, num in scored:  # scored は馬連と同じ EV 順
            vi = vidx(hid)
            if hi is not None and vi is not None and mkt:
                p_um = harville(mkt, [hi, vi])
                eo = est_odds(p_um, "馬単")
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
                        p_um = harville(mkt, [hi, vi_w2])
                        um_ret = est_odds(p_um, "馬単") * 100
                    else:
                        um_ret = 500.0
                    um_hit = 1
                    break

        row["umatan_bet"] = len(um_partners) * 100
        row["umatan_pts"] = len(um_partners)
        row["umatan_hit"] = um_hit
        row["umatan_ret"] = um_ret

        # 3連複プール
        pool = []
        for _, r in partner_rows.iterrows():
            vi = vidx(str(r["horse_id"]))
            if vi is not None:
                pool.append((str(r["horse_id"]), str(int(r["horse_number"])), vi))

        sf_combos = []
        for (hid_a, _, vi_a), (hid_b, _, vi_b) in combinations(pool, 2):
            if hi is not None and mkt:
                p = prob_trio(mkt, hi, vi_a, vi_b)
                eo = est_odds(p, "3連複")
                if eo < TORIKAMI_THR:
                    continue
            else:
                eo = 999.0
            sf_combos.append((hid_a, hid_b, eo))
        sf_combos.sort(key=lambda x: -x[2])
        sf_combos = sf_combos[:MAX_SF]
        if sf_combos and synth_odds([e for *_, e in sf_combos]) < 1.0:
            sf_combos = []

        # 3連複的中チェック
        top3_ids = {hid for hid, fp in pos_map.items() if fp in (1, 2, 3)}
        sf_hit, sf_ret = 0, 0.0
        for hid_a, hid_b, _ in sf_combos:
            if {honmei_id, hid_a, hid_b} == top3_ids:
                if hi is not None and mkt:
                    vi_tops = [vidx(h) for h in top3_ids if vidx(h) is not None]
                    if len(vi_tops) == 3:
                        p_t = prob_trio(mkt, vi_tops[0], vi_tops[1], vi_tops[2])
                        sf_ret = est_odds(p_t, "3連複") * 100
                    else:
                        sf_ret = 500.0
                else:
                    sf_ret = 500.0
                sf_hit = 1
                break

        row["sanrenfuku_bet"] = len(sf_combos) * 100
        row["sanrenfuku_pts"] = len(sf_combos)
        row["sanrenfuku_hit"] = sf_hit
        row["sanrenfuku_ret"] = sf_ret

        # 3連単
        st_combos = []
        for (hid_a, _, vi_a), (hid_b, _, vi_b) in permutations(pool, 2):
            if hi is not None and mkt:
                p = prob_sanrentan(mkt, hi, vi_a, vi_b)
                eo = est_odds(p, "3連単")
                if eo < TORIKAMI_THR:
                    continue
            else:
                eo = 999.0
            st_combos.append((hid_a, hid_b, eo))
        st_combos.sort(key=lambda x: -x[2])
        st_combos = st_combos[:MAX_ST]
        if st_combos and synth_odds([e for *_, e in st_combos]) < 1.0:
            st_combos = []

        # 3連単的中チェック
        win3 = [hid for hid, fp in pos_map.items() if fp == 3]
        st_hit, st_ret = 0, 0.0
        if win1 and win2 and win3:
            w1, w2, w3 = win1[0], win2[0], win3[0]
            for hid_a, hid_b, _ in st_combos:
                if [honmei_id, hid_a, hid_b] == [w1, w2, w3]:
                    vi_st = [vidx(w1), vidx(w2), vidx(w3)]
                    if all(v is not None for v in vi_st) and mkt:
                        p_st = prob_sanrentan(mkt, vi_st[0], vi_st[1], vi_st[2])
                        st_ret = est_odds(p_st, "3連単") * 100
                    else:
                        st_ret = 1000.0
                    st_hit = 1
                    break

        row["sanrentan_bet"] = len(st_combos) * 100
        row["sanrentan_pts"] = len(st_combos)
        row["sanrentan_hit"] = st_hit
        row["sanrentan_ret"] = st_ret

    results.append(row)

    if (i + 1) % 200 == 0:
        done = len(results)
        buy  = sum(1 for r in results if not r["is_skip"])
        hits = sum(r["tansho_hit"] for r in results)
        bets = sum(r["tansho_bet"] for r in results)
        rets = sum(r["tansho_ret"] for r in results)
        roi  = rets / bets * 100 if bets > 0 else 0
        print(f"  {i+1}/{len(target_ids)} 処理済 | 買い:{buy}R 単勝{hits}hit ROI:{roi:.1f}%")

# ── 結果保存 ──────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nCSV保存: {OUT_CSV} ({len(df)} rows)")

# ── サマリー ──────────────────────────────────────────────────
buy = df[df["is_skip"] == 0]
print(f"\n{'='*60}")
print(f"バックテスト結果 ({DATE_FROM} 〜 2026-03-01)")
print(f"{'='*60}")
print(f"対象レース: {len(df)}R  買い: {len(buy)}R ({len(buy)/len(df)*100:.1f}%)")

def roi_str(bet, ret):
    return f"{ret/bet*100:.1f}%" if bet > 0 else "N/A"

print(f"\n{'馬券種':<8} {'ヒット':>6} {'賭金':>10} {'払戻(推定)':>12} {'ROI':>8}")
print("-" * 52)
for kind, label in [
    ("tansho", "単勝"), ("fukusho", "複勝"),
    ("baren", "馬連"), ("umatan", "馬単"),
    ("sanrenfuku", "3連複"), ("sanrentan", "3連単"),
]:
    h = int(buy[f"{kind}_hit"].sum())
    b = int(buy[f"{kind}_bet"].sum())
    r = float(buy[f"{kind}_ret"].sum())
    print(f"{label:<8} {h:>6}回  ¥{b:>8,}  ¥{int(r):>10,}  {roi_str(b, r):>8}")

print(f"\n※ 馬連・3連複・3連単の払戻は Harville 推定値（実際の払戻と異なる場合あり）")
print(f"※ 複勝払戻は単勝オッズ×0.35の近似値")

# コース別単勝ROI
print(f"\n{'='*60}")
print("コース別 単勝ROI（買いレース）")
print(f"{'='*60}")
for ct in ["ダート", "芝"]:
    sub = buy[buy["course_type"] == ct]
    if len(sub) >= 5:
        b = sub["tansho_bet"].sum()
        r = sub["tansho_ret"].sum()
        h = sub["tansho_hit"].sum()
        print(f"  {ct}: {len(sub)}R  単勝ROI={r/b*100:.1f}%  的中率={h/len(sub)*100:.1f}%")

# honmei_prob別
print(f"\n本命確率帯別 単勝ROI（全コース）")
print(f"{'='*60}")
for lo, hi_p in [(0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 1.0)]:
    sub = buy[(buy["honmei_prob"] >= lo) & (buy["honmei_prob"] < hi_p)]
    if len(sub) >= 3:
        b = sub["tansho_bet"].sum()
        r = sub["tansho_ret"].sum()
        h = sub["tansho_hit"].sum()
        print(f"  prob {lo:.2f}〜{hi_p:.2f}: {len(sub)}R  単勝ROI={r/b*100:.1f}%  的中率={h/len(sub)*100:.1f}%")
