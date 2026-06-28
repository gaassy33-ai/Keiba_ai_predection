"""
backtest_2024_lookback.py
=========================
2024年レースのバックテスト（lookbackモード・データリークなし）。
daily_batch.py と同一ロジック（モデル確率基準の買い目選択）を使用。

実行（約2〜3時間）:
    .venv/bin/python backtest_2024_lookback.py
"""
from __future__ import annotations

import sys, time
from itertools import combinations as _comb, permutations as _perm
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.engineer import FeatureEngineer
from src.model.trainer import ModelTrainer
from config.settings import settings

# ── 定数（daily_batch.py と完全一致） ────────────────────────
MIN_HONMEI_PROB, MARK_STRONG_PROB, MIN_CONFIDENCE_GAP = 0.30, 0.35, 0.05
EV_THRESHOLD = 1.05
ENTRIES_ADJ, ENTRIES_BASE, ENTRIES_CAP = 0.003, 10, 0.05
MAIDEN_KEYWORDS = ("新馬", "未勝利")

_SEASON_PROB_THRESHOLD = {1:0.38,2:0.38,3:0.33,4:0.30,5:0.30,6:0.30,
                           7:0.38,8:0.38,9:0.38,10:0.33,11:0.33,12:0.38}
_SUMMER_DIRT_SKIP_MONTHS = {7,8,9}
_MARK_O_SKIP_MONTHS      = {1,2,10}
_HIGH_VALUE_ODDS_MIN, _HIGH_VALUE_ODDS_MAX = 8.0, 15.0
_HIGH_VALUE_EV_THRESHOLD = 0.95
_BAD_SEASON_MONTHS       = {1,7,8,9,11,12}
_BAD_VENUE_SKIP_CODES    = {"02","04","09"}
_BAD_SEASON_MAX_DISTANCE = 1800
_BAD_MODEL_PROB_THRESHOLD, _BAD_MODEL_MARK_STRONG = 0.25, 0.28

EV_PARTNER_TOP_N = 7
MAX_BAREN_TICKETS, MAX_SANRENFUKU_TICKETS, MAX_SANRENTAN_TICKETS = 3, 7, 7
JRA_TAKE = {"馬連":0.225,"馬単":0.25,"3連複":0.225,"3連単":0.275}

Path("logs").mkdir(exist_ok=True)
logger.remove()
_fmt = "{time:HH:mm:ss} | {level:<7} | {message}"
logger.add(sys.stdout, level="INFO", format=_fmt, colorize=True)
logger.add("logs/backtest_2024_lookback.log", level="DEBUG", format=_fmt, rotation="20 MB")

def _parse_odds(v):
    try: return float(str(v).replace(",","").strip())
    except: return float("nan")

def _market_probs(ol):
    raw = np.array([1./max(o,1.01) for o in ol]); s=raw.sum()
    return (raw/s).tolist() if s>0 else raw.tolist()

def _harville(p,order):
    r,rem=1.,1.
    for idx in order:
        if rem<1e-9: return 0.
        r*=p[idx]/rem; rem-=p[idx]
    return r

def _prob_quinella(p,i,j): return _harville(p,[i,j])+_harville(p,[j,i])
def _prob_trio(p,i,j,k): return sum(_harville(p,list(o)) for o in _perm([i,j,k]))
def _prob_sanrentan(p,i,j,k): return _harville(p,[i,j,k])
def _synth_odds(ol): d=sum(1./max(o,1e-9) for o in ol); return 1./d if d>0 else 0.
def _est_odds(prob,bet="3連複"): return (1.-JRA_TAKE.get(bet,.225))/max(prob,.001)

def is_buy(h1p,gap,month,maiden,n,ev,odds,ct,jyo,dist,bad):
    if maiden: return False
    thr = _BAD_MODEL_PROB_THRESHOLD if bad else \
          _SEASON_PROB_THRESHOLD.get(month,MIN_HONMEI_PROB)+min(max(0,n-ENTRIES_BASE)*ENTRIES_ADJ,ENTRIES_CAP)
    if gap<MIN_CONFIDENCE_GAP or h1p<thr: return False
    ev_ok = True if np.isnan(ev) else \
            (ev>=_HIGH_VALUE_EV_THRESHOLD if _HIGH_VALUE_ODDS_MIN<=odds<=_HIGH_VALUE_ODDS_MAX else ev>=EV_THRESHOLD)
    if not ev_ok: return False
    if month in _SUMMER_DIRT_SKIP_MONTHS and ct=="ダート": return False
    if month in _BAD_SEASON_MONTHS and jyo in _BAD_VENUE_SKIP_CODES: return False
    if month in _BAD_SEASON_MONTHS and dist>=_BAD_SEASON_MAX_DISTANCE: return False
    return True

def main():
    t0 = time.time()
    logger.info("="*65)
    logger.info("バックテスト開始: 2024年（lookbackモード・モデル確率基準）")
    logger.info("="*65)

    all_res = pd.read_csv(ROOT/"data/raw/train_results.csv", dtype=str)
    meta    = pd.read_csv(ROOT/"data/raw/train_meta.csv",    dtype=str)
    meta["race_date"] = pd.to_datetime(
        meta["race_id"].str[:8].apply(lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s)>=8 else None),
        errors="coerce")
    meta = meta.dropna(subset=["race_date"]).sort_values("race_date")
    meta24 = meta[meta["race_date"].dt.year==2024]

    all_res["race_date"] = pd.to_datetime(
        all_res["race_id"].str[:8].apply(lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s)>=8 else None),
        errors="coerce")

    logger.info(f"対象: {len(meta24):,}R  ({meta24['race_date'].min().date()} 〜 {meta24['race_date'].max().date()})")
    logger.info(f"全履歴: {len(all_res):,}行")

    trainer = ModelTrainer.load(settings.model_path)
    bad_trainer = None
    bad_path = ROOT/"data/models/lgbm_model_bad_season.pkl"
    if bad_path.exists():
        bad_trainer = ModelTrainer.load(bad_path)
        logger.info("  不調期専用モデル読み込み完了")

    records, buy_count = [], 0
    # ── 日付ごとにFeatureEngineerを1回だけ構築（約24倍高速化） ──
    unique_dates = sorted(meta24["race_date"].unique())
    logger.info(f"ユニーク日付数: {len(unique_dates)}日")

    total_races = 0
    for date_idx, rd in enumerate(unique_dates):
        date_races = meta24[meta24["race_date"]==rd]
        hist_before = all_res[all_res["race_date"] < rd]

        try:
            fe = FeatureEngineer(hist_before)
        except Exception as e:
            logger.warning(f"skip date {rd}: {e}"); continue

        for _, row in date_races.iterrows():
            race_id = str(row["race_id"])
            total_races += 1
            entries = all_res[all_res["race_id"]==race_id].copy()
            if len(entries)<4: continue

            ct      = str(row.get("course_type",""))
            rname   = str(row.get("race_name",""))
            dist    = int(row.get("distance",0) or 0)
            gc      = int(row.get("ground_condition_code",-1) or -1)
            wx      = int(row.get("weather_code",-1) or -1)
            jyo     = str(race_id)[4:6]
            month   = rd.month
            if not ct or dist==0: continue

            edf = entries[["horse_id","horse_name","horse_number","frame_number",
                            "jockey_id","sex_age","weight_carried"]].copy()
            edf["sex"] = edf["sex_age"].str[0]
            edf["age"] = pd.to_numeric(edf["sex_age"].str[1:], errors="coerce")
            edf["weight_carried"] = pd.to_numeric(edf["weight_carried"], errors="coerce")
            edf["father"] = edf["mother_father"] = ""
            if "odds" in entries.columns:
                edf["odds"] = pd.to_numeric(entries["odds"].values, errors="coerce")

            rcc = FeatureEngineer._race_name_to_class_code(rname)
            try: vc = int(race_id[4:6])
            except: vc = -1

            try:
                feat = fe.build_entry_features(entry_df=edf, course_type=ct, distance=dist,
                    ground_condition_code=gc, weather_code=wx, race_class_code=rcc, venue_code=vc)
            except Exception as e:
                logger.debug(f"skip {race_id}: {e}"); continue

            bad = (bad_trainer is not None and month in _BAD_SEASON_MONTHS)
            act = bad_trainer if bad else trainer
            X = feat[FeatureEngineer.FEATURE_COLUMNS].apply(pd.to_numeric,errors="coerce").fillna(0)
            wp = act.model.predict(X, num_threads=1)
            pp = act.place_model.predict(X, num_threads=1) if act.place_model else None
            probs = 0.7*wp+0.3*pp if pp is not None else wp

            pred = feat[["horse_id","horse_name","horse_number"]].copy()
            pred["win_prob"] = probs
            pred = pred.sort_values("win_prob",ascending=False).reset_index(drop=True)
            if len(pred)<3: continue

            h1p=float(pred.iloc[0]["win_prob"]); h2p=float(pred.iloc[1]["win_prob"])
            gap=h1p-h2p; n_ent=len(entries)
            maiden=any(k in rname for k in MAIDEN_KEYWORDS)
            om={}
            if "odds" in feat.columns: om=feat.set_index("horse_id")["odds"].dropna().to_dict()
            h1id=str(pred.iloc[0]["horse_id"])
            h1od=float(om.get(h1id,float("nan")))
            h1ev=h1p*h1od if not np.isnan(h1od) else float("nan")

            buy = is_buy(h1p,gap,month,maiden,n_ent,h1ev,h1od,ct,jyo,dist,bad)
            ms = _BAD_MODEL_MARK_STRONG if bad else MARK_STRONG_PROB
            mark = "◎" if buy and h1p>=ms else ("○" if buy else "△")
            if mark=="○" and month in _MARK_O_SKIP_MONTHS: mark="△"; buy=False
            if not buy: continue
            buy_count+=1

            act2 = entries[["horse_id","horse_number","finish_position","odds"]].copy()
            act2["pos"]=pd.to_numeric(act2["finish_position"].str.extract(r"(\d+)")[0],errors="coerce")
            act2["of"]=act2["odds"].apply(_parse_odds)
            act2=act2.dropna(subset=["pos"]).sort_values("pos")
            if len(act2)<3: continue

            t3=act2.iloc[:3]
            p1id=str(t3.iloc[0]["horse_id"]); p2id=str(t3.iloc[1]["horse_id"]); p3id=str(t3.iloc[2]["horse_id"])
            p1n=str(int(float(t3.iloc[0]["horse_number"]))); p2n=str(int(float(t3.iloc[1]["horse_number"]))); p3n=str(int(float(t3.iloc[2]["horse_number"])))
            wod=float(t3.iloc[0]["of"])
            hnid=str(pred.iloc[0]["horse_id"]); hnum=str(int(float(pred.iloc[0]["horse_number"])))

            vids=pred["horse_id"].astype(str).tolist()
            ol=[float(om.get(hid,float("nan"))) for hid in vids]
            ol=[o if not np.isnan(o) and o>1. else 5. for o in ol]
            mkp=_market_probs(ol)
            pr_raw=pred["win_prob"].tolist(); ps=sum(pr_raw)
            mop=[p/ps for p in pr_raw] if ps>0 else pr_raw
            hi=vids.index(hnid) if hnid in vids else 0

            prows=pred[pred["horse_id"]!=hnid].head(EV_PARTNER_TOP_N)
            pool=[]
            for _,r2 in prows.iterrows():
                hid=str(r2["horse_id"]); num=str(int(float(r2["horse_number"])))
                vi=vids.index(hid) if hid in vids else None
                pool.append((num,vi))
            pidx={num:k+1 for k,(num,_) in enumerate(pool)}

            # 単勝
            th=int(hnid==p1id); tr_=wod*100 if th else 0.

            # 馬連
            bpool=[num for num,_ in pool[:3]]
            bc=[(hnum,p) for p in bpool]
            abr={p1n,p2n}
            bh=int(any(set([h,p])==abr for h,p in bc))
            br=0.
            if bh:
                vi1=vids.index(p1id) if p1id in vids else None
                vi2=vids.index(p2id) if p2id in vids else None
                if vi1 is not None and vi2 is not None: br=_est_odds(_prob_quinella(mkp,vi1,vi2),"馬連")*100
            bcost=len(bc)*100

            # 3連複
            sfa=[]
            for (na,via),(nb,vib) in _comb(pool,2):
                pa=pidx.get(na,1); pb=pidx.get(nb,2)
                mp=_prob_trio(mop,0,pa,pb)
                eo=(_est_odds(_prob_trio(mkp,hi,via,vib),"3連複") if hi is not None and via is not None and vib is not None
                    else _est_odds(mp,"3連複"))
                sfa.append((na,nb,mp,eo))
            sfa.sort(key=lambda x:-x[2])
            ss=sfa[:MAX_SANRENFUKU_TICKETS]
            sfest=[od for _,_,_,od in ss]
            sfc=([(a,b) for a,b,_,_ in ss] if (not sfest or _synth_odds(sfest)>=1.) else [])
            at3={p1n,p2n,p3n}
            sfh=int(any(set([hnum,a,b])==at3 for a,b in sfc))
            sfr=0.
            if sfh:
                v1=vids.index(p1id) if p1id in vids else None
                v2=vids.index(p2id) if p2id in vids else None
                v3=vids.index(p3id) if p3id in vids else None
                if v1 is not None and v2 is not None and v3 is not None: sfr=_est_odds(_prob_trio(mkp,v1,v2,v3),"3連複")*100
            sfcost=len(sfc)*100

            # 3連単
            sta=[]
            for (n2,v2_),(n3,v3_) in _perm(pool,2):
                p2_=pidx.get(n2,1); p3_=pidx.get(n3,2)
                mp=_prob_sanrentan(mop,0,p2_,p3_)
                eo=(_est_odds(_prob_sanrentan(mkp,hi,v2_,v3_),"3連単") if hi is not None and v2_ is not None and v3_ is not None
                    else _est_odds(mp,"3連単"))
                sta.append((n2,n3,mp,eo))
            sta.sort(key=lambda x:-x[2])
            sts=sta[:MAX_SANRENTAN_TICKETS]
            stest=[od for _,_,_,od in sts]
            stc=([(n2,n3) for n2,n3,_,_ in sts] if (not stest or _synth_odds(stest)>=1.) else [])
            sth=int(hnum==p1n and any(n2==p2n and n3==p3n for n2,n3 in stc))
            str_=0.
            if sth:
                v2_=vids.index(p2id) if p2id in vids else None
                v3_=vids.index(p3id) if p3id in vids else None
                if v2_ is not None and v3_ is not None: str_=_est_odds(_prob_sanrentan(mkp,hi,v2_,v3_),"3連単")*100
            stcost=len(stc)*100

            tc=100+bcost+sfcost+stcost; tret=tr_+br+sfr+str_
            records.append({"race_id":race_id,"race_date":str(rd)[:10],"race_name":rname,
                "race_month":month,"course_type":ct,"distance":dist,"mark":mark,"n_entries":n_ent,
                "h1_prob":round(h1p,4),"h2_prob":round(h2p,4),"gap":round(gap,4),
                "h1_odds":round(h1od,1) if not np.isnan(h1od) else None,
                "honmei_num":hnum,"honmei_name":pred.iloc[0]["horse_name"],
                "actual_1st":p1n,"actual_2nd":p2n,"actual_3rd":p3n,
                "tansho_hit":th,"tansho_ret":round(tr_,1),
                "baren_hit":bh,"baren_cost":bcost,"baren_ret":round(br,1),
                "sf_hit":sfh,"sf_cost":sfcost,"sf_ret":round(sfr,1),
                "st_hit":sth,"st_cost":stcost,"st_ret":round(str_,1),
                "total_cost":tc,"total_ret":round(tret,1)})

        if (date_idx+1)%10==0:
            e=(time.time()-t0)/60
            logger.info(f"  {date_idx+1}/{len(unique_dates)} 日付処理済み  買い{buy_count}R  ({e:.1f}分)")

    df=pd.DataFrame(records)
    logger.info(f"完了: 買い{len(df):,}R  ({(time.time()-t0)/60:.1f}分)")
    return df


def print_summary(df, title):
    b="="*68
    print(f"\n{b}\n  {title}")
    print(f"  期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")
    print(f"  買い対象: {len(df):,} R\n{b}")
    def roi(c,r): return r/c*100 if c>0 else 0.
    tc=df["total_cost"].sum(); tr=df["total_ret"].sum()
    print(f"\n  【総合】 投票:{int(tc):,}円  回収:{int(tr):,}円  ROI:{roi(tc,tr):.1f}%")
    print(f"\n  【馬券種別 ROI】")
    for bet,cc,rc,hc in [("単勝",None,"tansho_ret","tansho_hit"),("馬連","baren_cost","baren_ret","baren_hit"),
                          ("3連複","sf_cost","sf_ret","sf_hit"),("3連単","st_cost","st_ret","st_hit")]:
        c=len(df)*100 if cc is None else df[cc].sum(); r=df[rc].sum(); h=df[hc].sum()
        print(f"    {bet:<5}: 投票{int(c):>8,}円  回収{int(r):>9,}円  ROI{roi(c,r):>6.1f}%  的中{int(h):>3}/{len(df)}R ({h/len(df):.1%})")
    print(f"\n  【月別 ROI】")
    print(f"  {'月':>3}  {'R数':>4}  {'単勝':>7}  {'馬連':>7}  {'3連複':>7}  {'3連単':>7}  {'総合':>7}")
    for m,g in df.groupby("race_month"):
        print(f"  {m:>3}月  {len(g):>4}R  {roi(len(g)*100,g['tansho_ret'].sum()):>6.0f}%  "
              f"{roi(g['baren_cost'].sum(),g['baren_ret'].sum()):>6.0f}%  "
              f"{roi(g['sf_cost'].sum(),g['sf_ret'].sum()):>6.0f}%  "
              f"{roi(g['st_cost'].sum(),g['st_ret'].sum()):>6.0f}%  "
              f"{roi(g['total_cost'].sum(),g['total_ret'].sum()):>6.0f}%")
    print(f"\n  【芝/ダート別】")
    for ct,g in df.groupby("course_type"):
        print(f"    {ct}: {len(g)}R  ROI {roi(g['total_cost'].sum(),g['total_ret'].sum()):.1f}%  3連複的中{int(g['sf_hit'].sum())}R")
    print(f"\n  【確率差(gap)別 ROI】")
    df["gap_bin"]=pd.cut(df["gap"],bins=[0,.05,.10,.15,.25,1.],labels=["0-5%","5-10%","10-15%","15-25%","25%+"])
    for gb,g in df.groupby("gap_bin",observed=True):
        if len(g)==0: continue
        print(f"    {str(gb):>7}  {len(g):>4}R  3連複:{roi(g['sf_cost'].sum(),g['sf_ret'].sum()):>6.1f}%  "
              f"3連単:{roi(g['st_cost'].sum(),g['st_ret'].sum()):>6.1f}%  総合:{roi(g['total_cost'].sum(),g['total_ret'].sum()):>6.1f}%")
    for label,hcol,rcol in [("3連複","sf_hit","sf_ret"),("3連単","st_hit","st_ret")]:
        hits=df[df[hcol]==1].sort_values(rcol,ascending=False)
        if len(hits)>0:
            print(f"\n  【{label} 的中上位10件】")
            for _,r in hits.head(10).iterrows():
                print(f"    {r['race_date']} {r['race_name']}  ◎{r['honmei_num']}番{r['honmei_name']}  推定{r[rcol]:.0f}円")
    print(f"\n{b}\n")


if __name__=="__main__":
    df = main()
    if len(df)==0: print("買い対象なし"); sys.exit(0)
    print_summary(df, "バックテスト結果: 2024年（lookbackモード・モデル確率基準）")
    out=ROOT/"data/processed/backtest_2024_lookback.csv"
    df.to_csv(out,index=False)
    logger.info(f"詳細CSV保存: {out}")
