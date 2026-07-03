"""
beta_sweep_shortleg.py
 (4) 숏레그 가중 λ 스윕 → 중립화 완화 시 수익/변동성/Sharpe/MDD/순베타 변화.
     port(λ) = 롱레그 + λ·숏레그.  λ=1 현행(≈중립), λ=0 롱온리(순베타 최대).
 (5) 숏레그 부진 원인: 종목별 숏 누적손익 + 베타구간별 숏손익.
"""
import os, numpy as np, pandas as pd
OUT=os.path.dirname(os.path.abspath(__file__))
df=pd.read_parquet(os.path.join(OUT,"oos_signals.parquet"))
uni=[s for s in df.symbol.unique() if s!="SPY"]; N=len(uni)
# 종목 베타 (일별 vs SPY)
piv=df.pivot_table(index=df.datetime.dt.date,columns="symbol",values="close",aggfunc="last")
dret=np.log(piv).diff().dropna(how="all"); spy=dret["SPY"]
beta={s:(np.cov(*[x.values for x in [pd.concat([dret[s],spy],axis=1).dropna().iloc[:,0],
        pd.concat([dret[s],spy],axis=1).dropna().iloc[:,1]]])[0,1]/spy.var()) for s in uni}
beta=pd.Series(beta)
d=df[df.symbol.isin(uni)].copy()
d["cL"]=np.where(d.signal>0,d.signal*d.ret_fwd/N,0.0)
d["cS"]=np.where(d.signal<0,d.signal*d.ret_fwd/N,0.0)
d["bL"]=np.where(d.signal>0,d.signal*d.symbol.map(beta)/N,0.0)
d["bS"]=np.where(d.signal<0,d.signal*d.symbol.map(beta)/N,0.0)
g=d.groupby("datetime")[["cL","cS","bL","bS"]].sum()
g["date"]=g.index.date
def stats(daily):
    cum=(1+daily).cumprod(); ann=cum.iloc[-1]**(252/len(cum))-1
    vol=daily.std()*np.sqrt(252); sh=daily.mean()/daily.std()*np.sqrt(252)
    mdd=(cum/np.maximum.accumulate(cum)-1).min()
    return ann,vol,sh,mdd
print("="*74)
print(f"(4) 숏레그 가중 λ 스윕   port(λ)=롱+λ·숏    [N={N}, gross, 비용0]")
print("="*74)
print(f"{'λ':>5}{'순베타平':>9}{'연수익':>9}{'연변동성':>9}{'Sharpe':>8}{'MDD':>8}{'10%vol환산수익':>14}")
nb_full=(g['bL']+g['bS'])  # 30분봉 순베타 기여 합? -> 사용 위해 일평균 따로
for lam in [0.0,0.25,0.5,0.75,1.0,1.25]:
    daily=(g['cL']+lam*g['cS']).groupby(g['date']).sum()
    # 시점별 순베타: bL + lam*bS, 일평균의 평균
    nb=(g['bL']+lam*g['bS']); nb_daily=nb.groupby(g['date']).sum()  # 하루 합(근사 노출척도 아님) -> 평균 시점값 사용
    nb_mean=(g['bL']+lam*g['bS']).mean()*1  # 봉단위 평균 순베타
    ann,vol,sh,mdd=stats(daily)
    print(f"{lam:>5.2f}{nb_mean:>9.2f}{ann*100:>8.1f}%{vol*100:>8.1f}%{sh:>8.2f}{mdd*100:>7.1f}%{sh*0.10*100:>13.1f}%")
print("\n  ※ 10%vol환산수익 = Sharpe×10% = 변동성 10%로 레버리지 맞췄을 때 기대 연수익(레버 정규화 비교).")
print("="*74)
print("(5) 숏레그 부진 — 종목별 숏 누적손익 (signal<0 기여 합산)")
print("="*74)
sp=d.groupby("symbol")["cS"].sum().reindex(uni)  # 종목별 숏 누적기여(근사: 단순합)
tbl=pd.DataFrame({"short_pnl":sp,"beta":beta}).dropna().sort_values("short_pnl")
print("  최악 숏(손실 큰) 8개:")
for s,row in tbl.head(8).iterrows(): print(f"    {s:6} 숏손익 {row.short_pnl*100:+6.2f}%   beta {row.beta:+.2f}")
print("  최선 숏(이익 큰) 8개:")
for s,row in tbl.tail(8).iterrows(): print(f"    {s:6} 숏손익 {row.short_pnl*100:+6.2f}%   beta {row.beta:+.2f}")
# 베타구간별 숏손익
tbl["beta_bucket"]=pd.cut(tbl.beta,[-1,0.5,1.0,1.5,3.0],labels=["β<0.5","0.5-1.0","1.0-1.5","β>1.5"])
print("\n  베타구간별 숏레그 누적손익 합:")
for bk,v in tbl.groupby("beta_bucket",observed=True)["short_pnl"].sum().items():
    print(f"    {bk:8}: {v*100:+6.2f}%   (종목수 {int((tbl.beta_bucket==bk).sum())})")
