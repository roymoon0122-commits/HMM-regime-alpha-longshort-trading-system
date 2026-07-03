"""
short_mechanism.py — 숏 수익성의 베타 의존 '메커니즘' 분해 + 거래 특성 요약.

(A) carry vs skill 분해 (베타구간별):
    - drift      = E[ret_fwd]                  (무조건. 숏이 맞서는 시장 드리프트=carry)
    - E[ret|bear]= E[ret_fwd | signal<-0.1]    (알파가 약세로 본 시점의 실제 수익)
    - skill      = E[ret|bear] - drift          (음수면 = bear신호가 평균보다 더 하락 예측 = 진짜 숏엣지)
    - hit        = P(ret_fwd<0 | bear)          (숏 적중률)
    모두 봉당 → 연율화(×13×252)로 표기.
(B) 거래 특성: 종목별 부호전환수=거래수, 평균보유봉수, 롱/숏/플랫 시간비율, 회전율.
"""
import os, numpy as np, pandas as pd
OUT=os.path.dirname(os.path.abspath(__file__))
df=pd.read_parquet(os.path.join(OUT,"oos_signals.parquet"))
uni=[s for s in df.symbol.unique() if s!="SPY"]
ANN=13*252
# 베타
piv=df.pivot_table(index=df.datetime.dt.date,columns="symbol",values="close",aggfunc="last")
dret=np.log(piv).diff().dropna(how="all"); spy=dret["SPY"]
beta={}
for s in uni:
    j=pd.concat([dret[s],spy],axis=1).dropna(); beta[s]=np.cov(j.iloc[:,0],j.iloc[:,1])[0,1]/spy.var()
beta=pd.Series(beta)
d=df[df.symbol.isin(uni)].copy()
d["beta"]=d.symbol.map(beta)
d["bucket"]=pd.cut(d.beta,[-1,0.5,1.0,1.5,3.0],labels=["β<0.5","0.5-1.0","1.0-1.5","β>1.5"])
d=d.dropna(subset=["ret_fwd"])

print("="*78)
print("(A) 숏 메커니즘: carry(드리프트) vs skill(예측력)  — 연율화 %, 봉당E×13×252")
print("="*78)
print(f"{'bucket':9}{'종목':>4}{'drift(carry)':>13}{'E[ret|bear]':>13}{'skill':>9}{'숏적중%':>8}{'E[ret|bull]':>13}")
for bk in ["β<0.5","0.5-1.0","1.0-1.5","β>1.5"]:
    g=d[d.bucket==bk]
    bear=g[g.signal<-0.1]; bull=g[g.signal>0.1]
    drift=g.ret_fwd.mean()*ANN*100
    ebear=bear.ret_fwd.mean()*ANN*100
    ebull=bull.ret_fwd.mean()*ANN*100
    skill=ebear-drift
    hit=(bear.ret_fwd<0).mean()*100
    nsym=g.symbol.nunique()
    print(f"{bk:9}{nsym:>4}{drift:>11.0f}%{ebear:>12.0f}%{skill:>8.0f}%{hit:>8.1f}{ebull:>12.0f}%")
print("\n 해석 키:")
print("  - drift>0 클수록 = 숏이 맞서야 할 상승 carry가 큼.")
print("  - skill<0 (E[ret|bear]<drift) = bear신호가 평균보다 더 하락을 예측 = 구조적 숏엣지 존재.")
print("  - skill≈0 = bear신호에 하락 예측력 없음(그 구간 숏은 carry만 떠안음).")

print("\n"+"="*78)
print("(B) 거래 특성")
print("="*78)
rows=[]
for s in uni:
    g=df[df.symbol==s].sort_values("datetime")
    sgn=np.sign(g.signal.where(g.signal.abs()>0.1,0.0).values)
    flips=int((np.diff(sgn)!=0).sum())          # 부호/진입·청산 전환수
    nbar=len(g)
    hold=nbar/flips if flips>0 else np.nan
    frL=(g.signal>0.1).mean(); frS=(g.signal<-0.1).mean(); frF=1-frL-frS
    rows.append((s,beta[s],flips,hold,frL*100,frS*100,frF*100))
T=pd.DataFrame(rows,columns=["sym","beta","flips","hold_bars","long%","short%","flat%"])
print(f"전체 평균: 전환수 {T.flips.mean():.0f}회, 평균보유 {T.hold_bars.mean():.0f}봉"
      f"(≈{T.hold_bars.mean()/13:.1f}거래일), 롱시간 {T['long%'].mean():.0f}% / 숏 {T['short%'].mean():.0f}% / 플랫 {T['flat%'].mean():.0f}%")
print(f"OOS 7787봉(≈599거래일) 기준. 회전이 잦을수록 hold_bars↓.\n")
print("가장 자주 전환(고회전) 8종목:")
for _,r in T.sort_values('flips',ascending=False).head(8).iterrows():
    print(f"  {r.sym:6} β{r.beta:+.2f}  전환 {int(r.flips):>3}회  보유 {r.hold_bars:4.0f}봉  롱{r['long%']:.0f}/숏{r['short%']:.0f}/플랫{r['flat%']:.0f}%")
print("가장 드물게 전환(저회전) 8종목:")
for _,r in T.sort_values('flips').head(8).iterrows():
    print(f"  {r.sym:6} β{r.beta:+.2f}  전환 {int(r.flips):>3}회  보유 {r.hold_bars:4.0f}봉  롱{r['long%']:.0f}/숏{r['short%']:.0f}/플랫{r['flat%']:.0f}%")
print(f"\n숏 시간비율 높은 5종목:", T.sort_values('short%',ascending=False).head(5)[['sym','short%']].values.tolist())
print(f"롱 시간비율 높은 5종목:", T.sort_values('long%',ascending=False).head(5)[['sym','long%']].values.tolist())
