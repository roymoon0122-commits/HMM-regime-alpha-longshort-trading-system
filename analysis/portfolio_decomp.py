"""
portfolio_decomp.py — 등가중 포트폴리오 레벨 백테스트 + 시장베타/잔차 분해.

oos_signals.parquet (49 유니버스 + SPY, OOS 2024-01-01~2026-05-22, 30분봉)을 받아:
 1) 등가중 gross 포트폴리오 수익 = (1/N) Σ signal_i * ret_fwd_i
 2) 일별 집계 후 SPY에 회귀 → 시장베타 β, 잔차알파 α(연율), R²
 3) 북의 순베타(시점별) = (1/N) Σ signal_i * beta_i  (롱/숏 쏠림 진단)
 4) 롱레그 vs 숏레그 누적 P&L·샤프 (숏레그가 돈 버는지 = 내부중립 vs SPY오버레이 판단)
비용 미반영(gross). 분석 전용.
"""
import os, numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(OUT, "oos_signals.parquet"))
df["date"] = df["datetime"].dt.date

uni = [s for s in df["symbol"].unique() if s != "SPY"]
N = len(uni)
ANN_30 = 13 * 252  # 30분봉 연율화 계수

def sharpe(r, ann):
    r = r[np.isfinite(r)];
    return r.mean()/r.std()*np.sqrt(ann) if r.std()>0 else np.nan
def mdd(cum):
    return (cum/np.maximum.accumulate(cum) - 1).min()

# ── 1) 종목별 베타 (OOS 일별 수익률 vs SPY) ──
piv_close = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
dret = np.log(piv_close).diff().dropna(how="all")
spy = dret["SPY"]
beta = {}
for s in uni:
    j = pd.concat([dret[s], spy], axis=1, keys=["x","m"]).dropna()
    beta[s] = np.cov(j["x"], j["m"])[0,1]/np.var(j["m"]) if len(j)>30 else np.nan
beta = pd.Series(beta)

# ── 2) 등가중 gross 포트폴리오 30분봉 수익 ──
df["contrib"] = df["signal"] * df["ret_fwd"] / N
port = df[df.symbol.isin(uni)].groupby("datetime")["contrib"].sum()
# 롱/숏 레그 분리
df["contrib_L"] = np.where(df.signal>0, df.signal*df.ret_fwd/N, 0.0)
df["contrib_S"] = np.where(df.signal<0, df.signal*df.ret_fwd/N, 0.0)
legL = df[df.symbol.isin(uni)].groupby("datetime")["contrib_L"].sum()
legS = df[df.symbol.isin(uni)].groupby("datetime")["contrib_S"].sum()

# 시점별 북 순베타·총노출
df["sb"] = df["signal"] * df["symbol"].map(beta) / N
df["gross"] = df["signal"].abs() / N
netbeta = df[df.symbol.isin(uni)].groupby("datetime")["sb"].sum()
gross = df[df.symbol.isin(uni)].groupby("datetime")["gross"].sum()
nlong = df[(df.symbol.isin(uni))&(df.signal>0)].groupby("datetime").size()
nshort = df[(df.symbol.isin(uni))&(df.signal<0)].groupby("datetime").size()

# ── 3) 일별 집계 + SPY 회귀 ──
pdaily = port.groupby(port.index.date).sum()           # 포폴 일수익(근사: 합)
sdaily = spy.copy(); sdaily.index = pd.Index(sdaily.index)
al = pd.concat([pdaily.rename("p"), sdaily.rename("m")], axis=1).dropna()
b1, b0 = np.polyfit(al["m"], al["p"], 1)                # p = b1*m + b0
resid = al["p"] - (b1*al["m"] + b0)
r2 = np.corrcoef(al["m"], al["p"])[0,1]**2
alpha_ann = b0 * 252

print("="*64)
print(f"등가중 gross 포트폴리오  (N={N}, OOS {al.index.min()}~{al.index.max()}, {len(al)}일)")
print("="*64)
cum = (1+pdaily).cumprod()
print(f"  누적수익      : {(cum.iloc[-1]-1)*100:+.1f}%")
print(f"  연수익(기하)  : {(cum.iloc[-1]**(252/len(cum))-1)*100:+.1f}%")
print(f"  연변동성      : {pdaily.std()*np.sqrt(252)*100:.1f}%")
print(f"  Sharpe(일)    : {sharpe(pdaily.values,252):+.2f}")
print(f"  MDD           : {mdd(cum.values)*100:.1f}%")
print(f"  SPY 상관      : {np.corrcoef(al['m'],al['p'])[0,1]:+.2f}")
print()
print("── 시장베타 분해 (일별 회귀: port = β·SPY + α) ──")
print(f"  시장베타 β    : {b1:+.2f}")
print(f"  잔차알파 α    : {alpha_ann*100:+.1f}%/년  (시장중립 성분)")
print(f"  R²(시장설명력): {r2*100:.0f}%")
print(f"  → 시장설명 {r2*100:.0f}%, 잔차 {(1-r2)*100:.0f}%")
print()
print("── 북 순베타 / 쏠림 ──")
print(f"  평균 순베타   : {netbeta.mean():+.2f}   (|.|중앙값 {netbeta.abs().median():.2f})")
print(f"  순베타>0 비중 : {(netbeta>0).mean()*100:.0f}%  (net-long 시간비율)")
print(f"  평균 총노출   : {gross.mean()*100:.0f}%   롱종목 {nlong.mean():.0f} / 숏종목 {nshort.mean():.0f} (평균동시)")
print()
print("── 롱레그 vs 숏레그 (gross, 누적) ──")
cumL=(1+legL.groupby(legL.index.date).sum()).cumprod()
cumS=(1+legS.groupby(legS.index.date).sum()).cumprod()
print(f"  롱레그 누적   : {(cumL.iloc[-1]-1)*100:+.1f}%   Sharpe {sharpe(legL.groupby(legL.index.date).sum().values,252):+.2f}")
print(f"  숏레그 누적   : {(cumS.iloc[-1]-1)*100:+.1f}%   Sharpe {sharpe(legS.groupby(legS.index.date).sum().values,252):+.2f}")
print()
# SPY 동기간 buy&hold
spy_cum=(1+sdaily.reindex(al.index).fillna(0)).cumprod()
print(f"  [참고] SPY 동기간 누적: {(np.exp(spy.reindex(al.index).fillna(0).cumsum()).iloc[-1]-1)*100:+.1f}%")
# 저장
pd.DataFrame({"port_daily":pdaily}).to_csv(os.path.join(OUT,"port_daily.csv"))
print("\n베타 상위/하위:", beta.sort_values().round(2).to_dict())
