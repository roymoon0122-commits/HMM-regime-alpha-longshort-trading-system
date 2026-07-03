"""
walkforward_decomp.py — walk-forward(하락장 포함) 포트폴리오 분해 분석.

portfolio_decomp.py 의 walk-forward 버전. walkforward_signals.parquet(여러 윈도우의
test 세그먼트를 이어붙인 연속 OOS)을 받아, 윈도우(레짐)별로 + 전체연속으로:
  (a) 등가중 gross 포폴 누적·연수익·연변동성·Sharpe·MDD·MDD회복
  (b) 시점별 북 순베타 추이(레짐별 평균/중앙값, net-long 시간비율)
  (c) 롱레그 vs 숏레그 누적·Sharpe  →  숏레그가 하락장(2020/2022)에 실제로 돈 버는지
  (d) λ(숏 비중) 스윕: port = 롱 + λ·숏  →  '덜 헷지가 이긴다'가 하락장서도 유지되는지

베타는 윈도우별로 그 구간 SPY 일수익에 회귀해 산출(레짐별 베타 이동 반영).
비용 미반영(gross). 분석 전용. strategy/·config·live 미수정.

입력: analysis/walkforward_signals.parquet  (없으면 sig_parts_wf/*.parquet 결합 시도)
출력: 콘솔 표 + analysis/wf_port_daily.csv (윈도우 태그 포함 일별 포폴 수익)
"""
import os, glob, numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
ANN = 252
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_signals():
    p = os.path.join(OUT, "walkforward_signals.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    parts = glob.glob(os.path.join(OUT, "sig_parts_wf", "*.parquet"))
    frames = [pd.read_parquet(f) for f in parts]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("시그널 없음: walkforward_signals.parquet 또는 sig_parts_wf/ 가 비어있음")
    return pd.concat(frames, ignore_index=True)


def sharpe(r, ann=ANN):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan


def mdd(cum):
    cum = np.asarray(cum, float)
    return (cum / np.maximum.accumulate(cum) - 1).min()


def mdd_recovery_days(daily_idx, cum):
    """MDD 저점 이후 직전 고점 회복까지 걸린 거래일수(회복 못 하면 None)."""
    cum = np.asarray(cum, float)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    trough = int(np.argmin(dd))
    peak_val = peak[trough]
    after = np.where(cum[trough:] >= peak_val)[0]
    if len(after) == 0:
        return None
    return int(after[0])


def per_symbol_beta(df, uni):
    """윈도우 내부 일별 로그수익으로 종목별 시장베타(vs SPY) 산출."""
    piv = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    dret = np.log(piv).diff().dropna(how="all")
    if "SPY" not in dret:
        return pd.Series(dtype=float)
    spy = dret["SPY"]
    beta = {}
    for s in uni:
        if s not in dret:
            continue
        j = pd.concat([dret[s], spy], axis=1, keys=["x", "m"]).dropna()
        beta[s] = np.cov(j["x"], j["m"])[0, 1] / np.var(j["m"]) if len(j) > 30 else np.nan
    return pd.Series(beta)


def analyze_segment(df, label):
    """한 구간(윈도우 또는 전체연속)의 포폴 분해. df는 SPY 포함."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    uni = [s for s in df["symbol"].unique() if s != "SPY"]
    N = len(uni)
    beta = per_symbol_beta(df, uni)

    sub = df[df.symbol.isin(uni)].copy()
    sub["contrib"] = sub["signal"] * sub["ret_fwd"] / N
    sub["cL"] = np.where(sub.signal > 0, sub.signal * sub.ret_fwd / N, 0.0)
    sub["cS"] = np.where(sub.signal < 0, sub.signal * sub.ret_fwd / N, 0.0)
    sub["sb"] = sub["signal"] * sub["symbol"].map(beta) / N

    port = sub.groupby("datetime")["contrib"].sum()
    legL = sub.groupby("datetime")["cL"].sum()
    legS = sub.groupby("datetime")["cS"].sum()
    netbeta = sub.groupby("datetime")["sb"].sum()

    pdaily = port.groupby(port.index.date).sum()
    Ldaily = legL.groupby(legL.index.date).sum()
    Sdaily = legS.groupby(legS.index.date).sum()
    cum = (1 + pdaily).cumprod()

    # SPY 동기간
    spy_px = df[df.symbol == "SPY"].set_index("datetime")["close"]
    spy_d = np.log(spy_px).groupby(spy_px.index.date).last().diff().dropna()

    # 일별 회귀 (시장베타/잔차알파)
    al = pd.concat([pdaily.rename("p"), spy_d.rename("m")], axis=1).dropna()
    if len(al) > 5:
        b1, b0 = np.polyfit(al["m"], al["p"], 1)
        r2 = np.corrcoef(al["m"], al["p"])[0, 1] ** 2
        corr = np.corrcoef(al["m"], al["p"])[0, 1]
    else:
        b1 = b0 = r2 = corr = np.nan

    rec = mdd_recovery_days(pdaily.index, cum.values)
    spy_cum_ret = (np.exp(spy_d.reindex(pdaily.index).fillna(0).cumsum()).iloc[-1] - 1) * 100 if len(spy_d) else np.nan

    res = {
        "label": label, "N": N, "days": len(pdaily),
        "cum": (cum.iloc[-1] - 1) * 100,
        "ann": (cum.iloc[-1] ** (ANN / len(cum)) - 1) * 100 if len(cum) else np.nan,
        "vol": pdaily.std() * np.sqrt(ANN) * 100,
        "sharpe": sharpe(pdaily.values),
        "mdd": mdd(cum.values) * 100,
        "rec_days": rec,
        "corr": corr, "beta": b1, "alpha_ann": b0 * ANN * 100, "r2": r2 * 100,
        "netbeta_mean": netbeta.mean(), "netbeta_absmed": netbeta.abs().median(),
        "netlong_pct": (netbeta > 0).mean() * 100,
        "legL_cum": ((1 + Ldaily).cumprod().iloc[-1] - 1) * 100,
        "legL_sh": sharpe(Ldaily.values),
        "legS_cum": ((1 + Sdaily).cumprod().iloc[-1] - 1) * 100,
        "legS_sh": sharpe(Sdaily.values),
        "spy_cum": spy_cum_ret,
        "_pdaily": pdaily, "_Ldaily": Ldaily, "_Sdaily": Sdaily,
    }
    return res


def lambda_sweep(res):
    """port_λ = 롱레그 + λ·숏레그. 일별 합성 후 Sharpe/MDD/누적."""
    L, S = res["_Ldaily"], res["_Sdaily"]
    rows = []
    for lam in LAMBDAS:
        d = L.add(lam * S, fill_value=0.0)
        cum = (1 + d).cumprod()
        rows.append((lam, (cum.iloc[-1] - 1) * 100, sharpe(d.values), mdd(cum.values) * 100))
    return rows


def print_segment(res):
    print("=" * 70)
    print(f"[{res['label']}]  N={res['N']}, {res['days']}일")
    print("=" * 70)
    print(f"  누적 {res['cum']:+.1f}% / 연 {res['ann']:+.1f}% / 변동성 {res['vol']:.1f}% / "
          f"Sharpe {res['sharpe']:+.2f} / MDD {res['mdd']:.1f}%"
          + (f" / MDD회복 {res['rec_days']}일" if res['rec_days'] is not None else " / MDD미회복"))
    print(f"  SPY상관 {res['corr']:+.2f} | 시장β {res['beta']:+.2f} | 잔차α {res['alpha_ann']:+.1f}%/년 | R² {res['r2']:.0f}%")
    print(f"  순베타: 평균 {res['netbeta_mean']:+.2f} (|.|중앙값 {res['netbeta_absmed']:.2f}), net-long 시간 {res['netlong_pct']:.0f}%")
    print(f"  롱레그 {res['legL_cum']:+.1f}% (Sh {res['legL_sh']:+.2f}) | "
          f"숏레그 {res['legS_cum']:+.1f}% (Sh {res['legS_sh']:+.2f})  ← 하락장서 숏이 버는지")
    print(f"  [참고] SPY 동기간 B&H {res['spy_cum']:+.1f}%")
    print("  λ 스윕 (port = 롱 + λ·숏):  '덜 헷지(λ↓)가 이기는가?'")
    print(f"    {'λ':>5} {'누적%':>9} {'Sharpe':>8} {'MDD%':>8}")
    for lam, cum, sh, m in lambda_sweep(res):
        print(f"    {lam:>5.2f} {cum:>9.1f} {sh:>8.2f} {m:>8.1f}")
    print()


def main():
    df = load_signals()
    df["datetime"] = pd.to_datetime(df["datetime"])
    has_window = "window" in df.columns
    windows = sorted(df["window"].unique()) if has_window else ["all"]
    print(f"로드: {len(df):,}행, {df['symbol'].nunique()}종목, 윈도우 {windows}\n")

    saved = []
    # 윈도우(레짐)별
    for w in windows:
        seg = df[df["window"] == w] if has_window else df
        res = analyze_segment(seg, f"윈도우 {w}")
        print_segment(res)
        pd_ = res["_pdaily"].rename("port_daily").to_frame(); pd_["window"] = w
        saved.append(pd_)

    # 전체 연속 OOS (윈도우 2개 이상일 때만 의미)
    if has_window and len(windows) > 1:
        res_all = analyze_segment(df, "전체 연속 OOS (2020+2022+2024-25)")
        print_segment(res_all)

    pd.concat(saved).to_csv(os.path.join(OUT, "wf_port_daily.csv"))
    print("저장: analysis/wf_port_daily.csv")


if __name__ == "__main__":
    main()
