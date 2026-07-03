"""
포트폴리오 레벨 OOS 대시보드.

analysis/oos_signals.parquet 하나만 읽어 배분전략별 자산곡선과
SPY Buy&Hold 벤치마크를 Plotly 자기완결 HTML로 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = Path(__file__).resolve().parent / "oos_signals.parquet"
OUTPUT_PATH = ROOT / "results" / "portfolio_dashboard.html"
SPY_SYMBOL = "SPY"
TRADING_DAYS = 252
HOVER_TOP_N = 12
NET_BETA_CAP = 0.25
BETA_LOOKBACK_DAYS = 252
BETA_MIN_OBS = 126
SLIPPAGE_BPS = [0, 2, 5, 10]  # 편도 슬리피지 시나리오(bp)


@dataclass(frozen=True)
class WideData:
    sig_wide: pd.DataFrame
    ret_wide: pd.DataFrame
    close_daily: pd.DataFrame
    spy_close_daily: pd.Series
    universe: list[str]


@dataclass(frozen=True)
class PortfolioResult:
    name: str
    weights: pd.DataFrame
    daily_returns: pd.Series
    equity: pd.Series
    stats: dict[str, float]
    color: str


StrategyFn = Callable[[pd.DataFrame], pd.DataFrame]


def load(path: Path = INPUT_PATH) -> pd.DataFrame:
    """입력 parquet을 읽고 필수 컬럼을 검증한다."""
    df = pd.read_parquet(path)
    required = {"datetime", "symbol", "signal", "close", "ret_fwd"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[:, ["datetime", "symbol", "signal", "close", "ret_fwd"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    return df


def to_wide(df: pd.DataFrame) -> WideData:
    """long-format 신호를 전략 계산용 wide-format으로 변환한다."""
    symbols = sorted(df["symbol"].unique())
    if SPY_SYMBOL not in symbols:
        raise ValueError("SPY benchmark symbol is missing.")

    universe = [symbol for symbol in symbols if symbol != SPY_SYMBOL]
    if not universe:
        raise ValueError("No strategy universe symbols found.")

    df_with_date = df.assign(date=lambda x: _daily_index(x["datetime"]))
    uni_df = df[df["symbol"].isin(universe)]
    sig_wide = (
        uni_df.pivot(index="datetime", columns="symbol", values="signal")
        .sort_index()
        .reindex(columns=universe)
    )
    ret_wide = (
        uni_df.pivot(index="datetime", columns="symbol", values="ret_fwd")
        .sort_index()
        .reindex(columns=universe)
    )

    close_daily = (
        df_with_date.pivot_table(
            index="date",
            columns="symbol",
            values="close",
            aggfunc="last",
        )
        .sort_index()
        .reindex(columns=[*universe, SPY_SYMBOL])
    )
    spy_daily = close_daily[SPY_SYMBOL].dropna()
    return WideData(
        sig_wide=sig_wide,
        ret_wide=ret_wide,
        close_daily=close_daily,
        spy_close_daily=spy_daily,
        universe=universe,
    )


def _daily_index(index_like) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(index_like)).normalize()


def strat_equal_weight(sig_wide: pd.DataFrame) -> pd.DataFrame:
    """등가중: 각 종목 signal을 유니버스 수로 나눈 자본 비중."""
    n_symbols = len(sig_wide.columns)
    if n_symbols == 0:
        raise ValueError("sig_wide has no symbols.")
    return sig_wide.fillna(0.0) / n_symbols


# 전략 레지스트리 — 새 전략은 여기 한 줄만 추가하면 곡선이 붙는다.
STRATEGIES: list[tuple[str, StrategyFn, str]] = [
    ("Equal-Weight", strat_equal_weight, "#1f77b4"),
    # ("Inverse-Vol", strat_inverse_vol, "#ff7f0e"),   # 추후
    # ("Sharpe-Tilt", strat_sharpe_tilt, "#2ca02c"),   # 추후
]


def _stats_from_daily_returns(
    daily_returns: pd.Series,
    equity: pd.Series,
) -> dict[str, float]:
    r = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    std = r.std()
    sharpe = float(r.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan
    mdd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else np.nan
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan

    if len(equity) and equity.iloc[-1] > 0:
        annual_return = float(equity.iloc[-1] ** (TRADING_DAYS / len(equity)) - 1.0)
    else:
        annual_return = np.nan

    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "total_return": total_return,
        "annual_return": annual_return,
    }


def equity_and_stats(
    w_wide: pd.DataFrame,
    ret_wide: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """봉 수익을 일별 합산하고 자산곡선과 성과지표를 계산한다."""
    weights = w_wide.reindex_like(ret_wide).fillna(0.0)
    returns = ret_wide.fillna(0.0)

    r_bar = (weights * returns).sum(axis=1)
    r_day = r_bar.groupby(_daily_index(r_bar.index)).sum()
    equity = (1.0 + r_day).cumprod()
    stats = _stats_from_daily_returns(r_day, equity)
    return r_day, equity, stats


def stats_with_slippage(
    w_wide: pd.DataFrame,
    ret_wide: pd.DataFrame,
    slippage_bps: float,
) -> dict[str, float]:
    """편도 슬리피지(bp)를 반영한 성과지표를 계산한다.

    봉별 거래비용 = 회전율 Σ|Δw_i| × (slippage_bps / 1e4). 슬리피지 0bp면
    equity_and_stats와 동일한 결과가 나온다.
    """
    weights = w_wide.reindex_like(ret_wide).fillna(0.0)
    returns = ret_wide.fillna(0.0)

    r_bar = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (slippage_bps / 1e4)
    r_bar_net = r_bar - cost

    r_day = r_bar_net.groupby(_daily_index(r_bar_net.index)).sum()
    equity = (1.0 + r_day).cumprod()
    return _stats_from_daily_returns(r_day, equity)


def capm_decomposition(
    port_daily: pd.Series,
    market_daily: pd.Series,
) -> dict[str, float]:
    """포트 일별수익을 시장(SPY)에 CAPM식 선형회귀해 베타/알파를 분해한다.

    rp = a + b·rm. 누적 기여분은 일별수익 단리 합 기준으로,
    시장기여 = b·Σrm, 알파기여 = Σrp − b·Σrm (= a×거래일수)로 나눈다.
    """
    idx = port_daily.index.intersection(market_daily.index)
    rp = port_daily.reindex(idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rm = market_daily.reindex(idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(idx) < 2 or float(np.var(rm)) <= 0:
        return {k: np.nan for k in (
            "beta", "alpha_annual", "corr", "r2",
            "mkt_contrib", "alpha_contrib", "total", "mkt_share", "alpha_share",
        )}

    beta, alpha = np.polyfit(rm.to_numpy(), rp.to_numpy(), 1)
    corr = float(np.corrcoef(rp.to_numpy(), rm.to_numpy())[0, 1])
    total = float(rp.sum())
    mkt_contrib = float(beta * rm.sum())
    alpha_contrib = total - mkt_contrib
    return {
        "beta": float(beta),
        "alpha_annual": float(alpha * TRADING_DAYS),
        "corr": corr,
        "r2": corr ** 2,
        "mkt_contrib": mkt_contrib,
        "alpha_contrib": alpha_contrib,
        "total": total,
        "mkt_share": mkt_contrib / total if total else np.nan,
        "alpha_share": alpha_contrib / total if total else np.nan,
    }


def estimate_causal_rolling_betas(
    close_daily: pd.DataFrame,
    universe: list[str],
    lookback_days: int = BETA_LOOKBACK_DAYS,
    min_obs: int = BETA_MIN_OBS,
) -> pd.DataFrame:
    """오늘 이전 일별 수익률만 써서 SPY 대비 rolling beta를 계산한다."""
    daily_log_returns = np.log(close_daily).diff().replace([np.inf, -np.inf], np.nan)
    betas = pd.DataFrame(index=close_daily.index, columns=universe, dtype=float)

    for date in close_daily.index:
        # 해당 날짜 포지션에는 그 날짜 종가가 아직 없으므로 date 이전만 사용.
        hist = daily_log_returns.loc[daily_log_returns.index < date].tail(lookback_days)
        spy_returns = hist[SPY_SYMBOL]
        for symbol in universe:
            joined = pd.concat(
                [hist[symbol], spy_returns],
                axis=1,
                keys=["asset", "spy"],
            ).dropna()
            if len(joined) < min_obs or np.var(joined["spy"]) <= 0:
                continue
            betas.loc[date, symbol] = float(
                np.cov(joined["asset"], joined["spy"])[0, 1] / np.var(joined["spy"])
            )

    return betas


def _beta_frame_for_weights(
    w_wide: pd.DataFrame,
    beta: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Series/일별 DataFrame beta를 가중치 index/columns에 맞춘다."""
    if isinstance(beta, pd.Series):
        beta_series = beta.reindex(w_wide.columns)
        return pd.DataFrame(
            np.repeat(beta_series.to_numpy()[None, :], len(w_wide.index), axis=0),
            index=w_wide.index,
            columns=w_wide.columns,
        )

    beta_daily = beta.reindex(columns=w_wide.columns)
    direct = beta_daily.reindex(w_wide.index)
    if direct.notna().to_numpy().any() or w_wide.index.equals(beta_daily.index):
        return direct

    by_day = beta_daily.reindex(_daily_index(w_wide.index))
    by_day.index = w_wide.index
    return by_day


def net_beta_series(
    w_wide: pd.DataFrame,
    beta: pd.Series | pd.DataFrame,
) -> pd.Series:
    """가중치 행렬의 시점별 순베타(Σ w_i × beta_i)를 계산한다."""
    beta_frame = _beta_frame_for_weights(w_wide, beta)
    has_beta = beta_frame.notna().any(axis=1)
    net_beta = w_wide.fillna(0.0).mul(beta_frame.fillna(0.0), axis=1).sum(axis=1)
    return net_beta.where(has_beta)


def apply_net_beta_cap_wide(
    w_wide: pd.DataFrame,
    beta: pd.Series | pd.DataFrame,
    cap: float = NET_BETA_CAP,
) -> pd.DataFrame:
    """순베타 초과분을 만드는 포지션만 비례 축소한다.

    포지션 부호는 뒤집지 않고, 새 포지션도 만들지 않는다. 시점별 순베타가
    +cap을 넘으면 양의 베타 기여분만 줄이고, -cap보다 작으면 음의 베타
    기여분만 줄인다.
    """
    weights = w_wide.fillna(0.0).copy()
    beta_frame = _beta_frame_for_weights(weights, beta)
    has_beta = beta_frame.notna().any(axis=1)
    contrib = weights.mul(beta_frame.fillna(0.0), axis=1)
    net_beta = contrib.sum(axis=1)

    factors = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)

    over = has_beta & (net_beta > cap)
    if over.any():
        positive_contrib = contrib.where(contrib > 0.0, 0.0).sum(axis=1)
        positive_scale = 1.0 - (net_beta - cap) / positive_contrib.replace(0.0, np.nan)
        positive_scale = positive_scale.clip(lower=0.0, upper=1.0).fillna(1.0)
        over_positive = contrib.gt(0.0) & over.to_numpy()[:, None]
        positive_scale_wide = pd.DataFrame(
            np.repeat(positive_scale.to_numpy()[:, None], len(weights.columns), axis=1),
            index=weights.index,
            columns=weights.columns,
        )
        factors = factors.mask(over_positive, positive_scale_wide)

    under = has_beta & (net_beta < -cap)
    if under.any():
        negative_contrib_abs = -contrib.where(contrib < 0.0, 0.0).sum(axis=1)
        negative_scale = 1.0 - (-cap - net_beta) / negative_contrib_abs.replace(0.0, np.nan)
        negative_scale = negative_scale.clip(lower=0.0, upper=1.0).fillna(1.0)
        under_negative = contrib.lt(0.0) & under.to_numpy()[:, None]
        negative_scale_wide = pd.DataFrame(
            np.repeat(negative_scale.to_numpy()[:, None], len(weights.columns), axis=1),
            index=weights.index,
            columns=weights.columns,
        )
        factors = factors.mask(under_negative, negative_scale_wide)

    return weights * factors


def make_beta_cap_strategy(
    beta: pd.Series | pd.DataFrame,
    cap: float = NET_BETA_CAP,
) -> StrategyFn:
    """등가중 전략에 순베타 캡을 덧씌운 전략 함수를 만든다."""

    def strat_equal_weight_beta_cap(sig_wide: pd.DataFrame) -> pd.DataFrame:
        raw = strat_equal_weight(sig_wide)
        return apply_net_beta_cap_wide(raw, beta, cap)

    return strat_equal_weight_beta_cap


def _format_position(symbol: str, weight: float) -> str:
    return f"{symbol:<6} {weight:+7.1%}"


def _format_net_beta(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.2f}"


def _sorted_side(weights: pd.Series, long_side: bool) -> pd.Series:
    side = weights[weights > 0] if long_side else weights[weights < 0]
    order = side.abs().sort_values(ascending=False).index
    return side.loc[order].head(HOVER_TOP_N)


def build_hover(
    w_wide: pd.DataFrame,
    daily_returns: pd.Series,
    equity: pd.Series,
    beta: pd.Series | pd.DataFrame,
) -> pd.Series:
    """각 일별 포인트의 롱/숏 2열 보유표 hover 문자열을 만든다."""
    daily_weights = w_wide.fillna(0.0).groupby(_daily_index(w_wide.index)).last()
    daily_weights = daily_weights.reindex(equity.index).fillna(0.0)
    daily_returns = daily_returns.reindex(equity.index).fillna(0.0)
    net_beta = net_beta_series(daily_weights, beta)

    hover_texts: list[str] = []
    col_width = 16
    header = f"{'LONG':<{col_width}}  {'SHORT':<{col_width}}"
    divider = f"{'-' * col_width}  {'-' * col_width}"

    for date, weights in daily_weights.iterrows():
        longs = _sorted_side(weights, long_side=True)
        shorts = _sorted_side(weights, long_side=False)
        row_count = max(len(longs), len(shorts), 1)

        rows = []
        for i in range(row_count):
            left = (
                _format_position(str(longs.index[i]), float(longs.iloc[i]))
                if i < len(longs)
                else ""
            )
            right = (
                _format_position(str(shorts.index[i]), float(shorts.iloc[i]))
                if i < len(shorts)
                else ""
            )
            rows.append(f"{left:<{col_width}}  {right:<{col_width}}")

        cumulative = equity.loc[date] - 1.0
        day_return = daily_returns.loc[date]
        summary = (
            f"<b>{pd.Timestamp(date):%Y-%m-%d}</b>  "
            f"누적 {cumulative:+.1%} · 당일 {day_return:+.1%} · "
            f"순베타 {_format_net_beta(net_beta.loc[date])}"
        )
        hover_texts.append("<br>".join([summary, header, divider, *rows]))

    return pd.Series(hover_texts, index=equity.index)


def _benchmark_equity_and_stats(
    spy_close_daily: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    spy_close_daily = spy_close_daily.dropna().sort_index()
    if spy_close_daily.empty:
        raise ValueError("SPY daily close series is empty.")

    equity = spy_close_daily / float(spy_close_daily.iloc[0])
    daily_returns = spy_close_daily.pct_change().fillna(0.0)
    stats = _stats_from_daily_returns(daily_returns, equity)
    return daily_returns, equity, stats


def _legend_name(name: str, stats: dict[str, float]) -> str:
    return f"{name}  Sharpe {stats['sharpe']:.2f} · MDD {stats['mdd']:.1%}"


def _make_strategy_result(
    name: str,
    strategy: StrategyFn,
    color: str,
    data: WideData,
) -> PortfolioResult:
    weights = strategy(data.sig_wide).reindex_like(data.sig_wide).fillna(0.0)
    daily_returns, equity, stats = equity_and_stats(weights, data.ret_wide)
    return PortfolioResult(
        name=name,
        weights=weights,
        daily_returns=daily_returns,
        equity=equity,
        stats=stats,
        color=color,
    )


def make_figure(
    data: WideData,
    strategies: list[tuple[str, StrategyFn, str]] = STRATEGIES,
) -> tuple[go.Figure, list[PortfolioResult], dict[str, pd.Series | dict[str, float]]]:
    """전략 레지스트리를 순회해 원본/베타캡 자산곡선 그래프를 만든다."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Equal-Weight 원본 포트폴리오",
            (
                f"Equal-Weight 순베타 캡 {NET_BETA_CAP:.2f} 적용 "
                f"(rolling {BETA_LOOKBACK_DAYS}d, min {BETA_MIN_OBS}d, no lookahead)"
            ),
        ),
    )
    strategy_results: list[PortfolioResult] = []
    beta_by_date = estimate_causal_rolling_betas(data.close_daily, data.universe)

    for name, strategy, color in strategies:
        result = _make_strategy_result(name, strategy, color, data)
        strategy_results.append(result)
        hover = build_hover(
            result.weights,
            result.daily_returns,
            result.equity,
            beta_by_date,
        )

        fig.add_trace(
            go.Scatter(
                x=result.equity.index,
                y=result.equity,
                name=_legend_name(result.name, result.stats),
                mode="lines",
                line=dict(color=result.color, width=2.2),
                customdata=hover.to_numpy(),
                hovertemplate="%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    capped_strategies: list[tuple[str, StrategyFn, str]] = [
        (
            f"Equal-Weight BetaCap {NET_BETA_CAP:.2f}",
            make_beta_cap_strategy(beta_by_date, NET_BETA_CAP),
            "#d62728",
        ),
    ]
    for name, strategy, color in capped_strategies:
        result = _make_strategy_result(name, strategy, color, data)
        strategy_results.append(result)
        hover = build_hover(
            result.weights,
            result.daily_returns,
            result.equity,
            beta_by_date,
        )

        fig.add_trace(
            go.Scatter(
                x=result.equity.index,
                y=result.equity,
                name=_legend_name(result.name, result.stats),
                mode="lines",
                line=dict(color=color, width=2.2),
                customdata=hover.to_numpy(),
                hovertemplate="%{customdata}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    spy_daily_returns, spy_equity, spy_stats = _benchmark_equity_and_stats(
        data.spy_close_daily
    )
    spy_hover = [
        (
            f"<b>{pd.Timestamp(date):%Y-%m-%d}</b><br>"
            f"SPY Buy&Hold<br>"
            f"누적 {equity - 1.0:+.1%} · 당일 {spy_daily_returns.loc[date]:+.1%}"
        )
        for date, equity in spy_equity.items()
    ]
    fig.add_trace(
        go.Scatter(
            x=spy_equity.index,
            y=spy_equity,
            name=_legend_name("SPY Buy&Hold", spy_stats),
            mode="lines",
            line=dict(color="#888888", width=1.8, dash="dot"),
            customdata=np.array(spy_hover, dtype=object),
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=spy_equity.index,
            y=spy_equity,
            name=_legend_name("SPY Buy&Hold", spy_stats),
            mode="lines",
            line=dict(color="#888888", width=1.8, dash="dot"),
            customdata=np.array(spy_hover, dtype=object),
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    first_day = min(result.equity.index.min() for result in strategy_results)
    last_day = max(result.equity.index.max() for result in strategy_results)
    summary = "<br>".join(
        [
            (
                f"{result.name}: Sharpe {result.stats['sharpe']:.2f}, "
                f"MDD {result.stats['mdd']:.1%}, "
                f"누적 {result.stats['total_return']:+.1%}"
            )
            for result in strategy_results
        ]
        + [
            (
                f"SPY: Sharpe {spy_stats['sharpe']:.2f}, "
                f"MDD {spy_stats['mdd']:.1%}, "
                f"누적 {spy_stats['total_return']:+.1%}"
            )
        ]
    )

    fig.add_hline(
        y=1.0,
        line=dict(color="#888888", width=0.8, dash="dot"),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=1.0,
        line=dict(color="#888888", width=0.8, dash="dot"),
        row=2,
        col=1,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        text=summary,
        align="left",
        showarrow=False,
        font=dict(size=11, color="#222222"),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="rgba(44,62,80,0.18)",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                "포트폴리오 레벨 OOS 자산곡선 "
                f"({first_day:%Y-%m-%d} ~ {last_day:%Y-%m-%d})"
            ),
            font=dict(size=16),
        ),
        template="plotly_white",
        height=1120,
        margin=dict(l=70, r=35, t=95, b=55),
        hovermode="closest",
        hoverlabel=dict(font=dict(family="monospace", size=11)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    fig.update_xaxes(
        title_text="날짜",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="자산곡선 (시작=1.0)",
        tickformat=".2f",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="자산곡선 (시작=1.0)",
        tickformat=".2f",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        row=2,
        col=1,
    )

    benchmark = {
        "daily_returns": spy_daily_returns,
        "equity": spy_equity,
        "stats": spy_stats,
        "beta_by_date": beta_by_date,
    }
    return fig, strategy_results, benchmark


def _fmt_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.1%}"


def _fmt_num(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def _slippage_table_html(name: str, w_wide: pd.DataFrame, ret_wide: pd.DataFrame) -> str:
    """한 전략의 슬리피지 시나리오별 성과표 HTML 조각을 만든다."""
    rows = []
    for bps in SLIPPAGE_BPS:
        s = stats_with_slippage(w_wide, ret_wide, bps)
        label = f"{bps}bp" + (" (무비용)" if bps == 0 else "")
        rows.append(
            "<tr>"
            f"<td class='lbl'>{label}</td>"
            f"<td>{_fmt_pct(s['total_return'])}</td>"
            f"<td>{_fmt_pct(s['annual_return'])}</td>"
            f"<td>{_fmt_num(s['sharpe'])}</td>"
            f"<td>{_fmt_pct(s['mdd'])}</td>"
            "</tr>"
        )
    return (
        f"<div class='card'><h3>{name}</h3>"
        "<table><thead><tr>"
        "<th>슬리피지/편도</th><th>총수익</th><th>연수익</th>"
        "<th>Sharpe</th><th>MDD</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build_tables_html(
    strategy_results: list[PortfolioResult],
    ret_wide: pd.DataFrame,
    spy_stats: dict[str, float],
) -> str:
    """SPY 요약 1줄 + 전략별 슬리피지 성과표 블록을 만든다."""
    spy_line = (
        "<div class='spy'>벤치마크 · <b>SPY Buy&amp;Hold</b> "
        f"(슬리피지 무관): 총수익 {_fmt_pct(spy_stats['total_return'])} · "
        f"연수익 {_fmt_pct(spy_stats['annual_return'])} · "
        f"Sharpe {_fmt_num(spy_stats['sharpe'])} · "
        f"MDD {_fmt_pct(spy_stats['mdd'])}</div>"
    )
    tables = "".join(
        _slippage_table_html(r.name, r.weights, ret_wide) for r in strategy_results
    )
    style = (
        "<style>"
        ".dash-tables{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
        "max-width:1100px;margin:8px auto 4px;padding:0 16px;color:#222}"
        ".dash-tables h2{font-size:16px;margin:6px 0 10px}"
        ".dash-tables .spy{background:#f4f6f8;border:1px solid rgba(44,62,80,.15);"
        "border-radius:6px;padding:8px 12px;font-size:12.5px;margin-bottom:12px}"
        ".dash-tables .cards{display:flex;flex-wrap:wrap;gap:16px}"
        ".dash-tables .card{flex:1 1 320px;border:1px solid rgba(44,62,80,.15);"
        "border-radius:6px;padding:10px 12px}"
        ".dash-tables .card h3{font-size:13.5px;margin:2px 0 8px;color:#2c3e50}"
        ".dash-tables table{border-collapse:collapse;width:100%;font-size:12.5px}"
        ".dash-tables th,.dash-tables td{padding:5px 8px;text-align:right;"
        "border-bottom:1px solid rgba(0,0,0,.08)}"
        ".dash-tables th{background:#2c3e50;color:#fff;font-weight:600}"
        ".dash-tables td.lbl{text-align:left;font-weight:600}"
        ".dash-tables tbody tr:hover{background:#f0f4f8}"
        ".dash-tables .sub{color:#888;font-size:11px;margin-left:4px}"
        ".dash-tables .note{color:#888;font-size:11px;margin-top:8px}"
        "</style>"
    )
    return (
        f"{style}<div class='dash-tables'>"
        "<h2>슬리피지 민감도 (편도 bp · 수수료 0 가정)</h2>"
        f"{spy_line}<div class='cards'>{tables}</div></div>"
    )


def build_capm_table_html(
    strategy_results: list[PortfolioResult],
    market_daily: pd.Series,
) -> str:
    """방식별 CAPM 회귀 분해표(시장 추종도 + 베타/알파 기여분)를 만든다."""
    decomps = [
        (r.name, capm_decomposition(r.daily_returns, market_daily))
        for r in strategy_results
    ]

    header_cells = "".join(f"<th>{name}</th>" for name, _ in decomps)

    def contrib_cell(value: float, share: float) -> str:
        return f"{_fmt_pct(value)}<span class='sub'>({_fmt_pct(share)})</span>"

    metric_rows = [
        ("실현 베타 (β)", lambda d: _fmt_num(d["beta"])),
        ("연율 알파 (α)", lambda d: _fmt_pct(d["alpha_annual"])),
        ("상관계수 (ρ)", lambda d: _fmt_num(d["corr"])),
        ("결정계수 (R²)", lambda d: _fmt_num(d["r2"])),
        ("시장(β) 기여분", lambda d: contrib_cell(d["mkt_contrib"], d["mkt_share"])),
        ("알파(α) 기여분", lambda d: contrib_cell(d["alpha_contrib"], d["alpha_share"])),
    ]
    body = ""
    for label, fn in metric_rows:
        cells = "".join(f"<td>{fn(d)}</td>" for _, d in decomps)
        body += f"<tr><td class='lbl'>{label}</td>{cells}</tr>"

    return (
        "<div class='dash-tables'>"
        "<h2>CAPM 회귀 분해 (SPY 대비 · 무비용 일별수익 기준)</h2>"
        "<div class='card'><table><thead><tr><th>지표</th>"
        f"{header_cells}</tr></thead><tbody>{body}</tbody></table>"
        "<div class='note'>기여분 = 누적수익(단리 합)을 시장·알파로 분해, "
        "괄호는 총수익 대비 비중. 단일팩터 근사치이며 알파엔 종목선택 외 "
        "다른 팩터 노출도 섞여 있음.</div></div></div>"
    )


def main() -> None:
    df = load()
    data = to_wide(df)
    fig, strategy_results, benchmark = make_figure(data, STRATEGIES)

    tables_html = build_tables_html(
        strategy_results, data.ret_wide, benchmark["stats"]
    )
    capm_html = build_capm_table_html(
        strategy_results, benchmark["daily_returns"]
    )
    fig_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>포트폴리오 OOS 대시보드</title></head>"
        f"<body style='margin:0;background:#fff'>{tables_html}{capm_html}{fig_html}</body></html>"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(page, encoding="utf-8")

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Universe size: {len(data.universe)}")
    beta_by_date = benchmark["beta_by_date"]
    print(
        f"Beta: rolling {BETA_LOOKBACK_DAYS} trading days, "
        f"min_obs {BETA_MIN_OBS}, no lookahead"
    )
    for result in strategy_results:
        max_abs_net_beta = net_beta_series(result.weights, beta_by_date).abs().max()
        print(
            f"{result.name}: Sharpe {result.stats['sharpe']:.4f}, "
            f"MDD {result.stats['mdd']:.4%}, "
            f"Total {result.stats['total_return']:.4%}, "
            f"Annual {result.stats['annual_return']:.4%}, "
            f"MaxAbsNetBeta {max_abs_net_beta:.4f}"
        )
    spy_stats = benchmark["stats"]
    print(
        f"SPY Buy&Hold: Sharpe {spy_stats['sharpe']:.4f}, "
        f"MDD {spy_stats['mdd']:.4%}, "
        f"Total {spy_stats['total_return']:.4%}, "
        f"Annual {spy_stats['annual_return']:.4%}"
    )


if __name__ == "__main__":
    main()
