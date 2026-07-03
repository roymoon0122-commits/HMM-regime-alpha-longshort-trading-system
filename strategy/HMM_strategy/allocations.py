"""
allocations.py — 포트폴리오 종목별 자본 배분비율(allocation) 관리.

────────────────────────────────────────────────────────────────────
이 파일의 목적
────────────────────────────────────────────────────────────────────
지금 live_trade.py는 자본을 종목 수로 똑같이 나눈다(등가중):
    budget = equity / len(SYMBOLS)

장기적으로는 "종목마다 다른 비중을 부여하는 알파"로 확장할 예정이다.
그때 집행 코드(live_trade.py)를 건드리지 않고 배분비율만 갈아끼울 수 있도록,
배분 로직을 이 파일 한 곳에 모아 둔다.

★ 배분비율 자체는 아직 자리표시자(placeholder)다. 기본 동작은 등가중이며,
  본격적인 차등 비중 알파는 나중에 여기서 구현한다.
  단, live 순베타 cap 계산/적용에 필요한 순수 포트폴리오 함수는 이 파일에 둔다.

────────────────────────────────────────────────────────────────────
설계 원칙 (config.py 규약과 동일)
────────────────────────────────────────────────────────────────────
- 이 파일은 "배분비율의 출처(source of truth)" 역할만 한다.
- 호출하는 쪽(live_trade.py)이 get_budgets()를 호출해 종목별 예산을 받는다.
- 배분비율은 세 가지 방식으로 줄 수 있게 설계한다:
    1) None              → 등가중 (현재 기본)
    2) 정적 dict          → SYMBOL_ALLOCATIONS 에 {"AAPL": 0.2, ...} 처럼 고정
    3) 동적 dict(미래)    → 알파가 매 사이클 계산한 비율을 인자로 주입
  → 어느 방식이든 get_budgets()의 반환 형태(=종목별 달러 예산 dict)는 동일.

────────────────────────────────────────────────────────────────────
용어
────────────────────────────────────────────────────────────────────
- allocation(배분비율): 각 종목에 자본의 몇 %를 "할당"할지. 합이 1이면 풀투자.
  (신호 방향/크기와는 별개. 방향·강도는 전략 시그널 -1~+1 이 결정한다.)
- budget(예산): equity × allocation. 실제로 그 종목에 굴릴 달러 금액.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd


# ─── 정적 배분비율 (배분 알파 자리표시자) ─────────────────────────
# None  → 등가중 (종목 수로 균등 분배). 현재 기본.
# dict  → 종목별 고정 비율. 예: {"AAPL": 0.25, "MSFT": 0.15, ...}
#         합이 1을 넘으면 레버리지가 되므로 NORMALIZE_ALLOCATIONS 로 통제.
#
# ※ 본격적인 차등 비중 알파는 나중에 이 값을 동적으로 산출하도록 확장한다.
SYMBOL_ALLOCATIONS: Optional[Dict[str, float]] = None

# 배분비율 합이 1을 넘을 때(=레버리지) 처리.
#   True  → 합으로 나눠 정규화(합=1, 레버리지 없음)
#   False → 입력 비율을 그대로 사용(의도적 레버리지 허용)
NORMALIZE_ALLOCATIONS: bool = True


def get_allocations(
    symbols: List[str],
    allocations: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """종목 → 배분비율(0~1) dict 반환.

    Args:
        symbols:     대상 종목 리스트.
        allocations: None 이면 등가중. dict 이면 그 비율 사용
                     (symbols 에 있는데 dict 에 없는 종목은 0 으로 간주).
        normalize:   True 면 비율 합이 1을 넘을 때 합으로 나눠 정규화.

    Returns:
        {symbol: ratio} — 모든 symbols 에 대해 키가 존재.
    """
    if not symbols:
        return {}

    if allocations is None:
        w = 1.0 / len(symbols)
        return {s: w for s in symbols}

    ratios = {s: float(allocations.get(s, 0.0)) for s in symbols}
    total = sum(ratios.values())
    if normalize and total > 1.0 and total > 0:
        ratios = {s: r / total for s, r in ratios.items()}
    return ratios


def get_budgets(
    equity: float,
    symbols: List[str],
    allocations: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """종목 → 달러 예산(budget) dict 반환.

    budget[sym] = equity × allocation[sym]

    live_trade.py 는 현재 get_allocations()로 raw portfolio weight를 만들지만,
    예산 dict가 필요한 호출부는 이 함수로 같은 allocation source를 재사용할 수 있다.
    """
    ratios = get_allocations(symbols, allocations, normalize)
    return {s: equity * r for s, r in ratios.items()}


@dataclass(frozen=True)
class BetaCapResult:
    """순베타 캡 적용 결과."""

    adjusted_weights: Dict[str, float]
    raw_net_beta: Optional[float]
    adjusted_net_beta: Optional[float]
    capped: bool
    scale: float
    beta_symbols: List[str]
    missing_beta_symbols: List[str]


def _valid_float(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_beta_map(
    symbols: List[str],
    betas: Mapping[str, float],
) -> Dict[str, float]:
    """유효한 beta 값만 symbols 순서에 맞춰 반환한다."""
    return {
        symbol: float(betas[symbol])
        for symbol in symbols
        if symbol in betas and _valid_float(betas[symbol])
    }


def calculate_net_beta(
    weights: Mapping[str, float],
    betas: Mapping[str, float],
) -> Optional[float]:
    """Σ weight_i × beta_i. 유효 beta가 하나도 없으면 None."""
    total = 0.0
    has_beta = False
    for symbol, weight in weights.items():
        if (
            symbol not in betas
            or not _valid_float(weight)
            or not _valid_float(betas[symbol])
        ):
            continue
        total += float(weight) * float(betas[symbol])
        has_beta = True
    return total if has_beta else None


def apply_net_beta_cap(
    raw_weights: Mapping[str, float],
    betas: Mapping[str, float],
    cap: float,
) -> BetaCapResult:
    """순베타 초과 방향의 beta 기여 포지션만 비례 축소한다.

    - 포지션 부호 반전 없음.
    - 신규 반대 포지션 생성 없음.
    - 총 노출 증가 없음(축소 계수는 0~1).
    - beta가 없는 종목은 cap 계산에서 제외하고 원래 weight를 유지한다.
      live 주문 안정성을 위해 임의 beta fallback으로 목표주수를 흔들지 않는다.
    """
    if cap < 0:
        raise ValueError("cap must be non-negative")

    symbols = list(raw_weights.keys())
    weights = {
        symbol: float(weight) if _valid_float(weight) else 0.0
        for symbol, weight in raw_weights.items()
    }
    beta_map = clean_beta_map(symbols, betas)
    missing_beta_symbols = [symbol for symbol in symbols if symbol not in beta_map]

    raw_net_beta = calculate_net_beta(weights, beta_map)
    if raw_net_beta is None or abs(raw_net_beta) <= cap:
        return BetaCapResult(
            adjusted_weights=weights.copy(),
            raw_net_beta=raw_net_beta,
            adjusted_net_beta=raw_net_beta,
            capped=False,
            scale=1.0,
            beta_symbols=list(beta_map.keys()),
            missing_beta_symbols=missing_beta_symbols,
        )

    contributions = {
        symbol: weights[symbol] * beta
        for symbol, beta in beta_map.items()
    }
    adjusted = weights.copy()
    scale = 1.0

    if raw_net_beta > cap:
        side_symbols = [
            symbol for symbol, contribution in contributions.items()
            if contribution > 0.0
        ]
        side_contribution = sum(contributions[symbol] for symbol in side_symbols)
        if side_contribution > 0:
            scale = 1.0 - (raw_net_beta - cap) / side_contribution
            scale = min(1.0, max(0.0, scale))
            for symbol in side_symbols:
                adjusted[symbol] = weights[symbol] * scale
    else:
        side_symbols = [
            symbol for symbol, contribution in contributions.items()
            if contribution < 0.0
        ]
        side_contribution_abs = -sum(contributions[symbol] for symbol in side_symbols)
        if side_contribution_abs > 0:
            scale = 1.0 - (-cap - raw_net_beta) / side_contribution_abs
            scale = min(1.0, max(0.0, scale))
            for symbol in side_symbols:
                adjusted[symbol] = weights[symbol] * scale

    adjusted_net_beta = calculate_net_beta(adjusted, beta_map)
    return BetaCapResult(
        adjusted_weights=adjusted,
        raw_net_beta=raw_net_beta,
        adjusted_net_beta=adjusted_net_beta,
        capped=True,
        scale=scale,
        beta_symbols=list(beta_map.keys()),
        missing_beta_symbols=missing_beta_symbols,
    )


def estimate_capm_betas_from_daily_closes(
    close_daily: pd.DataFrame,
    symbols: List[str],
    benchmark_symbol: str = "SPY",
    as_of_date=None,
    lookback_days: int = 252,
    min_obs: int = 126,
) -> tuple[Dict[str, float], Dict[str, str]]:
    """오늘 이전 close-to-close 수익률만 사용해 CAPM beta를 추정한다.

    beta_i = Cov(r_i, r_benchmark) / Var(r_benchmark)

    Args:
        close_daily: 일별 종가 wide DataFrame. columns에는 symbols와 benchmark가 필요.
        symbols: beta를 추정할 대상 종목.
        benchmark_symbol: 시장 proxy. 기본 SPY.
        as_of_date: 사이클의 ET 날짜. 이 날짜 이상의 일별 수익률은 제외한다.
        lookback_days: 사용할 최대 일별 수익률 개수.
        min_obs: 종목별 최소 paired 관측치.

    Returns:
        (beta_map, missing_reasons)
    """
    missing_reasons: Dict[str, str] = {}
    if close_daily.empty:
        return {}, {symbol: "daily close frame empty" for symbol in symbols}
    if benchmark_symbol not in close_daily.columns:
        return {}, {
            symbol: f"benchmark {benchmark_symbol} missing"
            for symbol in symbols
        }

    closes = close_daily.sort_index().copy()
    closes.index = pd.DatetimeIndex(pd.to_datetime(closes.index)).normalize()
    closes = closes.groupby(level=0).last()
    if as_of_date is None:
        as_of = closes.index.max() + pd.Timedelta(days=1)
    else:
        as_of = pd.Timestamp(as_of_date).normalize()

    daily_returns = closes.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan)
    hist = daily_returns.loc[daily_returns.index < as_of].tail(lookback_days)
    spy_returns = hist[benchmark_symbol]
    betas: Dict[str, float] = {}

    spy_var = np.var(spy_returns.dropna())
    if not _valid_float(spy_var) or float(spy_var) <= 0:
        return {}, {symbol: "benchmark variance unavailable" for symbol in symbols}

    for symbol in symbols:
        if symbol not in hist.columns:
            missing_reasons[symbol] = "symbol close missing"
            continue
        joined = pd.concat(
            [hist[symbol], spy_returns],
            axis=1,
            keys=["asset", "benchmark"],
        ).dropna()
        if len(joined) < min_obs:
            missing_reasons[symbol] = f"obs {len(joined)} < min_obs {min_obs}"
            continue
        var_benchmark = np.var(joined["benchmark"])
        if not _valid_float(var_benchmark) or float(var_benchmark) <= 0:
            missing_reasons[symbol] = "paired benchmark variance unavailable"
            continue
        betas[symbol] = float(
            np.cov(joined["asset"], joined["benchmark"])[0, 1] / var_benchmark
        )

    return betas, missing_reasons
