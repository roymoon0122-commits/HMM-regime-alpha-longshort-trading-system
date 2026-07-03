"""
ADX, R² 지표 계산 (HMM_strategy 패키지 내 자급자족 구현).

────────────────────────────────────────────────────────────────────
설계 의도
────────────────────────────────────────────────────────────────────
기존 strategy/filters/adx.py, r2_filter.py 와는 별개의 구현.
HMM_strategy 패키지가 외부 모듈에 의존하지 않게 해서:
  - ETH/SOL/XRP 확장 시 영향 범위 한정
  - 파라미터 튜닝 시 부작용 격리

────────────────────────────────────────────────────────────────────
look-ahead bias 처리
────────────────────────────────────────────────────────────────────
이 파일의 함수들은 shift(1)을 적용하지 않는다.
  - 함수 내부에서: 인덱스 i의 결과는 "i봉까지의 데이터"로 계산된 값
  - 시점 정렬 책임은 호출하는 쪽 (window_features.py)에 있음
  - 이렇게 하면 같은 함수를 다양한 맥락에서 재사용 가능

────────────────────────────────────────────────────────────────────
파라미터 기본값
────────────────────────────────────────────────────────────────────
donchian_adx_r2_B.py (정확히는 strategy/filters/adx_and_Rsquare.py)에서
사용하는 값과 일치:
  ADX_PERIOD = 12, R2_PERIOD = 40
"""

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════
#  ADX (Average Directional Index)
# ════════════════════════════════════════════════════════════════

def compute_adx(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """
    ADX (Average Directional Index) 계산.

    ADX는 추세의 "강도"를 측정 (방향과 무관).
      - ADX > 25 : 추세 존재 (강세장 또는 약세장)
      - ADX < 20 : 횡보 (방향성 없음)

    계산 단계:
      1. True Range (TR)        : 봉의 실제 변동폭
      2. +DM / -DM              : 방향성 이동 (상승/하락)
      3. Smoothed TR / ±DM      : 지수 평활 (period봉)
      4. +DI / -DI              : 방향성 지수
      5. DX                     : (+DI - -DI) / (+DI + -DI) × 100
      6. ADX                    : DX의 지수 평활

    Args:
        df: 'high', 'low', 'close' 컬럼이 있는 DataFrame.
        period: 평활 기간 (기본 12).

    Returns:
        pd.Series: ADX 값 (df와 같은 인덱스).
                   초반 워밍업 구간(약 2 × period봉)은 NaN.
    """
    high  = df['high'].astype(float)
    low   = df['low'].astype(float)
    close = df['close'].astype(float)

    # ── 1. True Range (TR) ───────────────────────────────────
    # TR = max(고가-저가, |고가-전일종가|, |저가-전일종가|)
    # 갭 발생 시에도 정확한 변동폭을 잡기 위함
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ── 2. Directional Movement (+DM, -DM) ──────────────────
    # 상승폭이 하락폭보다 크면 → +DM, 반대면 → -DM
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # ── 3. Wilder 평활 (지수 가중 이동평균의 한 종류) ──────
    # alpha = 1/period 인 EMA와 동등 (Wilder 원래 공식)
    # pandas의 ewm(alpha=1/period, adjust=False)가 정확히 이걸 수행
    atr      = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    # ── 4. DX → ADX ─────────────────────────────────────────
    # DX = |+DI - -DI| / (+DI + -DI) × 100  → 0 ~ 100
    di_sum  = plus_di + minus_di
    # 0으로 나누기 방어:
    #   di_sum == 0 (가격 완전 정체) → 추세 강도 = 0 (NaN 아님)
    #   이렇게 처리해야 dropna 후에도 데이터가 유지됨
    di_diff = (plus_di - minus_di).abs()
    dx_raw = np.where(
        di_sum.values > 0,
        100 * di_diff.values / np.where(di_sum.values > 0, di_sum.values, 1),
        0.0,
    )
    dx = pd.Series(dx_raw, index=df.index)

    # ADX = DX의 평활
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    # ── 5. 워밍업 구간 NaN 처리 ─────────────────────────────
    # 초반 period봉은 평활값이 안정되지 않음 → 명시적으로 NaN 처리
    # (Wilder 원래 공식은 첫 period봉을 단순 평균으로 시작하지만,
    #  여기서는 단순화를 위해 첫 period봉을 NaN으로 마스킹)
    adx.iloc[:period] = np.nan

    return adx


# ════════════════════════════════════════════════════════════════
#  R² (Coefficient of Determination, 선형회귀 결정계수)
# ════════════════════════════════════════════════════════════════

def compute_r2(df: pd.DataFrame, period: int = 40) -> pd.Series:
    """
    R² (선형회귀 결정계수) 계산.

    R²는 가격 움직임이 얼마나 "직선적인가"를 측정.
      - R² ≈ 1.0 : 완벽한 직선 추세 (강한 추세)
      - R² ≈ 0.0 : 무작위 (횡보)
      - 방향(상승/하락)과는 무관

    각 시점 i에 대해:
      x = [0, 1, 2, ..., period-1]
      y = close[i-period+1 : i+1]
      → 선형 회귀 후 R² 계산

    Args:
        df: 'close' 컬럼이 있는 DataFrame.
        period: 회귀 윈도우 크기 (기본 40).

    Returns:
        pd.Series: R² 값 (0.0 ~ 1.0). 초반 period-1봉은 NaN.
    """
    close = df['close'].astype(float).values
    n = len(close)
    r2 = np.full(n, np.nan, dtype=np.float64)

    # x축은 모든 윈도우에서 동일 (0, 1, ..., period-1)
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    # Sxx = Σ(x - x_mean)² — x축 분산 합 (윈도우마다 동일)
    sxx = ((x - x_mean) ** 2).sum()

    # 각 시점 i에서 윈도우를 잘라 R² 계산
    # (벡터화 가능하지만, 가독성 우선으로 명시적 루프 사용)
    for i in range(period - 1, n):
        y = close[i - period + 1 : i + 1]
        y_mean = y.mean()

        # 회귀 기울기 b = Sxy / Sxx
        sxy = ((x - x_mean) * (y - y_mean)).sum()

        # SS_tot = Σ(y - y_mean)² — 총 변동
        ss_tot = ((y - y_mean) ** 2).sum()

        if ss_tot == 0:
            # 가격이 완전히 일정한 경우 → 변동 없음 → R² = 1로 정의
            # (선형 모델이 완벽히 설명함)
            r2[i] = 1.0
        else:
            # SS_res = Σ(y - y_pred)² — 잔차 제곱합
            # 선형 회귀의 경우: SS_res = SS_tot - b² × Sxx
            # → R² = 1 - SS_res/SS_tot = b² × Sxx / SS_tot
            b = sxy / sxx
            r2[i] = (b ** 2) * sxx / ss_tot

    return pd.Series(r2, index=df.index)


# ════════════════════════════════════════════════════════════════
#  Slope (선형회귀 기울기) — window_features의 'slope' 피처용
# ════════════════════════════════════════════════════════════════

def compute_slope(prices: np.ndarray) -> float:
    """
    가격 배열에 대한 선형회귀 기울기를 계산.

    "방향 + 속도"를 동시에 표현:
      - 양수 : 상승 추세
      - 음수 : 하락 추세
      - 절댓값 : 단위 봉당 가격 변화량

    Args:
        prices: 1차원 numpy 배열.

    Returns:
        float: 회귀 기울기.

    Note:
        window_features.py에서 윈도우 단위로 호출됨.
        (전체 시계열에 대한 rolling slope는 별도 구현하지 않음 — 윈도우
         피처 계산 시점에 한 번만 필요하므로)
    """
    n = len(prices)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = prices.mean()

    sxx = ((x - x_mean) ** 2).sum()
    sxy = ((x - x_mean) * (prices - y_mean)).sum()

    if sxx == 0:
        return 0.0
    return sxy / sxx
