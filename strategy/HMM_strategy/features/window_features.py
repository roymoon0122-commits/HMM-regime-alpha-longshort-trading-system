"""
롤링 윈도우 단위 9개 요약 피처 계산.

기획서 4-1 (HMM_regime_plan.md) 기반.

────────────────────────────────────────────────────────────────────
피처 목록
────────────────────────────────────────────────────────────────────
  1. cum_return       : 윈도우 내 누적 수익률
  2. volatility       : 봉별 수익률의 표준편차
  3. adx_mean         : 윈도우 내 ADX 평균
  4. r2_mean          : 윈도우 내 R² 평균
  5. adx_end          : 윈도우 마지막 봉의 ADX
  6. r2_end           : 윈도우 마지막 봉의 R²
  7. slope            : 종가에 대한 선형회귀 기울기
  8. max_drawdown     : 윈도우 내 최대 낙폭 (음수)
  9. up_candle_ratio  : 양봉 비율 (0 ~ 1)

────────────────────────────────────────────────────────────────────
인덱스 의미 (look-ahead bias 핵심)
────────────────────────────────────────────────────────────────────
반환 DataFrame의 각 행:
  - window_end_idx = i  →  "원본 df의 i번째 봉까지의 정보로 계산된 피처"
  - 즉 i번째 봉의 종가까지 본 상태
  - 이 피처는 i+1번째 봉부터 사용 가능 (i번째 봉 의사결정에는 사용 불가)
  - Phase 2에서 라벨 매칭 시 shift(-1) 등으로 시점 정렬

────────────────────────────────────────────────────────────────────
워밍업 처리
────────────────────────────────────────────────────────────────────
처음 약 (window_size + max(adx_period, r2_period)) 봉은 NaN이 발생.
계산 후 dropna()로 깨끗한 행만 반환.
"""

import numpy as np
import pandas as pd

from strategy.HMM_strategy.features.indicators import (
    compute_adx,
    compute_r2,
    compute_slope,
)


# 반환 DataFrame의 피처 컬럼 순서 (메타데이터 컬럼 제외)
FEATURE_COLUMNS = [
    'cum_return',
    'volatility',
    'adx_mean',
    'r2_mean',
    'adx_end',
    'r2_end',
    'slope',
    'max_drawdown',
    'up_candle_ratio',
]


def compute_window_features(
    df: pd.DataFrame,
    window_size: int = 60,
    step_size: int = 1,
    adx_period: int = 12,
    r2_period: int = 40,
) -> pd.DataFrame:
    """
    원본 OHLCV DataFrame → 윈도우 단위 피처 DataFrame.

    Args:
        df:
            'open', 'high', 'low', 'close' 컬럼이 있는 DataFrame.
            (선택적으로 'datetime' 컬럼이 있으면 결과에 시각 정보 포함)
            시간 오름차순 정렬되어 있어야 함.

        window_size:
            윈도우당 봉 수 (기본 60 = 4시간봉 기준 10일).
            인자로 가변 — 호출 시점에 자유롭게 변경 가능.

        step_size:
            윈도우 이동 간격 (기본 1 = 매 봉마다 새 윈도우).

        adx_period:
            ADX 계산 기간 (기본 12).

        r2_period:
            R² 계산 기간 (기본 40).

    Returns:
        DataFrame:
          - window_end_idx (int)        : 윈도우 마지막 봉의 원본 df 인덱스
          - window_end_time (datetime)  : 윈도우 마지막 봉 시각 (df에 datetime 컬럼이 있을 때만)
          - cum_return, volatility, adx_mean, r2_mean,
            adx_end, r2_end, slope, max_drawdown, up_candle_ratio
        모든 NaN 행은 dropna()로 제거된 상태.

    Example:
        >>> features = compute_window_features(df_4h, window_size=60)
        >>> features[['window_end_time', 'cum_return', 'adx_mean']].head()
    """
    # ── 0. 입력 검증 ────────────────────────────────────────────
    required_cols = {'open', 'high', 'low', 'close'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    if len(df) < window_size:
        raise ValueError(
            f"데이터가 윈도우 크기보다 작습니다: len(df)={len(df)}, "
            f"window_size={window_size}"
        )

    # 작업용 복사본 (원본 df 변경 방지)
    df = df.reset_index(drop=True).copy()

    has_datetime = 'datetime' in df.columns

    # ── 1. 전체 시계열에 대해 ADX, R² 미리 계산 ─────────────────
    # 윈도우 안에서 매번 다시 계산하지 않고 한 번만 계산해서 슬라이싱
    # → 속도 향상 + 코드 단순화
    adx_series = compute_adx(df, period=adx_period).values
    r2_series  = compute_r2(df, period=r2_period).values

    # 봉별 수익률 (close[i] / close[i-1] - 1)
    close = df['close'].astype(float).values
    open_ = df['open'].astype(float).values
    returns = np.full(len(df), np.nan, dtype=np.float64)
    returns[1:] = close[1:] / close[:-1] - 1.0

    # 양봉 여부 (close > open)
    is_up_candle = (close > open_).astype(np.float64)

    # ── 2. 윈도우 순회하며 9개 피처 계산 ────────────────────────
    n = len(df)
    rows = []

    # i는 "윈도우 마지막 봉 인덱스"
    # 첫 윈도우: i = window_size - 1 (0~window_size-1 봉을 묶음)
    for i in range(window_size - 1, n, step_size):
        win_start = i - window_size + 1  # 윈도우 첫 봉 인덱스
        win_end   = i + 1                # 슬라이싱 종료 (i 포함)

        # 윈도우 내 데이터 슬라이싱
        win_close   = close[win_start:win_end]
        win_returns = returns[win_start + 1:win_end]   # 첫 봉의 수익률은 NaN이라 제외
        win_adx     = adx_series[win_start:win_end]
        win_r2      = r2_series[win_start:win_end]
        win_up      = is_up_candle[win_start:win_end]

        # ── 피처 1: 누적 수익률 ────────────────────────────────
        # 윈도우 첫 봉 → 마지막 봉의 종가 변화율
        cum_return = win_close[-1] / win_close[0] - 1.0

        # ── 피처 2: 변동성 (봉별 수익률의 표준편차) ────────────
        # ddof=1 = 표본 표준편차 (n-1로 나눔)
        if len(win_returns) > 1:
            volatility = float(np.std(win_returns, ddof=1))
        else:
            volatility = np.nan

        # ── 피처 3, 5: ADX 평균 / 끝값 ─────────────────────────
        # nanmean 사용 — 워밍업 NaN 영향 배제
        adx_mean = float(np.nanmean(win_adx)) if not np.isnan(win_adx).all() else np.nan
        adx_end  = float(win_adx[-1])

        # ── 피처 4, 6: R² 평균 / 끝값 ──────────────────────────
        r2_mean = float(np.nanmean(win_r2)) if not np.isnan(win_r2).all() else np.nan
        r2_end  = float(win_r2[-1])

        # ── 피처 7: slope (선형회귀 기울기) ───────────────────
        slope = compute_slope(win_close)

        # ── 피처 8: max_drawdown (윈도우 내 최대 낙폭) ────────
        # 누적 최고가 대비 현재가의 비율 → 최댓값을 음수로 표현
        running_max = np.maximum.accumulate(win_close)
        drawdowns = win_close / running_max - 1.0   # 0 또는 음수
        max_drawdown = float(drawdowns.min())

        # ── 피처 9: 양봉 비율 ─────────────────────────────────
        up_candle_ratio = float(win_up.mean())

        # ── 행 구성 ────────────────────────────────────────────
        row = {
            'window_end_idx':  i,
            'cum_return':      cum_return,
            'volatility':      volatility,
            'adx_mean':        adx_mean,
            'r2_mean':         r2_mean,
            'adx_end':         adx_end,
            'r2_end':          r2_end,
            'slope':           slope,
            'max_drawdown':    max_drawdown,
            'up_candle_ratio': up_candle_ratio,
        }
        if has_datetime:
            row['window_end_time'] = df['datetime'].iloc[i]

        rows.append(row)

    # ── 3. DataFrame 구성 ──────────────────────────────────────
    features_df = pd.DataFrame(rows)

    # 컬럼 순서 정리 (메타 → 피처 순서)
    meta_cols = ['window_end_idx']
    if has_datetime:
        meta_cols.append('window_end_time')
    features_df = features_df[meta_cols + FEATURE_COLUMNS]

    # ── 4. 워밍업 NaN 제거 ─────────────────────────────────────
    # ADX/R² 워밍업 구간이 윈도우에 걸리면 adx_mean, r2_mean 등이 NaN
    features_df = features_df.dropna().reset_index(drop=True)

    return features_df
