"""
거래량(volume) 기반 윈도우 피처 — 탈시즌화 RVOL 위에 구축.

────────────────────────────────────────────────────────────────────
왜 RVOL(Relative Volume, 상대거래량)인가
────────────────────────────────────────────────────────────────────
30분봉 raw 거래량은 장중 U자형 시즌성에 지배된다.
  - 09:30 개장봉 / 15:30 마감봉 → 항상 거래량 폭발
  - 점심시간 봉 → 항상 바닥
그대로 피처로 쓰면 모델이 "거래가 활발한가"가 아니라
"지금이 몇 시 봉인가"를 학습해버린다 (국면과 무관한 노이즈).

해법: 각 봉을 "같은 시간대(time-of-day)의 과거 평소 거래량"으로 나눈다.

    RVOL_t = volume_t / (그 시간대 봉의 과거 N거래일 중앙값)

  - 09:30봉은 과거 09:30봉들의 중앙값으로 정규화
  - 점심봉은 점심봉 기준으로 정규화
  → U자형 시즌성 제거, "이 시각 치고 평소보다 붐비나?"만 남음.

분포: raw 거래량은 right-skew가 심함(로그정규). RVOL은 비율이라
1.0 중심으로 분포하고, log를 한 번 더 씌우면 0 중심 대칭에 가까워진다.
→ 본 모듈은 log(RVOL)을 기본 단위로 사용.

────────────────────────────────────────────────────────────────────
룩어헤드 바이어스 처리 (핵심)
────────────────────────────────────────────────────────────────────
1. 시즌 기준선(같은 시간대 과거 거래량)은 **현재 봉 이전**만 사용한다.
   - 같은 슬롯끼리 묶어 rolling(lookback_days).median() 후 shift(1).
   - shift(1) → 오늘 같은 슬롯 값은 자기 기준선에서 제외 (표준 RVOL 정의).
   - 미래의 같은-시간대 봉은 절대 포함되지 않음.
2. 윈도우 요약 피처는 compute_window_features와 동일한 인덱싱을 따른다.
   - window_end_idx = i  →  "봉 i까지의 정보로 계산된 피처"
   - 봉 i+1 의사결정에 사용 (EngineHMM이 1봉 시프트 처리).

────────────────────────────────────────────────────────────────────
이 모듈의 위치 (설계 의도)
────────────────────────────────────────────────────────────────────
거래량 피처는 HMM 클러스터링(국면 정의)에는 넣지 않는다.
  - HMM은 cum_return 순서로 Bull/Side/Bear를 매핑 → 거래량 축이 끼면
    국면 매핑이 불안정해짐.
거래량 피처는 meta-model 입력에만 넣는다 (supervised, 강건).
  → 국면 정의는 그대로 두고 "이 국면이 진짜인지 거래량으로 보정".
따라서 본 모듈은 compute_window_features와 **분리된** 독립 함수로 제공되고,
window_end_idx로 정렬해 meta 입력에서 합쳐진다.
"""

import numpy as np
import pandas as pd

from strategy.HMM_strategy.features.indicators import compute_slope


# 반환 DataFrame의 거래량 피처 컬럼 순서 (메타데이터 컬럼 제외)
VOLUME_FEATURE_COLUMNS = [
    'rvol_mean',        # 윈도우 내 log(RVOL) 평균 — 평소 대비 참여도 수준
    'rvol_slope',       # 윈도우 내 log(RVOL) 기울기 — 참여가 늘고 있나 빠지나
    'vol_price_corr',   # 봉별 수익률과 log(RVOL)의 상관 — 누적(+)/분산(-) 틸트
]


def compute_log_rvol(
    df: pd.DataFrame,
    lookback_days: int = 20,
    clip: float = 3.0,
) -> pd.Series:
    """
    탈시즌화된 log(RVOL) 시계열을 계산한다.

    각 봉을 같은 시간대(time-of-day) 슬롯으로 묶고, 그 슬롯의 과거
    lookback_days 거래일 거래량 중앙값으로 나눠 RVOL을 구한 뒤 log 변환.

    Args:
        df:
            'datetime'과 'volume' 컬럼이 있는 DataFrame.
            시간 오름차순 정렬되어 있어야 한다 (정규장 30분봉 가정).
        lookback_days:
            시즌 기준선 계산에 쓸 과거 거래일 수 (기본 20 ≈ 1개월).
        clip:
            log(RVOL)을 [-clip, +clip]으로 winsorize (기본 3.0 = RVOL 0.05~20배).
            반일장(조기폐장) 오후 슬롯은 거래량이 거의 0이라 RVOL이 0에 수렴,
            log가 -7까지 박히는 데이터 아티팩트가 발생한다. 이는 신호가 아니라
            노이즈이므로 잘라낸다. 실적일 거래량 폭증(3~5배=log 1.1~1.6)이나
            대형 뉴스(10배=log 2.3)는 clip=3.0 안쪽이라 보존됨.
            None이면 클립하지 않음.

    Returns:
        pd.Series:
            log(RVOL) 값 (df와 같은 인덱스).
            기준선이 부족한 초반 워밍업 구간(슬롯별 lookback_days일)은 NaN.

    룩어헤드 안전:
        슬롯별 rolling median에 shift(1)을 적용 → 현재 봉(및 미래 봉)은
        자신의 기준선 계산에서 제외된다.

    Example:
        >>> log_rvol = compute_log_rvol(df_30min, lookback_days=20)
        >>> # log_rvol > 0 : 그 시간대 치고 평소보다 활발
        >>> # log_rvol < 0 : 평소보다 한산
    """
    # ── 0. 입력 검증 ────────────────────────────────────────────
    required = {'datetime', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    if lookback_days < 1:
        raise ValueError(f"lookback_days는 1 이상이어야 합니다: {lookback_days}")

    work = df.reset_index(drop=True).copy()
    work['volume'] = work['volume'].astype(float)

    # ── 1. 시간대(슬롯) 식별 ────────────────────────────────────
    # 같은 시각(예: 09:30)끼리 한 그룹. 반일장(half-day)은 오후 슬롯이
    # 그냥 빠질 뿐, 존재하는 슬롯은 정상일의 같은-슬롯 이력과 매칭됨.
    dt = pd.to_datetime(work['datetime'])
    slot = dt.dt.strftime('%H:%M')   # 시간대 키 (분 단위)

    # ── 2. 슬롯별 과거 중앙값 기준선 (룩어헤드 안전) ────────────
    # groupby(slot).rolling은 그룹 내에서만 시간순 누적된다.
    # 슬롯 그룹의 각 행은 "그날 그 시각" → rolling(lookback_days)는
    # 과거 lookback_days일의 같은-슬롯 거래량. shift(1)로 현재일 제외.
    #
    # min_periods=lookback_days : 기준선이 충분히 안정되기 전엔 NaN.
    baseline = (
        work.groupby(slot, sort=False)['volume']
        .apply(
            lambda s: s.rolling(window=lookback_days, min_periods=lookback_days)
                       .median()
                       .shift(1)
        )
    )
    # groupby.apply는 (slot, 원래인덱스) MultiIndex 또는 정렬된 Series를
    # 반환할 수 있어 원래 인덱스 순서로 재정렬한다.
    baseline = baseline.reset_index(level=0, drop=True).sort_index()

    # ── 3. RVOL → log(RVOL) ────────────────────────────────────
    # 기준선이 0이거나 NaN이면 RVOL 정의 불가 → NaN.
    # 거래량 0은 데이터상 없지만(검증 완료), 방어적으로 NaN 처리.
    vol = work['volume'].values
    base = baseline.values
    with np.errstate(divide='ignore', invalid='ignore'):
        rvol = np.where((base > 0) & (vol > 0), vol / base, np.nan)
        log_rvol = np.log(rvol)

    # ── 4. winsorize — 반일장 등 데이터 아티팩트 절단 ──────────
    if clip is not None:
        log_rvol = np.clip(log_rvol, -abs(clip), abs(clip))

    return pd.Series(log_rvol, index=work.index, name='log_rvol')


def compute_volume_window_features(
    df: pd.DataFrame,
    window_size: int = 130,
    step_size: int = 1,
    lookback_days: int = 20,
    clip: float = 3.0,
) -> pd.DataFrame:
    """
    원본 OHLCV DataFrame → 윈도우 단위 거래량 요약 피처 DataFrame.

    compute_window_features와 **동일한 window_end_idx 인덱싱**을 사용하므로
    두 결과를 window_end_idx로 머지하면 시점이 정확히 정렬된다.

    Args:
        df:
            'datetime', 'open', 'high', 'low', 'close', 'volume' 컬럼.
            시간 오름차순 정렬.
        window_size:
            윈도우당 봉 수 (기본 130 — config WINDOW_SIZE와 맞출 것).
        step_size:
            윈도우 이동 간격 (기본 1).
        lookback_days:
            RVOL 시즌 기준선 거래일 수 (기본 20).

    Returns:
        DataFrame:
          - window_end_idx (int)
          - rvol_mean       : 윈도우 내 log(RVOL) 평균
          - rvol_slope      : 윈도우 내 log(RVOL) 선형회귀 기울기
          - vol_price_corr  : 봉별 수익률 vs log(RVOL) 상관
                              ( >0 누적/추세 컨퍼메이션, <0 분산/다이버전스 )
        NaN 행(워밍업 등)은 dropna()로 제거된 상태.

    Note:
        vol_price_corr 의미
          - 양수: 상승봉에 거래량이 실림(또는 하락봉에 빠짐) → 추세 신뢰(누적)
          - 음수: 상승봉이 얇은 거래량 위에서 나옴(또는 하락봉에 거래량 폭발)
                  → 가격-거래량 다이버전스 / 분산 경고
    """
    required = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    if len(df) < window_size:
        raise ValueError(
            f"데이터가 윈도우 크기보다 작습니다: len(df)={len(df)}, "
            f"window_size={window_size}"
        )

    work = df.reset_index(drop=True).copy()

    # ── 1. log(RVOL) 시계열 (룩어헤드 안전) ────────────────────
    log_rvol = compute_log_rvol(work, lookback_days=lookback_days, clip=clip).values

    # ── 2. 봉별 수익률 (다이버전스용) ──────────────────────────
    close = work['close'].astype(float).values
    returns = np.full(len(work), np.nan, dtype=np.float64)
    returns[1:] = close[1:] / close[:-1] - 1.0

    # ── 3. 윈도우 순회 (compute_window_features와 동일 인덱싱) ──
    n = len(work)
    rows = []
    for i in range(window_size - 1, n, step_size):
        win_start = i - window_size + 1
        win_end = i + 1

        win_lrvol = log_rvol[win_start:win_end]
        win_ret = returns[win_start:win_end]

        # 윈도우에 NaN(워밍업 등)이 끼면 피처 불안정 → NaN으로 두고
        # 하류에서 dropna. (compute_window_features의 워밍업 처리와 동일 철학)
        if np.isnan(win_lrvol).any():
            rvol_mean = np.nan
            rvol_slope = np.nan
            vol_price_corr = np.nan
        else:
            rvol_mean = float(np.mean(win_lrvol))
            rvol_slope = compute_slope(win_lrvol)

            # 수익률-거래량 상관: 두 계열 모두 유효한 쌍만 사용
            ret_pair = win_ret
            valid = ~np.isnan(ret_pair)
            if valid.sum() >= 2:
                a = ret_pair[valid]
                b = win_lrvol[valid]
                # 분산 0(상수)이면 상관 정의 불가 → 0(중립)
                if np.std(a) > 0 and np.std(b) > 0:
                    vol_price_corr = float(np.corrcoef(a, b)[0, 1])
                else:
                    vol_price_corr = 0.0
            else:
                vol_price_corr = np.nan

        rows.append({
            'window_end_idx': i,
            'rvol_mean': rvol_mean,
            'rvol_slope': rvol_slope,
            'vol_price_corr': vol_price_corr,
        })

    features_df = pd.DataFrame(rows)
    features_df = features_df[['window_end_idx'] + VOLUME_FEATURE_COLUMNS]
    features_df = features_df.dropna().reset_index(drop=True)

    return features_df
