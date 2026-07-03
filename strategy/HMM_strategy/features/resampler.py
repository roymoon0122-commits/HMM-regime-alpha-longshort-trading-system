"""
1분봉 CSV → 임의 타임프레임 OHLCV DataFrame 변환.

기존 run_backtest.py의 인라인 리샘플링 로직을 함수화한 것.
타임프레임을 인자로 받아 1H/4H/1D 등 자유롭게 실험 가능.

look-ahead bias 관점:
  - 리샘플링은 "과거 1분봉 N개를 묶어서 1개의 큰 봉"을 만드는 작업
  - 미래 데이터를 사용하지 않으므로 룩어헤드 위험 없음
  - 단, 리샘플링 후 봉의 timestamp가 "구간 시작 시각"인지 "종료 시각"인지
    의식해서 사용할 것 (pandas resample 기본은 "구간 시작 시각")
"""

import pandas as pd


def load_and_resample(
    csv_path: str,
    timeframe: str = '4H',
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    1분봉 CSV 파일을 읽어 지정된 타임프레임으로 리샘플링한다.

    Args:
        csv_path:
            1분봉 데이터 CSV 경로.
            CSV에는 'datetime', 'open', 'high', 'low', 'close', 'volume'
            컬럼이 있어야 한다.

        timeframe:
            pandas resample 문법 문자열.
            예: '1H' (1시간봉), '4H' (4시간봉), '1D' (일봉)
            기본값 '4H'는 기획서의 기준 타임프레임.

        start:
            데이터 시작 시각 문자열 (예: '2020-01-01').
            None이면 CSV의 처음부터 사용.

        end:
            데이터 종료 시각 문자열 (예: '2025-12-31').
            None이면 CSV의 끝까지 사용.

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
        - datetime: 각 봉의 "시작 시각" (pandas resample 기본 동작)
        - 시간 오름차순 정렬됨
        - 데이터가 없는 구간(NaN 봉)은 제거됨

    Example:
        >>> df = load_and_resample("data/historical/BTC_USDT_1m.csv",
        ...                        timeframe='4H',
        ...                        start='2024-01-01',
        ...                        end='2024-12-31')
        >>> df.head()
    """
    # ── 1. CSV 로드 ─────────────────────────────────────────────
    # parse_dates: 'datetime' 컬럼을 문자열 → datetime 객체로 자동 변환
    df = pd.read_csv(csv_path, parse_dates=['datetime'])

    # ── 2. 기간 필터링 (리샘플링 전에 미리 잘라서 메모리 절약) ───
    if start is not None:
        df = df[df['datetime'] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df['datetime'] <= pd.Timestamp(end)]

    # 빈 DataFrame 방어
    if len(df) == 0:
        raise ValueError(
            f"필터링 후 데이터가 없습니다. start={start}, end={end} 확인 필요"
        )

    # ── 3. 리샘플링 ─────────────────────────────────────────────
    # datetime을 인덱스로 설정해야 resample() 호출 가능
    df = df.set_index('datetime')

    # OHLCV 각 컬럼별 집계 방식:
    #   open   : 구간 첫 값
    #   high   : 구간 최대값
    #   low    : 구간 최소값
    #   close  : 구간 마지막 값
    #   volume : 구간 합계
    df_resampled = df.resample(timeframe).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    })

    # ── 4. NaN 봉 제거 (데이터 공백 구간) ───────────────────────
    # 거래소 점검 등으로 1분봉이 비어있으면 그 4시간봉은 NaN이 됨
    df_resampled = df_resampled.dropna()

    # ── 5. 인덱스 → 컬럼으로 복원 ──────────────────────────────
    df_resampled = df_resampled.reset_index()

    return df_resampled
