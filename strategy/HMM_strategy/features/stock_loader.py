"""
주식 분봉 parquet → 정규장(RTH) 리샘플 OHLCV 변환 어댑터.

────────────────────────────────────────────────────────────────────
왜 이 모듈이 필요한가 (coin features/resampler.py 와의 차이)
────────────────────────────────────────────────────────────────────
코인은 24시간 거래라 1분봉을 그냥 리샘플하면 됐다 (resampler.py).
미국 주식은 다르다:

  - 정규장(RTH, Regular Trading Hours)은 09:30~16:00 ET, 주중에만.
  - Alpaca 분봉 데이터에는 프리장/애프터장(ET 04:00~20:00)이 섞여 있다.
    이걸 그대로 리샘플하면 거래가 거의 없는 시간대 봉이 끼어들어
    변동성·국면 판단이 왜곡된다. → 정규장만 걸러낸 뒤 리샘플한다.
  - 입력 데이터의 timestamp 는 UTC 이고, 정규장 경계는 ET 기준이라
    시간대 변환이 필요하다 (서머타임은 변환 시 자동 처리됨).

룩어헤드(미래 정보 사용) 관점:
  리샘플은 "과거 N개 분봉을 묶어 1개 봉"으로 만드는 작업이라
  미래 데이터를 쓰지 않는다 → 룩어헤드 위험 없음.

────────────────────────────────────────────────────────────────────
입력 형식 (data/minute/*.parquet — Alpaca 다운로드 원본)
────────────────────────────────────────────────────────────────────
  - MultiIndex (symbol, timestamp), timestamp 는 UTC tz-aware
  - 컬럼: open, high, low, close, volume, trade_count, vwap
  - 파일당 1종목

출력 형식 (coin features/resampler.py 의 load_and_resample 과 동일)
  - 컬럼: datetime, open, high, low, close, volume
  - datetime: 각 봉의 "시작 시각", tz 없는(naive) 미국 동부시간
  - 시간 오름차순, 빈 봉(밤·주말·갭) 제거됨

────────────────────────────────────────────────────────────────────
타임프레임 권장값
────────────────────────────────────────────────────────────────────
정규장 6.5시간(09:30~16:00)은 '30min' 으로 나누면 09:30, 10:00 …
15:30 — 하루 정확히 13봉으로 떨어진다. '1h' 는 09:30 시작 때문에
첫 봉이 반토막(09:30~10:00)이 되므로 '30min' 을 권장한다.
"""

import datetime as _dt

import pandas as pd


# ─── 미국 정규장 (Regular Trading Hours) ─────────────────────────
RTH_START = _dt.time(9, 30)    # 09:30 ET (포함)
RTH_END   = _dt.time(16, 0)    # 16:00 ET (미포함)
MARKET_TZ = "America/New_York"

# OHLCV 리샘플 집계 방식 (coin resampler.py 와 동일)
OHLCV_AGG = {
    "open":   "first",   # 구간 첫 값
    "high":   "max",     # 구간 최댓값
    "low":    "min",     # 구간 최솟값
    "close":  "last",    # 구간 마지막 값
    "volume": "sum",     # 구간 합계
}

OUTPUT_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def load_minute_parquet(parquet_path: str, symbol: str = None) -> pd.DataFrame:
    """
    분봉 parquet 파일을 읽어 단일 종목 OHLCV DataFrame 으로 반환.

    Args:
        parquet_path:
            data/minute/*.parquet 경로.
        symbol:
            파일에 여러 종목이 들어있을 때 추출할 종목 심볼.
            None 이면 파일에 종목이 하나뿐이라고 가정한다.

    Returns:
        DatetimeIndex(UTC, tz-aware) + open/high/low/close/volume 컬럼.

    Raises:
        ValueError: 파일에 종목이 여럿인데 symbol 을 지정하지 않은 경우.
    """
    df = pd.read_parquet(parquet_path)

    # ── MultiIndex (symbol, timestamp) → 단일 종목 추출 ──────────
    if isinstance(df.index, pd.MultiIndex):
        symbols = df.index.get_level_values("symbol").unique()
        if symbol is None:
            if len(symbols) != 1:
                raise ValueError(
                    f"파일에 종목이 {len(symbols)}개 있습니다: {list(symbols)}. "
                    f"symbol 인자로 하나를 지정하세요."
                )
            symbol = symbols[0]
        df = df.xs(symbol, level="symbol")

    df = df.copy()

    # ── 인덱스를 tz-aware UTC 로 보정 ────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    missing = [c for c in ("open", "high", "low", "close", "volume")
               if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing} (파일: {parquet_path})")

    return df[["open", "high", "low", "close", "volume"]]


def resample_to_bars(
    df_minute: pd.DataFrame,
    timeframe: str = "30min",
    rth_only: bool = True,
) -> pd.DataFrame:
    """
    UTC 분봉 DataFrame → 정규장 필터 → timeframe 리샘플.

    Args:
        df_minute:
            DatetimeIndex(UTC, tz-aware) + OHLCV 컬럼.
            load_minute_parquet() 의 출력.
        timeframe:
            pandas 리샘플 문자열. 기본 '30min' (권장).
            '1h' 등도 가능하나 09:30 시작이라 첫 봉이 반토막 됨.
        rth_only:
            True 면 정규장(09:30~16:00 ET)만 남기고 리샘플.
            False 면 프리장/애프터장까지 전부 포함.

    Returns:
        datetime(naive ET) + OHLCV 컬럼 DataFrame, 시간 오름차순.

    Raises:
        ValueError: 입력이 비어있거나 정규장 필터 후 데이터가 없을 때.
    """
    if df_minute.empty:
        raise ValueError("입력 분봉 데이터가 비어 있습니다.")

    # ── 1. UTC → 미국 동부시간 (서머타임 자동 처리) ──────────────
    df = df_minute.copy()
    df.index = df.index.tz_convert(MARKET_TZ)

    # ── 2. 정규장 필터 (09:30 ≤ t < 16:00 ET) ───────────────────
    if rth_only:
        t = df.index.time
        mask = (t >= RTH_START) & (t < RTH_END)
        df = df[mask]
        if df.empty:
            raise ValueError(
                "정규장(09:30~16:00 ET) 필터 후 데이터가 없습니다. "
                "입력 데이터의 시간대(UTC 여부)를 확인하세요."
            )

    # ── 3. 리샘플 (OHLCV 집계) ──────────────────────────────────
    # 30min 버킷 경계는 자정 기준 :00/:30 → 09:30 이 자연스러운 경계.
    resampled = df.resample(timeframe).agg(OHLCV_AGG)

    # ── 4. 빈 봉 제거 (밤·주말·휴장·데이터 갭 → NaN) ────────────
    resampled = resampled.dropna(subset=["close"])

    # ── 5. datetime 을 tz 없는 컬럼으로 (downstream coin 코드 호환) ─
    resampled.index.name = "datetime"
    resampled = resampled.reset_index()
    resampled["datetime"] = resampled["datetime"].dt.tz_localize(None)

    return resampled[OUTPUT_COLUMNS]


def load_stock_bars(
    parquet_path: str,
    timeframe: str = "30min",
    rth_only: bool = True,
    start=None,
    end=None,
    symbol: str = None,
) -> pd.DataFrame:
    """
    분봉 parquet 로드 → 정규장 리샘플 → 기간 필터까지 한 번에.

    coin features/resampler.py 의 load_and_resample() 의 주식 버전.

    Args:
        parquet_path: 분봉 parquet 경로 (data/minute/*.parquet).
        timeframe:    리샘플 타임프레임 (기본 '30min').
        rth_only:     정규장만 사용할지 (기본 True).
        start, end:   기간 필터 (예: '2021-01-01'). None 이면 전체.
        symbol:       파일에 여러 종목일 때 추출할 심볼.

    Returns:
        datetime(naive ET) + OHLCV DataFrame.

    Raises:
        ValueError: 기간 필터 후 데이터가 없을 때.
    """
    df_min = load_minute_parquet(parquet_path, symbol=symbol)
    bars = resample_to_bars(df_min, timeframe=timeframe, rth_only=rth_only)

    if start is not None:
        bars = bars[bars["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        bars = bars[bars["datetime"] <= pd.Timestamp(end)]

    bars = bars.reset_index(drop=True)
    if bars.empty:
        raise ValueError(
            f"기간 필터 후 데이터가 없습니다. start={start}, end={end}"
        )
    return bars


def load_resampled_bars(parquet_path: str, start=None, end=None) -> pd.DataFrame:
    """
    이미 리샘플된 parquet(datetime + OHLCV)을 읽어 기간 필터만 적용.

    data/2_resample_bars.py 로 미리 만들어 둔 파일을 백테스트에서
    빠르게 로드할 때 사용한다. 매번 100만 행짜리 분봉을 다시
    리샘플하지 않아도 되므로 실행이 훨씬 빠르다.

    Args:
        parquet_path: 리샘플 완료된 parquet 경로 (data/30min/*.parquet).
        start, end:   기간 필터. None 이면 전체.

    Returns:
        datetime(naive) + OHLCV DataFrame.

    Raises:
        ValueError: 'datetime' 컬럼이 없을 때 (리샘플 파일이 아님).
    """
    bars = pd.read_parquet(parquet_path)
    if "datetime" not in bars.columns:
        raise ValueError(
            f"{parquet_path} 에 'datetime' 컬럼이 없습니다. "
            f"리샘플 완료된 파일(data/30min/*)이 맞는지 확인하세요."
        )
    bars = bars.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"])

    if start is not None:
        bars = bars[bars["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        bars = bars[bars["datetime"] <= pd.Timestamp(end)]

    return bars.reset_index(drop=True)


def resample_bars_df(
    raw_df: pd.DataFrame,
    symbol: str = None,
    timeframe: str = "30min",
    rth_only: bool = True,
) -> pd.DataFrame:
    """메모리에 있는 원본 분봉 DataFrame을 정규장 리샘플 봉으로 변환.

    load_stock_bars() 의 parquet 입력 버전을, 이미 메모리에 올라온
    DataFrame(예: alpaca-py 의 get_stock_bars().df — 라이브 데이터)에
    그대로 적용한다. 라이브 트레이딩에서 최신 봉을 만들 때 사용.

    Args:
        raw_df:
            (symbol, timestamp) MultiIndex 또는 timestamp 단일 인덱스 +
            open/high/low/close/volume 컬럼. timestamp 는 UTC 권장.
        symbol:
            MultiIndex 일 때 추출할 종목. None 이면 단일 종목 가정.
        timeframe:
            리샘플 타임프레임 (기본 '30min').
        rth_only:
            정규장만 사용할지 (기본 True).

    Returns:
        datetime(naive ET) + OHLCV DataFrame.
    """
    df = raw_df
    if isinstance(df.index, pd.MultiIndex):
        if symbol is not None and "symbol" in (df.index.names or []):
            df = df.xs(symbol, level="symbol")
        else:
            drop = [n for n in df.index.names if n != "timestamp"]
            if drop:
                df = df.droplevel(drop)
    df = df.copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    missing = [c for c in ("open", "high", "low", "close", "volume")
               if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    return resample_to_bars(
        df[["open", "high", "low", "close", "volume"]],
        timeframe=timeframe, rth_only=rth_only,
    )
