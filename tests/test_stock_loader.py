"""
stock_loader.py 단위 테스트.

검증 대상: 주식 분봉 → 정규장 리샘플 어댑터.
  - 정규장(09:30~16:00 ET) 필터
  - 30분봉 집계 정확성 (open/high/low/close/volume)
  - 서머타임(DST) 처리
  - datetime tz-naive 출력
  - parquet 라운드트립 (load_stock_bars / load_resampled_bars)

실행: pytest tests/test_stock_loader.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from strategy.HMM_strategy.features.stock_loader import (
    RTH_START,
    RTH_END,
    resample_to_bars,
    resample_bars_df,
    load_stock_bars,
    load_resampled_bars,
)


# ════════════════════════════════════════════════════════════
#  합성 데이터 헬퍼
# ════════════════════════════════════════════════════════════

def _make_minute_day(date_str: str) -> pd.DataFrame:
    """하루치 1분봉 합성 데이터.

    ET 04:00~20:00 (확장시간 포함) 16시간 = 960분.
    UTC tz-aware 인덱스로 생성한다.
    가격은 분 단위로 1씩 증가시켜(0,1,2,...) 집계 검증을 쉽게 한다.
    """
    start_et = pd.Timestamp(f"{date_str} 04:00", tz="America/New_York")
    idx_et = pd.date_range(start_et, periods=960, freq="1min")
    idx_utc = idx_et.tz_convert("UTC")
    n = len(idx_utc)
    df = pd.DataFrame(
        {
            "open":   np.arange(n, dtype=float),
            "high":   np.arange(n, dtype=float) + 0.5,
            "low":    np.arange(n, dtype=float) - 0.5,
            "close":  np.arange(n, dtype=float) + 0.1,
            "volume": np.ones(n, dtype=float),
        },
        index=idx_utc,
    )
    df.index.name = "timestamp"
    return df


# ════════════════════════════════════════════════════════════
#  정규장 필터
# ════════════════════════════════════════════════════════════

class TestRTHFilter:

    def test_all_bars_within_rth(self):
        """리샘플 출력의 모든 봉이 09:30~16:00 안에 있어야 한다."""
        bars = resample_to_bars(_make_minute_day("2024-03-15"))
        times = bars["datetime"].dt.time
        assert (times >= RTH_START).all()
        assert (times < RTH_END).all()

    def test_first_bar_is_0930(self):
        """첫 봉의 시작 시각은 정확히 09:30 이어야 한다."""
        bars = resample_to_bars(_make_minute_day("2024-03-15"))
        assert bars["datetime"].iloc[0].time() == dt.time(9, 30)

    def test_last_bar_is_1530(self):
        """마지막 봉은 15:30 (15:30~16:00 구간)."""
        bars = resample_to_bars(_make_minute_day("2024-03-15"))
        assert bars["datetime"].iloc[-1].time() == dt.time(15, 30)

    def test_full_day_has_13_bars(self):
        """정규장 6.5시간 / 30분 = 하루 13봉."""
        bars = resample_to_bars(_make_minute_day("2024-03-15"))
        assert len(bars) == 13

    def test_rth_false_includes_extended_hours(self):
        """rth_only=False 면 프리장/애프터장까지 포함되어 봉이 더 많다."""
        bars_rth = resample_to_bars(_make_minute_day("2024-03-15"),
                                    rth_only=True)
        bars_all = resample_to_bars(_make_minute_day("2024-03-15"),
                                    rth_only=False)
        assert len(bars_all) > len(bars_rth)


# ════════════════════════════════════════════════════════════
#  OHLCV 집계 정확성
# ════════════════════════════════════════════════════════════

class TestAggregation:

    def test_ohlc_values(self):
        """09:30 봉은 분봉 인덱스 330~359 를 집계한 값이어야 한다.

        ET 04:00 이 인덱스 0 → 09:30 은 5.5시간 뒤 = 330분.
        30분봉이므로 330~359 가 한 봉.
        """
        bars = resample_to_bars(_make_minute_day("2024-03-15"))
        first = bars.iloc[0]
        assert first["open"] == 330.0          # open[330]
        assert first["close"] == 359.0 + 0.1   # close[359]
        assert first["high"] == 359.0 + 0.5    # max high = high[359]
        assert first["low"] == 330.0 - 0.5     # min low = low[330]
        assert first["volume"] == 30.0         # 30개 분봉 합

    def test_columns_order(self):
        """출력 컬럼 순서는 datetime/open/high/low/close/volume."""
        bars = resample_to_bars(_make_minute_day("2024-03-15"))
        assert list(bars.columns) == [
            "datetime", "open", "high", "low", "close", "volume",
        ]

    def test_datetime_is_tz_naive(self):
        """downstream coin 코드 호환을 위해 datetime 은 tz 없는 값."""
        bars = resample_to_bars(_make_minute_day("2024-03-15"))
        assert bars["datetime"].dt.tz is None


# ════════════════════════════════════════════════════════════
#  서머타임 (DST) 처리
# ════════════════════════════════════════════════════════════

class TestDST:

    def test_summer_and_winter_both_start_0930(self):
        """여름(EDT, UTC-4)·겨울(EST, UTC-5) 모두 첫 봉이 09:30 ET.

        UTC→ET 변환이 DST 를 자동 처리하는지 검증.
        """
        summer = resample_to_bars(_make_minute_day("2024-07-15"))
        winter = resample_to_bars(_make_minute_day("2024-01-15"))
        assert summer["datetime"].iloc[0].time() == dt.time(9, 30)
        assert winter["datetime"].iloc[0].time() == dt.time(9, 30)

    def test_summer_and_winter_same_bar_count(self):
        """계절과 무관하게 하루 13봉."""
        summer = resample_to_bars(_make_minute_day("2024-07-15"))
        winter = resample_to_bars(_make_minute_day("2024-01-15"))
        assert len(summer) == len(winter) == 13


# ════════════════════════════════════════════════════════════
#  여러 날 / 갭 처리
# ════════════════════════════════════════════════════════════

class TestMultiDay:

    def test_two_days(self):
        """이틀치 → 26봉, 밤샘 구간 빈 봉은 제거됨."""
        d1 = _make_minute_day("2024-03-14")
        d2 = _make_minute_day("2024-03-15")
        both = pd.concat([d1, d2])
        bars = resample_to_bars(both)
        assert len(bars) == 26
        # 시간 오름차순
        assert bars["datetime"].is_monotonic_increasing

    def test_partial_day_no_crash(self):
        """절반만 있는 날(반장 등)도 에러 없이 처리된다."""
        day = _make_minute_day("2024-03-15")
        # 정규장 전반부만 남김 (09:30~12:00 정도)
        et = day.index.tz_convert("America/New_York")
        day = day[et.time < dt.time(12, 0)]
        bars = resample_to_bars(day)
        assert len(bars) > 0
        assert (bars["datetime"].dt.time < dt.time(12, 0)).all()


# ════════════════════════════════════════════════════════════
#  예외 처리
# ════════════════════════════════════════════════════════════

class TestErrors:

    def test_empty_input_raises(self):
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        with pytest.raises(ValueError):
            resample_to_bars(empty)


# ════════════════════════════════════════════════════════════
#  parquet 라운드트립 (load_stock_bars / load_resampled_bars)
# ════════════════════════════════════════════════════════════

class TestParquetRoundtrip:

    def _write_minute_parquet(self, tmp_path, symbol="TEST"):
        """(symbol, timestamp) MultiIndex 분봉 parquet 생성."""
        df = _make_minute_day("2024-03-15")
        df = df.copy()
        df["symbol"] = symbol
        df = df.set_index("symbol", append=True)
        df = df.reorder_levels(["symbol", "timestamp"])
        path = tmp_path / f"{symbol}_1min.parquet"
        df.to_parquet(path)
        return path

    def test_load_stock_bars(self, tmp_path):
        """분봉 parquet → load_stock_bars → 13봉."""
        path = self._write_minute_parquet(tmp_path)
        bars = load_stock_bars(str(path), timeframe="30min")
        assert len(bars) == 13
        assert list(bars.columns) == [
            "datetime", "open", "high", "low", "close", "volume",
        ]

    def test_load_stock_bars_period_filter(self, tmp_path):
        """start 필터가 동작한다."""
        path = self._write_minute_parquet(tmp_path)
        bars = load_stock_bars(str(path), start="2024-03-15 12:00")
        assert (bars["datetime"] >= pd.Timestamp("2024-03-15 12:00")).all()

    def test_load_resampled_bars(self, tmp_path):
        """리샘플 parquet 저장 후 load_resampled_bars 로 동일하게 로드."""
        path = self._write_minute_parquet(tmp_path)
        bars = load_stock_bars(str(path))
        rs_path = tmp_path / "TEST_30min.parquet"
        bars.to_parquet(rs_path, index=False)

        reloaded = load_resampled_bars(str(rs_path))
        pd.testing.assert_frame_equal(bars, reloaded)


# ════════════════════════════════════════════════════════════
#  resample_bars_df — 라이브 데이터(메모리 DataFrame) 변환
# ════════════════════════════════════════════════════════════

class TestResampleBarsDf:

    def test_multiindex_matches_resample_to_bars(self):
        """(symbol, timestamp) MultiIndex df → resample_to_bars와 동일 결과."""
        minute = _make_minute_day("2024-03-15")        # UTC 인덱스 + OHLCV
        mi = minute.copy()
        mi["symbol"] = "TEST"
        mi = mi.set_index("symbol", append=True)
        mi = mi.reorder_levels(["symbol", "timestamp"])
        out = resample_bars_df(mi, symbol="TEST")
        ref = resample_to_bars(minute)
        pd.testing.assert_frame_equal(out, ref)

    def test_single_index_input(self):
        """MultiIndex가 아닌 단일 timestamp 인덱스도 처리한다."""
        minute = _make_minute_day("2024-03-15")
        out = resample_bars_df(minute)
        assert len(out) == 13
        assert list(out.columns) == [
            "datetime", "open", "high", "low", "close", "volume",
        ]

    def test_missing_column_raises(self):
        minute = _make_minute_day("2024-03-15").drop(columns=["volume"])
        with pytest.raises(ValueError):
            resample_bars_df(minute)
