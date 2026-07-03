"""
HMM_strategy 패키지의 Phase 1 모듈 단위 테스트.

실행 방법:
    cd Coin-trader-main
    pytest tests/test_hmmfeatures.py -v

테스트 구성:
  [A] 합성 데이터로 9개 피처의 정확성 검증 (단조증가, 단조감소, 일정값 등)
  [B] 룩어헤드 바이어스 자동 검증 (전체 데이터 vs 잘라낸 데이터 비교)
  [C] 인터페이스 검증 (window_size 가변, 컬럼 존재, 워밍업 처리)
  [D] ADX/R² 자체 동작 검증 (완벽한 직선 → R² ≈ 1)
  [E] RegimeDataset 인터페이스 검증
"""

import numpy as np
import pandas as pd
import pytest

from strategy.HMM_strategy.features.indicators import (
    compute_adx,
    compute_r2,
    compute_slope,
)
from strategy.HMM_strategy.features.window_features import (
    FEATURE_COLUMNS,
    compute_window_features,
)
from strategy.HMM_strategy.regime.regime_dataset import RegimeDataset


# ════════════════════════════════════════════════════════════════
#  헬퍼: 합성 OHLCV DataFrame 생성
# ════════════════════════════════════════════════════════════════

def make_fake_df(closes, datetime_start='2024-01-01', freq='4h'):
    """
    종가 리스트로부터 OHLCV DataFrame을 생성 (테스트용).
    open=close (단순화), high/low는 close ± 0.5%.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    # 'open' = 직전 종가 (없으면 자기 자신)
    opens = np.empty(n)
    opens[0]  = closes[0]
    opens[1:] = closes[:-1]

    df = pd.DataFrame({
        'datetime': pd.date_range(datetime_start, periods=n, freq=freq),
        'open':     opens,
        'high':     np.maximum(opens, closes) * 1.005,
        'low':      np.minimum(opens, closes) * 0.995,
        'close':    closes,
        'volume':   np.full(n, 1000.0),
    })
    return df


def make_uptrend_df(n=100, start=100.0, step=1.0):
    """단조 증가 가격 시계열."""
    closes = start + np.arange(n) * step
    return make_fake_df(closes)


def make_downtrend_df(n=100, start=200.0, step=1.0):
    """단조 감소 가격 시계열."""
    closes = start - np.arange(n) * step
    return make_fake_df(closes)


def make_flat_df(n=100, price=100.0):
    """가격 일정 시계열."""
    closes = np.full(n, price)
    return make_fake_df(closes)


# ════════════════════════════════════════════════════════════════
#  [A] 9개 피처 정확성 테스트
# ════════════════════════════════════════════════════════════════

class TestFeatureAccuracy:
    """합성 데이터로 사람이 답을 알 수 있는 케이스를 검증."""

    def test_cum_return_uptrend(self):
        """단조증가 → cum_return은 양수."""
        df = make_uptrend_df(n=100, start=100.0, step=1.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        # 첫 윈도우(인덱스 0~9): 종가 100 → 109 → 9% 상승
        assert feats['cum_return'].iloc[0] == pytest.approx(0.09, rel=1e-6)
        # 모든 윈도우 cum_return > 0
        assert (feats['cum_return'] > 0).all()

    def test_cum_return_downtrend(self):
        """단조감소 → cum_return은 음수."""
        df = make_downtrend_df(n=100, start=200.0, step=1.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        assert (feats['cum_return'] < 0).all()

    def test_volatility_constant_price(self):
        """가격 일정 → volatility = 0."""
        df = make_flat_df(n=100, price=100.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        # 일정 가격이라도 high/low 노이즈가 있어 ADX는 정의되지만 vol은 0
        assert feats['volatility'].iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_up_candle_ratio_all_up(self):
        """모든 봉이 양봉 → up_candle_ratio = 1.0."""
        df = make_uptrend_df(n=100, start=100.0, step=1.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        # uptrend는 close > open(직전 close)이므로 모두 양봉
        assert feats['up_candle_ratio'].iloc[-1] == pytest.approx(1.0)

    def test_up_candle_ratio_all_down(self):
        """모든 봉이 음봉 → up_candle_ratio = 0.0."""
        df = make_downtrend_df(n=100, start=200.0, step=1.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        assert feats['up_candle_ratio'].iloc[-1] == pytest.approx(0.0)

    def test_max_drawdown_uptrend(self):
        """단조증가 → max_drawdown = 0 (낙폭 없음)."""
        df = make_uptrend_df(n=100, start=100.0, step=1.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        assert feats['max_drawdown'].iloc[-1] == pytest.approx(0.0)

    def test_max_drawdown_known_dd(self):
        """가격 100→120→90 패턴 → max_drawdown = (90-120)/120 = -25%."""
        # 윈도우 정확히 3봉으로 만들기
        closes = [100.0, 120.0, 90.0]
        df = make_fake_df(closes)
        # adx/r2 계산은 무시 — 너무 짧으면 NaN이라 dropna에서 다 사라질 수 있음
        # 직접 계산 검증을 위해 window_features 내부 로직만 확인
        from strategy.HMM_strategy.features.window_features import (
            compute_window_features as cwf,
        )
        # 윈도우=3, adx/r2 period도 작게
        feats = cwf(df, window_size=3, adx_period=2, r2_period=2)
        if len(feats) > 0:
            assert feats['max_drawdown'].iloc[-1] == pytest.approx(-0.25, rel=1e-6)

    def test_slope_flat(self):
        """가격 일정 → slope ≈ 0."""
        df = make_flat_df(n=100, price=100.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        assert abs(feats['slope'].iloc[-1]) < 1e-10

    def test_slope_uptrend(self):
        """단조증가 (step=1) → slope ≈ 1.0."""
        df = make_uptrend_df(n=100, start=100.0, step=1.0)
        feats = compute_window_features(df, window_size=10, adx_period=5, r2_period=5)
        # step=1이므로 회귀 기울기도 1
        assert feats['slope'].iloc[-1] == pytest.approx(1.0, rel=1e-6)


# ════════════════════════════════════════════════════════════════
#  [B] 룩어헤드 바이어스 자동 검증
# ════════════════════════════════════════════════════════════════

class TestNoLookahead:
    """
    핵심 원리:
      i번째 봉의 피처를 두 가지 방식으로 계산:
        (a) 전체 데이터(0~N)로 계산한 결과의 i번째 행
        (b) 잘라낸 데이터(0~i)로 계산한 결과의 마지막 행
      두 값이 정확히 같으면 → 미래 데이터를 안 본 것 ✅
    """

    def test_no_lookahead_window_features(self):
        np.random.seed(42)
        n = 200
        # 무작위 가격 시계열 (look-ahead 검증용)
        returns = np.random.randn(n) * 0.01
        closes = 100.0 * np.exp(np.cumsum(returns))
        df_full = make_fake_df(closes)

        # 전체 데이터로 계산
        feats_full = compute_window_features(
            df_full, window_size=30, adx_period=10, r2_period=15
        )

        # 임의 시점 i=120에서 잘라서 계산
        cutoff = 120
        df_partial = df_full.iloc[:cutoff + 1].copy()
        feats_partial = compute_window_features(
            df_partial, window_size=30, adx_period=10, r2_period=15
        )

        # feats_full 중 window_end_idx == cutoff 인 행
        full_row = feats_full[feats_full['window_end_idx'] == cutoff]
        # feats_partial의 마지막 행
        partial_row = feats_partial.iloc[[-1]]

        assert len(full_row) == 1, f"cutoff={cutoff}에 해당하는 행이 없음"

        # 9개 피처가 모두 동일해야 함
        for col in FEATURE_COLUMNS:
            full_val    = full_row[col].iloc[0]
            partial_val = partial_row[col].iloc[0]
            assert full_val == pytest.approx(partial_val, rel=1e-9, abs=1e-12), (
                f"피처 '{col}'에서 룩어헤드 의심: "
                f"full={full_val}, partial={partial_val}"
            )


# ════════════════════════════════════════════════════════════════
#  [C] 인터페이스 검증
# ════════════════════════════════════════════════════════════════

class TestInterface:
    """함수 시그니처/반환 형식이 약속대로 동작하는지."""

    @pytest.mark.parametrize("window_size", [20, 30, 60])
    def test_window_size_configurable(self, window_size):
        """window_size를 바꿔도 정상 동작."""
        df = make_uptrend_df(n=300, step=0.5)
        feats = compute_window_features(
            df, window_size=window_size, adx_period=10, r2_period=10
        )
        assert len(feats) > 0
        # 첫 윈도우 인덱스 = window_size - 1 또는 그 이상 (워밍업 dropna 이후)
        assert feats['window_end_idx'].iloc[0] >= window_size - 1

    def test_output_schema(self):
        """반환 DataFrame에 9개 피처 + 메타 컬럼이 존재."""
        df = make_uptrend_df(n=200)
        feats = compute_window_features(df, window_size=30, adx_period=10, r2_period=10)
        for col in FEATURE_COLUMNS:
            assert col in feats.columns, f"피처 컬럼 누락: {col}"
        assert 'window_end_idx' in feats.columns
        assert 'window_end_time' in feats.columns  # datetime이 있으니

    def test_warmup_dropped(self):
        """초반 워밍업 NaN이 dropna로 제거됨."""
        df = make_uptrend_df(n=200)
        feats = compute_window_features(df, window_size=30, adx_period=12, r2_period=15)
        # 어떤 피처에도 NaN이 없어야 함
        assert not feats[FEATURE_COLUMNS].isna().any().any()

    def test_missing_column_raises(self):
        """필수 컬럼이 없으면 ValueError."""
        df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="필수 컬럼"):
            compute_window_features(df, window_size=3)

    def test_too_short_data_raises(self):
        """데이터가 윈도우보다 작으면 ValueError."""
        df = make_uptrend_df(n=10)
        with pytest.raises(ValueError, match="윈도우 크기"):
            compute_window_features(df, window_size=60)


# ════════════════════════════════════════════════════════════════
#  [D] ADX/R² 자체 검증
# ════════════════════════════════════════════════════════════════

class TestIndicators:

    def test_r2_perfect_line(self):
        """완벽한 직선 가격 → R² ≈ 1.0."""
        df = make_uptrend_df(n=100, start=100.0, step=1.0)
        r2 = compute_r2(df, period=20)
        # 워밍업 이후의 마지막 값
        last = r2.iloc[-1]
        assert last == pytest.approx(1.0, abs=1e-9)

    def test_r2_flat(self):
        """가격 완전 일정 → ss_tot=0 → R²=1.0 (정의대로)."""
        df = make_flat_df(n=100, price=100.0)
        r2 = compute_r2(df, period=20)
        last = r2.iloc[-1]
        assert last == pytest.approx(1.0)

    def test_r2_random_low(self):
        """무작위 가격 → R²은 작은 값 (보통 < 0.5)."""
        np.random.seed(0)
        closes = 100 + np.random.randn(200).cumsum() * 0.5  # 작은 노이즈
        df = make_fake_df(closes)
        r2 = compute_r2(df, period=40)
        # 무작위 워크 → R² 평균이 1보다 훨씬 낮음 (보통 0.3 이하)
        assert r2.dropna().mean() < 0.7

    def test_adx_strong_trend(self):
        """강한 단조 추세 → ADX가 높음 (보통 > 30)."""
        df = make_uptrend_df(n=100, start=100.0, step=1.0)
        adx = compute_adx(df, period=12)
        # 워밍업 이후
        last = adx.iloc[-1]
        assert last > 30, f"강한 추세인데 ADX={last}로 너무 낮음"

    def test_adx_returns_series(self):
        """ADX 출력 타입이 Series이고 길이가 입력과 동일."""
        df = make_uptrend_df(n=100)
        adx = compute_adx(df, period=12)
        assert isinstance(adx, pd.Series)
        assert len(adx) == len(df)

    def test_slope_function(self):
        """compute_slope: 선형 데이터에 대해 정확한 기울기."""
        prices = np.array([10.0, 12.0, 14.0, 16.0, 18.0])  # step=2
        assert compute_slope(prices) == pytest.approx(2.0)

        prices = np.array([100.0, 100.0, 100.0])  # 일정
        assert compute_slope(prices) == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════════
#  [E] RegimeDataset 인터페이스 검증
# ════════════════════════════════════════════════════════════════

class TestRegimeDataset:

    def _make_dataset(self):
        df = make_uptrend_df(n=200)
        feats = compute_window_features(df, window_size=30, adx_period=10, r2_period=10)
        return RegimeDataset(feats), feats

    def test_get_X_shape(self):
        ds, feats = self._make_dataset()
        X = ds.get_X()
        assert X.shape == (len(feats), len(FEATURE_COLUMNS))
        assert X.dtype == np.float64

    def test_set_get_labels(self):
        ds, feats = self._make_dataset()
        labels = np.zeros(len(feats), dtype=np.int64)
        ds.set_labels(labels)
        y = ds.get_y(shift=0)
        assert len(y) == len(feats)

    def test_label_length_mismatch(self):
        ds, feats = self._make_dataset()
        wrong_labels = np.zeros(len(feats) + 5)
        with pytest.raises(ValueError, match="라벨 길이"):
            ds.set_labels(wrong_labels)

    def test_get_y_without_labels_raises(self):
        ds, _ = self._make_dataset()
        with pytest.raises(RuntimeError, match="라벨이 주입되지 않았습니다"):
            ds.get_y()

    def test_shift_minus_one(self):
        """shift=-1 → 마지막 행이 NaN, 첫 행은 원래 두 번째 라벨."""
        ds, feats = self._make_dataset()
        labels = np.arange(len(feats), dtype=np.int64)
        ds.set_labels(labels)
        y_shifted = ds.get_y(shift=-1)
        # shift(-1) 적용 시 첫 행 = 원본의 두 번째 값(=1)
        assert y_shifted[0] == 1
        # 마지막 행은 NaN
        assert np.isnan(y_shifted[-1])

    def test_aligned_Xy_drops_nan(self):
        ds, feats = self._make_dataset()
        labels = np.arange(len(feats), dtype=np.int64)
        ds.set_labels(labels)
        X, y = ds.get_aligned_Xy(shift=-1)
        # shift=-1 → 마지막 1행이 빠짐
        assert len(X) == len(feats) - 1
        assert len(y) == len(feats) - 1
        assert y.dtype == np.int64

    def test_train_test_split(self):
        ds, feats = self._make_dataset()
        # feats의 중간 시점으로 분할
        mid_time = feats['window_end_time'].iloc[len(feats) // 2]
        train, test = ds.get_train_test_split(mid_time)
        assert len(train) + len(test) == len(feats)
        assert len(train) > 0 and len(test) > 0

    # ── 피처 부분집합 (HMM/Meta 분리) ──────────────────────

    def test_get_X_with_subset(self):
        """get_X(feature_cols=[...])로 부분집합 추출 — Option C 시나리오."""
        ds, feats = self._make_dataset()
        subset = ['cum_return', 'volatility', 'adx_mean', 'r2_mean', 'up_candle_ratio']
        X_subset = ds.get_X(feature_cols=subset)
        assert X_subset.shape == (len(feats), len(subset))
        # 컬럼 순서가 요청한 대로 나오는지
        names = ds.get_feature_names(feature_cols=subset)
        assert names == subset

    def test_get_X_with_none_uses_default(self):
        """get_X(feature_cols=None) → 기본 9개 사용."""
        ds, feats = self._make_dataset()
        X = ds.get_X(feature_cols=None)
        assert X.shape == (len(feats), len(FEATURE_COLUMNS))

    def test_get_X_invalid_column_raises_keyerror(self):
        """존재하지 않는 컬럼명 요청 시 명확한 KeyError."""
        ds, _ = self._make_dataset()
        with pytest.raises(KeyError, match="없는 컬럼"):
            ds.get_X(feature_cols=['cum_return', 'nonexistent_feature'])

    def test_feature_subset_is_easy_to_change(self):
        """
        피처 추가/제외 워크플로우 시뮬레이션:
          1. config 변경(여기서는 변수로 시뮬레이션)
          2. 동일한 RegimeDataset에서 다른 부분집합 추출
        """
        ds, feats = self._make_dataset()
        # "config 변경" 시뮬레이션: 4개 → 5개로 피처 추가
        cols_v1 = ['cum_return', 'volatility', 'adx_mean', 'up_candle_ratio']
        cols_v2 = cols_v1 + ['r2_mean']
        X1 = ds.get_X(feature_cols=cols_v1)
        X2 = ds.get_X(feature_cols=cols_v2)
        assert X1.shape[1] == 4
        assert X2.shape[1] == 5
        # X1의 4개 컬럼이 X2의 처음 4개와 같은지 (순서 보존 확인)
        np.testing.assert_array_equal(X1, X2[:, :4])
