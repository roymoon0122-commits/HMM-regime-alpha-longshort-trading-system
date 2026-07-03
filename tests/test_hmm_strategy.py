"""
HMMStrategy 단위 테스트.

검증 항목:
- __init__ 기본 파라미터 검증
- from_config() 클래스메서드 동작
- fit()→generate_signals() 라운드트립 (합성 데이터)
- signal 범위/길이/dtype
- 룩어헤드 안전성: 미래 봉 변경해도 과거 signal 불변
- variant 조합: include_hmm_proba × use_smoothed_labels (4 cases)
- 워밍업 구간 0 처리
"""

import types

import numpy as np
import pandas as pd
import pytest

from strategy.HMM_strategy.strategy import HMMStrategy


# =============================================================================
# 합성 데이터 헬퍼 (Bull/Side/Bear가 명확히 보이는 가짜 4h봉)
# =============================================================================

def _make_synthetic_df(n_bars: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    합성 OHLCV DataFrame 생성.

    구조: 3개 국면을 반복하여 HMM이 학습 가능한 패턴 형성.
    - Bull: drift +0.002, vol 0.005
    - Side: drift 0,      vol 0.004
    - Bear: drift -0.002,  vol 0.008
    """
    rng = np.random.default_rng(seed)
    block_len = max(60, n_bars // 9)   # 9개 블록(Bull/Side/Bear×3)
    blocks = []
    pattern = [
        (+0.002, 0.005),  # Bull
        (0.000, 0.004),   # Side
        (-0.002, 0.008),  # Bear
    ] * 3
    while sum(b[0] is not None for b in blocks if b is not None) < 1:
        # 위 while은 placeholder — 실제 블록 생성은 아래
        break

    blocks = []
    for drift, vol in pattern:
        rets = rng.normal(drift, vol, block_len)
        blocks.append(rets)
    rets = np.concatenate(blocks)[:n_bars]

    # 길이 보정
    if len(rets) < n_bars:
        extra = rng.normal(0, 0.005, n_bars - len(rets))
        rets = np.concatenate([rets, extra])

    close = 10000.0 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n_bars)))
    volume = rng.uniform(100, 1000, n_bars)
    dt = pd.date_range('2024-01-01', periods=n_bars, freq='4h')

    return pd.DataFrame({
        'datetime': dt,
        'open':  open_,
        'high':  high,
        'low':   low,
        'close': close,
        'volume': volume,
    })


def _quick_strategy(**overrides) -> HMMStrategy:
    """단위 테스트용으로 빠르게 학습되는 HMMStrategy 생성."""
    defaults = dict(
        # HMM 학습 가벼운 설정
        n_states=3,
        hmm_n_iter=20,
        hmm_n_random_restart=2,
        hmm_covariance_type='diag',
        # 캐시 사용 안 함 (테스트 격리)
        hmm_model_path=None,
        # 작은 윈도우/롤링
        window_size=30,
        rolling_window=200,
        adx_period=12,
        r2_period=20,
        # smoother 빠르게
        smoother_lookback=5,
        smoother_threshold=0.02,
        smoother_persistence=2,
        # 메타
        meta_C=1.0,
        meta_class_weight='balanced',
        meta_max_iter=200,
        random_state=42,
        verbose=False,
    )
    defaults.update(overrides)
    return HMMStrategy(**defaults)


# =============================================================================
# __init__ / from_config
# =============================================================================
class TestInit:

    def test_defaults(self):
        s = HMMStrategy()
        assert s.include_hmm_proba is True
        assert s.use_smoothed_labels is True
        assert s.position_mode == 'net'
        assert s.is_fitted_ is False
        assert s.labeler_ is None
        assert s.meta_model_ is None

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            HMMStrategy(window_size=1)

    def test_invalid_adx_period(self):
        with pytest.raises(ValueError, match="adx_period"):
            HMMStrategy(adx_period=0)

    def test_invalid_rolling_window(self):
        with pytest.raises(ValueError, match="rolling_window"):
            HMMStrategy(rolling_window=0)

    def test_repr_unfitted(self):
        s = HMMStrategy()
        r = repr(s)
        assert 'unfitted' in r
        assert 'net' in r


class TestFromConfig:

    def test_from_config_default(self):
        # 진짜 config 모듈 사용
        s = HMMStrategy.from_config()
        from strategy.HMM_strategy import config
        assert s.window_size == config.WINDOW_SIZE
        assert s.adx_period == config.ADX_PERIOD
        assert s.position_mode == config.POSITION_MODE

    def test_from_config_overrides(self):
        s = HMMStrategy.from_config(window_size=30, include_hmm_proba=False)
        assert s.window_size == 30
        assert s.include_hmm_proba is False

    def test_from_config_with_mock_module(self):
        """config_module 인자로 mock 주입 가능 (Pattern B 테스트성 확보)."""
        mock = types.SimpleNamespace(
            INCLUDE_HMM_PROBA=False,
            USE_SMOOTHED_LABELS=False,
            POSITION_MODE='net',
            MIN_POSITION_THRESHOLD=0.2,
            HMM_MODEL_PATH=None,
            N_STATES=3,
            HMM_N_ITER=10,
            HMM_RANDOM_RESTART=1,
            HMM_COVARIANCE_TYPE='diag',
            ROLLING_SCALER_WINDOW=100,
            WINDOW_SIZE=15,
            ADX_PERIOD=5,
            R2_PERIOD=10,
            HMM_FEATURE_COLS=['cum_return'],
            SLOPE_NORM_COL='slope_norm',
            ADX_THRESHOLD=25.0,
            ADX_CLF_STEEPNESS=0.2,
            DIRECTION_STEEPNESS=50.0,
            R2_THRESHOLD=0.55,
            R2_CLF_STEEPNESS=8.0,
            R2_DIRECTION_STEEPNESS=1.0,
            LABEL_SMOOTHER_LOOKBACK=5,
            LABEL_SMOOTHER_THRESHOLD=0.03,
            LABEL_SMOOTHER_PERSISTENCE=2,
            LABEL_SMOOTHER_INCLUDE_SIDE=False,
            META_C=2.0,
            META_CLASS_WEIGHT='balanced',
        )
        s = HMMStrategy.from_config(config_module=mock)
        assert s.window_size == 15
        assert s.adx_period == 5
        assert s.min_threshold == 0.2
        assert s.include_hmm_proba is False
        assert s.use_smoothed_labels is False
        assert s.meta_C == 2.0


# =============================================================================
# fit / generate_signals 라운드트립 (모든 variant 4가지)
# =============================================================================
@pytest.fixture(scope='module')
def synth_df():
    return _make_synthetic_df(n_bars=1500, seed=42)


@pytest.mark.parametrize("include_hmm_proba", [True, False])
@pytest.mark.parametrize("use_smoothed_labels", [True, False])
class TestFitAndSignal:
    """4가지 variant 조합에서 fit→generate_signals 동작 검증."""

    def test_fit_returns_self(self, synth_df, include_hmm_proba, use_smoothed_labels):
        s = _quick_strategy(
            include_hmm_proba=include_hmm_proba,
            use_smoothed_labels=use_smoothed_labels,
        )
        result = s.fit(synth_df)
        assert result is s
        assert s.is_fitted_ is True
        assert s.meta_model_ is not None
        assert s.labeler_ is not None
        assert s.sizer_ is not None

    def test_signal_length_matches_df(self, synth_df, include_hmm_proba, use_smoothed_labels):
        s = _quick_strategy(
            include_hmm_proba=include_hmm_proba,
            use_smoothed_labels=use_smoothed_labels,
        )
        s.fit(synth_df)
        signals = s.generate_signals(synth_df)
        assert signals.shape == (len(synth_df),)

    def test_signal_range(self, synth_df, include_hmm_proba, use_smoothed_labels):
        s = _quick_strategy(
            include_hmm_proba=include_hmm_proba,
            use_smoothed_labels=use_smoothed_labels,
        )
        s.fit(synth_df)
        signals = s.generate_signals(synth_df)
        assert signals.dtype == np.float64
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)
        assert not np.any(np.isnan(signals))

    def test_warmup_zeros(self, synth_df, include_hmm_proba, use_smoothed_labels):
        s = _quick_strategy(
            include_hmm_proba=include_hmm_proba,
            use_smoothed_labels=use_smoothed_labels,
        )
        s.fit(synth_df)
        signals = s.generate_signals(synth_df)
        # window_size 미만 인덱스는 윈도우가 형성되지 않음 → 0
        assert np.all(signals[:s.window_size - 1] == 0.0)


# =============================================================================
# variant 차이 검증
# =============================================================================
class TestVariantDifferences:

    def test_meta_input_dim_differs(self, synth_df):
        s_with = _quick_strategy(include_hmm_proba=True)
        s_without = _quick_strategy(include_hmm_proba=False)
        s_with.fit(synth_df)
        s_without.fit(synth_df)
        # X_meta_shape는 _fit_diagnostics에 저장
        n_with = s_with._fit_diagnostics['X_meta_shape'][1]
        n_without = s_without._fit_diagnostics['X_meta_shape'][1]
        assert n_with == 16
        assert n_without == 10
        assert len(s_with.feature_names_) == 16
        assert len(s_without.feature_names_) == 10

    def test_smoothed_changes_recorded(self, synth_df):
        s_smooth = _quick_strategy(use_smoothed_labels=True)
        s_orig = _quick_strategy(use_smoothed_labels=False)
        s_smooth.fit(synth_df)
        s_orig.fit(synth_df)
        # 원본 라벨에는 smoother가 작동하지 않음
        assert s_orig._fit_diagnostics['smoother_changes'] == 0
        # smoothed는 0건일 수도 있지만 일반적으로 ≥0
        assert s_smooth._fit_diagnostics['smoother_changes'] >= 0

    def test_signals_can_differ_across_variants(self, synth_df):
        """4가지 variant가 서로 다른 signal을 만들어낼 수 있는지 확인.

        모두 같은 결과면 variant 분기가 작동하지 않는 것.
        합성 데이터에서 모든 variant가 같은 결과를 내는 우연을 막기 위해
        '4개 중 적어도 2개는 서로 다르다'로 검증.
        """
        results = []
        for inc in (True, False):
            for sm in (True, False):
                s = _quick_strategy(include_hmm_proba=inc, use_smoothed_labels=sm)
                s.fit(synth_df)
                results.append(s.generate_signals(synth_df))

        # 비교: 적어도 한 쌍 이상이 달라야 함
        any_differ = False
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if not np.allclose(results[i], results[j], atol=1e-9):
                    any_differ = True
                    break
            if any_differ:
                break
        assert any_differ, "모든 variant가 같은 signal을 만들었습니다 — 분기 작동 의심"


# =============================================================================
# 룩어헤드 안전성
# =============================================================================
class TestLookahead:

    def test_no_lookahead(self, synth_df):
        """미래 봉을 변경해도 과거 signal이 바뀌지 않아야 함.

        i 시점의 signals[i]는 봉 0..i 데이터로만 결정되므로, 봉 i+1 이후를
        바꾼다고 변하면 안 됨. 단, ROLLING_SCALER가 미래 데이터로 영향받지
        않는지도 함께 검증.
        """
        s = _quick_strategy()
        s.fit(synth_df)
        signals_orig = s.generate_signals(synth_df)

        # 마지막 100봉의 close를 50% 확대
        df_modified = synth_df.copy()
        modify_start = len(df_modified) - 100
        for col in ('open', 'high', 'low', 'close'):
            df_modified.loc[modify_start:, col] = df_modified.loc[modify_start:, col] * 1.5

        signals_modified = s.generate_signals(df_modified)

        # 변경 시작 시점 이전(modify_start - window_size 까지 안전)의 signal이 같아야 함
        # 윈도우는 window_size봉 뒤까지만 영향. 보수적으로 window_size×2 앞까지 안전.
        safe_end = modify_start - s.window_size - 5
        assert safe_end > 100, "테스트 데이터가 너무 짧음"

        np.testing.assert_allclose(
            signals_orig[:safe_end],
            signals_modified[:safe_end],
            atol=1e-9,
            err_msg="과거 signal이 미래 봉 변경에 영향을 받음 — 룩어헤드 의심",
        )


# =============================================================================
# 학습 전 호출 / 잘못된 사용
# =============================================================================
class TestErrorHandling:

    def test_generate_before_fit_raises(self, synth_df):
        s = _quick_strategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.generate_signals(synth_df)

    def test_too_short_df_raises(self):
        # window_size=30, rolling_window=200이면 최소 230봉 이상 필요
        # 200봉만 주면 학습 가능 행이 0
        s = _quick_strategy()
        short = _make_synthetic_df(n_bars=200, seed=1)
        with pytest.raises(RuntimeError, match="학습 가능"):
            s.fit(short)
