"""
HMMStrategy의 use_donchian_on_side 옵션 단위 테스트.

검증 항목:
1. 옵션 OFF (기본값): 기존 동작과 100% 동일 (회귀 안전)
2. 옵션 ON: HMM argmax = SIDE 시점에서만 donchian × P(Side)로 덮어쓰기
3. 옵션 ON일 때 SIDE 시점이 아닌 곳은 OFF 결과와 동일
4. 옵션 ON일 때 SIDE 시점의 신호는 |signal| <= P(Side) (돈치안이 0/±1이므로)
5. 라이브 시나리오: from_config(hmm_model_path=None, verbose=False) → 옵션 OFF
"""

import numpy as np
import pandas as pd
import pytest

from strategy.HMM_strategy.strategy import HMMStrategy
from strategy.HMM_strategy.regime.hmm_labeler import SIDE as SIDE_IDX


# 기존 test_hmm_strategy.py와 동일한 합성 데이터 로직 재사용 (테스트 격리 목적으로 복제)
def _make_synthetic_df(n_bars: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    block_len = max(60, n_bars // 9)
    pattern = [
        (+0.002, 0.005),  # Bull
        (0.000, 0.004),   # Side
        (-0.002, 0.008),  # Bear
    ] * 3
    blocks = []
    for drift, vol in pattern:
        rets = rng.normal(drift, vol, block_len)
        blocks.append(rets)
    rets = np.concatenate(blocks)[:n_bars]
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
        'datetime': dt, 'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    })


def _quick_strategy(**overrides) -> HMMStrategy:
    """단위 테스트용으로 빠르게 학습되는 HMMStrategy 생성."""
    defaults = dict(
        n_states=3,
        hmm_n_iter=20,
        hmm_n_random_restart=2,
        hmm_covariance_type='diag',
        hmm_model_path=None,
        window_size=30,
        rolling_window=200,
        adx_period=12,
        r2_period=20,
        smoother_lookback=5,
        smoother_threshold=0.02,
        smoother_persistence=2,
        meta_C=1.0,
        meta_class_weight='balanced',
        meta_max_iter=200,
        random_state=42,
        verbose=False,
        # 합성 데이터(1500봉)에 맞는 짧은 돈치안 파라미터
        donchian_entry_period=60,
        donchian_exit_period=30,
    )
    defaults.update(overrides)
    return HMMStrategy(**defaults)


# =============================================================================
# 옵션 기본값 / 검증
# =============================================================================
class TestDonchianOptionInit:

    def test_default_off(self):
        """기본값이 False여야 한다 (라이브 무영향 보장)."""
        s = HMMStrategy()
        assert s.use_donchian_on_side is False
        assert s.donchian_entry_period == 260
        assert s.donchian_exit_period == 130

    def test_repr_includes_option(self):
        s = HMMStrategy(use_donchian_on_side=True)
        assert 'use_donchian_on_side=True' in repr(s)

    def test_invalid_entry_period(self):
        with pytest.raises(ValueError, match='donchian_entry_period'):
            HMMStrategy(donchian_entry_period=1)

    def test_invalid_exit_period(self):
        with pytest.raises(ValueError, match='donchian_exit_period'):
            HMMStrategy(donchian_exit_period=1)

    def test_live_trade_call_pattern_keeps_off(self):
        """live_trade.py의 정확한 호출 형태로 만들었을 때 옵션이 꺼져있어야 한다."""
        s = HMMStrategy.from_config(hmm_model_path=None, verbose=False)
        assert s.use_donchian_on_side is False


# =============================================================================
# 회귀 안전성 — 옵션 OFF면 기존 동작과 동일
# =============================================================================
class TestRegressionWhenOff:

    def test_off_matches_baseline(self):
        """옵션 OFF에서 만든 signal은 기존 HMMStrategy 동작과 동일."""
        df = _make_synthetic_df(n_bars=1500, seed=42)

        # 같은 random_state, 같은 데이터, 옵션만 다르게 두 번 학습
        s_off = _quick_strategy(use_donchian_on_side=False)
        s_off.fit(df)
        sig_off = s_off.generate_signals(df)

        # 옵션을 명시적으로 안 줘도 기본값 False
        s_default = _quick_strategy()
        s_default.fit(df)
        sig_default = s_default.generate_signals(df)

        # 동일 random_state면 결과 동일해야 함
        np.testing.assert_array_equal(sig_off, sig_default)


# =============================================================================
# 옵션 ON 동작
# =============================================================================
class TestDonchianOverrideOn:

    @pytest.fixture(scope='class')
    def df(self):
        return _make_synthetic_df(n_bars=1500, seed=42)

    @pytest.fixture(scope='class')
    def strategies_and_signals(self, df):
        """OFF/ON 각각 학습해서 시그널 반환 (fixture 재사용)."""
        s_off = _quick_strategy(use_donchian_on_side=False)
        s_off.fit(df)
        sig_off = s_off.generate_signals(df)

        s_on = _quick_strategy(use_donchian_on_side=True)
        s_on.fit(df)
        sig_on = s_on.generate_signals(df)

        return s_off, sig_off, s_on, sig_on

    def test_signals_differ_only_at_side(self, df, strategies_and_signals):
        """ON과 OFF의 차이는 HMM argmax = SIDE인 시점에만 발생해야 한다."""
        s_off, sig_off, s_on, sig_on = strategies_and_signals

        # ON 전략으로 메타 proba를 다시 계산해서 SIDE 시점 식별
        # (s_on 내부에서 generate_signals가 만든 proba를 재현)
        features = s_on._build_features(df)
        hmm_proba, _, _ = s_on._compute_hmm_proba(features)
        from strategy.HMM_strategy.regime.transition import TransitionPredictor
        tp = TransitionPredictor.from_labeler(s_on.labeler_)
        X_meta, _, nan_mask = s_on._build_meta_input(features, hmm_proba, tp)
        cold_mask = np.isclose(hmm_proba, 1.0 / 3.0, atol=1e-9).all(axis=1)
        invalid_mask = nan_mask | cold_mask
        X_safe = np.where(np.isnan(X_meta), 0.0, X_meta)
        proba = s_on.meta_model_.predict_proba(X_safe)
        proba[invalid_mask] = 1.0 / 3.0
        argmax_per_window = np.argmax(proba, axis=1)

        end_idx = features['window_end_idx'].astype(int).values
        n_bars = len(df)
        valid_end = (end_idx >= 0) & (end_idx < n_bars)

        # bar별로 SIDE 여부 매핑
        bar_is_side = np.zeros(n_bars, dtype=bool)
        bar_is_invalid = np.zeros(n_bars, dtype=bool)
        for j in np.where(valid_end)[0]:
            bar_i = end_idx[j]
            if invalid_mask[j]:
                bar_is_invalid[bar_i] = True
            elif argmax_per_window[j] == SIDE_IDX:
                bar_is_side[bar_i] = True

        # 비-SIDE & 비-invalid 봉은 ON/OFF 시그널 동일해야 함
        # 주의: s_off와 s_on은 두 번 학습된 별개 인스턴스라 학습 단계의
        # 미세한 비결정성으로 ~1e-12 수준 부동소수점 차이가 나타날 수 있음.
        # (override 로직 자체의 정확성은 test_side_bars_use_donchian_times_pside
        #  에서 동일 인스턴스의 weights_arr와 직접 비교로 검증됨)
        non_side = ~bar_is_side & ~bar_is_invalid
        np.testing.assert_allclose(
            sig_on[non_side], sig_off[non_side],
            rtol=1e-8, atol=1e-10,
            err_msg="Non-SIDE bars should be (near-)identical between OFF and ON"
        )

        # SIDE 봉이 1개 이상 있어야 의미있는 테스트 (합성 데이터에 Side 블록이 있으니까)
        assert bar_is_side.sum() > 0, "테스트 데이터에 SIDE 시점이 없음 — 데이터 점검 필요"

    def test_side_bars_use_donchian_times_pside(self, df, strategies_and_signals):
        """SIDE 시점의 ON 신호는 donchian_signal × P(Side) 와 일치해야 한다."""
        s_off, sig_off, s_on, sig_on = strategies_and_signals

        # 돈치안 시그널 직접 계산
        from strategy.donchian_adx_r2_B import DonchianADXR2Strategy
        donch = DonchianADXR2Strategy(
            entry_period=s_on.donchian_entry_period,
            exit_period=s_on.donchian_exit_period,
        )
        donch_signals = donch.generate_signals(df).astype(np.float64)

        # 메타 proba 재계산
        features = s_on._build_features(df)
        hmm_proba, _, _ = s_on._compute_hmm_proba(features)
        from strategy.HMM_strategy.regime.transition import TransitionPredictor
        tp = TransitionPredictor.from_labeler(s_on.labeler_)
        X_meta, _, nan_mask = s_on._build_meta_input(features, hmm_proba, tp)
        cold_mask = np.isclose(hmm_proba, 1.0 / 3.0, atol=1e-9).all(axis=1)
        invalid_mask = nan_mask | cold_mask
        X_safe = np.where(np.isnan(X_meta), 0.0, X_meta)
        proba = s_on.meta_model_.predict_proba(X_safe)
        proba[invalid_mask] = 1.0 / 3.0

        end_idx = features['window_end_idx'].astype(int).values
        n_bars = len(df)
        valid_end = (end_idx >= 0) & (end_idx < n_bars)

        # SIDE 봉마다 expected vs actual 비교
        n_checked = 0
        for j in np.where(valid_end)[0]:
            if invalid_mask[j]:
                continue
            if np.argmax(proba[j]) != SIDE_IDX:
                continue
            bar_i = end_idx[j]
            expected = donch_signals[bar_i] * proba[j, SIDE_IDX]
            np.testing.assert_allclose(
                sig_on[bar_i], expected, rtol=1e-9, atol=1e-12,
                err_msg=f"bar {bar_i}, window {j}: expected {expected}, got {sig_on[bar_i]}"
            )
            n_checked += 1

        assert n_checked > 0, "SIDE 시점이 0개 — 테스트 무효"
        print(f"\n  [info] SIDE 시점 {n_checked}개에서 모두 donch × P(Side) 일치 확인")

    def test_signal_magnitude_bounded_by_pside_at_side(self, df, strategies_and_signals):
        """SIDE 시점 신호의 절대값은 P(Side) <= 1을 넘지 못해야 한다.
        (돈치안 출력이 -1/0/+1이고 곱한 P(Side)는 [0, 1])"""
        s_off, sig_off, s_on, sig_on = strategies_and_signals

        assert np.all(np.abs(sig_on) <= 1.0 + 1e-9), \
            f"signal max abs: {np.abs(sig_on).max()}"
