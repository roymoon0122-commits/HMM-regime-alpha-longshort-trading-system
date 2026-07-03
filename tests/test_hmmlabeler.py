"""
Phase 2 단위 테스트.

대상:
    - strategy.HMM_strategy.features.scaling.RollingStandardScaler
    - strategy.HMM_strategy.regime.hmm_labeler.HMMLabeler

테스트 철학:
    - 합성 데이터로 정답이 명확한 상황에서 검증 (실데이터 의존 없음)
    - 빠르게 돌아야 하므로 n_random_restart 작게 (기본 5)
    - 외부 라이브러리 동작은 신뢰하되, 우리가 추가한 layer만 검증
"""

import numpy as np
import pytest

from strategy.HMM_strategy.features.scaling import RollingStandardScaler
from strategy.HMM_strategy.regime.hmm_labeler import (
    HMMLabeler, BULL, SIDE, BEAR, REGIME_NAMES,
)


# ────────────────────────────────────────────────────────────────
# 합성 데이터 헬퍼
# ────────────────────────────────────────────────────────────────

def make_3regime_data(seed=42, n_per_state=300, noise=0.3):
    """
    명확히 구분되는 3개 분포로부터 합성 데이터 생성.

    Returns:
        X: shape (3 * n_per_state, 5), 피처 순서 = HMM_FEATURE_COLS
        cum_return: shape (3 * n_per_state,), X의 첫 컬럼과 동일
        true_labels: shape (3 * n_per_state,), 0=Bull / 1=Side / 2=Bear

    각 상태의 평균 (HMM_FEATURE_COLS = [cum_return, volatility, adx_mean, r2_mean, up_candle_ratio]):
        Bull : [+0.03,  0.012, 35, 0.65, 0.58]
        Side : [ 0.0,   0.009, 20, 0.35, 0.50]
        Bear : [-0.03,  0.018, 32, 0.60, 0.42]
    """
    rng = np.random.default_rng(seed)

    means = {
        BULL: np.array([+0.03,  0.012, 35.0, 0.65, 0.58]),
        SIDE: np.array([ 0.00,  0.009, 20.0, 0.35, 0.50]),
        BEAR: np.array([-0.03,  0.018, 32.0, 0.60, 0.42]),
    }
    # 각 피처별 분산 (피처 스케일 고려)
    stds = np.array([0.005, 0.002, 5.0, 0.05, 0.05]) * noise

    blocks = []
    label_blocks = []
    for regime in [BULL, SIDE, BEAR]:
        block = rng.normal(loc=means[regime], scale=stds, size=(n_per_state, 5))
        blocks.append(block)
        label_blocks.append(np.full(n_per_state, regime, dtype=np.int64))

    X = np.vstack(blocks)
    true_labels = np.concatenate(label_blocks)
    cum_return = X[:, 0].copy()
    return X, cum_return, true_labels


# ────────────────────────────────────────────────────────────────
# RollingStandardScaler
# ────────────────────────────────────────────────────────────────

class TestRollingStandardScaler:

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            RollingStandardScaler(window=1)
        with pytest.raises(ValueError):
            RollingStandardScaler(window=0)

    def test_output_shape(self):
        X = np.random.randn(100, 3)
        scaler = RollingStandardScaler(window=20)
        out = scaler.fit_transform(X)
        assert out.shape == X.shape

    def test_first_window_minus_one_is_nan(self):
        """첫 (window-1)행은 history 부족으로 NaN."""
        X = np.random.randn(100, 3)
        window = 20
        out = RollingStandardScaler(window=window).fit_transform(X)
        # 처음 window-1 행: NaN
        assert np.isnan(out[:window - 1]).all()
        # window번째 행부터는 finite
        assert not np.isnan(out[window - 1:]).any()

    def test_normalized_stats_approx_zero_one(self):
        """충분히 큰 N에서 rolling-normalized 값의 평균≈0, 표준편차≈1."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 2))
        out = RollingStandardScaler(window=50).fit_transform(X)
        valid = out[~np.isnan(out).any(axis=1)]
        np.testing.assert_allclose(valid.mean(axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(valid.std(axis=0), 1.0, atol=0.2)

    def test_constant_feature_no_inf_or_nan_after_warmup(self):
        """상수 피처(분산 0)는 warmup 후 0 반환 (inf/nan 없음)."""
        X = np.column_stack([
            np.ones(100),                        # 상수
            np.random.randn(100),                # 일반
        ])
        out = RollingStandardScaler(window=20).fit_transform(X)
        warmed = out[20:]   # 워밍업 이후
        assert np.isfinite(warmed).all()
        np.testing.assert_allclose(warmed[:, 0], 0.0)

    def test_dataframe_input_works(self):
        """입력이 DataFrame이어도 동일하게 작동."""
        import pandas as pd
        X_arr = np.random.randn(50, 2)
        X_df = pd.DataFrame(X_arr, columns=['a', 'b'])
        out_arr = RollingStandardScaler(window=10).fit_transform(X_arr)
        out_df = RollingStandardScaler(window=10).fit_transform(X_df)
        np.testing.assert_allclose(out_arr, out_df, equal_nan=True)

    def test_no_lookahead(self):
        """
        룩어헤드 검증: 시점 t의 정규화 결과는 t 이전 데이터를 바꾸기만 해도 변하지만,
        t 이후 데이터를 바꿔도 변하면 안 된다.
        """
        rng = np.random.default_rng(1)
        X1 = rng.standard_normal((100, 2))
        X2 = X1.copy()
        # 시점 50 이후를 완전히 다른 값으로 교체
        X2[60:] = rng.standard_normal((40, 2)) * 100

        out1 = RollingStandardScaler(window=20).fit_transform(X1)
        out2 = RollingStandardScaler(window=20).fit_transform(X2)

        # 시점 50까지의 결과는 동일해야 함 (window=20 → 시점 50의 input은 [31..50])
        # 50번째 시점 결과가 input[60:]에 영향받지 않음을 확인
        np.testing.assert_allclose(out1[:50], out2[:50], equal_nan=True)


# ────────────────────────────────────────────────────────────────
# HMMLabeler — 핵심 동작
# ────────────────────────────────────────────────────────────────

class TestHMMLabelerBasic:

    def test_fit_predict_runs(self):
        X, cr, _ = make_3regime_data()
        labeler = HMMLabeler(n_states=3, n_random_restart=5, n_iter=100)
        labeler.fit(X, cr)
        labels = labeler.predict(X)
        assert labels.shape == (len(X),)
        assert set(np.unique(labels)).issubset({BULL, SIDE, BEAR})

    def test_predict_proba_sums_to_one(self):
        X, cr, _ = make_3regime_data()
        labeler = HMMLabeler(n_random_restart=5, n_iter=100).fit(X, cr)
        proba = labeler.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_state_to_regime_mapping_by_cum_return(self):
        """매핑 후 Bull로 분류된 시점들의 cum_return 평균이 Bear보다 커야 함."""
        X, cr, _ = make_3regime_data()
        labeler = HMMLabeler(n_random_restart=5, n_iter=100).fit(X, cr)
        labels = labeler.predict(X)

        bull_mask = labels == BULL
        bear_mask = labels == BEAR
        assert bull_mask.sum() > 0, "Bull 라벨이 한 번도 안 나왔음 (학습 실패 의심)"
        assert bear_mask.sum() > 0, "Bear 라벨이 한 번도 안 나왔음 (학습 실패 의심)"
        assert cr[bull_mask].mean() > cr[bear_mask].mean()

    def test_recovers_three_regimes_on_synthetic_data(self):
        """
        합성 데이터의 진짜 라벨과 예측이 (재배치 후) 어느 정도 일치해야 한다.
        HMM은 시퀀스 의존성이 있어 perfect는 아니지만, 70%+ 일치는 기대.
        """
        X, cr, true_labels = make_3regime_data(noise=0.2)
        labeler = HMMLabeler(n_random_restart=10, n_iter=200).fit(X, cr)
        pred = labeler.predict(X)
        # 합성 데이터는 cum_return이 명확히 분리되어 있고 우리 매핑도 cum_return 기반이므로
        # 재배치 없이 직접 일치율 확인 가능
        accuracy = (pred == true_labels).mean()
        assert accuracy > 0.7, f"라벨 일치율 {accuracy:.2%} (70% 미만)"


class TestHMMLabelerErrors:

    def test_predict_before_fit_raises(self):
        labeler = HMMLabeler()
        with pytest.raises(RuntimeError, match="fit"):
            labeler.predict(np.zeros((10, 5)))

    def test_predict_proba_before_fit_raises(self):
        labeler = HMMLabeler()
        with pytest.raises(RuntimeError, match="fit"):
            labeler.predict_proba(np.zeros((10, 5)))

    def test_fit_with_nan_raises(self):
        X = np.random.randn(100, 5)
        X[10, 2] = np.nan
        cr = X[:, 0]
        labeler = HMMLabeler()
        with pytest.raises(ValueError, match="NaN"):
            labeler.fit(X, cr)

    def test_fit_length_mismatch_raises(self):
        X = np.random.randn(100, 5)
        cr = np.random.randn(50)
        with pytest.raises(ValueError, match="length"):
            HMMLabeler().fit(X, cr)

    def test_invalid_n_states_raises(self):
        with pytest.raises(ValueError):
            HMMLabeler(n_states=1)

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            HMMLabeler().save(str(tmp_path / "x.joblib"))


class TestHMMLabelerSaveLoad:

    def test_save_load_roundtrip(self, tmp_path):
        X, cr, _ = make_3regime_data()
        labeler1 = HMMLabeler(n_random_restart=5, n_iter=100).fit(X, cr)
        labels1 = labeler1.predict(X)
        proba1 = labeler1.predict_proba(X)

        path = tmp_path / "hmm.joblib"
        labeler1.save(str(path))

        labeler2 = HMMLabeler().load(str(path))
        labels2 = labeler2.predict(X)
        proba2 = labeler2.predict_proba(X)

        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_allclose(proba1, proba2)

    def test_load_restores_config(self, tmp_path):
        X, cr, _ = make_3regime_data()
        labeler1 = HMMLabeler(
            n_states=3,
            n_random_restart=5,
            n_iter=100,
            covariance_type='diag',
            random_state=123,
        ).fit(X, cr)
        path = tmp_path / "hmm.joblib"
        labeler1.save(str(path))

        labeler2 = HMMLabeler().load(str(path))
        assert labeler2.n_states == 3
        assert labeler2.covariance_type == 'diag'
        assert labeler2.random_state == 123


class TestHMMLabelerBIC:

    def test_bic_returns_valid_n(self):
        X, _, _ = make_3regime_data()
        labeler = HMMLabeler(n_random_restart=2, n_iter=80)
        best_n, bic_dict = labeler.select_n_states_by_bic(
            X, candidates=[2, 3, 4], n_restart_quick=2,
        )
        assert best_n in [2, 3, 4]
        assert all(n in bic_dict for n in [2, 3, 4])

    def test_bic_does_not_modify_self(self):
        """select_n_states_by_bic는 self.n_states를 변경하지 않아야 한다."""
        X, _, _ = make_3regime_data()
        labeler = HMMLabeler(n_states=3, n_random_restart=2, n_iter=80)
        labeler.select_n_states_by_bic(X, candidates=[2, 4], n_restart_quick=2)
        assert labeler.n_states == 3
        assert labeler.model_ is None


class TestHMMLabelerHistory:

    def test_fit_history_length(self):
        X, cr, _ = make_3regime_data()
        labeler = HMMLabeler(n_random_restart=5, n_iter=80).fit(X, cr)
        assert len(labeler.fit_history_) == 5
        for entry in labeler.fit_history_:
            assert 'restart_idx' in entry
            assert 'score' in entry
            assert 'converged' in entry

    def test_best_score_is_highest_converged(self):
        X, cr, _ = make_3regime_data()
        labeler = HMMLabeler(n_random_restart=5, n_iter=80).fit(X, cr)
        converged_scores = [
            h['score'] for h in labeler.fit_history_ if h['converged']
        ]
        assert labeler.best_score_ == max(converged_scores)
