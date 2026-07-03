"""
Phase 3 단위 테스트.

대상:
    - strategy.HMM_strategy.classifiers.base_classifier.BaseClassifier
    - strategy.HMM_strategy.classifiers.adx_classifier.ADXClassifier
    - strategy.HMM_strategy.classifiers.r2_classifier.R2Classifier
    - strategy.HMM_strategy.regime.transition.TransitionPredictor
    - strategy.HMM_strategy.meta_model.base_meta_model.BaseMetaModel
    - strategy.HMM_strategy.meta_model.logistic_meta_model.LogisticMetaModel

테스트 철학 (Phase 1/2와 동일):
    - 합성 데이터로 정답이 명확한 상황에서 검증
    - 외부 라이브러리(sklearn) 동작은 신뢰하되, 우리가 추가한 layer만 검증
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from strategy.HMM_strategy.classifiers.base_classifier import (
    BaseClassifier, BULL, SIDE, BEAR, REGIME_NAMES, REGIME_IDS,
)
from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier
from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier
from strategy.HMM_strategy.regime.transition import TransitionPredictor
from strategy.HMM_strategy.regime.label_smoother import RetrospectiveLabelSmoother
from strategy.HMM_strategy.meta_model.base_meta_model import BaseMetaModel
from strategy.HMM_strategy.meta_model.logistic_meta_model import LogisticMetaModel


# ════════════════════════════════════════════════════════════════
#  TestBaseClassifier — 추상 클래스 컨벤션
# ════════════════════════════════════════════════════════════════

class TestBaseClassifier:
    """추상 베이스 클래스의 인터페이스 강제 검증."""

    def test_abstract_class_cannot_instantiate(self):
        """추상 메서드 미구현 시 인스턴스화 불가."""
        with pytest.raises(TypeError):
            BaseClassifier()

    def test_regime_constants(self):
        """BULL=0, SIDE=1, BEAR=2 컨벤션 (HMMLabeler와 통일)."""
        assert BULL == 0
        assert SIDE == 1
        assert BEAR == 2
        assert REGIME_NAMES == {0: 'Bull', 1: 'Side', 2: 'Bear'}
        assert REGIME_IDS == [0, 1, 2]

    def test_batch_methods_auto_provided(self):
        """단일 메서드만 구현해도 batch 메서드가 동작."""
        class DummyClf(BaseClassifier):
            def predict(self, w):
                return BULL
            def predict_proba(self, w):
                return np.array([1.0, 0.0, 0.0])

        clf = DummyClf()
        df = pd.DataFrame({'a': [1, 2, 3]})
        labels = clf.predict_batch(df)
        proba = clf.predict_proba_batch(df)
        assert labels.shape == (3,)
        assert (labels == BULL).all()
        assert proba.shape == (3, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ════════════════════════════════════════════════════════════════
#  TestADXClassifier
# ════════════════════════════════════════════════════════════════

class TestADXClassifier:

    def test_strong_bull(self):
        clf = ADXClassifier()
        row = pd.Series({'adx_mean': 40.0, 'cum_return': 0.05})
        proba = clf.predict_proba(row)
        assert clf.predict(row) == BULL
        assert proba[BULL] > proba[SIDE]
        assert proba[BULL] > proba[BEAR]

    def test_strong_bear(self):
        clf = ADXClassifier()
        row = pd.Series({'adx_mean': 40.0, 'cum_return': -0.05})
        proba = clf.predict_proba(row)
        assert clf.predict(row) == BEAR
        assert proba[BEAR] > proba[BULL]
        assert proba[BEAR] > proba[SIDE]

    def test_sideways(self):
        clf = ADXClassifier()
        row = pd.Series({'adx_mean': 12.0, 'cum_return': 0.005})
        proba = clf.predict_proba(row)
        assert clf.predict(row) == SIDE
        assert proba[SIDE] > 0.7

    def test_proba_sums_to_one(self):
        clf = ADXClassifier()
        rng = np.random.default_rng(0)
        for _ in range(100):
            adx = rng.uniform(0, 80)
            ret = rng.uniform(-0.1, 0.1)
            row = pd.Series({'adx_mean': adx, 'cum_return': ret})
            assert abs(clf.predict_proba(row).sum() - 1.0) < 1e-10

    def test_batch_matches_single(self):
        clf = ADXClassifier()
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            'adx_mean': rng.uniform(0, 60, size=50),
            'cum_return': rng.uniform(-0.1, 0.1, size=50),
        })
        batch = clf.predict_proba_batch(df)
        single = np.vstack([clf.predict_proba(df.iloc[i]) for i in range(50)])
        assert np.allclose(batch, single)

    def test_nan_safe(self):
        clf = ADXClassifier()
        row = pd.Series({'adx_mean': np.nan, 'cum_return': 0.05})
        proba = clf.predict_proba(row)
        assert np.allclose(proba, [1/3, 1/3, 1/3])

    def test_invalid_steepness(self):
        with pytest.raises(ValueError):
            ADXClassifier(adx_steepness=0)
        with pytest.raises(ValueError):
            ADXClassifier(direction_steepness=-1)


# ════════════════════════════════════════════════════════════════
#  TestR2Classifier
# ════════════════════════════════════════════════════════════════

class TestR2Classifier:

    def test_strong_bull(self):
        clf = R2Classifier()
        row = pd.Series({'r2_mean': 0.75, 'slope_norm': 2.0})
        proba = clf.predict_proba(row)
        assert clf.predict(row) == BULL
        assert proba[BULL] > 0.5

    def test_strong_bear(self):
        clf = R2Classifier()
        row = pd.Series({'r2_mean': 0.75, 'slope_norm': -2.0})
        proba = clf.predict_proba(row)
        assert clf.predict(row) == BEAR

    def test_low_r2_means_side(self):
        clf = R2Classifier()
        row = pd.Series({'r2_mean': 0.20, 'slope_norm': 0.5})
        proba = clf.predict_proba(row)
        assert clf.predict(row) == SIDE
        assert proba[SIDE] > 0.7

    def test_proba_sums_to_one(self):
        clf = R2Classifier()
        rng = np.random.default_rng(2)
        for _ in range(100):
            r2 = rng.uniform(0, 1)
            slope_z = rng.uniform(-3, 3)
            row = pd.Series({'r2_mean': r2, 'slope_norm': slope_z})
            assert abs(clf.predict_proba(row).sum() - 1.0) < 1e-10

    def test_batch_matches_single(self):
        clf = R2Classifier()
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            'r2_mean': rng.uniform(0, 1, size=50),
            'slope_norm': rng.uniform(-3, 3, size=50),
        })
        batch = clf.predict_proba_batch(df)
        single = np.vstack([clf.predict_proba(df.iloc[i]) for i in range(50)])
        assert np.allclose(batch, single)

    def test_nan_safe(self):
        clf = R2Classifier()
        row = pd.Series({'r2_mean': 0.65, 'slope_norm': np.nan})
        proba = clf.predict_proba(row)
        assert np.allclose(proba, [1/3, 1/3, 1/3])

    def test_custom_slope_col(self):
        """slope_col 파라미터로 컬럼명 변경 가능."""
        clf = R2Classifier(slope_col='custom_slope')
        row = pd.Series({'r2_mean': 0.75, 'custom_slope': 2.0})
        proba = clf.predict_proba(row)
        assert clf.predict(row) == BULL


# ════════════════════════════════════════════════════════════════
#  TestTransitionPredictor
# ════════════════════════════════════════════════════════════════

class TestTransitionPredictor:

    def test_basic_forward(self):
        """예제 전이 행렬 — 단일 시점 예측."""
        transmat = np.array([
            [0.85, 0.13, 0.02],
            [0.05, 0.90, 0.05],
            [0.02, 0.13, 0.85],
        ])
        predictor = TransitionPredictor(transmat)
        # current = 100% Bull → next ≈ first row of transmat
        result = predictor.predict_next(np.array([1.0, 0.0, 0.0]))
        assert np.allclose(result, transmat[0])

    def test_batch_matches_single(self):
        transmat = np.array([
            [0.85, 0.13, 0.02],
            [0.05, 0.90, 0.05],
            [0.02, 0.13, 0.85],
        ])
        predictor = TransitionPredictor(transmat)
        rng = np.random.default_rng(5)
        # 합 = 1인 무작위 확률 벡터 10개
        batch = rng.dirichlet(np.ones(3), size=10)
        out_batch = predictor.predict_next_batch(batch)
        out_single = np.vstack([predictor.predict_next(p) for p in batch])
        assert np.allclose(out_batch, out_single)
        # 각 행 합 = 1
        assert np.allclose(out_batch.sum(axis=1), 1.0)

    def test_invalid_shape(self):
        """정사각 아닌 행렬 거부."""
        with pytest.raises(ValueError):
            TransitionPredictor(np.array([[0.5, 0.5]]))

    def test_invalid_row_sum(self):
        """행 합 ≠ 1 거부."""
        bad = np.array([
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.4],
            [0.0, 0.5, 0.5],
        ])
        with pytest.raises(ValueError):
            TransitionPredictor(bad)

    def test_negative_values_rejected(self):
        bad = np.array([
            [1.5, -0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        with pytest.raises(ValueError):
            TransitionPredictor(bad)

    def test_from_labeler_remapping(self):
        """from_labeler가 [Bull, Side, Bear] 순서로 transmat 재배열."""
        # 모의 labeler 객체 — 내부 state 순서가 무작위라고 가정
        # state 0 = Bear, state 1 = Bull, state 2 = Side 매핑
        class FakeModel:
            transmat_ = np.array([
                [0.80, 0.10, 0.10],   # Bear → ...
                [0.20, 0.70, 0.10],   # Bull → ...
                [0.15, 0.15, 0.70],   # Side → ...
            ])

        class FakeLabeler:
            n_states = 3
            model_ = FakeModel()
            # regime_to_state_[regime_id] = internal_state_id
            regime_to_state_ = {0: 1, 1: 2, 2: 0}  # Bull→1, Side→2, Bear→0

        predictor = TransitionPredictor.from_labeler(FakeLabeler())
        # 기대 결과: 행/열을 [1, 2, 0] 순서로 재배열
        # remapped[i][j] = original[state_order[i]][state_order[j]]
        # remapped[0][0] = original[1][1] = 0.70 (Bull → Bull)
        # remapped[0][1] = original[1][2] = 0.10 (Bull → Side)
        # remapped[2][2] = original[0][0] = 0.80 (Bear → Bear)
        assert np.isclose(predictor.transmat[0, 0], 0.70)
        assert np.isclose(predictor.transmat[0, 1], 0.10)
        assert np.isclose(predictor.transmat[2, 2], 0.80)
        # 각 행 합 = 1 보장
        assert np.allclose(predictor.transmat.sum(axis=1), 1.0)


# ════════════════════════════════════════════════════════════════
#  TestBaseMetaModel
# ════════════════════════════════════════════════════════════════

class TestBaseMetaModel:

    def test_abstract_class_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseMetaModel()


# ════════════════════════════════════════════════════════════════
#  TestLogisticMetaModel
# ════════════════════════════════════════════════════════════════

def _make_synthetic_meta_data(seed=42, n=600, n_features=16):
    """메타 모델 학습용 합성 데이터.

    cum_return 인덱스(6)에 클래스별 시그널 주입:
        Bull → +1.5
        Side → 0
        Bear → -1.5
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = rng.standard_normal((n, n_features))
    X[y == 0, 6] += 1.5
    X[y == 2, 6] -= 1.5
    return X, y


class TestLogisticMetaModel:

    def test_fit_predict_proba(self):
        X, y = _make_synthetic_meta_data()
        meta = LogisticMetaModel()
        meta.fit(X, y)
        proba = meta.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_returns_int_labels(self):
        X, y = _make_synthetic_meta_data()
        meta = LogisticMetaModel().fit(X, y)
        pred = meta.predict(X)
        assert pred.dtype == np.int64
        # 모든 예측이 0/1/2 안에 있음
        assert set(np.unique(pred)).issubset({0, 1, 2})

    def test_synthetic_signal_recovered(self):
        """주입한 cum_return 시그널이 계수에 잡혀야 함."""
        X, y = _make_synthetic_meta_data()
        feat_names = [f'feat_{i}' for i in range(16)]
        feat_names[6] = 'cum_return_proxy'
        meta = LogisticMetaModel(feature_names=feat_names).fit(X, y)
        coef = meta.get_coef_summary()
        # Bull 행에서 cum_return_proxy 계수는 양수
        assert coef.loc['Bull', 'cum_return_proxy'] > 0.5
        # Bear 행에서는 음수
        assert coef.loc['Bear', 'cum_return_proxy'] < -0.5

    def test_save_load_roundtrip(self):
        X, y = _make_synthetic_meta_data()
        meta = LogisticMetaModel().fit(X, y)
        proba_orig = meta.predict_proba(X)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            meta.save(path)
            meta2 = LogisticMetaModel()
            meta2.load(path)
            proba_loaded = meta2.predict_proba(X)
            assert np.allclose(proba_orig, proba_loaded)
        finally:
            os.remove(path)

    def test_predict_before_fit_raises(self):
        meta = LogisticMetaModel()
        with pytest.raises(RuntimeError):
            meta.predict_proba(np.zeros((1, 16)))

    def test_invalid_label_rejected(self):
        """y에 0/1/2 외 값이 있으면 거부."""
        X, _ = _make_synthetic_meta_data()
        bad_y = np.array([0, 1, 2, 5] + [0] * (len(X) - 4))
        meta = LogisticMetaModel()
        with pytest.raises(ValueError):
            meta.fit(X, bad_y)

    def test_nan_in_X_rejected(self):
        X, y = _make_synthetic_meta_data()
        X[0, 0] = np.nan
        meta = LogisticMetaModel()
        with pytest.raises(ValueError):
            meta.fit(X, y)

    def test_invalid_C(self):
        with pytest.raises(ValueError):
            LogisticMetaModel(C=0)

    def test_feature_names_length_check(self):
        X, y = _make_synthetic_meta_data()
        meta = LogisticMetaModel(feature_names=['a', 'b'])  # 잘못된 길이
        with pytest.raises(ValueError):
            meta.fit(X, y)

    def test_coef_summary_shape(self):
        X, y = _make_synthetic_meta_data()
        meta = LogisticMetaModel().fit(X, y)
        coef = meta.get_coef_summary()
        # 3 rows (Bull/Side/Bear), 16 features + 1 intercept = 17 columns
        assert coef.shape == (3, 17)
        assert list(coef.index) == ['Bull', 'Side', 'Bear']
        assert '_intercept' in coef.columns


# ════════════════════════════════════════════════════════════════
#  TestRetrospectiveLabelSmoother
# ════════════════════════════════════════════════════════════════

BULL_, SIDE_, BEAR_ = 0, 1, 2  # 별칭 (위에서 import한 BULL과 동일)


class TestRetrospectiveLabelSmoother:

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            RetrospectiveLabelSmoother(lookback=0)
        with pytest.raises(ValueError):
            RetrospectiveLabelSmoother(threshold=0)
        with pytest.raises(ValueError):
            RetrospectiveLabelSmoother(persistence_check=0)

    def test_no_transitions_no_change(self):
        """전환이 없으면 라벨 그대로."""
        labels = np.array([BULL_] * 20)
        returns = np.zeros(20)
        smoother = RetrospectiveLabelSmoother()
        smoothed, log = smoother.smooth(labels, returns)
        assert np.array_equal(smoothed, labels)
        assert log == []

    def test_bear_transition_with_crash(self):
        """Bull → Bear 전환 + lookback 안에 폭락 → backdate 발생."""
        # 인덱스: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        # 라벨:  B B B B B B B B B B  X  X  X  X  X   (X=Bear=2)
        # 전환점 = 10. 폭락은 인덱스 7에 -10%
        labels = np.array([BULL_]*10 + [BEAR_]*5)
        returns = np.zeros(15)
        returns[7] = -0.10  # 7번째 봉에서 10% 폭락

        smoother = RetrospectiveLabelSmoother(
            lookback=10, threshold=0.05, persistence_check=3,
        )
        smoothed, log = smoother.smooth(labels, returns)

        # 7번째부터 Bear로 backdate되었는지
        assert smoothed[7] == BEAR_
        assert smoothed[8] == BEAR_
        assert smoothed[9] == BEAR_
        # 7 미만은 여전히 Bull
        assert smoothed[6] == BULL_
        # log 1건
        assert len(log) == 1
        assert log[0]['original_t'] == 10
        assert log[0]['backdated_to'] == 7
        assert log[0]['shift'] == 3
        assert log[0]['new_label'] == BEAR_

    def test_bull_transition_with_spike(self):
        """Bear → Bull 전환 + 폭등 → backdate."""
        labels = np.array([BEAR_]*10 + [BULL_]*5)
        returns = np.zeros(15)
        returns[6] = 0.08  # 6번째 봉 8% 폭등

        smoother = RetrospectiveLabelSmoother()
        smoothed, log = smoother.smooth(labels, returns)
        assert smoothed[6] == BULL_
        assert smoothed[5] == BEAR_  # 6 미만은 여전히 Bear
        assert log[0]['backdated_to'] == 6

    def test_threshold_not_met_no_backdate(self):
        """폭락이 임계값 미달 → backdate 안 함."""
        labels = np.array([BULL_]*10 + [BEAR_]*5)
        returns = np.zeros(15)
        returns[7] = -0.03  # 3% 폭락, 임계값 5% 미달

        smoother = RetrospectiveLabelSmoother(threshold=0.05)
        smoothed, log = smoother.smooth(labels, returns)
        assert np.array_equal(smoothed, labels)  # 변화 없음
        assert log == []

    def test_persistence_check_filters_flicker(self):
        """깜빡이는 라벨 (1봉만 바뀜) → backdate 안 함."""
        # 9 Bull → 1 Bear → 5 Bull (Bear는 깜빡임)
        labels = np.array([BULL_]*9 + [BEAR_] + [BULL_]*5)
        returns = np.zeros(15)
        returns[7] = -0.10

        smoother = RetrospectiveLabelSmoother(persistence_check=3)
        smoothed, log = smoother.smooth(labels, returns)
        assert np.array_equal(smoothed, labels)
        assert log == []

    def test_side_excluded_by_default(self):
        """기본 설정에서 SIDE 전환은 backdate 제외."""
        labels = np.array([BULL_]*10 + [SIDE_]*5)
        returns = np.zeros(15)
        returns[7] = -0.10

        smoother = RetrospectiveLabelSmoother(include_side=False)
        smoothed, log = smoother.smooth(labels, returns)
        assert np.array_equal(smoothed, labels)
        assert log == []

    def test_side_included_when_requested(self):
        """include_side=True 면 SIDE도 backdate."""
        labels = np.array([BULL_]*10 + [SIDE_]*5)
        returns = np.zeros(15)
        returns[7] = -0.10

        smoother = RetrospectiveLabelSmoother(include_side=True)
        smoothed, log = smoother.smooth(labels, returns)
        # backdate 발생
        assert smoothed[7] == SIDE_
        assert len(log) == 1

    def test_lookback_limit(self):
        """lookback 범위 밖의 폭락은 무시."""
        # 폭락이 인덱스 0, 전환점은 12 → lookback 10이면 2~12만 봄
        labels = np.array([BULL_]*12 + [BEAR_]*3)
        returns = np.zeros(15)
        returns[0] = -0.20  # lookback 범위 밖

        smoother = RetrospectiveLabelSmoother(lookback=10, threshold=0.05)
        smoothed, log = smoother.smooth(labels, returns)
        # lookback 안에 폭락 없으니 backdate 없음
        assert np.array_equal(smoothed, labels)
        assert log == []

    def test_wrong_direction_no_backdate(self):
        """Bull → Bear 전환인데 lookback 안에 폭등만 있으면 backdate 안 함."""
        labels = np.array([BULL_]*10 + [BEAR_]*5)
        returns = np.zeros(15)
        returns[7] = 0.10   # 폭등 (Bear 전환과 방향 반대)

        smoother = RetrospectiveLabelSmoother()
        smoothed, log = smoother.smooth(labels, returns)
        assert np.array_equal(smoothed, labels)
        assert log == []

    def test_length_mismatch_raises(self):
        smoother = RetrospectiveLabelSmoother()
        with pytest.raises(ValueError):
            smoother.smooth(np.zeros(10), np.zeros(15))

    def test_summarize_changes(self):
        labels = np.array([BULL_]*10 + [BEAR_]*5 + [BULL_]*10 + [BEAR_]*5)
        returns = np.zeros(30)
        returns[7] = -0.10
        returns[22] = -0.08

        smoother = RetrospectiveLabelSmoother()
        smoothed, log = smoother.smooth(labels, returns)
        summary = smoother.summarize_changes(log)
        assert summary['n_backdates'] == 2
        assert summary['mean_shift'] > 0
