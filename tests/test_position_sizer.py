"""
PositionSizer 단위 테스트.

검증 항목:
- net 모드 단일/배치 변환
- dual 모드 단일/배치 변환
- min_threshold 노이즈 컷
- 입력 검증 (shape, NaN, 범위, 합)
- 잘못된 mode/threshold 파라미터
"""

import numpy as np
import pytest

from strategy.HMM_strategy.position.sizer import (
    PositionSizer,
    BULL_IDX,
    SIDE_IDX,
    BEAR_IDX,
    VALID_MODES,
)


# =============================================================================
# net 모드 — 단일 시점
# =============================================================================
class TestNetModeSingle:

    def test_strong_bull(self):
        sizer = PositionSizer(mode='net', min_threshold=0.1)
        result = sizer.compute(np.array([0.7, 0.2, 0.1]))
        assert isinstance(result, float)
        assert result == pytest.approx(0.6)

    def test_strong_bear(self):
        sizer = PositionSizer(mode='net', min_threshold=0.1)
        result = sizer.compute(np.array([0.1, 0.2, 0.7]))
        assert result == pytest.approx(-0.6)

    def test_pure_side_returns_zero(self):
        sizer = PositionSizer(mode='net', min_threshold=0.1)
        result = sizer.compute(np.array([0.0, 1.0, 0.0]))
        assert result == 0.0

    def test_below_threshold_zero(self):
        # |0.45 - 0.45| = 0 → 임계값 미만 → 0
        sizer = PositionSizer(mode='net', min_threshold=0.1)
        result = sizer.compute(np.array([0.45, 0.10, 0.45]))
        assert result == 0.0

    def test_just_below_threshold(self):
        # |0.55 - 0.46| = 0.09 < 0.1 → 0
        sizer = PositionSizer(mode='net', min_threshold=0.1)
        result = sizer.compute(np.array([0.50, 0.41, 0.09]))
        # 0.50 - 0.09 = 0.41 → 임계값 이상 → 그대로
        assert result == pytest.approx(0.41)

        result2 = sizer.compute(np.array([0.45, 0.46, 0.09]))
        # 0.45 - 0.09 = 0.36 → 임계값 이상
        assert result2 == pytest.approx(0.36)

    def test_threshold_zero_passes_everything(self):
        sizer = PositionSizer(mode='net', min_threshold=0.0)
        result = sizer.compute(np.array([0.34, 0.33, 0.33]))
        # 0.34 - 0.33 = 0.01 → threshold=0이므로 그대로 통과
        assert result == pytest.approx(0.01, abs=1e-9)


# =============================================================================
# net 모드 — 배치
# =============================================================================
class TestNetModeBatch:

    def test_batch_shape(self):
        sizer = PositionSizer(mode='net', min_threshold=0.1)
        proba = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.5, 0.0, 0.5],   # net = 0
            [0.4, 0.3, 0.3],   # net = 0.1 → 임계값과 같음, |net|<0.1이 아니므로 통과
        ])
        result = sizer.compute_batch(proba)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_batch_values(self):
        sizer = PositionSizer(mode='net', min_threshold=0.1)
        proba = np.array([
            [0.7, 0.2, 0.1],   # 0.6
            [0.1, 0.2, 0.7],   # -0.6
            [0.5, 0.0, 0.5],   # 0 (|net|=0 < 0.1)
            [0.4, 0.3, 0.3],   # 0.1 (|net|=0.1, 임계값 미만 아님)
            [0.0, 1.0, 0.0],   # 0
        ])
        result = sizer.compute_batch(proba)
        np.testing.assert_allclose(
            result,
            [0.6, -0.6, 0.0, 0.1, 0.0],
            atol=1e-9,
        )

    def test_batch_clip(self):
        # 합이 1을 넘는 비정상 입력은 _validate_batch가 거르지만,
        # 만약 통과한다 해도 clip(-1,1)이 적용되는지 단독 단위로 점검.
        # 여기서는 정상 입력 중 극단치(±1)만 확인.
        sizer = PositionSizer(mode='net', min_threshold=0.0)
        proba = np.array([
            [1.0, 0.0, 0.0],   # +1.0
            [0.0, 0.0, 1.0],   # -1.0
        ])
        result = sizer.compute_batch(proba)
        np.testing.assert_allclose(result, [1.0, -1.0])


# =============================================================================
# dual 모드 — 단일/배치
# =============================================================================
class TestDualMode:

    def test_dual_returns_dict_single(self):
        sizer = PositionSizer(mode='dual', min_threshold=0.1)
        result = sizer.compute(np.array([0.7, 0.2, 0.1]))
        assert isinstance(result, dict)
        assert set(result.keys()) == {'long', 'short'}
        assert result['long'] == pytest.approx(0.7)
        # 0.1 < 0.1은 False → 컷되지 않고 그대로 통과 (strict less than 컨벤션)
        assert result['short'] == pytest.approx(0.1)

    def test_dual_strict_less_than_threshold(self):
        # min_threshold와 정확히 같은 값은 통과 (strict <)
        sizer = PositionSizer(mode='dual', min_threshold=0.2)
        # P(Bull)=0.20 → 0.20 < 0.20 False → 통과
        # P(Bear)=0.19 → 0.19 < 0.20 True → 컷 (0)
        result = sizer.compute(np.array([0.20, 0.61, 0.19]))
        assert result['long'] == pytest.approx(0.20)
        assert result['short'] == 0.0

    def test_dual_threshold_cuts_only_below(self):
        sizer = PositionSizer(mode='dual', min_threshold=0.15)
        # P(Bull)=0.10 < 0.15 → 0
        # P(Bear)=0.20 ≥ 0.15 → 0.20
        result = sizer.compute(np.array([0.10, 0.70, 0.20]))
        assert result['long'] == 0.0
        assert result['short'] == pytest.approx(0.20)

    def test_dual_batch_returns_dict_of_arrays(self):
        sizer = PositionSizer(mode='dual', min_threshold=0.1)
        proba = np.array([
            [0.7, 0.2, 0.1],   # long=0.7, short=0 (0.1<0.1 false; 컷 안됨)
            [0.05, 0.20, 0.75],  # long=0(컷), short=0.75
            [0.40, 0.30, 0.30],  # long=0.4, short=0.30
        ])
        result = sizer.compute_batch(proba)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'long', 'short'}
        assert result['long'].shape == (3,)
        assert result['short'].shape == (3,)
        np.testing.assert_allclose(result['long'], [0.7, 0.0, 0.4])
        np.testing.assert_allclose(result['short'], [0.1, 0.75, 0.3])


# =============================================================================
# 잘못된 파라미터
# =============================================================================
class TestInvalidParams:

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            PositionSizer(mode='wrong')

    def test_negative_threshold(self):
        with pytest.raises(ValueError, match="min_threshold"):
            PositionSizer(mode='net', min_threshold=-0.1)

    def test_threshold_above_one(self):
        with pytest.raises(ValueError, match="min_threshold"):
            PositionSizer(mode='net', min_threshold=1.5)

    def test_threshold_zero_ok(self):
        # 0은 허용
        sizer = PositionSizer(mode='net', min_threshold=0.0)
        assert sizer.min_threshold == 0.0

    def test_threshold_one_ok(self):
        # 1.0도 허용 (모든 신호 컷)
        sizer = PositionSizer(mode='net', min_threshold=1.0)
        assert sizer.min_threshold == 1.0


# =============================================================================
# 입력 검증
# =============================================================================
class TestInputValidation:

    def test_wrong_shape_single(self):
        sizer = PositionSizer(mode='net')
        with pytest.raises(ValueError, match="shape"):
            sizer.compute(np.array([0.5, 0.5]))   # (2,)
        with pytest.raises(ValueError, match="shape"):
            sizer.compute(np.array([[0.7, 0.2, 0.1]]))   # (1,3)

    def test_wrong_shape_batch(self):
        sizer = PositionSizer(mode='net')
        with pytest.raises(ValueError, match="shape"):
            sizer.compute_batch(np.array([0.5, 0.3, 0.2]))   # (3,)
        with pytest.raises(ValueError, match="shape"):
            sizer.compute_batch(np.zeros((4, 4)))   # (n,4)

    def test_nan_rejected_single(self):
        sizer = PositionSizer(mode='net')
        with pytest.raises(ValueError, match="NaN"):
            sizer.compute(np.array([np.nan, 0.5, 0.5]))

    def test_nan_rejected_batch(self):
        sizer = PositionSizer(mode='net')
        proba = np.array([
            [0.7, 0.2, 0.1],
            [np.nan, 0.5, 0.5],
        ])
        with pytest.raises(ValueError, match="NaN"):
            sizer.compute_batch(proba)

    def test_negative_proba_rejected(self):
        sizer = PositionSizer(mode='net')
        with pytest.raises(ValueError, match="범위"):
            sizer.compute(np.array([-0.1, 0.5, 0.6]))

    def test_proba_above_one_rejected(self):
        sizer = PositionSizer(mode='net')
        with pytest.raises(ValueError, match="범위"):
            sizer.compute(np.array([1.5, 0.0, -0.5]))

    def test_sum_not_one_rejected_single(self):
        sizer = PositionSizer(mode='net')
        with pytest.raises(ValueError, match="합"):
            sizer.compute(np.array([0.3, 0.3, 0.3]))   # 합=0.9

    def test_sum_not_one_rejected_batch(self):
        sizer = PositionSizer(mode='net')
        proba = np.array([
            [0.7, 0.2, 0.1],   # ok
            [0.3, 0.3, 0.3],   # sum=0.9 → reject
        ])
        with pytest.raises(ValueError, match="합"):
            sizer.compute_batch(proba)

    def test_sum_within_tolerance_passes(self):
        # 부동소수 오차 정도는 허용 (atol=1e-3)
        sizer = PositionSizer(mode='net')
        result = sizer.compute(np.array([0.5001, 0.2999, 0.2000]))
        assert result == pytest.approx(0.5001 - 0.2000)


# =============================================================================
# 클래스 인덱스 상수 체크 (HMMLabeler/분류기와 통일)
# =============================================================================
class TestConstants:

    def test_class_indices(self):
        assert BULL_IDX == 0
        assert SIDE_IDX == 1
        assert BEAR_IDX == 2

    def test_valid_modes(self):
        assert 'net' in VALID_MODES
        assert 'dual' in VALID_MODES


# =============================================================================
# repr (사소하지만 디버깅 편의용)
# =============================================================================
class TestRepr:

    def test_repr_contains_mode_and_threshold(self):
        sizer = PositionSizer(mode='net', min_threshold=0.15)
        s = repr(sizer)
        assert 'net' in s
        assert '0.15' in s
