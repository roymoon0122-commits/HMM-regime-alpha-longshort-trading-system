"""
PositionSizer — 메타 모델 확률 → 포지션 비중 변환.

────────────────────────────────────────────────────────────────────
핵심 책임
────────────────────────────────────────────────────────────────────
입력:  [P(Bull), P(Side), P(Bear)]  (각 행은 합 ≈ 1)
출력:  포지션 비중
       - mode='net'  → float 단일 값 (-1.0 ~ +1.0)
                         net = P(Bull) - P(Bear)
                         |net| < min_threshold 이면 0으로 컷 (노이즈 제거)
       - mode='dual' → dict {'long': P(Bull), 'short': P(Bear)}
                         (각각에 min_threshold 적용)

────────────────────────────────────────────────────────────────────
설계 원칙 — Pattern B (config 직접 import 금지)
────────────────────────────────────────────────────────────────────
함수 내부에서 config를 직접 import 하지 않음. 호출자(caller)가
config.POSITION_MODE / config.MIN_POSITION_THRESHOLD 등을 인자로
명시 전달함. (config.py 파일 상단 사용 원칙 참조)

────────────────────────────────────────────────────────────────────
룩어헤드 안전성
────────────────────────────────────────────────────────────────────
PositionSizer는 입력으로 받은 확률 외에는 어떤 데이터에도 접근하지
않음. 따라서 룩어헤드 위험 없음. 호출자가 t시점의 확률을 입력하면
t시점의 비중이 그대로 출력됨.

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy import config
    from strategy.HMM_strategy.position.sizer import PositionSizer

    sizer = PositionSizer(
        mode=config.POSITION_MODE,                  # 'net'
        min_threshold=config.MIN_POSITION_THRESHOLD,  # 0.1
    )

    # 단일 시점
    weight = sizer.compute(np.array([0.7, 0.2, 0.1]))
    # → 0.6  (= 0.7 - 0.1)

    # 배치 (메타 모델 출력 전체)
    proba_batch = meta_model.predict_proba(X_meta)   # shape (n, 3)
    weights = sizer.compute_batch(proba_batch)       # shape (n,)
"""

from typing import Union, Dict
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 클래스 라벨 인덱스 (HMMLabeler / 분류기와 통일)
# ─────────────────────────────────────────────────────────────────
BULL_IDX = 0
SIDE_IDX = 1
BEAR_IDX = 2

VALID_MODES = ('net', 'dual')


class PositionSizer:
    """확률 → 포지션 비중 변환기.

    Parameters
    ----------
    mode : {'net', 'dual'}
        'net'  : 단일 float 비중 (-1.0 ~ +1.0). P(Bull) - P(Bear).
        'dual' : {'long': float, 'short': float} 분리형 (양방향 동시 보유 가능).
    min_threshold : float, default 0.1
        net 모드: |net| < min_threshold 이면 0으로 처리 (노이즈 컷).
        dual 모드: long/short 각각 min_threshold 미만이면 0으로 처리.
        0~1 범위.
    """

    def __init__(
        self,
        mode: str = 'net',
        min_threshold: float = 0.1,
    ):
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode는 {VALID_MODES} 중 하나여야 합니다. 받은 값: {mode!r}"
            )
        if not (0.0 <= min_threshold <= 1.0):
            raise ValueError(
                f"min_threshold는 [0, 1] 범위여야 합니다. 받은 값: {min_threshold}"
            )

        self.mode = mode
        self.min_threshold = float(min_threshold)

    # ─────────────────────────────────────────────────────────────
    # 단일 시점 변환
    # ─────────────────────────────────────────────────────────────
    def compute(self, proba: np.ndarray) -> Union[float, Dict[str, float]]:
        """단일 확률 벡터 → 포지션 비중.

        Parameters
        ----------
        proba : np.ndarray, shape (3,)
            [P(Bull), P(Side), P(Bear)]. 합이 1에 가까워야 함 (±1e-3 허용).

        Returns
        -------
        - mode='net'  : float (-1.0 ~ +1.0)
        - mode='dual' : {'long': float, 'short': float}
        """
        proba = self._validate_single(proba)

        if self.mode == 'net':
            return self._compute_net_single(proba)
        else:  # 'dual'
            return self._compute_dual_single(proba)

    # ─────────────────────────────────────────────────────────────
    # 배치 변환 (벡터화)
    # ─────────────────────────────────────────────────────────────
    def compute_batch(self, proba_batch: np.ndarray) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """배치 확률 행렬 → 포지션 비중 배열.

        Parameters
        ----------
        proba_batch : np.ndarray, shape (n, 3)
            행마다 [P(Bull), P(Side), P(Bear)].

        Returns
        -------
        - mode='net'  : np.ndarray (n,) float64
        - mode='dual' : {'long': np.ndarray (n,), 'short': np.ndarray (n,)}
        """
        proba_batch = self._validate_batch(proba_batch)

        if self.mode == 'net':
            return self._compute_net_batch(proba_batch)
        else:  # 'dual'
            return self._compute_dual_batch(proba_batch)

    # ─────────────────────────────────────────────────────────────
    # 내부 — net 모드
    # ─────────────────────────────────────────────────────────────
    def _compute_net_single(self, proba: np.ndarray) -> float:
        net = float(proba[BULL_IDX] - proba[BEAR_IDX])
        if abs(net) < self.min_threshold:
            return 0.0
        # |net|이 1을 넘는 일은 확률 정의상 불가능하지만 안전하게 클립
        return float(np.clip(net, -1.0, 1.0))

    def _compute_net_batch(self, proba_batch: np.ndarray) -> np.ndarray:
        net = proba_batch[:, BULL_IDX] - proba_batch[:, BEAR_IDX]
        # 임계값 미만 → 0
        net = np.where(np.abs(net) < self.min_threshold, 0.0, net)
        return np.clip(net, -1.0, 1.0)

    # ─────────────────────────────────────────────────────────────
    # 내부 — dual 모드
    # ─────────────────────────────────────────────────────────────
    def _compute_dual_single(self, proba: np.ndarray) -> Dict[str, float]:
        long_w = float(proba[BULL_IDX])
        short_w = float(proba[BEAR_IDX])
        if long_w < self.min_threshold:
            long_w = 0.0
        if short_w < self.min_threshold:
            short_w = 0.0
        return {'long': long_w, 'short': short_w}

    def _compute_dual_batch(self, proba_batch: np.ndarray) -> Dict[str, np.ndarray]:
        long_w = proba_batch[:, BULL_IDX].copy()
        short_w = proba_batch[:, BEAR_IDX].copy()
        long_w = np.where(long_w < self.min_threshold, 0.0, long_w)
        short_w = np.where(short_w < self.min_threshold, 0.0, short_w)
        return {'long': long_w, 'short': short_w}

    # ─────────────────────────────────────────────────────────────
    # 입력 검증
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_single(proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=np.float64)
        if proba.shape != (3,):
            raise ValueError(
                f"proba shape는 (3,) 여야 합니다. 받은 shape: {proba.shape}"
            )
        if np.any(np.isnan(proba)):
            raise ValueError("proba에 NaN 포함됨.")
        if np.any(proba < -1e-9) or np.any(proba > 1.0 + 1e-9):
            raise ValueError(
                f"proba 각 원소는 [0, 1] 범위여야 합니다. 받은 값: {proba}"
            )
        s = proba.sum()
        if not np.isclose(s, 1.0, atol=1e-3):
            raise ValueError(
                f"proba 합은 1이어야 합니다 (±1e-3 허용). 받은 합: {s:.6f}"
            )
        return proba

    @staticmethod
    def _validate_batch(proba_batch: np.ndarray) -> np.ndarray:
        proba_batch = np.asarray(proba_batch, dtype=np.float64)
        if proba_batch.ndim != 2 or proba_batch.shape[1] != 3:
            raise ValueError(
                f"proba_batch shape는 (n, 3) 여야 합니다. "
                f"받은 shape: {proba_batch.shape}"
            )
        if np.any(np.isnan(proba_batch)):
            raise ValueError("proba_batch에 NaN 포함됨.")
        if np.any(proba_batch < -1e-9) or np.any(proba_batch > 1.0 + 1e-9):
            raise ValueError("proba_batch 각 원소는 [0, 1] 범위여야 합니다.")
        sums = proba_batch.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-3):
            bad_idx = int(np.argmax(np.abs(sums - 1.0)))
            raise ValueError(
                f"proba_batch 각 행의 합은 1이어야 합니다 (±1e-3 허용). "
                f"가장 어긋난 행: idx={bad_idx}, sum={sums[bad_idx]:.6f}"
            )
        return proba_batch

    # ─────────────────────────────────────────────────────────────
    # 디버깅 편의
    # ─────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (f"PositionSizer(mode={self.mode!r}, "
                f"min_threshold={self.min_threshold})")
