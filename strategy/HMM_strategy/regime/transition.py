"""
TransitionPredictor — HMM 전이 행렬 기반 다음 윈도우 사전확률 계산.

────────────────────────────────────────────────────────────────────
핵심 아이디어
────────────────────────────────────────────────────────────────────
HMM이 학습 후 가지는 transmat (전이 행렬)은:

    transmat[i, j] = P(next_state = j | current_state = i)

각 행 합 = 1.0. 예를 들어 3-state HMM이면:

           → Bull   → Side   → Bear
    Bull   [ 0.85,   0.13,    0.02  ]
    Side   [ 0.05,   0.90,    0.05  ]
    Bear   [ 0.02,   0.13,    0.85  ]

이걸 사용하면 현재 사후확률 P_t = [P(Bull), P(Side), P(Bear)]에서
다음 윈도우의 사전확률 P_{t+1|t}를 계산할 수 있다:

    P_{t+1|t} = P_t @ transmat              # shape (3,) @ (3,3) = (3,)

예: P_t = [0.7, 0.2, 0.1]
    P_{t+1|t} = [0.7*0.85 + 0.2*0.05 + 0.1*0.02,   # → Bull
                 0.7*0.13 + 0.2*0.90 + 0.1*0.13,   # → Side
                 0.7*0.02 + 0.2*0.05 + 0.1*0.85]   # → Bear
              = [0.607, 0.284, 0.109]

이 값이 메타 모델 입력 피처로 들어간다.
HMM 사후확률(현재)과 함께 시간적 동역학(다음 시점 예상)도 메타 모델에
제공하기 위함.

────────────────────────────────────────────────────────────────────
상태 매핑 — 왜 별도로 처리하는가?
────────────────────────────────────────────────────────────────────
hmmlearn의 GaussianHMM.transmat_은 "내부 state ID" 기준이다.
내부 state ID는 학습 결과에 따라 무작위:
  - 어떤 학습 결과: state 0 = Bear, state 1 = Bull, state 2 = Side
  - 다른 학습 결과: state 0 = Bull, state 1 = Bear, state 2 = Side

HMMLabeler는 이걸 cum_return 평균 기준으로 자동 매핑해서:
  - regime ID 0 = Bull (사용자가 보는 라벨)
  - regime ID 1 = Side
  - regime ID 2 = Bear

따라서 TransitionPredictor를 쓰려면 transmat을
[Bull, Side, Bear] 순서로 재배열해야 한다 — 그래야 사용자가 받는 사후확률
(predict_proba가 Bull/Side/Bear 순서로 반환)과 곱셈 차원이 맞는다.

from_labeler 클래스 메서드가 이 재배열을 자동으로 처리한다.

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy.regime.hmm_labeler import HMMLabeler
    from strategy.HMM_strategy.regime.transition import TransitionPredictor

    labeler = HMMLabeler()
    labeler.load("models/hmm_btc.joblib")

    predictor = TransitionPredictor.from_labeler(labeler)
    print(predictor.transmat)  # Bull/Side/Bear 순서로 재배열됨

    # 단일 시점
    proba_t = labeler.predict_proba(X)[100]      # shape (3,)
    proba_next = predictor.predict_next(proba_t)  # shape (3,)

    # 배치
    proba_batch = labeler.predict_proba(X)              # shape (n, 3)
    next_batch = predictor.predict_next_batch(proba_batch)  # shape (n, 3)
"""

import numpy as np

from strategy.HMM_strategy.classifiers.base_classifier import (
    BULL, SIDE, BEAR,
)


class TransitionPredictor:
    """
    마르코프 전이 행렬 기반 다음 윈도우 사전확률 계산기.

    Args:
        transmat: shape (n_states, n_states) 전이 행렬.
                  transmat[i, j] = P(next=j | current=i)
                  ★ 이미 [Bull, Side, Bear] 순서로 재배열된 행렬을 받아야 함.
                    HMMLabeler에서 가져올 때는 from_labeler 클래스 메서드 사용 권장.
                  각 행 합 ≈ 1.0 (수치 오차 허용 범위 내)

    Raises:
        ValueError: 정사각 행렬이 아니거나 행 합이 1이 아닌 경우
    """

    # 행 합 = 1.0 검증 시 허용 오차 (hmmlearn이 가끔 1e-15 오차 줌)
    ROW_SUM_TOLERANCE = 1e-6

    def __init__(self, transmat: np.ndarray):
        transmat = np.asarray(transmat, dtype=np.float64)

        # 검증 1: 2차원 정사각 행렬
        if transmat.ndim != 2 or transmat.shape[0] != transmat.shape[1]:
            raise ValueError(
                f"transmat must be a square 2D matrix, got shape {transmat.shape}"
            )

        # 검증 2: 각 행 합 = 1
        row_sums = transmat.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=self.ROW_SUM_TOLERANCE):
            raise ValueError(
                f"transmat rows must sum to 1.0 (within tol={self.ROW_SUM_TOLERANCE}); "
                f"got row sums: {row_sums.tolist()}"
            )

        # 검증 3: 음수 없음
        if (transmat < 0).any():
            raise ValueError("transmat contains negative entries — invalid probabilities")

        self.transmat = transmat
        self.n_states = transmat.shape[0]

    # ── 단일 시점 ──────────────────────────────────────────────
    def predict_next(self, current_proba: np.ndarray) -> np.ndarray:
        """
        현재 사후확률 → 다음 윈도우 사전확률.

        Args:
            current_proba: shape (n_states,), 합 ≈ 1.0
                ★ 컬럼 순서가 transmat과 일치해야 함 (Bull/Side/Bear).

        Returns:
            shape (n_states,), 합 ≈ 1.0
        """
        current_proba = np.asarray(current_proba, dtype=np.float64).ravel()
        if current_proba.shape[0] != self.n_states:
            raise ValueError(
                f"current_proba length {current_proba.shape[0]} "
                f"does not match n_states {self.n_states}"
            )
        return current_proba @ self.transmat

    # ── 배치 ─────────────────────────────────────────────────
    def predict_next_batch(self, current_proba_batch: np.ndarray) -> np.ndarray:
        """
        여러 시점 한 번에 처리.

        Args:
            current_proba_batch: shape (n, n_states)

        Returns:
            shape (n, n_states)
        """
        current_proba_batch = np.asarray(current_proba_batch, dtype=np.float64)
        if current_proba_batch.ndim != 2 or current_proba_batch.shape[1] != self.n_states:
            raise ValueError(
                f"Expected shape (n, {self.n_states}), got {current_proba_batch.shape}"
            )
        return current_proba_batch @ self.transmat

    # ── HMMLabeler에서 자동 추출 + 재배열 ──────────────────────
    @classmethod
    def from_labeler(cls, labeler) -> 'TransitionPredictor':
        """
        HMMLabeler에서 transmat을 추출하고 [Bull, Side, Bear] 순서로 재배열.

        Args:
            labeler: 학습 완료된 HMMLabeler 인스턴스.

        Returns:
            TransitionPredictor — Bull/Side/Bear 순서로 재배열된 transmat 보유.
        """
        if labeler.model_ is None or labeler.regime_to_state_ is None:
            raise ValueError(
                "HMMLabeler must be fit before extracting transition matrix"
            )

        raw_transmat = labeler.model_.transmat_

        # regime_to_state_ = {0(Bull): X, 1(Side): Y, 2(Bear): Z} 형태
        # 우리는 transmat을 [Bull, Side, Bear] 순서로 재배열해야 함
        # → 내부 state 인덱스 순서: [X, Y, Z]
        regime_order = [BULL, SIDE, BEAR]

        # n_states != 3 인 경우는 일반화 (BIC 결과 등)
        # regime_to_state_의 모든 키 사용
        if labeler.n_states != 3:
            # 정렬된 regime ID 순서로 추출 (0, 1, 2, ...)
            regime_order = sorted(labeler.regime_to_state_.keys())

        state_order = [labeler.regime_to_state_[r] for r in regime_order]

        # numpy의 ix_로 행/열 동시 재배열
        # transmat[ix_(state_order, state_order)][i, j]
        #   = raw_transmat[state_order[i], state_order[j]]
        transmat_remapped = raw_transmat[np.ix_(state_order, state_order)]

        return cls(transmat_remapped)
