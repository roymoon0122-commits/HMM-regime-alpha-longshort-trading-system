"""
BaseMetaModel — 메타 모델 추상 베이스 클래스.

────────────────────────────────────────────────────────────────────
역할
────────────────────────────────────────────────────────────────────
모든 메타 모델 (LogisticMetaModel, 추후 XGBoost/NN 등)이 따라야 할
공통 인터페이스를 정의. Python의 abc 모듈로 강제한다.

하위 클래스가 구현해야 할 4개 메서드:
    1. fit(X, y)              — 학습
    2. predict_proba(X)       — (n, 3) 확률 반환
    3. save(path)             — 디스크 저장
    4. load(path)             — 디스크 로드

자동 제공 메서드 (재정의 가능):
    - predict(X)              — argmax of predict_proba

────────────────────────────────────────────────────────────────────
입력 X의 의미 (Phase 3 기획)
────────────────────────────────────────────────────────────────────
shape (n_samples, n_features) 행렬. 호출자(verify_meta_model.py 등)가
다음 피처들을 가로로 연결해서 만든다:

    [ADX 분류기 출력      : 3개]   adx_p_bull, adx_p_side, adx_p_bear
    [R²  분류기 출력      : 3개]   r2_p_bull,  r2_p_side,  r2_p_bear
    [윈도우 피처          : 4개]   cum_return, volatility, adx_mean, r2_mean
    [HMM 사후확률         : 3개]   hmm_p_bull, hmm_p_side, hmm_p_bear
    [마르코프 전이 후 확률 : 3개]   trans_p_bull, trans_p_side, trans_p_bear

총 16개 피처 예상. 메타 모델은 이 X를 받아 다음 윈도우의 국면을 예측.

라벨 y: 다음 윈도우의 HMM 라벨 (0=Bull, 1=Side, 2=Bear).
RegimeDataset.get_y(shift=-1)로 얻음.

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy.meta_model.logistic_meta_model import LogisticMetaModel

    meta = LogisticMetaModel()
    meta.fit(X_train, y_train)
    proba = meta.predict_proba(X_test)   # (n, 3)
    pred  = meta.predict(X_test)         # (n,) — 0/1/2
    meta.save("models/meta_logistic.joblib")
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseMetaModel(ABC):
    """
    메타 모델 추상 베이스 클래스.

    하위 클래스는 다음 4개 메서드를 구현해야 한다:
        - fit(X, y) -> self
        - predict_proba(X) -> np.ndarray, shape (n, 3)
        - save(path) -> None
        - load(path) -> None

    클래스 라벨 컨벤션 (변경 금지):
        0 = Bull
        1 = Side
        2 = Bear
    BaseClassifier, HMMLabeler와 동일.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMetaModel':
        """
        메타 모델 학습.

        Args:
            X: shape (n_samples, n_features) — 메타 입력 피처 행렬
            y: shape (n_samples,) — 정수 라벨 (0=Bull, 1=Side, 2=Bear)
                다음 윈도우의 HMM 라벨 (RegimeDataset.get_y(shift=-1)로 생성)

        Returns:
            self (메서드 체이닝 가능)
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        클래스별 확률 예측.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            np.ndarray, shape (n_samples, 3) — [P_Bull, P_Side, P_Bear]
            각 행 합 = 1.0
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        가장 확률 높은 클래스 예측 (자동 제공, 하위 재정의 가능).

        Args:
            X: shape (n_samples, n_features)

        Returns:
            np.ndarray, shape (n_samples,) — 정수 라벨 (0/1/2)
        """
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    @abstractmethod
    def save(self, path: str) -> None:
        """모델을 디스크에 저장."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """디스크에서 모델 로드 (in-place)."""
        ...
