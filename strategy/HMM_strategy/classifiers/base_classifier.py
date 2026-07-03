"""
Base Classifier 추상 베이스 클래스.

────────────────────────────────────────────────────────────────────
역할
────────────────────────────────────────────────────────────────────
모든 윈도우 단위 국면 분류기(ADX, R², 추후 추가될 것들)가 따라야 할
공통 인터페이스를 정의한다. Python의 abc 모듈로 두 메서드를
하위 클래스에 강제한다:

    1. predict(window)        → int            (BULL/SIDE/BEAR)
    2. predict_proba(window)  → np.ndarray (3,) [P_Bull, P_Side, P_Bear]

배치 메서드(predict_batch / predict_proba_batch)는 자동 제공되므로
하위 클래스는 단일 메서드만 구현하면 된다.

────────────────────────────────────────────────────────────────────
국면 ID 정수값 (HMMLabeler와 통일)
────────────────────────────────────────────────────────────────────
BULL = 0
SIDE = 1
BEAR = 2

기획서(4-3절)는 1/0/-1을 제안했으나, HMMLabeler가 이미 0/1/2를
사용하므로 메타 모델 라벨 정합성을 위해 0/1/2로 통일한다.
변경사항은 phase3_work_report.md에 기록.

────────────────────────────────────────────────────────────────────
입력 'window'의 형태
────────────────────────────────────────────────────────────────────
pd.Series 한 행 — compute_window_features() 출력의 한 행을 그대로 전달.
인덱스(피처명)로 접근: window['adx_mean'], window['cum_return'] 등.

배치는 pd.DataFrame을 받아 행마다 단일 메서드를 호출한다.

────────────────────────────────────────────────────────────────────
사용 예시 (Phase 3 메타 모델에서)
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier
    from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier

    classifiers = [ADXClassifier(threshold=25), R2Classifier(threshold=0.55)]

    for clf in classifiers:
        proba_arr = clf.predict_proba_batch(features_df)  # (n, 3)
        # → 메타 모델 입력 X에 추가
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ─── 국면 ID 정수값 (HMMLabeler와 통일) ────────────────────────
BULL = 0
SIDE = 1
BEAR = 2

REGIME_NAMES = {BULL: 'Bull', SIDE: 'Side', BEAR: 'Bear'}
REGIME_IDS = [BULL, SIDE, BEAR]    # 확률 배열 컬럼 순서 고정


class BaseClassifier(ABC):
    """
    윈도우 단위 국면 분류기의 추상 베이스 클래스.

    하위 클래스가 반드시 구현해야 할 메서드:
        - predict(window: pd.Series) -> int
        - predict_proba(window: pd.Series) -> np.ndarray (shape: (3,))

    배치 메서드 (자동 제공, 하위 클래스 재정의 가능):
        - predict_batch(windows: pd.DataFrame) -> np.ndarray (shape: (n,))
        - predict_proba_batch(windows: pd.DataFrame) -> np.ndarray (shape: (n, 3))
    """

    @abstractmethod
    def predict(self, window: pd.Series) -> int:
        """
        단일 윈도우의 국면 라벨.

        Args:
            window: pd.Series — 한 윈도우의 피처들
                (compute_window_features() 출력의 한 행)

        Returns:
            int — BULL(0) / SIDE(1) / BEAR(2)
        """
        ...

    @abstractmethod
    def predict_proba(self, window: pd.Series) -> np.ndarray:
        """
        단일 윈도우의 국면 확률.

        Args:
            window: pd.Series — 한 윈도우의 피처들

        Returns:
            np.ndarray, shape (3,) — [P_Bull, P_Side, P_Bear]
            합 = 1.0
        """
        ...

    # ── 배치 메서드 (자동 제공) ─────────────────────────────────
    def predict_batch(self, windows: pd.DataFrame) -> np.ndarray:
        """
        여러 윈도우 한 번에 라벨 예측.

        Args:
            windows: pd.DataFrame — 각 행이 한 윈도우

        Returns:
            np.ndarray, shape (n,) — 정수 라벨 배열
        """
        return np.array(
            [self.predict(row) for _, row in windows.iterrows()],
            dtype=np.int64,
        )

    def predict_proba_batch(self, windows: pd.DataFrame) -> np.ndarray:
        """
        여러 윈도우 한 번에 확률 예측.

        Args:
            windows: pd.DataFrame — 각 행이 한 윈도우

        Returns:
            np.ndarray, shape (n, 3) — [[P_Bull, P_Side, P_Bear], ...]
            각 행 합 = 1.0
        """
        return np.vstack(
            [self.predict_proba(row) for _, row in windows.iterrows()]
        )

    # ── 분류기 식별자 ──────────────────────────────────────────
    @property
    def name(self) -> str:
        """
        분류기 이름 — 메타 모델 피처명 prefix로 사용.

        예: ADXClassifier → "ADXClassifier_P_Bull", ...

        하위 클래스에서 더 간결한 이름이 필요하면 재정의:
            class ADXClassifier(BaseClassifier):
                @property
                def name(self):
                    return "adx"
        """
        return self.__class__.__name__
