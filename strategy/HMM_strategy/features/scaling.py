"""
Rolling 윈도우 기반 StandardScaler.

────────────────────────────────────────────────────────────────────
왜 Rolling Scaler가 필요한가?
────────────────────────────────────────────────────────────────────
sklearn의 StandardScaler는 전체 학습 데이터의 mean/std를 한 번에 계산한다.
시계열에서는 이게 두 가지 문제를 일으킬 수 있다:

1. 시간 드리프트 (temporal drift)
   2018년 BTC ~3,000불, 2024년 ~70,000불 같이 자산 절대 스케일이 크게 변하면
   전체 평균/표준편차에 과거 저가 시기가 섞여 들어가 현재 정규화를 왜곡한다.

2. 국면 구조 왜곡
   같은 "Bull 국면"인데 2020년 데이터는 양의 z-score, 2024년 데이터는 음의 z-score를
   받게 되면 HMM이 둘을 다른 국면으로 학습할 위험이 있다.

RollingStandardScaler는 시점 t에서 [t-window+1, t] 구간의 mean/std로 정규화하여
이 두 문제를 동시에 해결한다. 단, 처음 (window-1) 행은 history 부족으로 NaN이 된다
("cold start" 손실 — 호출자가 dropna 처리).

────────────────────────────────────────────────────────────────────
룩어헤드 안전성
────────────────────────────────────────────────────────────────────
시점 t의 정규화에 [t-window+1, t]만 사용 — 미래 정보 누출 없음.
시점 t의 데이터 자체는 t의 정규화에 포함되지만, 이 결과는 t+1 이후의 모델
판단에 사용되므로 룩어헤드가 아니다 (Phase 1 보고서 3-4 룩어헤드 규칙 준수).
"""

import numpy as np
import pandas as pd


class RollingStandardScaler:
    """
    Rolling-window StandardScaler.

    각 시점 t에서:
        mean_t = X[t-window+1 : t+1].mean()
        std_t  = X[t-window+1 : t+1].std()
        scaled[t] = (X[t] - mean_t) / std_t

    Args:
        window: 정규화 기준 윈도우 크기 (봉 수)

    Raises:
        ValueError: window <= 1인 경우
    """

    def __init__(self, window: int):
        if window <= 1:
            raise ValueError(f"window must be > 1, got {window}")
        self.window = window

    def fit_transform(self, X) -> np.ndarray:
        """
        Rolling 정규화 적용.

        Args:
            X: pd.DataFrame 또는 np.ndarray, shape (n_samples, n_features)

        Returns:
            np.ndarray, shape (n_samples, n_features)
            처음 (window-1) 행은 NaN — 호출자가 dropna 또는 슬라이싱으로 제거.
            나머지 행은 (X[t] - rolling_mean) / rolling_std 결과.

        주의:
            - 윈도우 내 분산이 0(상수 피처)인 경우 std=0 → 분모 1로 대체하여 0 반환
              ("변동 없음 = 정규화 후에도 0" 의미적으로 일관)
            - X가 DataFrame이면 컬럼 순서/이름 보존 안 됨 (np.ndarray 반환)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            X = X.copy()  # 원본 보호

        # min_periods=window: 처음 (window-1)행은 NaN
        rolling = X.rolling(self.window, min_periods=self.window)
        mean = rolling.mean()
        std = rolling.std()  # 기본 ddof=1 (표본 표준편차)

        # 0 division 방지: std가 0에 가까우면 1로 대체
        # (X - mean = 0이므로 결과는 0이 되어 의미적으로 일관)
        std_safe = std.where(std > 1e-12, other=1.0)

        scaled = (X - mean) / std_safe
        return scaled.values

    def __repr__(self):
        return f"RollingStandardScaler(window={self.window})"
