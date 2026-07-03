"""
ADXClassifier — ADX 기반 윈도우 단위 국면 분류기.

────────────────────────────────────────────────────────────────────
분류 로직 (Soft, Sigmoid 기반)
────────────────────────────────────────────────────────────────────
ADX는 추세의 "강도"만 알려주고 방향은 모른다. 그래서 다음 2단계로
Bull/Side/Bear 3-way 확률을 계산한다:

    Step 1. P_trend = sigmoid(adx_steepness × (adx - threshold))
            → ADX가 임계값(25)을 넘는 정도를 0~1로 부드럽게 변환
            → 추세장(=Bull or Bear)일 가능성

    Step 2. bull_share = sigmoid(direction_steepness × cum_return)
            → cum_return의 부호와 크기를 0~1로 변환
            → 추세장 안에서 Bull vs Bear 분배 비율

    Step 3. P_Side = 1 - P_trend
            P_Bull = P_trend × bull_share
            P_Bear = P_trend × (1 - bull_share)

세 확률의 합 = 1.0 (수학적으로 보장).

────────────────────────────────────────────────────────────────────
시그모이드 기울기 (steepness) 의미
────────────────────────────────────────────────────────────────────
시그모이드는 σ(x) = 1 / (1 + exp(-x)) 형태로, 입력 x의 절댓값이
클수록 0 또는 1에 가까워진다. 기울기 곱셈은 "임계값 근처에서
얼마나 급격히 변하는가"를 조절한다.

ADX_CLF_STEEPNESS = 0.2 (기본값):
    ADX = 20 → σ(-1.0) ≈ 0.27 (Side 가능성 73%)
    ADX = 25 → σ(0.0)  = 0.50 (반반)
    ADX = 30 → σ(+1.0) ≈ 0.73 (추세 가능성 73%)
    ADX = 40 → σ(+3.0) ≈ 0.95 (강한 추세)

DIRECTION_STEEPNESS = 50 (기본값):
    cum_return = -3% → σ(-1.5) ≈ 0.18 (18% Bull, 82% Bear)
    cum_return =  0% → σ(0.0)  = 0.50 (반반)
    cum_return = +3% → σ(+1.5) ≈ 0.82 (82% Bull)

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy import config
    from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier

    clf = ADXClassifier(
        threshold=config.ADX_THRESHOLD,
        adx_steepness=config.ADX_CLF_STEEPNESS,
        direction_steepness=config.DIRECTION_STEEPNESS,
    )

    # 단일 윈도우
    label = clf.predict(features_df.iloc[100])         # 0/1/2
    proba = clf.predict_proba(features_df.iloc[100])   # [0.72, 0.12, 0.16]

    # 배치
    proba_batch = clf.predict_proba_batch(features_df) # (n, 3)
"""

import numpy as np
import pandas as pd

from strategy.HMM_strategy.classifiers.base_classifier import (
    BaseClassifier, BULL, SIDE, BEAR, REGIME_IDS,
)


def _sigmoid(x: float) -> float:
    """
    수치 안정성을 보장하는 시그모이드.

    np.exp(매우 큰 음수) = inf 가 되어 오버플로우 발생할 수 있으므로,
    x의 부호에 따라 식을 다르게 계산한다.
    """
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


class ADXClassifier(BaseClassifier):
    """
    ADX 기반 국면 분류기 (sigmoid soft probability).

    Args:
        threshold: ADX 추세/횡보 경계값 (기본 25)
        adx_steepness: 추세 sigmoid 기울기 (기본 0.2)
            작을수록 부드럽게, 클수록 급격하게 임계값 근처에서 변함.
        direction_steepness: 방향 sigmoid 기울기 (기본 50.0)
            cum_return의 단위가 작으므로 큰 값을 사용.
        adx_col: 윈도우 피처 중 ADX 값을 가져올 컬럼명 (기본 'adx_mean')
        return_col: 방향 판단에 쓸 컬럼명 (기본 'cum_return')

    필요한 입력 컬럼:
        - window[adx_col]: ADX 값 (기본 'adx_mean')
        - window[return_col]: 누적 수익률 (기본 'cum_return')
    """

    def __init__(
        self,
        threshold: float = 25.0,
        adx_steepness: float = 0.2,
        direction_steepness: float = 50.0,
        adx_col: str = 'adx_mean',
        return_col: str = 'cum_return',
    ):
        if adx_steepness <= 0:
            raise ValueError(f"adx_steepness must be > 0, got {adx_steepness}")
        if direction_steepness <= 0:
            raise ValueError(
                f"direction_steepness must be > 0, got {direction_steepness}"
            )
        self.threshold = threshold
        self.adx_steepness = adx_steepness
        self.direction_steepness = direction_steepness
        self.adx_col = adx_col
        self.return_col = return_col

    @property
    def name(self) -> str:
        return "adx"

    def predict_proba(self, window: pd.Series) -> np.ndarray:
        """
        단일 윈도우의 [P_Bull, P_Side, P_Bear] 계산.

        Returns:
            np.ndarray, shape (3,), 합 = 1.0
        """
        adx = float(window[self.adx_col])
        cum_ret = float(window[self.return_col])

        # NaN 안전: NaN이면 완전 횡보(중립) 반환
        if np.isnan(adx) or np.isnan(cum_ret):
            return np.array([1/3, 1/3, 1/3], dtype=np.float64)

        # Step 1. 추세 강도 (Side 반대 확률)
        p_trend = _sigmoid(self.adx_steepness * (adx - self.threshold))

        # Step 2. 방향 분배 (Bull vs Bear 비율)
        bull_share = _sigmoid(self.direction_steepness * cum_ret)

        # Step 3. 최종 확률 합성 — 합 = 1 보장
        p_side = 1.0 - p_trend
        p_bull = p_trend * bull_share
        p_bear = p_trend * (1.0 - bull_share)

        return np.array([p_bull, p_side, p_bear], dtype=np.float64)

    def predict(self, window: pd.Series) -> int:
        """
        가장 확률이 높은 국면 반환. 동률 시 BULL → SIDE → BEAR 우선순위.
        """
        proba = self.predict_proba(window)
        # argmax는 첫 번째 최댓값 인덱스 반환 (REGIME_IDS 순서)
        return int(REGIME_IDS[int(np.argmax(proba))])

    # ── 배치 메서드 — 벡터화로 성능 향상 ──────────────────────────
    def predict_proba_batch(self, windows: pd.DataFrame) -> np.ndarray:
        """
        벡터화된 배치 처리 (행마다 호출하는 기본 구현보다 빠름).
        """
        adx = windows[self.adx_col].to_numpy(dtype=np.float64)
        cum_ret = windows[self.return_col].to_numpy(dtype=np.float64)

        # NaN 마스크 — 나중에 균등분포로 채움
        nan_mask = np.isnan(adx) | np.isnan(cum_ret)

        # Step 1 + 2: 시그모이드 (벡터화)
        # _sigmoid 자체가 스칼라 함수이므로 numpy의 expit 같은 안정 버전 사용
        # 여기서는 직접 vectorize
        with np.errstate(over='ignore'):  # exp 오버플로우 무시
            p_trend = 1.0 / (1.0 + np.exp(-self.adx_steepness * (adx - self.threshold)))
            bull_share = 1.0 / (1.0 + np.exp(-self.direction_steepness * cum_ret))

        # Step 3
        p_side = 1.0 - p_trend
        p_bull = p_trend * bull_share
        p_bear = p_trend * (1.0 - bull_share)

        out = np.column_stack([p_bull, p_side, p_bear])

        # NaN 처리: 균등분포
        if nan_mask.any():
            out[nan_mask] = 1/3

        return out
