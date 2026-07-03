"""
R2Classifier — R² 기반 윈도우 단위 국면 분류기.

────────────────────────────────────────────────────────────────────
ADXClassifier와의 차이점
────────────────────────────────────────────────────────────────────
1. 추세 강도 지표: r2_mean (직선성, 0~1 범위) — ADX 대신
2. 방향 지표:    slope_norm (정규화된 slope) — cum_return 대신
3. steepness:    R²는 0~1이라 큰 기울기, slope_norm은 z-score라 1.0

────────────────────────────────────────────────────────────────────
정규화된 slope (slope_norm)
────────────────────────────────────────────────────────────────────
slope (회귀 기울기)는 가격($/bar) 단위라 절대 스케일이 시간에 따라
크게 변한다 (BTC 3,000 vs 70,000 시기). 분류기에 넣기 전 호출자가
RollingStandardScaler로 z-score화해야 한다.

호출자(verify_meta_model.py 등)의 책임:
    from strategy.HMM_strategy.features.scaling import RollingStandardScaler
    scaler = RollingStandardScaler(window=2200)
    features['slope_norm'] = scaler.fit_transform(
        features[['slope']].values
    ).flatten()
    # 처음 (window-1)행은 NaN — predict_proba가 NaN-safe로 처리

분류기 자체는 정규화 안 함 (Single Responsibility).

────────────────────────────────────────────────────────────────────
분류 로직 (Soft, Sigmoid 기반)
────────────────────────────────────────────────────────────────────
    Step 1. P_trend = sigmoid(r2_steepness × (r2_mean - threshold))
    Step 2. bull_share = sigmoid(direction_steepness × slope_norm)
    Step 3. P_Side = 1 - P_trend
            P_Bull = P_trend × bull_share
            P_Bear = P_trend × (1 - bull_share)

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy import config
    from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier

    clf = R2Classifier(
        threshold=config.R2_THRESHOLD,
        r2_steepness=config.R2_CLF_STEEPNESS,
        direction_steepness=config.R2_DIRECTION_STEEPNESS,
    )

    # 호출자가 미리 features['slope_norm'] 계산해 둔 상태여야 함
    proba_batch = clf.predict_proba_batch(features)
"""

import numpy as np
import pandas as pd

from strategy.HMM_strategy.classifiers.base_classifier import (
    BaseClassifier, BULL, SIDE, BEAR, REGIME_IDS,
)


class R2Classifier(BaseClassifier):
    """
    R² 기반 국면 분류기 (sigmoid soft probability).

    Args:
        threshold: R² 추세/횡보 경계값 (기본 0.55)
        r2_steepness: R² sigmoid 기울기 (기본 8.0, R²가 0~1 범위라 큰 값)
        direction_steepness: 정규화된 slope sigmoid 기울기 (기본 1.0, z-score 단위)
        r2_col: 윈도우 피처 중 R² 컬럼명 (기본 'r2_mean')
        slope_col: 정규화된 slope 컬럼명 (기본 'slope_norm')
            ★ 호출자가 RollingStandardScaler로 미리 정규화한 컬럼이어야 함

    필요한 입력 컬럼:
        - window[r2_col]: R² 평균값 (기본 'r2_mean')
        - window[slope_col]: 정규화된 slope (기본 'slope_norm')
    """

    def __init__(
        self,
        threshold: float = 0.55,
        r2_steepness: float = 8.0,
        direction_steepness: float = 1.0,
        r2_col: str = 'r2_mean',
        slope_col: str = 'slope_norm',
    ):
        if r2_steepness <= 0:
            raise ValueError(f"r2_steepness must be > 0, got {r2_steepness}")
        if direction_steepness <= 0:
            raise ValueError(
                f"direction_steepness must be > 0, got {direction_steepness}"
            )
        self.threshold = threshold
        self.r2_steepness = r2_steepness
        self.direction_steepness = direction_steepness
        self.r2_col = r2_col
        self.slope_col = slope_col

    @property
    def name(self) -> str:
        return "r2"

    def predict_proba(self, window: pd.Series) -> np.ndarray:
        """
        단일 윈도우의 [P_Bull, P_Side, P_Bear] 계산.

        Returns:
            np.ndarray, shape (3,), 합 = 1.0
        """
        r2 = float(window[self.r2_col])
        slope_z = float(window[self.slope_col])

        # NaN 안전: NaN이면 균등분포 (rolling cold start 등에서 발생)
        if np.isnan(r2) or np.isnan(slope_z):
            return np.array([1/3, 1/3, 1/3], dtype=np.float64)

        # Step 1. 추세 강도 (R² 기반)
        # 수치 안정성: x의 부호로 분기
        x_trend = self.r2_steepness * (r2 - self.threshold)
        if x_trend >= 0:
            p_trend = 1.0 / (1.0 + np.exp(-x_trend))
        else:
            ez = np.exp(x_trend)
            p_trend = ez / (1.0 + ez)

        # Step 2. 방향 분배 (정규화된 slope 기반)
        x_dir = self.direction_steepness * slope_z
        if x_dir >= 0:
            bull_share = 1.0 / (1.0 + np.exp(-x_dir))
        else:
            ez = np.exp(x_dir)
            bull_share = ez / (1.0 + ez)

        # Step 3. 최종 확률 합성
        p_side = 1.0 - p_trend
        p_bull = p_trend * bull_share
        p_bear = p_trend * (1.0 - bull_share)

        return np.array([p_bull, p_side, p_bear], dtype=np.float64)

    def predict(self, window: pd.Series) -> int:
        """가장 확률이 높은 국면 반환."""
        proba = self.predict_proba(window)
        return int(REGIME_IDS[int(np.argmax(proba))])

    # ── 배치 메서드 — 벡터화로 성능 향상 ──────────────────────────
    def predict_proba_batch(self, windows: pd.DataFrame) -> np.ndarray:
        """벡터화된 배치 처리."""
        r2 = windows[self.r2_col].to_numpy(dtype=np.float64)
        slope_z = windows[self.slope_col].to_numpy(dtype=np.float64)

        nan_mask = np.isnan(r2) | np.isnan(slope_z)

        with np.errstate(over='ignore'):
            p_trend = 1.0 / (1.0 + np.exp(-self.r2_steepness * (r2 - self.threshold)))
            bull_share = 1.0 / (1.0 + np.exp(-self.direction_steepness * slope_z))

        p_side = 1.0 - p_trend
        p_bull = p_trend * bull_share
        p_bear = p_trend * (1.0 - bull_share)

        out = np.column_stack([p_bull, p_side, p_bear])

        if nan_mask.any():
            out[nan_mask] = 1/3

        return out
