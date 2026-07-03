"""
HMMStrategy — Phase 1~4의 모든 모듈을 묶는 메인 통합 클래스.

────────────────────────────────────────────────────────────────────
역할
────────────────────────────────────────────────────────────────────
BaseStrategy를 상속해서 EngineHMM(또는 일반 Engine)에 그대로 꽂히는
전략 클래스. fit()으로 한 번 학습하고, generate_signals()로 시점별
포지션 비중(-1.0 ~ +1.0)을 반환한다.

────────────────────────────────────────────────────────────────────
파이프라인
────────────────────────────────────────────────────────────────────
    df (4h OHLCV)
      ├─ compute_window_features              (Phase 1)
      ├─ RollingStandardScaler (slope_norm)   (Phase 2)
      ├─ HMMLabeler (load 또는 fit)            (Phase 2)
      ├─ ADXClassifier / R2Classifier         (Phase 3)
      ├─ TransitionPredictor.predict_next     (Phase 3)
      ├─ X_meta = stack(...)                  (variant 따라 10/16 피처)
      ├─ (학습 시) RetrospectiveLabelSmoother (Phase 3, 옵션)
      ├─ LogisticMetaModel                    (Phase 3)
      └─ PositionSizer.compute_batch          (Phase 4)
           ↓
       signals: float [-1.0, +1.0]

────────────────────────────────────────────────────────────────────
시점 정렬 (룩어헤드 방지)
────────────────────────────────────────────────────────────────────
window_end_idx=i 윈도우의 비중 → signals[i]에 저장.
EngineHMM이 자동으로 signals[i] → open[i+1]에 체결.

즉 봉 i 종가까지의 정보로 결정한 비중이 봉 i+1 시가에 체결됨.
1단계 시프트 = 룩어헤드 없음 + 정보 1봉 낭비 없음 + 실제 거래 흐름 일치.

────────────────────────────────────────────────────────────────────
config.py와의 관계 (Pattern B 준수 + from_config 헬퍼)
────────────────────────────────────────────────────────────────────
- 클래스 자체는 config를 직접 import하지 않음 (Pattern B).
- 모든 파라미터는 합리적 default를 갖는 인자로 받음.
- HMMStrategy.from_config(**overrides) 클래스메서드로 config.py 값을
  한꺼번에 적용 가능. 일부만 덮어쓰고 싶으면 overrides 인자로.

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    # 1. config.py 그대로 사용
    strategy = HMMStrategy.from_config()
    strategy.fit(df_btc_4h)
    signals = strategy.generate_signals(df_btc_4h)

    # 2. variant 비교용 (HMM 사후확률 제외)
    strategy_v2 = HMMStrategy.from_config(include_hmm_proba=False)

    # 3. 단위 테스트용 (모든 인자 명시 주입)
    strategy = HMMStrategy(
        window_size=30, include_hmm_proba=True, use_smoothed_labels=False, ...
    )
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import BaseStrategy
from strategy.HMM_strategy.features.window_features import compute_window_features
from strategy.HMM_strategy.features.volume_features import (
    compute_volume_window_features,
    VOLUME_FEATURE_COLUMNS,
)
from strategy.HMM_strategy.features.scaling import RollingStandardScaler
from strategy.HMM_strategy.regime.hmm_labeler import HMMLabeler, BULL, SIDE, BEAR
from strategy.HMM_strategy.regime.transition import TransitionPredictor
from strategy.HMM_strategy.regime.label_smoother import RetrospectiveLabelSmoother
from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier
from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier
from strategy.HMM_strategy.meta_model.logistic_meta_model import LogisticMetaModel
from strategy.HMM_strategy.position.sizer import PositionSizer


class HMMStrategy(BaseStrategy):
    """
    HMM 국면 분류 알파의 메인 통합 전략.

    상태 (fit 후 채워짐):
        labeler_:        HMMLabeler 인스턴스
        meta_model_:     LogisticMetaModel 인스턴스
        sizer_:          PositionSizer 인스턴스
        feature_names_:  메타 입력 X의 피처 이름 (variant에 따라 10 또는 16개)
        is_fitted_:      bool — fit() 호출 여부
    """

    def __init__(
        self,
        # ─── variant 스위치 (Phase 4 핵심) ─────────────────────
        include_hmm_proba: bool = True,
        use_smoothed_labels: bool = True,
        # ─── SIDE 국면 위임 (실험용 옵션, 기본 OFF) ─────────────
        # True면 HMM argmax == SIDE 시점에 한해 donchian_adx_r2_B.py의
        # 시그널을 P(Side) 가중치로 곱해 사용. 나머지 시점은 기존 HMM 동작.
        # 기본값 False라서 from_config(...)를 그대로 호출하는 live_trade.py는
        # 영향 없음. 백테스트 스크립트에서 from_config(use_donchian_on_side=True)
        # 로 override해 실험.
        use_donchian_on_side: bool = False,
        donchian_entry_period: int = 260,   # 30분봉 ≈ 20거래일 (13봉/day × 20)
        donchian_exit_period: int = 130,    # 30분봉 ≈ 10거래일
        # ─── 포지션 사이저 ─────────────────────────────────────
        position_mode: str = 'net',
        min_threshold: float = 0.1,
        # ─── HMM 라벨러 ────────────────────────────────────────
        hmm_model_path: Optional[str] = None,    # 캐시 경로 (있으면 load)
        n_states: int = 3,
        hmm_n_iter: int = 200,
        hmm_n_random_restart: int = 30,
        hmm_covariance_type: str = 'diag',
        # ─── 정규화 ────────────────────────────────────────────
        rolling_window: int = 2200,
        # ─── 윈도우 피처 ───────────────────────────────────────
        window_size: int = 60,
        adx_period: int = 12,
        r2_period: int = 40,
        # ─── HMM 학습용 피처 컬럼 ──────────────────────────────
        hmm_feature_cols: Optional[list] = None,
        slope_norm_col: str = 'slope_norm',
        # ─── ADX 분류기 ────────────────────────────────────────
        adx_threshold: float = 25.0,
        adx_steepness: float = 0.2,
        adx_direction_steepness: float = 50.0,
        # ─── R² 분류기 ─────────────────────────────────────────
        r2_threshold: float = 0.55,
        r2_steepness: float = 8.0,
        r2_direction_steepness: float = 1.0,
        # ─── Label Smoother ────────────────────────────────────
        smoother_lookback: int = 10,
        smoother_threshold: float = 0.03,
        smoother_persistence: int = 3,
        smoother_include_side: bool = False,
        # ─── 메타 모델 ─────────────────────────────────────────
        meta_C: float = 1.0,
        meta_class_weight: Optional[str] = 'balanced',
        meta_max_iter: int = 1000,
        meta_random_state: int = 42,
        # ─── 거래량 피처 (meta 입력 전용, HMM 미투입) ──────────
        include_volume: bool = False,
        volume_lookback_days: int = 20,
        volume_clip: float = 3.0,
        # ─── 재현성 ────────────────────────────────────────────
        random_state: int = 42,
        verbose: bool = False,
    ):
        # ── 기본 검증 ─────────────────────────────────────────
        if hmm_feature_cols is None:
            hmm_feature_cols = [
                'cum_return', 'volatility', 'adx_mean', 'r2_mean', 'up_candle_ratio',
            ]
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if adx_period < 1:
            raise ValueError(f"adx_period must be >= 1, got {adx_period}")
        if r2_period < 1:
            raise ValueError(f"r2_period must be >= 1, got {r2_period}")
        if rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1, got {rolling_window}")
        if donchian_entry_period < 2:
            raise ValueError(f"donchian_entry_period must be >= 2, got {donchian_entry_period}")
        if donchian_exit_period < 2:
            raise ValueError(f"donchian_exit_period must be >= 2, got {donchian_exit_period}")

        # ── 파라미터 저장 ─────────────────────────────────────
        self.include_hmm_proba = bool(include_hmm_proba)
        self.use_smoothed_labels = bool(use_smoothed_labels)

        # SIDE 위임 옵션
        self.use_donchian_on_side = bool(use_donchian_on_side)
        self.donchian_entry_period = int(donchian_entry_period)
        self.donchian_exit_period = int(donchian_exit_period)

        self.position_mode = position_mode
        self.min_threshold = float(min_threshold)

        self.hmm_model_path = hmm_model_path
        self.n_states = n_states
        self.hmm_n_iter = hmm_n_iter
        self.hmm_n_random_restart = hmm_n_random_restart
        self.hmm_covariance_type = hmm_covariance_type

        self.rolling_window = rolling_window
        self.window_size = window_size
        self.adx_period = adx_period
        self.r2_period = r2_period

        self.hmm_feature_cols = list(hmm_feature_cols)
        self.slope_norm_col = slope_norm_col

        self.adx_threshold = adx_threshold
        self.adx_steepness = adx_steepness
        self.adx_direction_steepness = adx_direction_steepness

        self.r2_threshold = r2_threshold
        self.r2_steepness = r2_steepness
        self.r2_direction_steepness = r2_direction_steepness

        self.smoother_lookback = smoother_lookback
        self.smoother_threshold = smoother_threshold
        self.smoother_persistence = smoother_persistence
        self.smoother_include_side = smoother_include_side

        self.meta_C = meta_C
        self.meta_class_weight = meta_class_weight
        self.meta_max_iter = meta_max_iter
        self.meta_random_state = meta_random_state

        self.include_volume = bool(include_volume)
        self.volume_lookback_days = int(volume_lookback_days)
        self.volume_clip = volume_clip

        self.random_state = random_state
        self.verbose = verbose

        # ── 학습 후 채워짐 ────────────────────────────────────
        self.labeler_: Optional[HMMLabeler] = None
        self.meta_model_: Optional[LogisticMetaModel] = None
        self.sizer_: Optional[PositionSizer] = None
        self.feature_names_: Optional[list] = None
        self.is_fitted_: bool = False
        # 디버깅/검증용 (테스트에서 활용)
        self._fit_diagnostics: dict = {}

    # ─────────────────────────────────────────────────────────────
    # config 기반 팩토리
    # ─────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config_module=None, **overrides) -> 'HMMStrategy':
        """config.py 값을 한꺼번에 적용해 HMMStrategy 인스턴스 생성.

        일부만 덮어쓸 때:
            HMMStrategy.from_config(include_hmm_proba=False)
            HMMStrategy.from_config(use_smoothed_labels=False, window_size=30)

        Pattern B 노트:
        - 일반 메서드(__init__, fit, generate_signals)는 config 직접 import 안 함.
        - 이 클래스메서드만 명시적으로 config를 가져와 default를 채움.
        - 단위 테스트 시 config_module 인자로 mock을 주입할 수 있음.
        """
        if config_module is None:
            from strategy.HMM_strategy import config as config_module

        params = dict(
            include_hmm_proba=getattr(config_module, 'INCLUDE_HMM_PROBA', True),
            use_smoothed_labels=getattr(config_module, 'USE_SMOOTHED_LABELS', True),
            position_mode=config_module.POSITION_MODE,
            min_threshold=config_module.MIN_POSITION_THRESHOLD,
            hmm_model_path=config_module.HMM_MODEL_PATH,
            n_states=config_module.N_STATES,
            hmm_n_iter=config_module.HMM_N_ITER,
            hmm_n_random_restart=config_module.HMM_RANDOM_RESTART,
            hmm_covariance_type=config_module.HMM_COVARIANCE_TYPE,
            rolling_window=config_module.ROLLING_SCALER_WINDOW,
            window_size=config_module.WINDOW_SIZE,
            adx_period=config_module.ADX_PERIOD,
            r2_period=config_module.R2_PERIOD,
            hmm_feature_cols=list(config_module.HMM_FEATURE_COLS),
            slope_norm_col=config_module.SLOPE_NORM_COL,
            adx_threshold=config_module.ADX_THRESHOLD,
            adx_steepness=config_module.ADX_CLF_STEEPNESS,
            adx_direction_steepness=config_module.DIRECTION_STEEPNESS,
            r2_threshold=config_module.R2_THRESHOLD,
            r2_steepness=config_module.R2_CLF_STEEPNESS,
            r2_direction_steepness=config_module.R2_DIRECTION_STEEPNESS,
            smoother_lookback=config_module.LABEL_SMOOTHER_LOOKBACK,
            smoother_threshold=config_module.LABEL_SMOOTHER_THRESHOLD,
            smoother_persistence=config_module.LABEL_SMOOTHER_PERSISTENCE,
            smoother_include_side=config_module.LABEL_SMOOTHER_INCLUDE_SIDE,
            meta_C=getattr(config_module, 'META_C', 1.0),
            meta_class_weight=getattr(config_module, 'META_CLASS_WEIGHT', 'balanced'),
            include_volume=getattr(config_module, 'INCLUDE_VOLUME', False),
            volume_lookback_days=getattr(config_module, 'VOLUME_LOOKBACK_DAYS', 20),
            volume_clip=getattr(config_module, 'VOLUME_CLIP', 3.0),
        )
        params.update(overrides)
        return cls(**params)

    # ─────────────────────────────────────────────────────────────
    # 공통 파이프라인 — 윈도우 피처 + slope_norm + HMM 사후확률
    # ─────────────────────────────────────────────────────────────
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV df → 윈도우 피처 + slope_norm 컬럼."""
        features = compute_window_features(
            df,
            window_size=self.window_size,
            adx_period=self.adx_period,
            r2_period=self.r2_period,
        )

        # slope_norm: rolling z-score (R²Classifier 입력)
        slope_scaler = RollingStandardScaler(window=self.rolling_window)
        slope_norm = slope_scaler.fit_transform(features[['slope']].values).flatten()
        features = features.copy()
        features[self.slope_norm_col] = slope_norm
        return features

    def _compute_hmm_proba(self, features: pd.DataFrame) -> tuple:
        """
        HMM 입력을 정규화한 뒤 라벨러로 사후확률 계산.

        Returns:
            hmm_proba:   shape (n, 3) — Bull/Side/Bear 순서
            hmm_cold:    shape (n,)   — bool, cold start 행 (정보 없음 → 1/3 채움)
            X_hmm_for_fit: shape (n, n_hmm_feats) — HMM 학습/추론용 정규화된 X
        """
        X_hmm_raw = features[self.hmm_feature_cols].values
        scaler = RollingStandardScaler(window=self.rolling_window)
        X_hmm_scaled = scaler.fit_transform(X_hmm_raw)

        # cold start 행: NaN 있는 행
        nan_mask = np.isnan(X_hmm_scaled).any(axis=1)
        X_hmm_safe = np.where(np.isnan(X_hmm_scaled), 0.0, X_hmm_scaled)

        if self.labeler_ is None:
            raise RuntimeError("HMM labeler not initialized — call _ensure_labeler first")

        hmm_proba = self.labeler_.predict_proba(X_hmm_safe)
        # cold start 시점은 정보 없음 → 균등 분포로 마스킹
        hmm_proba[nan_mask] = 1.0 / 3.0
        return hmm_proba, nan_mask, X_hmm_scaled

    def _ensure_labeler(
        self,
        features: pd.DataFrame,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """HMM 라벨러를 캐시에서 로드하거나 새로 학습.

        Args:
            features: 윈도우 피처 df (HMM 입력 X 계산용)
            df: 원본 OHLCV df (선택). 학습 메타데이터에 datetime 범위/길이를
                기록하기 위해 받음. None이면 메타데이터에서 해당 항목만 빠짐.
        """
        # 캐시 시도
        if self.hmm_model_path is not None:
            cache_path = Path(self.hmm_model_path)
            if cache_path.exists():
                self.labeler_ = HMMLabeler()
                self.labeler_.load(str(cache_path))
                if self.verbose:
                    print(f"[HMMStrategy] HMM 캐시 로드: {cache_path}")
                return

        # 새로 학습
        if self.verbose:
            print(f"[HMMStrategy] HMM 새로 학습 "
                  f"(restart={self.hmm_n_random_restart}, n_states={self.n_states})...")

        X_hmm_raw = features[self.hmm_feature_cols].values
        scaler = RollingStandardScaler(window=self.rolling_window)
        X_hmm_scaled = scaler.fit_transform(X_hmm_raw)
        cum_return_full = features['cum_return'].values

        valid = ~np.isnan(X_hmm_scaled).any(axis=1)
        if not valid.any():
            raise RuntimeError("정규화 후 학습 가능한 행이 0개. rolling_window/데이터 길이 점검 필요.")

        labeler = HMMLabeler(
            n_states=self.n_states,
            n_iter=self.hmm_n_iter,
            n_random_restart=self.hmm_n_random_restart,
            covariance_type=self.hmm_covariance_type,
            random_state=self.random_state,
        )

        # labeler가 자동 캡처하지 못하는 정보를 training_context로 보강.
        # n_samples, n_features는 fit() 내부에서 자동 캡처되므로 여기선 생략.
        training_context = {
            'hmm_feature_cols': list(self.hmm_feature_cols),
            'rolling_scaler_window': int(self.rolling_window),
        }
        if df is not None and 'datetime' in df.columns and len(df) > 0:
            try:
                training_context['date_start'] = str(df['datetime'].iloc[0])
                training_context['date_end'] = str(df['datetime'].iloc[-1])
                training_context['n_total_bars'] = int(len(df))
            except Exception:
                pass

        labeler.fit(
            X_hmm_scaled[valid],
            cum_return_full[valid],
            training_context=training_context,
        )
        self.labeler_ = labeler

        # 캐시 저장 — save() 내부에서 created_at/env/config_snapshot/training_data가
        # 모두 자동 박제됨 (호출자는 path만 넘기면 됨).
        if self.hmm_model_path is not None:
            cache_path = Path(self.hmm_model_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            labeler.save(str(cache_path))
            if self.verbose:
                tm = labeler.training_metadata_
                print(f"[HMMStrategy] HMM 캐시 저장: {cache_path}")
                print(f"[HMMStrategy]   메타 자동 박제: created_at={tm.get('created_at')}, "
                      f"config 항목 {len(tm.get('config_snapshot', {}))}개")

    def _build_meta_input(
        self,
        features: pd.DataFrame,
        hmm_proba: np.ndarray,
        transition_predictor: TransitionPredictor,
        df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """
        분류기/전이/HMM 사후확률을 합쳐 메타 입력 X 행렬 생성.

        variant:
            include_hmm_proba=True  → 16개 피처 (분류기 6 + 윈도우 4 + HMM 3 + 전이 3)
            include_hmm_proba=False → 10개 피처 (분류기 6 + 윈도우 4)
            include_volume=True     → 위에 거래량 피처 3개 추가 (RVOL 기반)

        거래량 피처는 HMM 클러스터링이 아니라 **meta 입력에만** 붙는다.
        RVOL 기준선(volume_lookback_days)이 길어 워밍업이 더 늦으므로
        단순 위치 결합이 아니라 window_end_idx 기준 left-join으로 정렬한다.
        매칭 안 되는 초반 행은 NaN → nan_mask가 학습/추론에서 제외.

        Args:
            df: 원본 OHLCV df. include_volume=True일 때만 필요(거래량 계산용).

        Returns:
            X_meta: shape (n, n_features)
            feature_names: list[str]
            nan_mask: shape (n,) bool — X_meta에 NaN 있는 행
        """
        # ── ADX/R² 분류기 ─────────────────────────────────────
        adx_clf = ADXClassifier(
            threshold=self.adx_threshold,
            adx_steepness=self.adx_steepness,
            direction_steepness=self.adx_direction_steepness,
        )
        adx_proba = adx_clf.predict_proba_batch(features)

        r2_clf = R2Classifier(
            threshold=self.r2_threshold,
            r2_steepness=self.r2_steepness,
            direction_steepness=self.r2_direction_steepness,
            slope_col=self.slope_norm_col,
        )
        r2_proba = r2_clf.predict_proba_batch(features)

        # ── 윈도우 피처 4개 ───────────────────────────────────
        win_feats = features[['cum_return', 'volatility', 'adx_mean', 'r2_mean']].values

        # ── 결합 — variant 분기 ────────────────────────────────
        if self.include_hmm_proba:
            trans_proba = transition_predictor.predict_next_batch(hmm_proba)
            X_meta = np.hstack([adx_proba, r2_proba, win_feats, hmm_proba, trans_proba])
            feature_names = [
                'adx_p_bull', 'adx_p_side', 'adx_p_bear',
                'r2_p_bull', 'r2_p_side', 'r2_p_bear',
                'cum_return', 'volatility', 'adx_mean', 'r2_mean',
                'hmm_p_bull', 'hmm_p_side', 'hmm_p_bear',
                'trans_p_bull', 'trans_p_side', 'trans_p_bear',
            ]
        else:
            X_meta = np.hstack([adx_proba, r2_proba, win_feats])
            feature_names = [
                'adx_p_bull', 'adx_p_side', 'adx_p_bear',
                'r2_p_bull', 'r2_p_side', 'r2_p_bear',
                'cum_return', 'volatility', 'adx_mean', 'r2_mean',
            ]

        # ── 거래량 피처 (variant: include_volume) ──────────────
        # window_end_idx 기준 left-join으로 정렬 → 위치 어긋남 방지.
        if self.include_volume:
            if df is None:
                raise ValueError(
                    "include_volume=True 인데 df가 전달되지 않았습니다 "
                    "(거래량 계산에 원본 OHLCV 필요)."
                )
            vol_wf = compute_volume_window_features(
                df,
                window_size=self.window_size,
                lookback_days=self.volume_lookback_days,
                clip=self.volume_clip,
            )
            # features의 window_end_idx 순서에 맞춰 reindex (없는 행 → NaN)
            vol_aligned = (
                vol_wf.set_index('window_end_idx')
                .reindex(features['window_end_idx'].astype(int).values)
            )
            vol_arr = vol_aligned[list(VOLUME_FEATURE_COLUMNS)].values
            X_meta = np.hstack([X_meta, vol_arr])
            feature_names = feature_names + list(VOLUME_FEATURE_COLUMNS)

        nan_mask = np.isnan(X_meta).any(axis=1)
        return X_meta, feature_names, nan_mask

    # ─────────────────────────────────────────────────────────────
    # 학습
    # ─────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> 'HMMStrategy':
        """
        df로 전체 파이프라인 학습.

        1. 윈도우 피처 + slope_norm
        2. HMM 라벨러 (캐시 또는 신규 학습)
        3. HMM 사후확률
        4. 메타 입력 X 구성 (variant 따라)
        5. 학습 라벨 y = next_window_label (옵션: smoothed)
        6. cold start/NaN 행 제거 후 LogisticMetaModel.fit

        Returns:
            self
        """
        if self.verbose:
            print(f"[HMMStrategy.fit] 시작 (variant: include_hmm_proba="
                  f"{self.include_hmm_proba}, use_smoothed_labels="
                  f"{self.use_smoothed_labels})")

        # 1. 피처
        features = self._build_features(df)

        # 2. HMM 라벨러 (df도 같이 넘겨서 학습 메타데이터에 datetime 범위 기록)
        self._ensure_labeler(features, df=df)

        # 3. HMM 사후확률
        hmm_proba, hmm_cold, _ = self._compute_hmm_proba(features)

        # 4. 메타 입력
        transition_predictor = TransitionPredictor.from_labeler(self.labeler_)
        X_meta, feature_names, nan_mask_meta = self._build_meta_input(
            features, hmm_proba, transition_predictor, df=df,
        )

        # 5. 학습 라벨 y = label[t+1]
        hmm_labels = np.argmax(hmm_proba, axis=1).astype(np.int64)

        if self.use_smoothed_labels:
            # 윈도우의 마지막 1봉 수익률 (smoother 입력)
            # window_end_idx[i] 봉의 수익률
            close = df['close'].to_numpy(dtype=np.float64)
            end_idx = features['window_end_idx'].astype(int).values
            # 봉 j의 수익률 = (close[j]-close[j-1])/close[j-1]
            ret = np.zeros(len(close))
            ret[1:] = np.diff(close) / close[:-1]
            last_bar_returns = ret[end_idx]

            smoother = RetrospectiveLabelSmoother(
                lookback=self.smoother_lookback,
                threshold=self.smoother_threshold,
                persistence_check=self.smoother_persistence,
                include_side=self.smoother_include_side,
            )
            smoothed_labels, change_log = smoother.smooth(hmm_labels, last_bar_returns)
            base_labels = smoothed_labels
            self._fit_diagnostics['smoother_changes'] = len(change_log)
        else:
            base_labels = hmm_labels
            self._fit_diagnostics['smoother_changes'] = 0

        # next-window shift
        y_full = np.full(len(base_labels), -1, dtype=np.int64)
        y_full[:-1] = base_labels[1:]
        y_nan = (y_full == -1)

        # 6. 학습 가능 행 마스크
        # cold start: hmm_proba가 1/3 균등으로 마킹된 시점
        cold_mask = np.isclose(hmm_proba, 1.0 / 3.0, atol=1e-9).all(axis=1)
        final_mask = ~(nan_mask_meta | y_nan | cold_mask)
        n_valid = int(final_mask.sum())
        if n_valid < 100:
            raise RuntimeError(
                f"학습 가능 행이 너무 적습니다 ({n_valid}개). "
                f"window_size/rolling_window/데이터 길이를 점검하세요."
            )

        X_train = X_meta[final_mask]
        y_train = y_full[final_mask]

        # 7. 메타 모델 학습
        self.meta_model_ = LogisticMetaModel(
            C=self.meta_C,
            class_weight=self.meta_class_weight,
            max_iter=self.meta_max_iter,
            random_state=self.meta_random_state,
            feature_names=feature_names,
        )
        self.meta_model_.fit(X_train, y_train)
        self.feature_names_ = feature_names

        # 8. 포지션 사이저
        self.sizer_ = PositionSizer(
            mode=self.position_mode,
            min_threshold=self.min_threshold,
        )

        # 진단 정보
        self._fit_diagnostics.update({
            'n_total': len(features),
            'n_train': n_valid,
            'X_meta_shape': X_meta.shape,
            'label_dist': {
                'Bull': float(np.mean(y_train == BULL)),
                'Side': float(np.mean(y_train == SIDE)),
                'Bear': float(np.mean(y_train == BEAR)),
            },
        })
        self.is_fitted_ = True

        if self.verbose:
            d = self._fit_diagnostics
            print(f"[HMMStrategy.fit] 완료: 학습 행 {d['n_train']:,}, "
                  f"X_meta {d['X_meta_shape']}, "
                  f"라벨 분포 {d['label_dist']}")

        return self

    # ─────────────────────────────────────────────────────────────
    # 시그널 생성 (BaseStrategy 인터페이스)
    # ─────────────────────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        df의 각 봉에 대해 포지션 비중(-1.0 ~ +1.0) 시그널 생성.

        시점 정렬:
            signals[i] = "봉 i 종가까지의 정보로 결정한 비중".
            EngineHMM이 자동으로 signals[i] → open[i+1]에 체결.

        Returns:
            np.ndarray, shape (len(df),), dtype float64
            워밍업/cold start 구간은 0.0.
        """
        if not self.is_fitted_:
            raise RuntimeError("HMMStrategy.fit()을 먼저 호출하세요.")

        n_bars = len(df)
        signals = np.zeros(n_bars, dtype=np.float64)

        # 1. 윈도우 피처 + slope_norm
        features = self._build_features(df)

        # 2. HMM 사후확률
        hmm_proba, hmm_cold, _ = self._compute_hmm_proba(features)

        # 3. 메타 입력
        transition_predictor = TransitionPredictor.from_labeler(self.labeler_)
        X_meta, _, nan_mask_meta = self._build_meta_input(
            features, hmm_proba, transition_predictor, df=df,
        )

        # cold start나 NaN이 있는 행은 메타 모델에 넣을 수 없으므로
        # 일단 0으로 채워 임시 추론 후 마스킹
        cold_mask = np.isclose(hmm_proba, 1.0 / 3.0, atol=1e-9).all(axis=1)
        invalid_mask = nan_mask_meta | cold_mask

        X_meta_safe = np.where(np.isnan(X_meta), 0.0, X_meta)
        proba = self.meta_model_.predict_proba(X_meta_safe)   # (n, 3)

        # 무효 행은 균등 분포 → sizer가 net=0으로 처리
        proba[invalid_mask] = 1.0 / 3.0

        # 4. 비중 변환
        weights = self.sizer_.compute_batch(proba)
        if isinstance(weights, dict):
            # dual 모드: 일단 net으로 합산 (long - short)
            # (PortfolioContinuous는 양방향 동시 보유를 지원하지 않으므로)
            weights_arr = weights['long'] - weights['short']
        else:
            weights_arr = weights

        # 4.5. (옵션) SIDE 시점에 한해 돈치안 시그널로 덮어쓰기
        #
        # 설계 요지:
        #   - donchian_adx_r2_B.py의 generate_signals(df)를 전체 df에 대해
        #     한 번 돌려서 그 출력(-1/0/+1)을 얻는다. 돈치안 내부 상태머신
        #     (base_position / counter / SL / 채널이탈)은 자체적으로 진화하므로
        #     우리 쪽에서 별도 상태 관리 안 함 (지난번 버그 회피 핵심).
        #   - HMM argmax가 SIDE인 시점만 signal = donch[i] * P(Side)[i]로 덮어씀.
        #   - HMM argmax가 Bull/Bear이면 기존 HMM 시그널 그대로.
        #   - invalid(cold start/NaN) 시점은 0 유지.
        #
        # 시점 매핑 주의:
        #   - meta proba는 윈도우 인덱스 j마다 1개. window_end_idx[j]로 봉 i에 매핑.
        #   - 돈치안 시그널은 봉 인덱스 i마다 1개 (np.ndarray, shape=(n_bars,)).
        #   - 따라서 SIDE 판단(j 기준)과 donch 시그널 추출(i 기준) 둘 다 매 윈도우 j에서 수행.
        if self.use_donchian_on_side:
            # 지연 import: 옵션 OFF면 돈치안 모듈 로딩 자체 안 함 → 라이브 무영향 강화
            from strategy.donchian_adx_r2_B import DonchianADXR2Strategy
            from strategy.HMM_strategy.regime.hmm_labeler import SIDE as _SIDE_IDX

            donch_strat = DonchianADXR2Strategy(
                entry_period=self.donchian_entry_period,
                exit_period=self.donchian_exit_period,
            )
            donch_signals = donch_strat.generate_signals(df).astype(np.float64)
            # 안전: 길이 검증
            if len(donch_signals) != n_bars:
                raise RuntimeError(
                    f"돈치안 시그널 길이 불일치: {len(donch_signals)} != {n_bars}"
                )

            argmax_per_window = np.argmax(proba, axis=1)
            p_side_per_window = proba[:, _SIDE_IDX]

            end_idx = features['window_end_idx'].astype(int).values
            valid_end = (end_idx >= 0) & (end_idx < n_bars)
            for j in np.where(valid_end)[0]:
                bar_i = end_idx[j]
                if invalid_mask[j]:
                    # cold start/NaN → 0 유지 (기존 동작과 동일)
                    signals[bar_i] = 0.0
                elif argmax_per_window[j] == _SIDE_IDX:
                    # SIDE: 돈치안 시그널 × P(Side)
                    signals[bar_i] = donch_signals[bar_i] * p_side_per_window[j]
                else:
                    # Bull/Bear: 기존 HMM 동작
                    signals[bar_i] = weights_arr[j]
        else:
            # 5. df 길이로 매핑 (기존 동작 — 옵션 OFF일 때 동일 경로)
            end_idx = features['window_end_idx'].astype(int).values
            # 안전: end_idx가 n_bars 범위 안에 있는지
            valid_end = (end_idx >= 0) & (end_idx < n_bars)
            for j in np.where(valid_end)[0]:
                signals[end_idx[j]] = weights_arr[j]

        return signals

    # ─────────────────────────────────────────────────────────────
    # 디버깅 편의
    # ─────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (f"HMMStrategy({status}, "
                f"include_hmm_proba={self.include_hmm_proba}, "
                f"use_smoothed_labels={self.use_smoothed_labels}, "
                f"use_donchian_on_side={self.use_donchian_on_side}, "
                f"position_mode={self.position_mode!r})")
