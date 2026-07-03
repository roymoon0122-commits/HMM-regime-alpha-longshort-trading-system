"""
(X, y) 데이터셋 빌더 — Phase 2/3가 받을 입력 형식.

────────────────────────────────────────────────────────────────────
역할
────────────────────────────────────────────────────────────────────
window_features.py가 만든 윈도우 피처 DataFrame을 감싸서:
  - X (피처 행렬) 추출
  - y (라벨 벡터) 정렬 — 라벨은 외부에서 주입받음
  - 시점별 정합성 보장 (룩어헤드 방지)

────────────────────────────────────────────────────────────────────
Phase 1 vs Phase 2~3 활성화 범위
────────────────────────────────────────────────────────────────────
[Phase 1 — 지금]
  - __init__, get_X, get_y, __len__ 활성
  - 라벨은 None일 수 있음 (HMM이 아직 없으므로)

[Phase 2]
  - HMMLabeler가 만든 라벨을 set_labels()로 주입
  - get_y(shift=-1)로 "다음 윈도우 라벨"을 정답으로 사용 → 예측 문제 변환

[Phase 3]
  - get_train_test_split() 본격 사용 (TimeSeriesSplit과 결합)
"""

import numpy as np
import pandas as pd

from strategy.HMM_strategy.features.window_features import FEATURE_COLUMNS


class RegimeDataset:
    """
    윈도우 피처 + 라벨을 (X, y) 형식으로 관리하는 컨테이너.

    Example (Phase 1 — 라벨 없이 X만):
        >>> features = compute_window_features(df_4h, window_size=60)
        >>> ds = RegimeDataset(features)
        >>> X = ds.get_X()
        >>> X.shape
        (n_windows, 9)

    Example (Phase 2 이후 — 라벨 주입):
        >>> ds.set_labels(hmm_labels)        # Viterbi 결과
        >>> X = ds.get_X()
        >>> y = ds.get_y(shift=-1)            # 다음 윈도우 라벨로 정렬
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        feature_cols: list = None,
    ):
        """
        Args:
            features_df:
                compute_window_features() 출력 DataFrame.
                'window_end_idx'와 9개 피처 컬럼이 있어야 함.

            feature_cols:
                X로 추출할 컬럼 목록.
                None이면 FEATURE_COLUMNS (기본 9개) 사용.
        """
        if feature_cols is None:
            feature_cols = list(FEATURE_COLUMNS)

        # 입력 검증
        missing = set(feature_cols) - set(features_df.columns)
        if missing:
            raise ValueError(f"feature_cols에 없는 컬럼이 있음: {missing}")

        self.features_df  = features_df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.labels       = None   # Phase 2에서 set_labels()로 주입됨

    # ── 라벨 관리 ──────────────────────────────────────────────

    def set_labels(self, labels: np.ndarray) -> None:
        """
        외부에서 만든 라벨(HMM Viterbi 결과 등)을 주입.

        Args:
            labels: shape (n_windows,) 인 정수 배열.
                    값 의미: 0=Bull, 1=Side, 2=Bear (또는 사용자 정의).
                    features_df의 행 수와 정확히 일치해야 함.
        """
        labels = np.asarray(labels)
        if len(labels) != len(self.features_df):
            raise ValueError(
                f"라벨 길이 불일치: labels={len(labels)}, "
                f"features={len(self.features_df)}"
            )
        self.labels = labels

    # ── X / y 추출 ─────────────────────────────────────────────

    def get_X(self, feature_cols: list = None) -> np.ndarray:
        """
        피처 행렬 X를 numpy 배열로 반환.

        피처 추가/제외는 이 메서드에 feature_cols 인자만 바꿔서 호출하면 됨.
        config.HMM_FEATURE_COLS 또는 config.META_FEATURE_COLS를 그대로 넘기는
        용법을 권장.

        Args:
            feature_cols:
                추출할 컬럼 목록.
                - None이면 __init__에서 지정한 컬럼 사용 (기본 9개)
                - 리스트로 부분집합을 넘기면 그 순서대로 컬럼을 뽑음

        Returns:
            shape (n_windows, n_features) numpy 배열.

        Raises:
            KeyError: feature_cols에 features_df에 없는 컬럼명이 있는 경우.

        Example:
            >>> from strategy.HMM_strategy import config
            >>> X_hmm  = ds.get_X(feature_cols=config.HMM_FEATURE_COLS)
            >>> X_meta = ds.get_X(feature_cols=config.META_FEATURE_COLS)
        """
        cols = feature_cols if feature_cols is not None else self.feature_cols

        # 명시적 검증: 잘못된 컬럼명을 빨리 잡음
        missing = [c for c in cols if c not in self.features_df.columns]
        if missing:
            available = list(self.features_df.columns)
            raise KeyError(
                f"feature_cols에 없는 컬럼이 있음: {missing}. "
                f"사용 가능한 컬럼: {available}"
            )

        return self.features_df[cols].to_numpy(dtype=np.float64)

    def get_feature_names(self, feature_cols: list = None) -> list:
        """
        get_X가 반환하는 numpy 배열의 컬럼 순서를 반환.

        모델 계수 시각화/해석 시 어떤 피처가 어떤 인덱스에 해당하는지
        알아야 할 때 유용함 (Phase 3 메타 모델에서 사용 예정).
        """
        return list(feature_cols) if feature_cols is not None else list(self.feature_cols)

    def get_y(self, shift: int = 0) -> np.ndarray:
        """
        라벨 벡터 y를 반환. shift로 시점 정렬을 조정.

        Args:
            shift:
                0  → 윈도우 i의 피처 X[i]에 라벨 y[i]를 매칭 (현재 시점)
                -1 → 윈도우 i의 피처 X[i]에 라벨 y[i+1] 매칭 (예측 문제)
                1  → 윈도우 i의 피처 X[i]에 라벨 y[i-1] 매칭

                Phase 2에서 "다음 윈도우의 국면"을 예측하려면 shift=-1.

        Returns:
            shape (n_windows,) numpy 배열.
            shift로 인해 양 끝의 |shift|개 행은 NaN(또는 미정의 값)이 됨.

        Note:
            shift된 결과의 NaN 행은 호출자가 X와 함께 dropna 처리해야 함.
            (이 클래스는 단순 시점 정렬만 담당)
        """
        if self.labels is None:
            raise RuntimeError(
                "라벨이 주입되지 않았습니다. set_labels()를 먼저 호출하세요."
            )
        if shift == 0:
            return self.labels.copy()

        # pandas Series로 변환해서 shift 활용 (NaN 자동 처리)
        s = pd.Series(self.labels, dtype='float64')
        # numpy의 shift는 음수가 "위로 밀기"인데 pandas와 부호가 같음
        return s.shift(shift).to_numpy()

    def get_aligned_Xy(self, shift: int = -1) -> tuple:
        """
        X와 y를 shift 적용 후 NaN 제거한 짝으로 반환 (편의 함수).

        Args:
            shift: get_y의 shift와 동일.

        Returns:
            (X, y) 튜플.
            shift로 인한 NaN 행이 제거된 상태.
        """
        X = self.get_X()
        y = self.get_y(shift=shift)

        # NaN 마스크 (y가 NaN인 행 제외)
        valid = ~np.isnan(y)
        return X[valid], y[valid].astype(np.int64)

    # ── 메타 정보 ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.features_df)

    def get_window_end_times(self) -> pd.Series:
        """각 윈도우의 마지막 봉 시각 (시각화/리포트용)."""
        if 'window_end_time' not in self.features_df.columns:
            raise KeyError("features_df에 'window_end_time' 컬럼이 없음")
        return self.features_df['window_end_time']

    # ── 학습/검증 분할 (Phase 3에서 본격 사용) ─────────────────

    def get_train_test_split(self, train_end_date) -> tuple:
        """
        학습/검증 분할 — 시간 기준으로 자른다.

        Args:
            train_end_date: 학습 종료일 (str 또는 pd.Timestamp).
                            이 날짜까지가 학습, 이후가 검증.

        Returns:
            (train_dataset, test_dataset) — 둘 다 RegimeDataset 인스턴스.

        Note:
            Phase 3에서 TimeSeriesSplit과 함께 본격 사용 예정.
            Phase 1에서는 인터페이스만 마련 (구현 동작은 가능).
        """
        if 'window_end_time' not in self.features_df.columns:
            raise KeyError(
                "시간 기준 분할에는 'window_end_time' 컬럼이 필요함. "
                "원본 df에 'datetime' 컬럼이 있어야 합니다."
            )

        cutoff = pd.Timestamp(train_end_date)
        train_mask = self.features_df['window_end_time'] <= cutoff

        train_features = self.features_df[train_mask].reset_index(drop=True)
        test_features  = self.features_df[~train_mask].reset_index(drop=True)

        train_ds = RegimeDataset(train_features, feature_cols=self.feature_cols)
        test_ds  = RegimeDataset(test_features,  feature_cols=self.feature_cols)

        # 라벨도 같이 분할 (있으면)
        if self.labels is not None:
            train_ds.set_labels(self.labels[train_mask.values])
            test_ds.set_labels(self.labels[~train_mask.values])

        return train_ds, test_ds
