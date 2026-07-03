"""
LogisticMetaModel — Multinomial Logistic Regression 기반 메타 모델.

────────────────────────────────────────────────────────────────────
모델 개요
────────────────────────────────────────────────────────────────────
sklearn의 LogisticRegression(multi_class='multinomial')을 래핑.
softmax 함수를 통해 [P_Bull, P_Side, P_Bear]를 출력한다.

수식:
    P(class = k | x) = exp(w_k · x + b_k) / Σ_j exp(w_j · x + b_j)

여기서 w_k, b_k는 클래스 k의 가중치 벡터와 절편. 학습 시
크로스 엔트로피 손실을 최소화하는 w, b를 찾는다.

────────────────────────────────────────────────────────────────────
내부 구조
────────────────────────────────────────────────────────────────────
    LogisticMetaModel
       │
       ├── StandardScaler        ← 피처 스케일 통일 (예: cum_return vs adx_mean)
       │
       └── LogisticRegression    ← 실제 학습/예측

scaler는 fit 시 X로부터 학습되고, predict 시 같은 통계로 transform.
TimeSeriesSplit 사용 시 매 fold마다 새 모델을 만들어 룩어헤드 방지
(이건 호출자 책임 — 메타 모델 자체는 fit-once 가정).

────────────────────────────────────────────────────────────────────
하이퍼파라미터
────────────────────────────────────────────────────────────────────
C : float
    L2 정규화 강도의 역수.
    - 작을수록 강한 규제 (계수가 0 쪽으로 수축, 단순한 모델)
    - 클수록 약한 규제 (계수 자유, 복잡한 모델)
    기본 1.0 (sklearn 기본값과 동일).

class_weight : str | dict | None
    'balanced'면 라벨 빈도 역수로 가중. 우리 데이터는
    Bull 24.6% / Side 48.2% / Bear 27.1% 로 약간 불균형하므로
    초기엔 'balanced' 권장.

────────────────────────────────────────────────────────────────────
계수 해석
────────────────────────────────────────────────────────────────────
fit 후 model.coef_ 는 shape (n_classes, n_features).
get_coef_summary()로 클래스별 계수 + 절편을 DataFrame으로 받을 수 있다.
양수 큰 값 → 그 클래스에 강한 영향. 음수 큰 값 → 반대 영향.

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    meta = LogisticMetaModel(C=1.0, class_weight='balanced',
                              feature_names=['adx_p_bull', 'adx_p_side', ...])
    meta.fit(X_train, y_train)
    proba = meta.predict_proba(X_test)
    print(meta.get_coef_summary())
    meta.save("models/meta.joblib")
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from strategy.HMM_strategy.meta_model.base_meta_model import BaseMetaModel


class LogisticMetaModel(BaseMetaModel):
    """
    Multinomial Logistic Regression 메타 모델.

    Args:
        C: L2 정규화 강도의 역수 (기본 1.0)
        max_iter: 최대 학습 반복 (기본 1000)
        class_weight: 'balanced' | None | dict (기본 'balanced')
        random_state: 랜덤 시드 (기본 42)
        feature_names: 피처 이름 list (기본 None — 자동 생성)
            ★ get_coef_summary()의 컬럼명에 사용. 시각화/해석에 중요.

    학습 후 채워지는 속성:
        scaler_: 학습된 StandardScaler
        model_: 학습된 LogisticRegression
        feature_names_: 학습 시 사용된 피처 이름
        n_features_: 피처 수
    """

    # 클래스 라벨 (BaseClassifier, HMMLabeler와 통일)
    CLASS_NAMES = ['Bull', 'Side', 'Bear']
    EXPECTED_CLASSES = np.array([0, 1, 2])

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        random_state: int = 42,
        feature_names: list = None,
    ):
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.feature_names = feature_names

        # 학습 후 채워지는 속성
        self.scaler_ = None
        self.model_ = None
        self.feature_names_ = None
        self.n_features_ = None

    # ── 학습 ──────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticMetaModel':
        """
        학습 시 내부 동작:
            1. StandardScaler를 X에 fit_transform → 정규화된 X
            2. LogisticRegression(multinomial) 을 정규화된 X와 y로 학습
            3. 학습된 두 모델을 self.scaler_, self.model_에 저장
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}"
            )

        # NaN/Inf 검증 — fit 전에 미리 잡아야 sklearn 에러보다 친절
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf — caller must clean first")

        # 라벨이 0/1/2 안에 있는지 확인 (다른 라벨은 일관성 깨짐)
        unique_y = np.unique(y)
        if not np.isin(unique_y, self.EXPECTED_CLASSES).all():
            raise ValueError(
                f"y must contain only [0, 1, 2], got {unique_y.tolist()}"
            )

        # ── 1. StandardScaler 학습 + 변환 ─────────────────────
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # ── 2. LogisticRegression 학습 ─────────────────────────
        # sklearn 1.5+에서 multi_class 파라미터는 deprecated.
        # lbfgs solver는 자동으로 multinomial(softmax) 모드 사용.
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver='lbfgs',
        )
        self.model_.fit(X_scaled, y)

        # ── 3. 메타 정보 저장 ──────────────────────────────────
        self.n_features_ = X.shape[1]
        if self.feature_names is None:
            self.feature_names_ = [f'feat_{i}' for i in range(self.n_features_)]
        else:
            if len(self.feature_names) != self.n_features_:
                raise ValueError(
                    f"feature_names length {len(self.feature_names)} "
                    f"!= n_features {self.n_features_}"
                )
            self.feature_names_ = list(self.feature_names)

        return self

    # ── 예측 ──────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        [P_Bull, P_Side, P_Bear] shape (n, 3) 반환.

        주의: sklearn은 학습 시 등장한 클래스 ID 순서로 컬럼을 출력.
        라벨이 항상 [0, 1, 2] 모두 등장한다면 자동으로 [Bull, Side, Bear] 순.
        희박한 케이스(한 fold에 한 클래스가 0개)에서 컬럼 누락 가능 →
        EXPECTED_CLASSES 기준으로 reindex 처리.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )

        X_scaled = self.scaler_.transform(X)
        proba = self.model_.predict_proba(X_scaled)

        # 학습 데이터에 없던 클래스가 있으면 컬럼이 누락됨 → 0으로 채워서 reindex
        present_classes = self.model_.classes_
        if not np.array_equal(present_classes, self.EXPECTED_CLASSES):
            full_proba = np.zeros((X.shape[0], 3), dtype=np.float64)
            for i, cls in enumerate(present_classes):
                full_proba[:, int(cls)] = proba[:, i]
            return full_proba

        return proba

    # ── 계수 시각화 ────────────────────────────────────────────
    def get_coef_summary(self) -> pd.DataFrame:
        """
        학습된 계수를 DataFrame으로 반환.

        Returns:
            pd.DataFrame, shape (3, n_features + 1)
            인덱스: ['Bull', 'Side', 'Bear']
            컬럼: feature_names + ['_intercept']
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # sklearn은 등장한 클래스 ID 순서로 coef_를 줌. 누락 클래스가 있으면
        # 그 행을 0으로 채워 [Bull, Side, Bear] 순서로 재배열.
        present = self.model_.classes_
        full_coef = np.zeros((3, self.n_features_), dtype=np.float64)
        full_int = np.zeros(3, dtype=np.float64)
        for i, cls in enumerate(present):
            full_coef[int(cls), :] = self.model_.coef_[i, :]
            full_int[int(cls)] = self.model_.intercept_[i]

        df = pd.DataFrame(
            full_coef,
            index=self.CLASS_NAMES,
            columns=self.feature_names_,
        )
        df['_intercept'] = full_int
        return df

    # ── 저장 / 로드 ────────────────────────────────────────────
    def save(self, path: str) -> None:
        """joblib로 전체 인스턴스 저장."""
        if self.model_ is None:
            raise RuntimeError("Cannot save unfitted model.")
        payload = {
            'C': self.C,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'feature_names_': self.feature_names_,
            'n_features_': self.n_features_,
            'scaler_': self.scaler_,
            'model_': self.model_,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> None:
        """저장된 모델 in-place 로드."""
        payload = joblib.load(path)
        self.C = payload['C']
        self.max_iter = payload['max_iter']
        self.class_weight = payload['class_weight']
        self.random_state = payload['random_state']
        self.feature_names = payload['feature_names']
        self.feature_names_ = payload['feature_names_']
        self.n_features_ = payload['n_features_']
        self.scaler_ = payload['scaler_']
        self.model_ = payload['model_']
