"""
HMM 기반 국면 분류기 (Regime Labeler).

────────────────────────────────────────────────────────────────────
역할
────────────────────────────────────────────────────────────────────
윈도우 피처 행렬 X(n_samples, n_features)를 받아 각 행에 Bull/Side/Bear
라벨을 자동으로 부착한다. 내부적으로 hmmlearn의 GaussianHMM을 학습하고,
다음 5가지 부가 기능을 얹는다:

    1. K-means 초기화        — 평균값을 데이터 기반 좋은 시작점에 둠
    2. Random Restart        — 다양한 시작점에서 학습 후 최고 모델 선택
    3. 수렴 실패 모델 제외    — 신뢰할 수 없는 모델은 best 선정에서 제외
    4. BIC 기반 상태 수 선택  — n_states를 데이터에 맞게 결정 (옵션)
    5. 상태→국면 자동 매핑    — cum_return 평균 기준으로 Bull/Side/Bear 부여

────────────────────────────────────────────────────────────────────
스케일링 책임 분리 (중요)
────────────────────────────────────────────────────────────────────
이 클래스는 X에 어떤 정규화가 되어 있는지 모른다.
호출자가 'none' / 'global' / 'rolling' 모드로 X를 미리 정규화한 뒤 전달해야 한다.
이유: 한 클래스에 너무 많은 책임을 주지 않기 위함 (Single Responsibility).
검증/실험 단계(Phase 2)에서 모드 비교가 깔끔해진다.

X의 NaN 행은 호출자가 미리 제거해야 한다 (rolling cold start 등).

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy.regime.hmm_labeler import HMMLabeler, BULL, SIDE, BEAR

    labeler = HMMLabeler(n_states=3, n_random_restart=30)
    labeler.fit(X, cum_return)        # X와 cum_return은 같은 길이의 1D/2D 배열
    labels = labeler.predict(X)        # 0=Bull, 1=Side, 2=Bear
    proba  = labeler.predict_proba(X)  # shape (n, 3), 컬럼 순서 [Bull, Side, Bear]
    labeler.save("hmm_btc.joblib")
"""

from datetime import datetime
from typing import Optional

import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans


# ─── 국면 ID 정의 ──────────────────────────────────────────────
# predict() 결과의 정수 의미
BULL = 0
SIDE = 1
BEAR = 2

REGIME_NAMES = {BULL: 'Bull', SIDE: 'Side', BEAR: 'Bear'}


# ─── 캐시 메타데이터 자동 캡처 헬퍼 (모듈 레벨) ─────────────────
# save() 내부에서 호출되어 캐시 파일에 박제할 메타데이터를 자동 생성한다.
# 호출자(HMMStrategy 등)는 이 함수들을 직접 부를 필요 없음.

def _capture_env_info() -> dict:
    """현재 환경의 Python/numpy/sklearn/hmmlearn 버전 정보를 dict로 캡처.

    캐시 파일이 만들어진 시점의 라이브러리 버전을 영구 박제하기 위함.
    어느 import라도 실패하면 그 항목만 빠지거나 전체가 빈 dict로 fallback.
    """
    info = {}
    try:
        import sys
        info['python'] = sys.version.split()[0]
    except Exception:
        pass
    for pkg_name in ('numpy', 'sklearn', 'hmmlearn'):
        try:
            mod = __import__(pkg_name)
            info[pkg_name] = getattr(mod, '__version__', 'unknown')
        except Exception:
            pass
    return info


def _snapshot_upper_snake(cfg_module) -> dict:
    """주어진 config 모듈에서 모든 UPPER_SNAKE 상수를 dict로 스냅샷.

    예시) WINDOW_SIZE=60, ADX_THRESHOLD=25 같은 상수만 모음.
    함수/모듈/언더스코어로 시작하는 항목은 제외.

    Args:
        cfg_module: 스냅샷할 config 모듈. None이면 빈 dict 반환.

    Returns:
        {'WINDOW_SIZE': 60, 'ADX_PERIOD': 12, ...} 형태.
        joblib 직렬화 가능한 기본 타입은 그대로, 그 외엔 str()로 변환됨.
    """
    if cfg_module is None:
        return {}

    import types
    snap = {}
    for name in dir(cfg_module):
        if name.startswith('_'):
            continue
        if not name.isupper():
            continue
        val = getattr(cfg_module, name)
        if callable(val) or isinstance(val, types.ModuleType):
            continue
        # joblib 직렬화 가능한 기본 타입은 그대로, 그 외엔 str()로
        if isinstance(val, (int, float, str, bool, type(None))):
            snap[name] = val
        elif isinstance(val, (list, tuple)):
            snap[name] = list(val)
        elif isinstance(val, dict):
            snap[name] = dict(val)
        else:
            snap[name] = str(val)
    return snap


class HMMLabeler:
    """
    Gaussian HMM 기반 국면 분류기.

    Args:
        n_states: HMM 상태 수 (기본 3 = Bull/Side/Bear)
        n_iter: Baum-Welch 최대 반복 (수렴 안 하면 여기서 끊김)
        n_random_restart: 다양한 K-means 초기점에서 재학습할 횟수
        covariance_type: 'diag' | 'full' | 'spherical' (hmmlearn 인자)
        random_state: 재현 가능성용 시드 (각 restart는 random_state + i 사용)

    학습 후 채워지는 속성 (trailing underscore = sklearn 컨벤션):
        model_:           hmmlearn.GaussianHMM (best 모델)
        state_to_regime_: {raw_state_id: regime_id} — Viterbi 출력을 0/1/2로 변환
        regime_to_state_: 역방향 매핑
        best_score_:      best 모델의 log-likelihood
        fit_history_:     각 restart의 (idx, score, converged) 기록
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        n_random_restart: int = 30,
        covariance_type: str = 'diag',
        random_state: int = 42,
    ):
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        if n_random_restart < 1:
            raise ValueError(f"n_random_restart must be >= 1, got {n_random_restart}")

        self.n_states = n_states
        self.n_iter = n_iter
        self.n_random_restart = n_random_restart
        self.covariance_type = covariance_type
        self.random_state = random_state

        # 학습 후 채워짐
        self.model_ = None
        self.state_to_regime_ = None
        self.regime_to_state_ = None
        self.best_score_ = None
        self.fit_history_ = []
        # 신규(2026-05): 학습 시점 메타데이터 자동 캡처 시스템.
        # - _training_context_ : fit() 시점에 자동 캡처되는 학습 컨텍스트
        #                        (n_samples, n_features 자동 + 호출자가 보강 가능)
        # - training_metadata_ : save() 시점에 자동 조립되어 캐시에 동봉되는 dict.
        #                        load() 시 캐시에서 복원되어 노출됨.
        # 구 형식 캐시(이 키 없음)는 빈 dict로 처리되어 호환성 유지.
        self._training_context_ = {}
        self.training_metadata_ = {}

    # ─── 내부 헬퍼 ──────────────────────────────────────────────

    def _initial_transmat(self, n: int) -> np.ndarray:
        """
        전이 행렬 초기값.

        n=3일 때는 기획서 4-2의 도메인 지식 값 사용:
            대각선 0.85 (국면 지속성), Side 행 대칭, Bull↔Bear 직접 전환 0.03

        n!=3일 때는 일반화: 대각선 0.95, off-diagonal 균등 분포.
        """
        if n == 3:
            return np.array([
                [0.85, 0.12, 0.03],   # Bull → ...
                [0.08, 0.84, 0.08],   # Side → ...
                [0.03, 0.12, 0.85],   # Bear → ...
            ])
        else:
            P = np.full((n, n), 0.05 / max(n - 1, 1))
            np.fill_diagonal(P, 0.95)
            return P

    def _kmeans_init(self, X: np.ndarray, n: int, rs: int) -> np.ndarray:
        """K-means 클러스터 중심을 HMM means_init으로 사용."""
        km = KMeans(n_clusters=n, random_state=rs, n_init=10)
        km.fit(X)
        return km.cluster_centers_

    def _build_hmm(self, n: int, rs: int) -> GaussianHMM:
        """기본 GaussianHMM 인스턴스 생성 (means/transmat/startprob는 호출자가 설정)."""
        return GaussianHMM(
            n_components=n,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=rs,
            init_params='c',  # 'c' = 공분산만 자동 초기화. 나머지(s,t,m)는 우리가 설정.
            params='stmc',    # 학습 중 갱신할 파라미터: s,t,m,c 모두
        )

    def _fit_one(self, X: np.ndarray, n: int, restart_idx: int) -> tuple:
        """
        한 번 학습. 실패하면 (None, -inf, False) 반환.

        Returns:
            (model, log_likelihood, converged)
        """
        rs = self.random_state + restart_idx

        try:
            means_init = self._kmeans_init(X, n, rs)
            model = self._build_hmm(n, rs)
            model.startprob_ = np.full(n, 1.0 / n)
            model.transmat_ = self._initial_transmat(n)
            model.means_ = means_init
            model.fit(X)
            score = model.score(X)
            converged = bool(model.monitor_.converged)
            return (model, score, converged)
        except Exception:
            return (None, -np.inf, False)

    def _map_states_to_regimes(
        self,
        raw_states: np.ndarray,
        cum_return: np.ndarray,
    ) -> dict:
        """
        Raw state ID → 국면 ID 매핑 dict 생성.

        cum_return 평균이 가장 높은 상태 → Bull(0)
        가장 낮은 상태                  → Bear(2)
        나머지                          → Side(1) (n=3일 때)

        Args:
            raw_states: shape (n_samples,) — model.predict(X) 결과
            cum_return: shape (n_samples,) — 원본(unscaled) cum_return

        Returns:
            {raw_state_id: regime_id} dict
        """
        # 각 raw state의 cum_return 평균
        state_means = {}
        for state_id in range(self.n_states):
            mask = (raw_states == state_id)
            if mask.sum() == 0:
                # 한 번도 방문 안 한 ghost state — 0으로 처리 (중간 위치)
                state_means[state_id] = 0.0
            else:
                state_means[state_id] = float(cum_return[mask].mean())

        # 평균 내림차순 정렬: 가장 높은 게 0번 (Bull)
        sorted_states = sorted(state_means.keys(), key=lambda s: -state_means[s])

        mapping = {}
        if self.n_states == 3:
            mapping[sorted_states[0]] = BULL
            mapping[sorted_states[1]] = SIDE
            mapping[sorted_states[2]] = BEAR
        else:
            # n!=3 일반화: 1등 = Bull, 꼴등 = Bear, 나머지 = Side
            for rank, state_id in enumerate(sorted_states):
                if rank == 0:
                    mapping[state_id] = BULL
                elif rank == len(sorted_states) - 1:
                    mapping[state_id] = BEAR
                else:
                    mapping[state_id] = SIDE

        return mapping

    def _bic(self, log_likelihood: float, n_samples: int, n: int, n_features: int) -> float:
        """
        BIC 계산.

        BIC = -2 * log_likelihood + k * log(N)

        파라미터 수 k:
            - means: n * n_features
            - covariances: n * n_features (diag) or n * n_features*(n_features+1)/2 (full)
            - transmat: n * (n - 1) (각 행 합 = 1 제약)
            - startprob: n - 1 (합 = 1 제약)
        """
        if self.covariance_type == 'diag':
            cov_params = n * n_features
        elif self.covariance_type == 'full':
            cov_params = n * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'spherical':
            cov_params = n
        else:
            cov_params = n * n_features  # fallback

        k = n * n_features + cov_params + n * (n - 1) + (n - 1)
        return -2.0 * log_likelihood + k * np.log(n_samples)

    # ─── 공개 API ──────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        cum_return: np.ndarray,
        training_context: Optional[dict] = None,
    ) -> 'HMMLabeler':
        """
        HMM 학습 + 상태→국면 자동 매핑까지 한 번에.

        Args:
            X: shape (n_samples, n_features), NaN 없어야 함
            cum_return: shape (n_samples,), 원본(unscaled) cum_return.
                        상태→국면 매핑에만 사용. X 안의 cum_return 컬럼이
                        이미 정규화돼 있을 수 있어 별도로 받음.
            training_context: (선택) save() 시 캐시에 함께 박제할 추가 컨텍스트 dict.
                              labeler가 자동 캡처하지 못하는 정보(예: 학습 데이터의
                              datetime 범위, 피처 컬럼명 등)를 호출자가 보강할 때 사용.
                              n_samples, n_features는 자동 캡처되므로 따로 안 넣어도 됨.
                              예: {'date_start': '2020-01-01', 'date_end': '2023-12-31'}

        Returns:
            self (chaining 지원)

        Raises:
            ValueError: shape 불일치 또는 NaN 존재
            RuntimeError: 모든 restart가 수렴 실패한 경우
        """
        X = np.asarray(X, dtype=np.float64)
        cum_return = np.asarray(cum_return, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != cum_return.shape[0]:
            raise ValueError(
                f"X and cum_return length mismatch: {X.shape[0]} vs {cum_return.shape[0]}"
            )
        if np.isnan(X).any():
            raise ValueError(
                "X contains NaN. Did you forget to drop rolling-scaler cold-start rows?"
            )

        # Random Restart 루프
        self.fit_history_ = []
        best_model = None
        best_score = -np.inf

        for i in range(self.n_random_restart):
            model, score, converged = self._fit_one(X, self.n_states, i)
            self.fit_history_.append({
                'restart_idx': i,
                'score': score,
                'converged': converged,
            })
            # 수렴한 모델만 best 후보
            if converged and score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            n_failed = sum(1 for h in self.fit_history_ if not h['converged'])
            raise RuntimeError(
                f"All {self.n_random_restart} random restarts failed to converge "
                f"({n_failed} not converged). "
                f"Try increasing n_iter, or check feature scaling."
            )

        self.model_ = best_model
        self.best_score_ = best_score

        # 상태→국면 매핑
        raw_states = best_model.predict(X)
        self.state_to_regime_ = self._map_states_to_regimes(raw_states, cum_return)
        self.regime_to_state_ = {v: k for k, v in self.state_to_regime_.items()}

        # 학습 컨텍스트 자동 캡처 (save() 시 메타데이터에 동봉됨).
        # 자동: n_samples, n_features. 추가: 호출자가 training_context로 넘긴 정보.
        self._training_context_ = {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
        }
        if training_context:
            self._training_context_.update(training_context)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi 라벨 (이미 매핑된 0/1/2).

        Args:
            X: shape (n_samples, n_features)

        Returns:
            shape (n_samples,), dtype int. 값은 BULL=0 / SIDE=1 / BEAR=2.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        raw_states = self.model_.predict(X)
        # state_to_regime은 모든 raw state ID에 대해 매핑이 정의되어 있음
        labels = np.array([self.state_to_regime_[s] for s in raw_states], dtype=np.int64)
        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        각 시점의 국면별 사후확률.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            shape (n_samples, 3), 컬럼 순서 [P(Bull), P(Side), P(Bear)].
            각 행의 합은 1.0.

        주의:
            n_states != 3일 때도 결과는 항상 (n, 3)이며, 같은 regime으로
            매핑된 여러 raw state의 확률은 합산된다.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        raw_proba = self.model_.predict_proba(X)  # (n, n_states), 컬럼=raw_state_id

        # 컬럼 재배열: raw_state_id 순서 → [Bull, Side, Bear] 순서
        result = np.zeros((X.shape[0], 3), dtype=np.float64)
        for raw_state_id, regime_id in self.state_to_regime_.items():
            result[:, regime_id] += raw_proba[:, raw_state_id]
        return result

    def select_n_states_by_bic(
        self,
        X: np.ndarray,
        candidates=range(2, 6),
        n_restart_quick: int = 5,
    ) -> tuple:
        """
        BIC가 가장 낮은 n_states 후보를 반환.

        부수 효과 없음 (self.n_states 변경 안 함). 결과를 본 뒤 사용자가
        새 HMMLabeler를 원하는 n_states로 만들어 fit하는 흐름.

        Args:
            X: shape (n_samples, n_features)
            candidates: 시도할 n_states 후보들 (기본 2~5)
            n_restart_quick: BIC 비교 시에는 빠른 비교가 목적이므로
                             정식 학습보다 적은 restart 사용 (기본 5)

        Returns:
            (best_n_states, {n: bic_value, ...})
            학습 실패한 후보의 BIC는 inf로 표기됨
        """
        X = np.asarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("X contains NaN.")

        n_samples, n_features = X.shape
        bic_results = {}

        for n in candidates:
            # 빠른 비교 — restart 적게
            best_score = -np.inf
            for i in range(n_restart_quick):
                rs = self.random_state + i
                try:
                    means_init = self._kmeans_init(X, n, rs)
                    model = self._build_hmm(n, rs)
                    model.startprob_ = np.full(n, 1.0 / n)
                    model.transmat_ = self._initial_transmat(n)
                    model.means_ = means_init
                    model.fit(X)
                    if model.monitor_.converged:
                        score = model.score(X)
                        if score > best_score:
                            best_score = score
                except Exception:
                    continue

            if best_score == -np.inf:
                bic_results[int(n)] = float('inf')
            else:
                bic_results[int(n)] = self._bic(best_score, n_samples, n, n_features)

        # 가장 작은 BIC가 베스트
        best_n = min(bic_results, key=bic_results.get)
        return (best_n, bic_results)

    def save(self, path: str, config_module=None) -> None:
        """
        모델 + 매핑 + 학습 메타데이터를 joblib으로 저장.

        scaler는 저장하지 않음 — 호출자(verify 스크립트, 전략 클래스)가 별도 관리.

        ────────────────────────────────────────────────────────────
        자동 캡처되는 메타데이터 (호출자가 따로 안 넣어도 됨)
        ────────────────────────────────────────────────────────────
        joblib에 'training_metadata' 키로 다음을 자동 박제:
            'created_at'       : 저장 시각 (ISO 8601 문자열, 분/초 단위)
            'env'              : Python/numpy/sklearn/hmmlearn 버전
            'config_snapshot'  : config_module의 모든 UPPER_SNAKE 상수
            'training_data'    : fit() 시점에 캡처된 컨텍스트
                                 (n_samples, n_features 자동 + fit()에 넘긴
                                  training_context 인자 내용)

        Args:
            path: 저장 경로
            config_module: (선택, 보통 안 넘김) 스냅샷할 config 모듈.
                None이면 strategy.HMM_strategy.config을 자동 import 시도.
                테스트 시 mock 모듈 주입 또는 다른 config를 박제하고 싶을 때만 명시.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first before save().")

        # config 자동 로드 (테스트 시 인자로 주입 가능 — Pattern B 호환).
        # ImportError나 다른 이유로 실패하면 config_snapshot이 빈 dict가 됨.
        if config_module is None:
            try:
                from strategy.HMM_strategy import config as config_module
            except ImportError:
                config_module = None

        # 메타데이터 자동 조립
        training_metadata = {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'env': _capture_env_info(),
            'config_snapshot': _snapshot_upper_snake(config_module),
            'training_data': dict(self._training_context_),
        }
        # 인스턴스 속성에도 캐시 (저장 후 곧바로 verbose 출력 등에 활용 가능)
        self.training_metadata_ = training_metadata

        joblib.dump({
            'model': self.model_,
            'state_to_regime': self.state_to_regime_,
            'regime_to_state': self.regime_to_state_,
            'best_score': self.best_score_,
            'fit_history': self.fit_history_,
            'config': {
                'n_states': self.n_states,
                'n_iter': self.n_iter,
                'n_random_restart': self.n_random_restart,
                'covariance_type': self.covariance_type,
                'random_state': self.random_state,
            },
            'training_metadata': training_metadata,
        }, path)

    def load(self, path: str) -> 'HMMLabeler':
        """저장된 모델 로드. self를 반환해서 chaining 지원."""
        data = joblib.load(path)
        self.model_ = data['model']
        self.state_to_regime_ = data['state_to_regime']
        self.regime_to_state_ = data['regime_to_state']
        self.best_score_ = data['best_score']
        self.fit_history_ = data['fit_history']

        cfg = data.get('config', {})
        self.n_states = cfg.get('n_states', self.n_states)
        self.n_iter = cfg.get('n_iter', self.n_iter)
        self.n_random_restart = cfg.get('n_random_restart', self.n_random_restart)
        self.covariance_type = cfg.get('covariance_type', self.covariance_type)
        self.random_state = cfg.get('random_state', self.random_state)

        # 신규(2026-05): training_metadata 복원.
        # 구 형식 캐시는 이 키가 없으므로 빈 dict로 안전 처리.
        self.training_metadata_ = data.get('training_metadata', {})
        return self

    def __repr__(self):
        fitted = "fitted" if self.model_ is not None else "not fitted"
        return (
            f"HMMLabeler(n_states={self.n_states}, "
            f"covariance_type='{self.covariance_type}', "
            f"n_random_restart={self.n_random_restart}, {fitted})"
        )
