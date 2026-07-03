"""
Retrospective Label Smoother — HMM 라벨 backdate 도구.

────────────────────────────────────────────────────────────────────
배경 — 왜 필요한가?
────────────────────────────────────────────────────────────────────
HMM Viterbi는 국면 지속성(persistence)을 강하게 가정해서, 급격한
전환(폭락/폭등)이 발생해도 라벨을 늦게 바꾼다. 예를 들어:

    [9봉 상승] [1봉 -10% 폭락] [후속 5봉 하락]
                ↓
    HMM:  Bull Bull Bull Bull Bull Bull Bull Bull Bull Bull Bull Side Bear Bear Bear
                                                          ↑ HMM은 폭락 후에야 전환
                                                실제로는: ↓ 여기서 Bull 끝났어야 함

이 라벨로 학습하면 메타 모델이 "전환점"을 잘못된 시점으로 학습.

────────────────────────────────────────────────────────────────────
해결 — 후향적 라벨 보정 (Retrospective Smoothing)
────────────────────────────────────────────────────────────────────
1. HMM 라벨에서 전환점(label[t-1] != label[t]) 찾기
2. 후속 N봉(persistence) 모두 새 국면이면 진짜 전환으로 인정
3. 전환점에서 K봉(lookback) 뒤로 돌아가, |마지막 1봉 수익률|이
   threshold 초과한 봉이 있으면 그 시점으로 라벨을 backdate
4. SIDE 전환은 점진적이라 기본 제외 (옵션으로 포함 가능)

────────────────────────────────────────────────────────────────────
룩어헤드 안전성 (중요)
────────────────────────────────────────────────────────────────────
이 기법은 학습용 정답지(y)만 개선한다. 예측 모델 입력(X)에는
어떤 미래 정보도 들어가지 않는다.

룩어헤드 = "예측 시점에 미래 데이터를 보는 것"
라벨링  = "정답을 만드는 일 — 미래 데이터 사용 OK"

기존 코드에서도 y = label[t+1] (shift -1)로 미래 라벨을 정답으로
쓰고 있어 이미 같은 원칙. Smoother는 그 원칙의 확장이다.

────────────────────────────────────────────────────────────────────
사용 예시
────────────────────────────────────────────────────────────────────
    from strategy.HMM_strategy.regime.label_smoother import RetrospectiveLabelSmoother

    smoother = RetrospectiveLabelSmoother(
        lookback=10, threshold=0.05, persistence_check=3,
    )
    smoothed_labels, change_log = smoother.smooth(
        hmm_labels=labels,
        last_bar_returns=returns,
    )

    # 변경된 시점 분석
    print(f"Total backdates: {len(change_log)}")
    for log in change_log[:5]:
        print(f"  Window {log['original_t']} → backdated to {log['backdated_to']}")
"""

import numpy as np

from strategy.HMM_strategy.regime.hmm_labeler import BULL, SIDE, BEAR


class RetrospectiveLabelSmoother:
    """
    HMM Viterbi 라벨을 폭락/폭등 시점으로 backdate.

    Args:
        lookback: 전환점에서 backdate할 최대 봉 수 (K). 기본 10.
        threshold: |1봉 수익률| 임계값. 기본 0.05 (5%).
        persistence_check: 전환 인정에 필요한 후속 일관성 봉 수. 기본 3.
        include_side: SIDE 전환도 backdate할지. 기본 False.

    Raises:
        ValueError: 잘못된 파라미터
    """

    def __init__(
        self,
        lookback: int = 10,
        threshold: float = 0.05,
        persistence_check: int = 3,
        include_side: bool = False,
    ):
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if persistence_check < 1:
            raise ValueError(
                f"persistence_check must be >= 1, got {persistence_check}"
            )
        self.lookback = lookback
        self.threshold = threshold
        self.persistence_check = persistence_check
        self.include_side = include_side

    def smooth(self, hmm_labels: np.ndarray, last_bar_returns: np.ndarray):
        """
        HMM 라벨을 backdate.

        Args:
            hmm_labels: shape (n,) — HMM Viterbi 정수 라벨 (0=Bull, 1=Side, 2=Bear)
            last_bar_returns: shape (n,) — 각 윈도우의 마지막 1봉 수익률
                             (윈도우 i가 봉 j로 끝나면 (close[j]-close[j-1])/close[j-1])

        Returns:
            (smoothed_labels, change_log)
            smoothed_labels: shape (n,) — backdate 적용된 라벨
            change_log: list of dict — 각 backdate 이벤트의 메타데이터
                {
                    'original_t': int,        # 원래 전환점
                    'backdated_to': int,       # backdate된 시점
                    'old_label': int,          # 전환 전 라벨
                    'new_label': int,          # 전환 후 라벨 (backdate된 라벨)
                    'shock_return': float,     # 그 시점의 수익률
                    'shift': int,              # backdate된 봉 수
                }
        """
        hmm_labels = np.asarray(hmm_labels, dtype=np.int64).ravel()
        last_bar_returns = np.asarray(last_bar_returns, dtype=np.float64).ravel()

        if len(hmm_labels) != len(last_bar_returns):
            raise ValueError(
                f"hmm_labels and last_bar_returns length mismatch: "
                f"{len(hmm_labels)} vs {len(last_bar_returns)}"
            )

        smoothed = hmm_labels.copy()
        n = len(smoothed)
        change_log = []

        # 전환점 찾기 — label[t-1] != label[t]
        for t in range(1, n):
            old_lbl = hmm_labels[t - 1]
            new_lbl = hmm_labels[t]

            if old_lbl == new_lbl:
                continue  # 전환 아님

            # SIDE로 가는 전환은 기본 제외
            if not self.include_side and new_lbl == SIDE:
                continue

            # SIDE → ? 인 경우도 처리 가능 (예: Side → Bear)
            # → 이 경우는 새 라벨이 BULL이나 BEAR이므로 위 체크 통과

            # 안전장치 1: 후속 persistence_check봉 모두 새 국면인가?
            check_end = t + self.persistence_check
            if check_end > n:
                continue
            if not np.all(hmm_labels[t:check_end] == new_lbl):
                continue  # 깜빡임 — 무시

            # lookback 범위에서 폭락/폭등 후보 찾기
            start = max(0, t - self.lookback)
            candidate_returns = last_bar_returns[start:t]

            # NaN 안전 처리
            if not np.isfinite(candidate_returns).any():
                continue

            # 방향 일치 검증 + 임계값
            if new_lbl == BEAR:
                # 가장 음수인 봉 (가장 큰 폭락)
                shock_value = np.nanmin(candidate_returns)
                if shock_value < -self.threshold:
                    idx = start + int(np.nanargmin(candidate_returns))
                    smoothed[idx:t] = new_lbl
                    change_log.append({
                        'original_t': t,
                        'backdated_to': idx,
                        'old_label': int(old_lbl),
                        'new_label': int(new_lbl),
                        'shock_return': float(shock_value),
                        'shift': t - idx,
                    })
            elif new_lbl == BULL:
                # 가장 양수인 봉 (가장 큰 폭등)
                shock_value = np.nanmax(candidate_returns)
                if shock_value > self.threshold:
                    idx = start + int(np.nanargmax(candidate_returns))
                    smoothed[idx:t] = new_lbl
                    change_log.append({
                        'original_t': t,
                        'backdated_to': idx,
                        'old_label': int(old_lbl),
                        'new_label': int(new_lbl),
                        'shock_return': float(shock_value),
                        'shift': t - idx,
                    })
            elif new_lbl == SIDE:
                # include_side=True인 경우만 도달. SIDE는 점진적 전환이라
                # 큰 충격 신호가 약함. 임의로 "음수든 양수든 절댓값이 크면"
                # 전환점으로 잡음.
                abs_returns = np.abs(candidate_returns)
                if np.nanmax(abs_returns) > self.threshold:
                    idx = start + int(np.nanargmax(abs_returns))
                    smoothed[idx:t] = new_lbl
                    change_log.append({
                        'original_t': t,
                        'backdated_to': idx,
                        'old_label': int(old_lbl),
                        'new_label': int(new_lbl),
                        'shock_return': float(candidate_returns[idx - start]),
                        'shift': t - idx,
                    })

        return smoothed, change_log

    def summarize_changes(self, change_log) -> dict:
        """change_log 통계 요약 (보고용)."""
        if not change_log:
            return {
                'n_backdates': 0,
                'mean_shift': 0.0,
                'mean_shock': 0.0,
                'by_direction': {},
            }
        shifts = np.array([c['shift'] for c in change_log])
        shocks = np.array([c['shock_return'] for c in change_log])

        # 방향별 분리 (Bull로 backdate vs Bear로 backdate)
        from collections import Counter
        direction_counter = Counter([c['new_label'] for c in change_log])

        return {
            'n_backdates': len(change_log),
            'mean_shift': float(shifts.mean()),
            'median_shift': float(np.median(shifts)),
            'max_shift': int(shifts.max()),
            'mean_shock_abs': float(np.abs(shocks).mean()),
            'max_shock_abs': float(np.abs(shocks).max()),
            'by_direction': dict(direction_counter),
        }
