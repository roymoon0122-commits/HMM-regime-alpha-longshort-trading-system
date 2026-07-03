"""
HMM 라벨러 캐시 빌더 — Phase 3 진입 전 준비 작업.

────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일
────────────────────────────────────────────────────────────────────
1. BTC 1분봉 → 4시간봉 리샘플
2. 윈도우 피처 9개 계산
3. HMM_FEATURE_COLS 5개 추출 + RollingStandardScaler 정규화
4. HMMLabeler(n_states=3, restart=30) 학습
5. 결과를 config.HMM_MODEL_PATH (기본: models/hmm_btc.joblib) 에 저장
6. 라벨 분포 (Bull/Side/Bear %) 출력

────────────────────────────────────────────────────────────────────
사용 방법
────────────────────────────────────────────────────────────────────
    cd Coin-trader-main

    # 캐시가 없으면 학습, 있으면 그냥 종료 (기본 동작)
    python -m strategy.HMM_strategy.scripts.build_hmm_cache

    # 강제 재학습 (피처/스케일러/restart 변경 후)
    python -m strategy.HMM_strategy.scripts.build_hmm_cache --retrain

    # 빠른 검증 (restart 적게)
    python -m strategy.HMM_strategy.scripts.build_hmm_cache --retrain --n-restart 5

    # 다른 경로로 저장
    python -m strategy.HMM_strategy.scripts.build_hmm_cache --out models/hmm_v2.joblib

────────────────────────────────────────────────────────────────────
왜 이 스크립트가 필요한가?
────────────────────────────────────────────────────────────────────
Phase 3에서 메타 모델을 학습할 때마다 HMM을 처음부터 재학습하는 것은
시간 낭비다 (30 restart × 13,500행 ≈ 14초). 한 번 학습해서 저장해두고,
필요할 때만 --retrain 으로 새로 학습하는 워크플로우가 효율적이다.

라벨러 캐시 = HMMLabeler 인스턴스 전체 (joblib 직렬화)
            = GaussianHMM 모델 + state→regime 매핑 + fit 이력

────────────────────────────────────────────────────────────────────
재학습이 필요한 경우 (참고)
────────────────────────────────────────────────────────────────────
- HMM_FEATURE_COLS 변경
- SCALER_MODE 또는 ROLLING_SCALER_WINDOW 변경
- N_STATES, HMM_COVARIANCE_TYPE 변경
- WINDOW_SIZE, ADX_PERIOD, R2_PERIOD 등 피처 계산 파라미터 변경
- 데이터 자체가 갱신됨

위 중 하나라도 바꿨으면 --retrain 필수.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.resampler import load_and_resample
from strategy.HMM_strategy.features.window_features import compute_window_features
from strategy.HMM_strategy.features.scaling import RollingStandardScaler
from strategy.HMM_strategy.regime.hmm_labeler import (
    HMMLabeler, BULL, SIDE, BEAR, REGIME_NAMES,
)


# ════════════════════════════════════════════════════════════════
#  CLI 인자
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HMM 라벨러 캐시 빌더 (Phase 3 준비)"
    )
    p.add_argument('--csv-path', default=config.DATA_PATH,
                   help=f"1분봉 CSV 경로 (기본: {config.DATA_PATH})")
    p.add_argument('--timeframe', default=config.TIMEFRAME,
                   help=f"리샘플링 타임프레임 (기본: {config.TIMEFRAME})")
    p.add_argument('--window-size', type=int, default=config.WINDOW_SIZE,
                   help=f"윈도우 봉 수 (기본: {config.WINDOW_SIZE})")
    p.add_argument('--rolling-window', type=int, default=config.ROLLING_SCALER_WINDOW,
                   help=f"Rolling scaler 윈도우 (기본: {config.ROLLING_SCALER_WINDOW})")
    p.add_argument('--n-states', type=int, default=config.N_STATES,
                   help=f"HMM 상태 수 (기본: {config.N_STATES})")
    p.add_argument('--n-restart', type=int, default=config.HMM_RANDOM_RESTART,
                   help=f"Random restart 횟수 (기본: {config.HMM_RANDOM_RESTART})")
    p.add_argument('--n-iter', type=int, default=config.HMM_N_ITER,
                   help=f"Baum-Welch 반복 (기본: {config.HMM_N_ITER})")
    p.add_argument('--covariance', default=config.HMM_COVARIANCE_TYPE,
                   choices=['diag', 'full', 'spherical', 'tied'],
                   help=f"공분산 타입 (기본: {config.HMM_COVARIANCE_TYPE})")
    p.add_argument('--random-state', type=int, default=42,
                   help="랜덤 시드 (기본: 42)")
    p.add_argument('--out', default=config.HMM_MODEL_PATH,
                   help=f"저장 경로 (기본: {config.HMM_MODEL_PATH})")
    p.add_argument('--retrain', action='store_true',
                   help="이미 캐시가 있어도 강제 재학습")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  메인 로직
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_path = Path(args.out)

    # ── 캐시가 이미 있으면 종료 ────────────────────────────────
    if out_path.exists() and not args.retrain:
        print(f"✓ 캐시가 이미 존재: {out_path}")
        print(f"  재학습이 필요하면 --retrain 옵션을 사용하세요.")

        # 빠른 sanity check — 캐시 로드 가능한지만 확인
        try:
            test_labeler = HMMLabeler()
            test_labeler.load(str(out_path))
            print(f"✓ 캐시 로드 성공: n_states={test_labeler.n_states}, "
                  f"covariance={test_labeler.covariance_type}")
        except Exception as e:
            print(f"⚠ 캐시 로드 실패: {e}")
            print(f"  --retrain 으로 재학습을 권장합니다.")
            sys.exit(1)
        return

    # ── 1. 데이터 로드 + 리샘플 ───────────────────────────────
    print(f"[1/5] 데이터 로드: {args.csv_path}")
    t0 = time.time()
    df = load_and_resample(args.csv_path, timeframe=args.timeframe)
    print(f"      → {len(df):,}봉 (타임프레임 {args.timeframe}), "
          f"{time.time()-t0:.1f}초")

    # ── 2. 윈도우 피처 계산 ───────────────────────────────────
    print(f"[2/5] 윈도우 피처 계산 (window_size={args.window_size})")
    t0 = time.time()
    features = compute_window_features(
        df,
        window_size=args.window_size,
        adx_period=config.ADX_PERIOD,
        r2_period=config.R2_PERIOD,
    )
    print(f"      → {len(features):,}개 윈도우, {time.time()-t0:.1f}초")

    # ── 3. HMM_FEATURE_COLS 추출 + Rolling 정규화 ─────────────
    print(f"[3/5] HMM 피처 추출 + RollingStandardScaler "
          f"(window={args.rolling_window})")
    print(f"      → 사용 피처: {config.HMM_FEATURE_COLS}")
    X_raw = features[config.HMM_FEATURE_COLS].values

    scaler = RollingStandardScaler(window=args.rolling_window)
    X_scaled = scaler.fit_transform(X_raw)

    # cold start NaN 행 제거 (rolling이 초반 ~window-1행을 NaN으로 만듦)
    valid_mask = ~np.isnan(X_scaled).any(axis=1)
    X = X_scaled[valid_mask]
    cum_return = features['cum_return'].values[valid_mask]

    print(f"      → 학습 가능 행수: {len(X):,} (cold start 제거 "
          f"{(~valid_mask).sum():,}행)")

    # ── 4. HMM 학습 ──────────────────────────────────────────
    print(f"[4/5] HMM 학습 (n_states={args.n_states}, "
          f"restart={args.n_restart}, covariance={args.covariance})")
    t0 = time.time()
    labeler = HMMLabeler(
        n_states=args.n_states,
        n_iter=args.n_iter,
        n_random_restart=args.n_restart,
        covariance_type=args.covariance,
        random_state=args.random_state,
    )
    labeler.fit(X, cum_return)
    elapsed = time.time() - t0

    converged_count = sum(1 for _, _, c in labeler.fit_history_ if c)
    print(f"      → {elapsed:.1f}초, "
          f"수렴 {converged_count}/{args.n_restart}회, "
          f"best log-likelihood: {labeler.best_score_:,.0f}")

    # 라벨 분포 출력
    labels = labeler.predict(X)
    n = len(labels)
    print(f"      → 라벨 분포:")
    for regime_id in [BULL, SIDE, BEAR]:
        cnt = (labels == regime_id).sum()
        print(f"        {REGIME_NAMES[regime_id]:>5}: {cnt:>6,} "
              f"({100*cnt/n:5.1f}%)")

    # ── 5. 저장 ──────────────────────────────────────────────
    print(f"[5/5] 캐시 저장: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeler.save(str(out_path))
    file_size_kb = out_path.stat().st_size / 1024
    print(f"      → 저장 완료 ({file_size_kb:.1f} KB)")

    print()
    print("═" * 60)
    print(" Phase 3 메타 모델 학습 시 다음 코드로 로드 가능:")
    print("═" * 60)
    print(f"""
    from strategy.HMM_strategy.regime.hmm_labeler import HMMLabeler
    labeler = HMMLabeler()
    labeler.load("{out_path}")
    labels = labeler.predict(X)        # 0/1/2
    proba  = labeler.predict_proba(X)  # (n, 3)
    transmat = labeler.model_.transmat_  # 전이 행렬 (3, 3)
    """)


if __name__ == '__main__':
    main()
