"""
Phase 3 메타 모델 검증 스크립트.

────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일
────────────────────────────────────────────────────────────────────
1. BTC 4h 데이터 → 윈도우 피처 (Phase 1 모듈)
2. HMM 캐시 로드 (없으면 학습), slope을 rolling z-score로 정규화
3. Base Classifier(ADX, R²) 출력 계산
4. HMM 사후확률 + 전이행렬 사전확률 계산
5. 메타 입력 X (16개 피처) + 라벨 y(다음 윈도우 HMM 라벨) 구성
6. TimeSeriesSplit 5-fold 학습/검증 (gap=WINDOW_SIZE)
7. 정확도, 혼동행렬, 계수 시각화

────────────────────────────────────────────────────────────────────
실행 방법
────────────────────────────────────────────────────────────────────
    cd Coin-trader-main

    # 기본 실행 (HMM 캐시 사용)
    python -m strategy.HMM_strategy.scripts.verify_meta_model

    # HMM 재학습 후 검증
    python -m strategy.HMM_strategy.scripts.verify_meta_model --retrain-hmm

    # 헤드리스 + PNG 저장
    python -m strategy.HMM_strategy.scripts.verify_meta_model --no-show \\
        --save-cm conf_matrix.png --save-coef coefficients.png

    # regularization 강도 변경
    python -m strategy.HMM_strategy.scripts.verify_meta_model --C 0.1

────────────────────────────────────────────────────────────────────
출력 해석
────────────────────────────────────────────────────────────────────
혼동행렬: 행=실제 라벨, 열=예측 라벨. 대각선이 강할수록 정확.
계수 그래프: 양수 큰 값=그 클래스에 강한 양의 영향, 음수 큰 값=반대.
            예: ADX 분류기 P_Bull 출력의 Bull 행 계수가 +1.5라면
            → ADX 분류기가 Bull이라고 할수록 메타 모델도 Bull로 더 예측.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.resampler import load_and_resample
from strategy.HMM_strategy.features.window_features import compute_window_features
from strategy.HMM_strategy.features.scaling import RollingStandardScaler
from strategy.HMM_strategy.regime.hmm_labeler import HMMLabeler, BULL, SIDE, BEAR, REGIME_NAMES
from strategy.HMM_strategy.regime.regime_dataset import RegimeDataset
from strategy.HMM_strategy.regime.transition import TransitionPredictor
from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier
from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier
from strategy.HMM_strategy.meta_model.logistic_meta_model import LogisticMetaModel


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 메타 모델 검증 스크립트")
    # 데이터 / 피처
    p.add_argument('--csv-path', default=config.DATA_PATH)
    p.add_argument('--timeframe', default=config.TIMEFRAME)
    p.add_argument('--window-size', type=int, default=config.WINDOW_SIZE)
    p.add_argument('--rolling-window', type=int, default=config.ROLLING_SCALER_WINDOW)
    # HMM
    p.add_argument('--hmm-cache', default=config.HMM_MODEL_PATH,
                   help=f"HMM 캐시 경로 (기본: {config.HMM_MODEL_PATH})")
    p.add_argument('--retrain-hmm', action='store_true',
                   help="HMM 캐시 무시하고 재학습")
    p.add_argument('--n-restart', type=int, default=config.HMM_RANDOM_RESTART)
    # 메타 모델
    p.add_argument('--C', type=float, default=1.0,
                   help="LogisticRegression 정규화 강도 역수 (기본 1.0)")
    p.add_argument('--class-weight', default='balanced',
                   choices=['balanced', 'none'],
                   help="클래스 가중 (기본 balanced)")
    p.add_argument('--ts-splits', type=int, default=config.TS_SPLIT_N,
                   help=f"TimeSeriesSplit fold 수 (기본 {config.TS_SPLIT_N})")
    # 출력
    p.add_argument('--save-cm', default=None, help="혼동행렬 PNG 저장 경로")
    p.add_argument('--save-coef', default=None, help="계수 막대그래프 PNG 저장 경로")
    p.add_argument('--no-show', action='store_true', help="화면 안 띄우기 (헤드리스)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  HMM 라벨러 준비 (캐시 사용 or 재학습)
# ════════════════════════════════════════════════════════════════

def prepare_hmm_labeler(args, X_full, cum_return_full):
    """캐시가 있고 --retrain-hmm 아니면 로드, 아니면 학습 후 저장."""
    cache_path = Path(args.hmm_cache)

    if cache_path.exists() and not args.retrain_hmm:
        labeler = HMMLabeler()
        labeler.load(str(cache_path))
        print(f"      → 캐시 로드: {cache_path}")
        return labeler

    # 재학습
    labeler = HMMLabeler(
        n_states=config.N_STATES,
        n_iter=config.HMM_N_ITER,
        n_random_restart=args.n_restart,
        covariance_type=config.HMM_COVARIANCE_TYPE,
        random_state=42,
    )
    print(f"      → HMM 재학습 (restart={args.n_restart})...")
    t0 = time.time()
    labeler.fit(X_full, cum_return_full)
    print(f"      → 학습 완료 ({time.time()-t0:.1f}초)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labeler.save(str(cache_path))
    print(f"      → 캐시 저장: {cache_path}")
    return labeler


# ════════════════════════════════════════════════════════════════
#  메타 입력 X 구성 — 핵심 로직
# ════════════════════════════════════════════════════════════════

def build_meta_features(features, labeler, transition_predictor):
    """
    16개 피처 행렬 X와 피처 이름을 반환.

    Args:
        features: pd.DataFrame — 윈도우 피처 (slope_norm 컬럼 포함된 상태)
        labeler:  학습된 HMMLabeler
        transition_predictor: TransitionPredictor

    Returns:
        X: shape (n, 16)
        feature_names: list of str (16개)
    """
    # ── 1. ADX 분류기 출력 ─────────────────────────────────────
    adx_clf = ADXClassifier(
        threshold=config.ADX_THRESHOLD,
        adx_steepness=config.ADX_CLF_STEEPNESS,
        direction_steepness=config.DIRECTION_STEEPNESS,
    )
    adx_proba = adx_clf.predict_proba_batch(features)   # (n, 3)

    # ── 2. R² 분류기 출력 ──────────────────────────────────────
    r2_clf = R2Classifier(
        threshold=config.R2_THRESHOLD,
        r2_steepness=config.R2_CLF_STEEPNESS,
        direction_steepness=config.R2_DIRECTION_STEEPNESS,
        slope_col=config.SLOPE_NORM_COL,
    )
    r2_proba = r2_clf.predict_proba_batch(features)     # (n, 3)

    # ── 3. 윈도우 피처 4개 ─────────────────────────────────────
    win_feats = features[['cum_return', 'volatility', 'adx_mean', 'r2_mean']].values

    # ── 4. HMM 사후확률 ────────────────────────────────────────
    # X_hmm은 HMM 학습 시 사용한 5개 피처 (정규화 필요)
    # HMMLabeler 자체는 raw X를 받아 정규화는 호출자 책임
    # 여기서도 같은 RollingStandardScaler 사용
    X_hmm_raw = features[config.HMM_FEATURE_COLS].values
    hmm_scaler = RollingStandardScaler(window=config.ROLLING_SCALER_WINDOW)
    X_hmm_scaled = hmm_scaler.fit_transform(X_hmm_raw)
    # HMM은 NaN 있으면 에러 → cold start 행은 임의로 0 채우고 나중에 마스킹
    nan_mask_hmm = np.isnan(X_hmm_scaled).any(axis=1)
    X_hmm_safe = np.where(np.isnan(X_hmm_scaled), 0.0, X_hmm_scaled)
    hmm_proba = labeler.predict_proba(X_hmm_safe)        # (n, 3) — Bull/Side/Bear 순서
    hmm_proba[nan_mask_hmm] = 1/3                         # cold start 무효화

    # ── 5. 전이행렬 사전확률 ───────────────────────────────────
    trans_proba = transition_predictor.predict_next_batch(hmm_proba)  # (n, 3)

    # ── 6. 가로 결합 ───────────────────────────────────────────
    X = np.hstack([adx_proba, r2_proba, win_feats, hmm_proba, trans_proba])

    feature_names = [
        'adx_p_bull', 'adx_p_side', 'adx_p_bear',
        'r2_p_bull', 'r2_p_side', 'r2_p_bear',
        'cum_return', 'volatility', 'adx_mean', 'r2_mean',
        'hmm_p_bull', 'hmm_p_side', 'hmm_p_bear',
        'trans_p_bull', 'trans_p_side', 'trans_p_bear',
    ]

    # nan_mask: HMM cold start 또는 윈도우 피처 NaN
    nan_mask = np.isnan(X).any(axis=1)
    return X, feature_names, nan_mask, hmm_proba


# ════════════════════════════════════════════════════════════════
#  TimeSeriesSplit 학습 / 평가
# ════════════════════════════════════════════════════════════════

def time_series_cv(X, y, feature_names, n_splits, gap, C, class_weight,
                    is_transition=None):
    """
    TimeSeriesSplit 학습/검증.

    각 fold:
        - train: 처음~중간
        - gap: train과 test 사이 (룩어헤드 방지)
        - test: 다음 청크
    매 fold마다 새 LogisticMetaModel 생성.

    Args:
        is_transition: shape (n,) bool — y가 현재 라벨과 다른지 (전환 시점)
            None이면 전환 평가 생략.

    Returns:
        results: list of dict (fold별 결과)
        avg_cm: 평균 혼동행렬 (3, 3)
        last_meta: 마지막 fold의 메타 모델 (계수 시각화용)
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import confusion_matrix, f1_score

    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    cw = None if class_weight == 'none' else class_weight

    results = []
    cm_sum = np.zeros((3, 3), dtype=np.float64)
    last_meta = None

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        meta = LogisticMetaModel(
            C=C,
            class_weight=cw,
            feature_names=feature_names,
        )
        meta.fit(X[train_idx], y[train_idx])
        pred = meta.predict(X[test_idx])
        acc = (pred == y[test_idx]).mean()

        # 클래스별 F1 (macro = 클래스 균등 평균; 불균형 데이터에 더 의미 있음)
        f1_macro = f1_score(y[test_idx], pred, labels=[0, 1, 2],
                             average='macro', zero_division=0)

        # 혼동행렬 (행=실제, 열=예측, 라벨 0/1/2 강제 지정)
        cm = confusion_matrix(y[test_idx], pred, labels=[0, 1, 2])
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_sum += cm_norm

        # 전환 시점에서의 정확도 — 메타 모델의 진짜 가치 평가
        trans_acc = None
        n_trans = 0
        if is_transition is not None:
            test_trans = is_transition[test_idx]
            n_trans = int(test_trans.sum())
            if n_trans > 0:
                trans_acc = (pred[test_trans] == y[test_idx][test_trans]).mean()

        results.append({
            'fold': fold_idx,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_trans': n_trans,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'transition_accuracy': trans_acc,
            'cm_norm': cm_norm,
        })
        last_meta = meta

    avg_cm = cm_sum / n_splits
    return results, avg_cm, last_meta


# ════════════════════════════════════════════════════════════════
#  시각화
# ════════════════════════════════════════════════════════════════

def plot_confusion_matrix(avg_cm, save_path=None, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_cm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    classes = ['Bull', 'Side', 'Bear']
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Average Confusion Matrix (row-normalized)')

    # 각 셀에 숫자 표시
    for i in range(3):
        for j in range(3):
            color = 'white' if avg_cm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{avg_cm[i, j]:.2%}', ha='center', va='center', color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"      → 혼동행렬 저장: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_coefficients(meta, save_path=None, show=True):
    import matplotlib.pyplot as plt

    coef = meta.get_coef_summary()
    feat_cols = [c for c in coef.columns if c != '_intercept']
    coef_only = coef[feat_cols]

    fig, ax = plt.subplots(figsize=(14, 5))
    n_feats = len(feat_cols)
    width = 0.27
    x = np.arange(n_feats)
    colors = {'Bull': '#e74c3c', 'Side': '#95a5a6', 'Bear': '#3498db'}

    for i, cls in enumerate(['Bull', 'Side', 'Bear']):
        ax.bar(x + (i - 1) * width, coef_only.loc[cls].values,
               width, label=cls, color=colors[cls])

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_cols, rotation=45, ha='right')
    ax.set_ylabel('Coefficient (z-scored input scale)')
    ax.set_title('LogisticMetaModel Coefficients per Class')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"      → 계수 그래프 저장: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # 헤드리스: matplotlib Agg 백엔드 강제
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')

    # ── [1] 데이터 + 윈도우 피처 ─────────────────────────────
    print(f"[1/5] 데이터 로드 + 윈도우 피처")
    df = load_and_resample(args.csv_path, timeframe=args.timeframe)
    features = compute_window_features(
        df,
        window_size=args.window_size,
        adx_period=config.ADX_PERIOD,
        r2_period=config.R2_PERIOD,
    )
    print(f"      → 원본 봉: {len(df):,}, 윈도우: {len(features):,}")

    # slope_norm 컬럼 추가 (R²Classifier 입력용)
    slope_scaler = RollingStandardScaler(window=args.rolling_window)
    slope_norm = slope_scaler.fit_transform(
        features[['slope']].values
    ).flatten()
    features[config.SLOPE_NORM_COL] = slope_norm
    cold_start_count = np.isnan(slope_norm).sum()
    print(f"      → slope_norm 추가 (cold start NaN: {cold_start_count:,}행)")

    # ── [2] HMM 라벨러 준비 ──────────────────────────────────
    print(f"[2/5] HMM 라벨러 준비")
    # HMM 학습용 X 미리 준비 (재학습 가능성 대비)
    X_hmm_raw = features[config.HMM_FEATURE_COLS].values
    hmm_scaler_for_fit = RollingStandardScaler(window=args.rolling_window)
    X_hmm_scaled_for_fit = hmm_scaler_for_fit.fit_transform(X_hmm_raw)
    cum_return_full = features['cum_return'].values
    valid_for_hmm = ~np.isnan(X_hmm_scaled_for_fit).any(axis=1)
    labeler = prepare_hmm_labeler(
        args,
        X_hmm_scaled_for_fit[valid_for_hmm],
        cum_return_full[valid_for_hmm],
    )
    transition_predictor = TransitionPredictor.from_labeler(labeler)
    print(f"      → 전이행렬 대각선: {np.diag(transition_predictor.transmat).round(3)}")

    # ── [3-4] 메타 입력 X + 라벨 y ───────────────────────────
    print(f"[3/5] Base Classifier + 전이행렬 사전확률 + HMM 사후확률 계산")
    X_meta, feature_names, nan_mask_meta, hmm_proba_full = build_meta_features(
        features, labeler, transition_predictor,
    )
    print(f"      → X_meta shape: {X_meta.shape}, NaN 있는 행: {nan_mask_meta.sum():,}")

    print(f"[4/5] 라벨 y 구성 (다음 윈도우 HMM 라벨, shift=-1)")
    # 현재 시점 t의 HMM 라벨 — argmax of hmm_proba_full
    hmm_labels_full = np.argmax(hmm_proba_full, axis=1).astype(np.int64)

    # 다음 윈도우 라벨로 시프트: y[t] = label[t+1]
    # shift(-1)는 마지막 행에 NaN(여기선 -1로 표시) 발생
    y_full = np.full(len(hmm_labels_full), -1, dtype=np.int64)
    y_full[:-1] = hmm_labels_full[1:]
    y_nan_mask = (y_full == -1)

    # 학습에서 제외할 행:
    #   (1) X_meta에 NaN 있는 행
    #   (2) y가 없는 행 (마지막 행)
    #   (3) HMM cold start 행 (hmm_proba가 1/3로 채워진 시점 — 정보 없음)
    hmm_cold = np.isclose(hmm_proba_full, 1/3, atol=1e-9).all(axis=1)
    final_mask = ~(nan_mask_meta | y_nan_mask | hmm_cold)
    X = X_meta[final_mask]
    y = y_full[final_mask]
    # 전환 시점 마스크 (다음 라벨이 현재와 다른 시점) — 평가용
    current_label = hmm_labels_full[final_mask]
    is_transition = (y != current_label)
    print(f"      → 최종 학습 가능: X={X.shape}, y={y.shape}")
    print(f"      → 라벨 분포: Bull={np.mean(y==BULL):.1%}, "
          f"Side={np.mean(y==SIDE):.1%}, Bear={np.mean(y==BEAR):.1%}")

    # ── [5] TimeSeriesSplit 학습/검증 ─────────────────────────
    print(f"[5/5] TimeSeriesSplit 학습 ({args.ts_splits} folds, gap={args.window_size})")
    print(f"      C={args.C}, class_weight={args.class_weight}")
    results, avg_cm, last_meta = time_series_cv(
        X, y, feature_names,
        n_splits=args.ts_splits,
        gap=args.window_size,
        C=args.C,
        class_weight=args.class_weight,
        is_transition=is_transition,
    )

    # 결과 출력
    print()
    print(f"      {'Fold':>4}  {'train':>7}  {'test':>7}  {'acc':>7}  "
          f"{'F1(macro)':>10}  {'trans':>6}  {'trans_acc':>10}")
    for r in results:
        ta = f"{r['transition_accuracy']:.3f}" if r['transition_accuracy'] is not None else "-"
        print(f"      {r['fold']:>4}  {r['n_train']:>7,}  "
              f"{r['n_test']:>7,}  {r['accuracy']:>7.3f}  "
              f"{r['f1_macro']:>10.3f}  {r['n_trans']:>6,}  {ta:>10}")
    accs = [r['accuracy'] for r in results]
    f1s = [r['f1_macro'] for r in results]
    trans_accs = [r['transition_accuracy'] for r in results
                   if r['transition_accuracy'] is not None]
    print(f"      {'─' * 70}")
    print(f"      평균 정확도:        {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"      평균 F1 (macro):    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    if trans_accs:
        print(f"      평균 전환시점 정확도: {np.mean(trans_accs):.3f} ± {np.std(trans_accs):.3f}")
        print(f"      → 전환 시점은 다음 라벨이 현재와 다른 시점. 이 정확도가 메타 모델의 진짜 가치.")

    # naive baseline (다수 클래스 항상 예측 시 정확도)
    most_common = np.argmax(np.bincount(y))
    naive_acc = np.mean(y == most_common)
    print(f"      참고 — 다수클래스({REGIME_NAMES[int(most_common)]}) 예측 baseline: {naive_acc:.3f}")
    # persistence baseline (현재 라벨 그대로 예측 시 정확도)
    persistence_acc = np.mean(y == hmm_labels_full[final_mask])
    print(f"      참고 — 현재 라벨 유지(persistence) baseline: {persistence_acc:.3f}")

    # 평균 혼동행렬
    print()
    print(f"[혼동행렬 (행=실제, 열=예측)]")
    print(f"             Bull    Side    Bear")
    for i, name in enumerate(['Bull', 'Side', 'Bear']):
        print(f"        {name}  {avg_cm[i,0]:.3f}  {avg_cm[i,1]:.3f}  {avg_cm[i,2]:.3f}")

    # 계수 텍스트 출력
    print()
    print("[메타 모델 계수 (마지막 fold, StandardScaler 정규화 후 스케일)]")
    coef_df = last_meta.get_coef_summary()
    feat_cols = [c for c in coef_df.columns if c != '_intercept']
    col_w = max(len(c) for c in feat_cols + ['Feature']) + 2
    header = f"  {'Feature':<{col_w}}{'Bull':>10}{'Side':>10}{'Bear':>10}"
    print(header)
    print("  " + "─" * (col_w + 30))
    for feat in feat_cols:
        vals = coef_df[feat]
        print(f"  {feat:<{col_w}}{vals['Bull']:>10.4f}{vals['Side']:>10.4f}{vals['Bear']:>10.4f}")
    print("  " + "─" * (col_w + 30))
    intercepts = coef_df['_intercept']
    print(f"  {'_intercept':<{col_w}}{intercepts['Bull']:>10.4f}{intercepts['Side']:>10.4f}{intercepts['Bear']:>10.4f}")

    # 시각화 (plt.show()는 GUI 창을 열고 블로킹됨 — 창을 닫아야 다음으로 진행)
    print()
    show = not args.no_show
    plot_confusion_matrix(avg_cm, save_path=args.save_cm, show=show)
    plot_coefficients(last_meta, save_path=args.save_coef, show=show)


if __name__ == '__main__':
    main()
