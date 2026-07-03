"""
Window Size Sweep — 윈도우 크기 변화에 따른 시스템 동작 비교.

────────────────────────────────────────────────────────────────────
실행 방법
────────────────────────────────────────────────────────────────────
    cd Coin-trader-main

    # 기본: 60/30/15봉 비교, PNG 저장
    python -m strategy.HMM_strategy.scripts.sweep_window_size \\
        --out-dir outputs/window_sweep --no-show

    # 특정 윈도우만
    python -m strategy.HMM_strategy.scripts.sweep_window_size \\
        --windows 60 30 --no-show

    # restart 적게 (빠른 테스트)
    python -m strategy.HMM_strategy.scripts.sweep_window_size \\
        --n-restart 5 --no-show

────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일 (각 윈도우마다)
────────────────────────────────────────────────────────────────────
1. 윈도우 피처 계산 (window_size 변경)
2. HMM 학습 (rolling 모드)
3. 전이 예측기 + base classifier 출력
4. 메타 모델 5-fold TimeSeriesSplit 학습
5. 시각화 4종 PNG 생성:
   (a) HMM 라벨 + 메타 예측 → BTC 가격 오버레이 (2 패널)
   (b) 혼동행렬
   (c) 계수 막대그래프
6. 콘솔에 비교 표 출력 (정확도, 전환정확도, 전이행렬 대각선 등)

────────────────────────────────────────────────────────────────────
출력 파일 (예: --out-dir outputs/window_sweep)
────────────────────────────────────────────────────────────────────
    outputs/window_sweep/
    ├── win60_overlay.png       # HMM + 메타 예측 오버레이
    ├── win60_confusion.png
    ├── win60_coefficients.png
    ├── win30_overlay.png
    ├── win30_confusion.png
    ├── win30_coefficients.png
    ├── win15_overlay.png
    ├── win15_confusion.png
    ├── win15_coefficients.png
    └── summary.txt              # 비교 표 (콘솔 출력 사본)
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
from strategy.HMM_strategy.regime.hmm_labeler import (
    HMMLabeler, BULL, SIDE, BEAR, REGIME_NAMES,
)
from strategy.HMM_strategy.regime.transition import TransitionPredictor
from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier
from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier
from strategy.HMM_strategy.meta_model.logistic_meta_model import LogisticMetaModel


# 색상 (verify_hmm_labels.py와 동일 컨벤션)
REGIME_COLORS = {
    BULL: '#e74c3c',   # 빨강
    SIDE: '#95a5a6',   # 회색
    BEAR: '#3498db',   # 파랑
}


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Window Size Sweep — 60/30/15봉 비교")
    p.add_argument('--csv-path', default=config.DATA_PATH)
    p.add_argument('--timeframe', default=config.TIMEFRAME)
    p.add_argument('--windows', type=int, nargs='+', default=[60, 30, 15],
                   help="비교할 윈도우 크기 리스트 (기본 60 30 15)")
    p.add_argument('--rolling-window', type=int, default=config.ROLLING_SCALER_WINDOW)
    p.add_argument('--n-restart', type=int, default=config.HMM_RANDOM_RESTART)
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--ts-splits', type=int, default=config.TS_SPLIT_N)
    p.add_argument('--out-dir', default='outputs/window_sweep')
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  핵심 — 단일 윈도우 크기에 대해 전체 파이프라인 실행
# ════════════════════════════════════════════════════════════════

def run_pipeline(df, window_size, args):
    """
    단일 윈도우 크기로 전체 파이프라인 실행.

    Returns:
        dict — {
            'window_size', 'features', 'hmm_labels', 'meta_pred', 'mask',
            'fold_results', 'avg_cm', 'meta_model',
            'transmat_diag', 'label_dist'
        }
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import confusion_matrix

    print(f"\n{'═' * 60}")
    print(f"  윈도우 크기 = {window_size}봉 ({window_size * 4}h = "
          f"{window_size * 4 / 24:.1f}일)")
    print(f"{'═' * 60}")

    # ── 1. 윈도우 피처 ─────────────────────────────────────
    t0 = time.time()
    features = compute_window_features(
        df, window_size=window_size,
        adx_period=config.ADX_PERIOD, r2_period=config.R2_PERIOD,
    )
    print(f"  [1/5] 윈도우 피처 계산: {len(features):,}개 ({time.time()-t0:.1f}s)")

    # slope_norm 추가
    slope_scaler = RollingStandardScaler(window=args.rolling_window)
    features = features.copy()
    features['slope_norm'] = slope_scaler.fit_transform(
        features[['slope']].values
    ).flatten()

    # ── 2. HMM 학습 ────────────────────────────────────────
    t0 = time.time()
    X_hmm_raw = features[config.HMM_FEATURE_COLS].values
    hmm_scaler = RollingStandardScaler(window=args.rolling_window)
    X_hmm_scaled = hmm_scaler.fit_transform(X_hmm_raw)
    cum_return_full = features['cum_return'].values
    valid_for_hmm = ~np.isnan(X_hmm_scaled).any(axis=1)

    labeler = HMMLabeler(
        n_states=config.N_STATES,
        n_iter=config.HMM_N_ITER,
        n_random_restart=args.n_restart,
        covariance_type=config.HMM_COVARIANCE_TYPE,
        random_state=42,
    )
    labeler.fit(X_hmm_scaled[valid_for_hmm], cum_return_full[valid_for_hmm])
    print(f"  [2/5] HMM 학습 (restart={args.n_restart}): {time.time()-t0:.1f}s, "
          f"log-lik={labeler.best_score_:,.0f}")

    transition_predictor = TransitionPredictor.from_labeler(labeler)
    transmat_diag = np.diag(transition_predictor.transmat).copy()
    print(f"        전이행렬 대각선: {transmat_diag.round(3)}")

    # ── 3. Base Classifier + HMM 사후확률 + 전이 사전확률 ────
    t0 = time.time()
    adx_proba = ADXClassifier().predict_proba_batch(features)
    r2_proba = R2Classifier().predict_proba_batch(features)
    win_feats = features[['cum_return', 'volatility', 'adx_mean', 'r2_mean']].values

    nan_mask_hmm = np.isnan(X_hmm_scaled).any(axis=1)
    X_hmm_safe = np.where(np.isnan(X_hmm_scaled), 0.0, X_hmm_scaled)
    hmm_proba = labeler.predict_proba(X_hmm_safe)
    hmm_proba[nan_mask_hmm] = 1/3
    trans_proba = transition_predictor.predict_next_batch(hmm_proba)

    X_meta = np.hstack([adx_proba, r2_proba, win_feats, hmm_proba, trans_proba])
    feature_names = [
        'adx_p_bull', 'adx_p_side', 'adx_p_bear',
        'r2_p_bull', 'r2_p_side', 'r2_p_bear',
        'cum_return', 'volatility', 'adx_mean', 'r2_mean',
        'hmm_p_bull', 'hmm_p_side', 'hmm_p_bear',
        'trans_p_bull', 'trans_p_side', 'trans_p_bear',
    ]
    print(f"  [3/5] 피처 결합: X_meta shape {X_meta.shape} ({time.time()-t0:.1f}s)")

    # 라벨 + 마스크
    hmm_labels_full = np.argmax(hmm_proba, axis=1).astype(np.int64)
    y_full = np.full(len(hmm_labels_full), -1, dtype=np.int64)
    y_full[:-1] = hmm_labels_full[1:]

    nan_mask_meta = np.isnan(X_meta).any(axis=1)
    hmm_cold = np.isclose(hmm_proba, 1/3, atol=1e-9).all(axis=1)
    final_mask = ~(nan_mask_meta | (y_full == -1) | hmm_cold)
    X = X_meta[final_mask]
    y = y_full[final_mask]
    current_label = hmm_labels_full[final_mask]
    is_transition = (y != current_label)

    label_dist = {
        'bull': float(np.mean(hmm_labels_full[final_mask] == BULL)),
        'side': float(np.mean(hmm_labels_full[final_mask] == SIDE)),
        'bear': float(np.mean(hmm_labels_full[final_mask] == BEAR)),
    }

    # ── 4. 메타 모델 5-fold ─────────────────────────────────
    t0 = time.time()
    cv = TimeSeriesSplit(n_splits=args.ts_splits, gap=window_size)
    fold_results = []
    cm_sum = np.zeros((3, 3), dtype=np.float64)
    last_meta = None
    last_pred = None
    last_test_idx = None

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        meta = LogisticMetaModel(
            C=args.C, class_weight='balanced',
            feature_names=feature_names,
        )
        meta.fit(X[train_idx], y[train_idx])
        pred = meta.predict(X[test_idx])

        acc = (pred == y[test_idx]).mean()
        cm = confusion_matrix(y[test_idx], pred, labels=[0, 1, 2])
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_sum += cm_norm

        test_trans = is_transition[test_idx]
        n_trans = int(test_trans.sum())
        trans_acc = (pred[test_trans] == y[test_idx][test_trans]).mean() if n_trans > 0 else None

        fold_results.append({
            'fold': fold_idx, 'n_train': len(train_idx), 'n_test': len(test_idx),
            'accuracy': acc, 'n_trans': n_trans, 'transition_accuracy': trans_acc,
        })
        last_meta = meta
        last_pred = pred
        last_test_idx = test_idx

    avg_cm = cm_sum / args.ts_splits
    print(f"  [4/5] 메타 모델 5-fold ({time.time()-t0:.1f}s)")

    # 메타 예측을 features 순서에 맞춰 풀 길이로 복원 (시각화용)
    # 학습 데이터에 포함되지 않은 시점은 -1 (회색으로 처리하거나 HMM 라벨 사용)
    full_meta_pred = np.full(len(features), -1, dtype=np.int64)
    # final_mask가 True인 행들의 features 인덱스
    feature_idx_in_X = np.where(final_mask)[0]
    # 마지막 fold의 test 부분 메타 예측
    full_meta_pred[feature_idx_in_X[last_test_idx]] = last_pred

    # ── 5. 결과 정리 ────────────────────────────────────────
    accs = [r['accuracy'] for r in fold_results]
    trans_accs = [r['transition_accuracy'] for r in fold_results
                   if r['transition_accuracy'] is not None]
    persistence = float(np.mean(y == current_label))

    print(f"  [5/5] 결과:")
    print(f"        평균 정확도:        {np.mean(accs):.3f}")
    print(f"        평균 전환시점 정확도: {np.mean(trans_accs):.3f}")
    print(f"        Persistence baseline: {persistence:.3f}")
    print(f"        라벨 분포: Bull={label_dist['bull']:.1%}, "
          f"Side={label_dist['side']:.1%}, Bear={label_dist['bear']:.1%}")

    return {
        'window_size': window_size,
        'features': features,
        'hmm_labels': hmm_labels_full,
        'meta_pred_full': full_meta_pred,
        'final_mask': final_mask,
        'fold_results': fold_results,
        'avg_cm': avg_cm,
        'meta_model': last_meta,
        'transmat_diag': transmat_diag,
        'label_dist': label_dist,
        'persistence': persistence,
        'mean_acc': float(np.mean(accs)),
        'mean_trans_acc': float(np.mean(trans_accs)),
    }


# ════════════════════════════════════════════════════════════════
#  시각화 1 — BTC 가격 + 국면 오버레이 (HMM + 메타 2 패널)
# ════════════════════════════════════════════════════════════════

def plot_overlay(df, result, save_path, show=True):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    features = result['features']
    hmm_labels = result['hmm_labels']
    meta_pred = result['meta_pred_full']

    df_times = df['datetime'].values
    df_close = df['close'].values
    times = features['window_end_time'].values

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    titles = [
        f'HMM Viterbi 라벨 (현재 시점 국면)',
        f'메타 모델 예측 (다음 윈도우 국면, 마지막 fold test 구간)',
    ]
    label_arrays = [hmm_labels, meta_pred]

    for ax, labels, title in zip(axes, label_arrays, titles):
        # BTC 가격
        ax.plot(df_times, df_close, color='black', linewidth=0.5, alpha=0.85)
        ax.set_yscale('log')
        ax.set_ylabel('BTC Price (log)')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=10, loc='left')

        # 국면 색칠
        valid = labels >= 0   # -1은 데이터 없음 (메타 예측에서 발생)
        if valid.sum() == 0:
            continue
        valid_idx = np.where(valid)[0]
        prev_label = labels[valid_idx[0]]
        seg_start_t = times[valid_idx[0]]
        prev_t = seg_start_t

        for i in valid_idx[1:]:
            cur_label = labels[i]
            cur_t = times[i]
            if cur_label != prev_label:
                ax.axvspan(seg_start_t, prev_t,
                           color=REGIME_COLORS[int(prev_label)],
                           alpha=0.18, zorder=0)
                prev_label = cur_label
                seg_start_t = cur_t
            prev_t = cur_t
        ax.axvspan(seg_start_t, prev_t,
                   color=REGIME_COLORS[int(prev_label)], alpha=0.18, zorder=0)

        legend_handles = [
            Patch(facecolor=REGIME_COLORS[BULL], alpha=0.35, label='Bull'),
            Patch(facecolor=REGIME_COLORS[SIDE], alpha=0.35, label='Side'),
            Patch(facecolor=REGIME_COLORS[BEAR], alpha=0.35, label='Bear'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"Window Size = {result['window_size']}봉 "
        f"({result['window_size'] * 4 / 24:.1f}일) | "
        f"Acc={result['mean_acc']:.3f}, "
        f"TransAcc={result['mean_trans_acc']:.3f}, "
        f"Persistence={result['persistence']:.3f}",
        fontsize=11, fontweight='bold',
    )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"      → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  시각화 2 — 혼동행렬
# ════════════════════════════════════════════════════════════════

def plot_confusion_matrix(result, save_path, show=True):
    import matplotlib.pyplot as plt

    avg_cm = result['avg_cm']
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(avg_cm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    classes = ['Bull', 'Side', 'Bear']
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f"Confusion Matrix (window={result['window_size']}, "
                 f"Acc={result['mean_acc']:.3f})")

    for i in range(3):
        for j in range(3):
            color = 'white' if avg_cm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{avg_cm[i, j]:.2%}',
                    ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"      → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  시각화 3 — 계수 막대그래프
# ════════════════════════════════════════════════════════════════

def plot_coefficients(result, save_path, show=True):
    import matplotlib.pyplot as plt

    meta = result['meta_model']
    coef = meta.get_coef_summary()
    feat_cols = [c for c in coef.columns if c != '_intercept']
    coef_only = coef[feat_cols]

    fig, ax = plt.subplots(figsize=(13, 4.5))
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
    ax.set_ylabel('Coefficient (z-scored input)')
    ax.set_title(f"Logistic Meta Coefficients (window={result['window_size']})")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"      → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  요약 표
# ════════════════════════════════════════════════════════════════

def print_summary(results, save_path=None):
    """비교 표 출력 + 파일 저장."""
    lines = []
    lines.append("=" * 90)
    lines.append("  윈도우 크기 비교 요약")
    lines.append("=" * 90)
    header = (f"  {'window':>7}  {'days':>6}  "
              f"{'acc':>6}  {'trans_acc':>10}  {'persist':>8}  "
              f"{'tm_diag':>22}  "
              f"{'Bull%':>6}  {'Side%':>6}  {'Bear%':>6}")
    lines.append(header)
    lines.append("  " + "-" * 88)
    for r in results:
        diag_str = " ".join([f"{d:.3f}" for d in r['transmat_diag']])
        lines.append(
            f"  {r['window_size']:>7}  "
            f"{r['window_size']*4/24:>6.1f}  "
            f"{r['mean_acc']:>6.3f}  "
            f"{r['mean_trans_acc']:>10.3f}  "
            f"{r['persistence']:>8.3f}  "
            f"[{diag_str}]  "
            f"{r['label_dist']['bull']:>6.1%}  "
            f"{r['label_dist']['side']:>6.1%}  "
            f"{r['label_dist']['bear']:>6.1%}"
        )
    lines.append("=" * 90)
    text = "\n".join(lines)
    print()
    print(text)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(text + "\n", encoding='utf-8')
        print(f"\n📁 요약 저장: {save_path}")


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 한 번만 로드
    print(f"데이터 로드: {args.csv_path}")
    t0 = time.time()
    df = load_and_resample(args.csv_path, timeframe=args.timeframe)
    print(f"  → {len(df):,}봉, {time.time()-t0:.1f}s")

    # 각 윈도우 크기로 실행
    all_results = []
    for ws in args.windows:
        result = run_pipeline(df, ws, args)
        all_results.append(result)

        # 시각화 저장
        prefix = f"win{ws}"
        plot_overlay(df, result, save_path=str(out_dir / f"{prefix}_overlay.png"),
                      show=not args.no_show)
        plot_confusion_matrix(result, save_path=str(out_dir / f"{prefix}_confusion.png"),
                              show=not args.no_show)
        plot_coefficients(result, save_path=str(out_dir / f"{prefix}_coefficients.png"),
                          show=not args.no_show)

    # 비교 요약
    print_summary(all_results, save_path=str(out_dir / "summary.txt"))

    print(f"\n✅ 모든 결과: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
