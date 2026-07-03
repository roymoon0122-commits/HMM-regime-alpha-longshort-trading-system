"""
Retrospective Label Smoother 검증 스크립트.

────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일
────────────────────────────────────────────────────────────────────
1. BTC 4h 데이터 + 윈도우 피처 + HMM 라벨 (캐시 사용)
2. 각 윈도우의 마지막 1봉 수익률 계산
3. RetrospectiveLabelSmoother로 라벨 backdate
4. before/after 비교:
   (a) backdate 통계 (몇 건, 평균 shift, 방향별)
   (b) BTC 가격 오버레이 (Original vs Smoothed)
   (c) 메타 모델 5-fold 학습 (Original 라벨 vs Smoothed 라벨)
   (d) 정확도/전환정확도/계수 비교 표

────────────────────────────────────────────────────────────────────
실행 방법
────────────────────────────────────────────────────────────────────
    cd Coin-trader-main

    # 기본 실행
    python -m strategy.HMM_strategy.scripts.verify_label_smoother --no-show \\
        --out-dir outputs/label_smoother

    # 임계값 변경
    python -m strategy.HMM_strategy.scripts.verify_label_smoother \\
        --threshold 0.07 --no-show

    # SIDE 전환도 backdate
    python -m strategy.HMM_strategy.scripts.verify_label_smoother \\
        --include-side --no-show
"""

import argparse
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
from strategy.HMM_strategy.regime.label_smoother import RetrospectiveLabelSmoother
from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier
from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier
from strategy.HMM_strategy.meta_model.logistic_meta_model import LogisticMetaModel


REGIME_COLORS = {
    BULL: '#e74c3c',   # 빨강
    SIDE: '#95a5a6',   # 회색
    BEAR: '#3498db',   # 파랑
}


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Retrospective Label Smoother 검증")
    p.add_argument('--csv-path', default=config.DATA_PATH)
    p.add_argument('--timeframe', default=config.TIMEFRAME)
    p.add_argument('--window-size', type=int, default=config.WINDOW_SIZE)
    p.add_argument('--rolling-window', type=int, default=config.ROLLING_SCALER_WINDOW)
    p.add_argument('--hmm-cache', default=config.HMM_MODEL_PATH)
    # smoother 파라미터
    p.add_argument('--lookback', type=int, default=config.LABEL_SMOOTHER_LOOKBACK)
    p.add_argument('--threshold', type=float, default=config.LABEL_SMOOTHER_THRESHOLD)
    p.add_argument('--persistence', type=int, default=config.LABEL_SMOOTHER_PERSISTENCE)
    p.add_argument('--include-side', action='store_true')
    # 메타 모델
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--ts-splits', type=int, default=config.TS_SPLIT_N)
    # 출력
    p.add_argument('--out-dir', default='outputs/label_smoother')
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  메타 모델 5-fold 평가 헬퍼
# ════════════════════════════════════════════════════════════════

def evaluate_meta_model(X, y, current_label, feature_names, n_splits, gap, C):
    """
    주어진 X, y (라벨)로 5-fold 학습 + 메트릭 반환.

    Returns:
        dict — accuracy, transition_accuracy, persistence, last_meta, avg_cm
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import confusion_matrix

    is_transition = (y != current_label)
    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    accs, trans_accs = [], []
    cm_sum = np.zeros((3, 3), dtype=np.float64)
    last_meta = None

    for train_idx, test_idx in cv.split(X):
        meta = LogisticMetaModel(
            C=C, class_weight='balanced', feature_names=feature_names,
        )
        meta.fit(X[train_idx], y[train_idx])
        pred = meta.predict(X[test_idx])

        accs.append((pred == y[test_idx]).mean())
        cm = confusion_matrix(y[test_idx], pred, labels=[0, 1, 2])
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_sum += cm_norm

        test_trans = is_transition[test_idx]
        if test_trans.sum() > 0:
            trans_accs.append((pred[test_trans] == y[test_idx][test_trans]).mean())
        last_meta = meta

    persistence = float(np.mean(y == current_label))
    return {
        'mean_acc': float(np.mean(accs)),
        'mean_trans_acc': float(np.mean(trans_accs)) if trans_accs else 0.0,
        'persistence': persistence,
        'last_meta': last_meta,
        'avg_cm': cm_sum / n_splits,
    }


# ════════════════════════════════════════════════════════════════
#  시각화
# ════════════════════════════════════════════════════════════════

def plot_overlay_compare(df, features, labels_orig, labels_smooth,
                          save_path, show=True):
    """BTC 가격 + Original vs Smoothed 라벨 2 패널."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    df_times = df['datetime'].values
    df_close = df['close'].values
    times = features['window_end_time'].values

    for ax, labels, title in zip(
        axes,
        [labels_orig, labels_smooth],
        ['Original HMM Viterbi Labels', 'Smoothed Labels (backdate applied)'],
    ):
        ax.plot(df_times, df_close, color='black', linewidth=0.5, alpha=0.85)
        ax.set_yscale('log')
        ax.set_ylabel('BTC (log)')
        ax.set_title(title, fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)

        prev = labels[0]
        seg_start = times[0]
        for i in range(1, len(labels)):
            if labels[i] != prev:
                ax.axvspan(seg_start, times[i],
                           color=REGIME_COLORS[int(prev)], alpha=0.18, zorder=0)
                prev = labels[i]
                seg_start = times[i]
        ax.axvspan(seg_start, times[-1],
                   color=REGIME_COLORS[int(prev)], alpha=0.18, zorder=0)

        legend_handles = [
            Patch(facecolor=REGIME_COLORS[BULL], alpha=0.35, label='Bull'),
            Patch(facecolor=REGIME_COLORS[SIDE], alpha=0.35, label='Side'),
            Patch(facecolor=REGIME_COLORS[BEAR], alpha=0.35, label='Bear'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=0.9)

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


def plot_coefficients_compare(meta_orig, meta_smooth, save_path, show=True):
    """Original vs Smoothed 메타 모델 계수 비교."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, meta, title in zip(
        axes, [meta_orig, meta_smooth],
        ['Coefficients — Original Labels', 'Coefficients — Smoothed Labels'],
    ):
        coef = meta.get_coef_summary()
        feat_cols = [c for c in coef.columns if c != '_intercept']
        coef_only = coef[feat_cols]

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
        ax.set_title(title, fontsize=10, loc='left')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"      → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_compare(cm_orig, cm_smooth, save_path, show=True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cm, title in zip(
        axes, [cm_orig, cm_smooth],
        ['Confusion — Original', 'Confusion — Smoothed'],
    ):
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        classes = ['Bull', 'Side', 'Bear']
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title, fontsize=10)
        for i in range(3):
            for j in range(3):
                color = 'white' if cm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm[i, j]:.2%}',
                        ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"      → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_shift_distribution(change_log, save_path, show=True):
    """backdate된 봉 수의 분포."""
    import matplotlib.pyplot as plt

    if not change_log:
        return
    shifts = np.array([c['shift'] for c in change_log])
    by_bull = [c['shift'] for c in change_log if c['new_label'] == BULL]
    by_bear = [c['shift'] for c in change_log if c['new_label'] == BEAR]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0.5, max(shifts) + 1.5, 1)
    ax.hist([by_bull, by_bear], bins=bins, stacked=True,
            label=['Backdated to Bull', 'Backdated to Bear'],
            color=[REGIME_COLORS[BULL], REGIME_COLORS[BEAR]], alpha=0.85)
    ax.set_xlabel('Shift (bars backdated)')
    ax.set_ylabel('Event count')
    ax.set_title(f'Backdate Shift Distribution (total {len(change_log)} events)')
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
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] 데이터 + 피처 ─────────────────────────────────
    print(f"[1/6] 데이터 로드 + 윈도우 피처")
    df = load_and_resample(args.csv_path, timeframe=args.timeframe)
    features = compute_window_features(
        df, window_size=args.window_size,
        adx_period=config.ADX_PERIOD, r2_period=config.R2_PERIOD,
    )

    # 마지막 1봉 수익률 — 각 윈도우의 window_end_idx 시점의 봉 수익률
    bar_returns = df['close'].pct_change().values
    end_indices = features['window_end_idx'].astype(int).values
    last_bar_returns = bar_returns[end_indices]
    print(f"      → 윈도우: {len(features):,}, last_bar_returns 준비")

    # slope_norm
    slope_scaler = RollingStandardScaler(window=args.rolling_window)
    features = features.copy()
    features['slope_norm'] = slope_scaler.fit_transform(
        features[['slope']].values
    ).flatten()

    # ── [2] HMM 라벨 (캐시 사용) ──────────────────────────
    print(f"[2/6] HMM 라벨러 로드: {args.hmm_cache}")
    labeler = HMMLabeler()
    labeler.load(args.hmm_cache)
    transition_predictor = TransitionPredictor.from_labeler(labeler)

    # HMM 사후확률 계산용 X
    X_hmm_raw = features[config.HMM_FEATURE_COLS].values
    hmm_scaler = RollingStandardScaler(window=args.rolling_window)
    X_hmm_scaled = hmm_scaler.fit_transform(X_hmm_raw)
    nan_mask_hmm = np.isnan(X_hmm_scaled).any(axis=1)
    X_hmm_safe = np.where(np.isnan(X_hmm_scaled), 0.0, X_hmm_scaled)
    hmm_proba = labeler.predict_proba(X_hmm_safe)
    hmm_proba[nan_mask_hmm] = 1/3

    # 원본 HMM 라벨 (Viterbi)
    labels_orig = np.argmax(hmm_proba, axis=1).astype(np.int64)

    # ── [3] Smoother 적용 ─────────────────────────────────
    print(f"[3/6] Retrospective Smoother 적용")
    print(f"      lookback={args.lookback}, threshold={args.threshold:.0%}, "
          f"persistence={args.persistence}, include_side={args.include_side}")
    smoother = RetrospectiveLabelSmoother(
        lookback=args.lookback,
        threshold=args.threshold,
        persistence_check=args.persistence,
        include_side=args.include_side,
    )
    labels_smooth, change_log = smoother.smooth(labels_orig, last_bar_returns)
    summary = smoother.summarize_changes(change_log)
    print(f"      → backdate 이벤트: {summary['n_backdates']}건")
    if summary['n_backdates']:
        print(f"        평균 shift: {summary['mean_shift']:.1f}봉 "
              f"(median {summary.get('median_shift', 0):.1f}, "
              f"max {summary.get('max_shift', 0)}봉)")
        print(f"        평균 |shock|: {summary['mean_shock_abs']:.2%} "
              f"(max {summary['max_shock_abs']:.2%})")
        print(f"        방향별: {summary['by_direction']}")

    # 라벨 분포 비교
    n = len(labels_orig)
    print(f"      라벨 분포 변화 (전체 {n:,}봉):")
    for regime_id in [BULL, SIDE, BEAR]:
        orig_cnt = (labels_orig == regime_id).sum()
        smooth_cnt = (labels_smooth == regime_id).sum()
        diff = smooth_cnt - orig_cnt
        print(f"        {REGIME_NAMES[regime_id]:>4}: {orig_cnt:>5,} → "
              f"{smooth_cnt:>5,} ({diff:+d})")

    # ── [4] 메타 입력 X 구성 ─────────────────────────────
    print(f"[4/6] 메타 입력 X 구성 (16개 피처)")
    adx_proba = ADXClassifier().predict_proba_batch(features)
    r2_proba = R2Classifier().predict_proba_batch(features)
    win_feats = features[['cum_return', 'volatility', 'adx_mean', 'r2_mean']].values
    trans_proba = transition_predictor.predict_next_batch(hmm_proba)

    X_meta = np.hstack([adx_proba, r2_proba, win_feats, hmm_proba, trans_proba])
    feature_names = [
        'adx_p_bull', 'adx_p_side', 'adx_p_bear',
        'r2_p_bull', 'r2_p_side', 'r2_p_bear',
        'cum_return', 'volatility', 'adx_mean', 'r2_mean',
        'hmm_p_bull', 'hmm_p_side', 'hmm_p_bear',
        'trans_p_bull', 'trans_p_side', 'trans_p_bear',
    ]

    # 메타 학습용 마스크 + y 구성
    nan_mask_meta = np.isnan(X_meta).any(axis=1)
    hmm_cold = np.isclose(hmm_proba, 1/3, atol=1e-9).all(axis=1)

    # y_orig = labels_orig[t+1], y_smooth = labels_smooth[t+1]
    y_orig_full = np.full(n, -1, dtype=np.int64)
    y_orig_full[:-1] = labels_orig[1:]
    y_smooth_full = np.full(n, -1, dtype=np.int64)
    y_smooth_full[:-1] = labels_smooth[1:]
    y_nan_mask = (y_orig_full == -1)

    final_mask = ~(nan_mask_meta | y_nan_mask | hmm_cold)
    X = X_meta[final_mask]
    y_orig = y_orig_full[final_mask]
    y_smooth = y_smooth_full[final_mask]
    current_label_orig = labels_orig[final_mask]
    current_label_smooth = labels_smooth[final_mask]
    print(f"      → 학습 가능: X={X.shape}")

    # ── [5] 메타 모델 학습 — Original vs Smoothed ─────────
    print(f"[5/6] 메타 모델 5-fold 학습")
    print(f"      Original 라벨로 학습...")
    res_orig = evaluate_meta_model(
        X, y_orig, current_label_orig, feature_names,
        args.ts_splits, args.window_size, args.C,
    )
    print(f"      Smoothed 라벨로 학습...")
    res_smooth = evaluate_meta_model(
        X, y_smooth, current_label_smooth, feature_names,
        args.ts_splits, args.window_size, args.C,
    )

    # 비교 표
    print()
    print("═" * 65)
    print(f"  메트릭 비교 (Original vs Smoothed)")
    print("═" * 65)
    print(f"  {'Metric':<25}  {'Original':>12}  {'Smoothed':>12}  {'Diff':>9}")
    print(f"  {'-'*60}")
    rows = [
        ('평균 정확도',       res_orig['mean_acc'],       res_smooth['mean_acc']),
        ('전환시점 정확도',   res_orig['mean_trans_acc'], res_smooth['mean_trans_acc']),
        ('Persistence base',  res_orig['persistence'],    res_smooth['persistence']),
    ]
    for name, a, b in rows:
        diff = b - a
        print(f"  {name:<25}  {a:>12.3f}  {b:>12.3f}  {diff:>+9.3f}")
    print("═" * 65)

    # ── [6] 시각화 ────────────────────────────────────────
    print()
    print(f"[6/6] 시각화")
    show = not args.no_show
    plot_overlay_compare(
        df, features, labels_orig, labels_smooth,
        save_path=str(out_dir / "overlay_compare.png"), show=show,
    )
    plot_coefficients_compare(
        res_orig['last_meta'], res_smooth['last_meta'],
        save_path=str(out_dir / "coefficients_compare.png"), show=show,
    )
    plot_confusion_compare(
        res_orig['avg_cm'], res_smooth['avg_cm'],
        save_path=str(out_dir / "confusion_compare.png"), show=show,
    )
    plot_shift_distribution(
        change_log,
        save_path=str(out_dir / "shift_distribution.png"), show=show,
    )

    # 요약 텍스트 저장
    summary_text = (
        f"Label Smoother 검증 요약\n"
        f"{'=' * 50}\n"
        f"파라미터:\n"
        f"  lookback={args.lookback}, threshold={args.threshold:.0%}\n"
        f"  persistence={args.persistence}, include_side={args.include_side}\n\n"
        f"Backdate 통계:\n"
        f"  총 이벤트:    {summary['n_backdates']}\n"
    )
    if summary['n_backdates']:
        summary_text += (
            f"  평균 shift:   {summary['mean_shift']:.2f}봉\n"
            f"  median shift: {summary.get('median_shift', 0):.1f}봉\n"
            f"  max shift:    {summary.get('max_shift', 0)}봉\n"
            f"  평균 |shock|: {summary['mean_shock_abs']:.2%}\n"
            f"  방향별:       {summary['by_direction']}\n"
        )
    summary_text += (
        f"\n메타 모델 비교:\n"
        f"  {'Metric':<22}  {'Original':>10}  {'Smoothed':>10}  {'Diff':>8}\n"
        f"  {'-' * 56}\n"
        f"  {'평균 정확도':<22}  {res_orig['mean_acc']:>10.3f}  "
        f"{res_smooth['mean_acc']:>10.3f}  "
        f"{res_smooth['mean_acc']-res_orig['mean_acc']:>+8.3f}\n"
        f"  {'전환시점 정확도':<22}  {res_orig['mean_trans_acc']:>10.3f}  "
        f"{res_smooth['mean_trans_acc']:>10.3f}  "
        f"{res_smooth['mean_trans_acc']-res_orig['mean_trans_acc']:>+8.3f}\n"
        f"  {'Persistence base':<22}  {res_orig['persistence']:>10.3f}  "
        f"{res_smooth['persistence']:>10.3f}  "
        f"{res_smooth['persistence']-res_orig['persistence']:>+8.3f}\n"
    )
    (out_dir / "summary.txt").write_text(summary_text, encoding='utf-8')
    print(f"      → {out_dir / 'summary.txt'}")
    print()
    print(f"✅ 모든 결과: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
