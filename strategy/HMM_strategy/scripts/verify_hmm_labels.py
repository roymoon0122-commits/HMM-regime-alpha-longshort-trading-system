"""
HMM 학습 결과 검증 스크립트 — 3가지 정규화 모드 비교.

────────────────────────────────────────────────────────────────────
실행 방법
────────────────────────────────────────────────────────────────────
    cd Coin-trader-main
    python -m strategy.HMM_strategy.scripts.verify_hmm_labels

빠른 검증 (restart 적게):
    python -m strategy.HMM_strategy.scripts.verify_hmm_labels --n-restart 5

특정 모드만:
    python -m strategy.HMM_strategy.scripts.verify_hmm_labels --modes rolling

PNG 저장 + 화면 안 띄우기:
    python -m strategy.HMM_strategy.scripts.verify_hmm_labels \\
        --save-overlay regimes.png --save-stats stats.png --no-show

────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일
────────────────────────────────────────────────────────────────────
1. BTC 4h 데이터 → 윈도우 피처 (Phase 1 모듈 활용)
2. 세 가지 모드로 X 준비:
   - 'none'    : 정규화 없음 (raw)
   - 'global'  : sklearn StandardScaler (전체 평균/표준편차)
   - 'rolling' : RollingStandardScaler (과거 1년 평균/표준편차)
3. 모든 모드를 같은 데이터 길이(rolling cold start 이후)로 학습 — 공정 비교
4. 각 모드별 HMMLabeler 학습 + 결과 비교

────────────────────────────────────────────────────────────────────
사용자가 봐야 할 점
────────────────────────────────────────────────────────────────────
- 가격 차트 위에 색칠된 국면이 직관적으로 말이 되는가?
  (Bull = 상승장 = 빨강 / Bear = 하락장 = 파랑 / Side = 횡보 = 회색)
- 세 모드 간 라벨이 얼마나 다른가? (전이 행렬, 일치율 비교)
- 어느 모드가 "Bear 시장 진입"을 더 빨리 포착하는가?
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler

from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.resampler import load_and_resample
from strategy.HMM_strategy.features.window_features import compute_window_features
from strategy.HMM_strategy.features.scaling import RollingStandardScaler
from strategy.HMM_strategy.regime.hmm_labeler import (
    HMMLabeler, BULL, SIDE, BEAR, REGIME_NAMES,
)


# ────────────────────────────────────────────────────────────────
#  국면별 색상 (가격 차트 오버레이용)
# ────────────────────────────────────────────────────────────────
REGIME_COLORS = {
    BULL: '#e74c3c',   # 빨강 (상승)
    SIDE: '#95a5a6',   # 회색 (횡보)
    BEAR: '#3498db',   # 파랑 (하락)
}


# ════════════════════════════════════════════════════════════════
#  CLI 인자
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HMM 라벨링 검증 — 3가지 정규화 모드 비교"
    )
    p.add_argument('--csv-path', default=config.DATA_PATH,
                   help=f"1분봉 CSV 경로 (기본: {config.DATA_PATH})")
    p.add_argument('--timeframe', default=config.TIMEFRAME,
                   help=f"리샘플링 타임프레임 (기본: {config.TIMEFRAME})")
    p.add_argument('--window-size', type=int, default=config.WINDOW_SIZE,
                   help=f"윈도우 크기 (기본: {config.WINDOW_SIZE})")
    p.add_argument('--adx-period', type=int, default=config.ADX_PERIOD,
                   help=f"ADX 기간 (기본: {config.ADX_PERIOD})")
    p.add_argument('--r2-period', type=int, default=config.R2_PERIOD,
                   help=f"R² 기간 (기본: {config.R2_PERIOD})")

    # HMM 관련
    p.add_argument('--n-states', type=int, default=config.N_STATES,
                   help=f"HMM 상태 수 (기본: {config.N_STATES})")
    p.add_argument('--n-restart', type=int, default=config.HMM_RANDOM_RESTART,
                   help=f"Random Restart 횟수 (기본: {config.HMM_RANDOM_RESTART})")
    p.add_argument('--n-iter', type=int, default=config.HMM_N_ITER,
                   help=f"Baum-Welch 최대 반복 (기본: {config.HMM_N_ITER})")
    p.add_argument('--rolling-window', type=int, default=config.ROLLING_SCALER_WINDOW,
                   help=f"Rolling scaler 윈도우 (기본: {config.ROLLING_SCALER_WINDOW})")

    # 모드 선택
    p.add_argument('--modes', nargs='+',
                   default=['none', 'global', 'rolling'],
                   choices=['none', 'global', 'rolling'],
                   help="비교할 모드 (기본: 모두)")

    # 출력
    p.add_argument('--save-overlay', default=None,
                   help="가격+국면 오버레이 PNG 저장 경로")
    p.add_argument('--save-stats', default=None,
                   help="통계 그래프 PNG 저장 경로")
    p.add_argument('--no-show', action='store_true',
                   help="그래프 창 띄우지 않기")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  데이터 준비
# ════════════════════════════════════════════════════════════════

def prepare_features(args) -> pd.DataFrame:
    """원본 데이터 → 윈도우 피처 DataFrame."""
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ CSV 파일을 찾을 수 없음: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📥 데이터 로드 + 리샘플링 ({args.timeframe})...")
    df = load_and_resample(
        csv_path=str(csv_path),
        timeframe=args.timeframe,
    )
    print(f"   → {len(df):,}봉 ({df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]})")

    print(f"🧮 윈도우 피처 계산 (window_size={args.window_size})...")
    features = compute_window_features(
        df,
        window_size=args.window_size,
        adx_period=args.adx_period,
        r2_period=args.r2_period,
    )
    print(f"   → {len(features):,}개 윈도우 생성")
    return df, features


def prepare_X_for_modes(features: pd.DataFrame, args) -> dict:
    """
    각 모드별로 (X, cum_return, valid_idx) 준비.

    공정 비교를 위해 모든 모드가 rolling cold start 이후의 동일한 행 범위를 사용.

    Returns:
        dict: {mode_name: {'X': np.ndarray, 'cum_return': np.ndarray, 'valid_idx': np.ndarray}}
        valid_idx는 features DataFrame에서 사용된 행의 정수 인덱스 (시각화용)
    """
    feat_cols = config.HMM_FEATURE_COLS
    X_raw_full = features[feat_cols].values            # (n_full, 5)
    cum_full = features['cum_return'].values          # (n_full,)

    cold_start = args.rolling_window - 1               # rolling이 NaN 반환하는 행 수
    n_full = len(features)

    result = {}
    valid_range = slice(cold_start, n_full)            # 모든 모드 공통 학습 범위
    valid_idx = np.arange(cold_start, n_full)

    print(f"\n📐 학습 데이터 범위: features[{cold_start}:{n_full}] = "
          f"{n_full - cold_start:,}행 (모든 모드 공통)")

    for mode in args.modes:
        if mode == 'none':
            X = X_raw_full[valid_range].copy()
        elif mode == 'global':
            # 전체 학습 범위로 fit (cold start 제외 후의 데이터)
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw_full[valid_range])
        elif mode == 'rolling':
            # 전체 데이터에 rolling 적용 후 cold start 제거
            scaler = RollingStandardScaler(window=args.rolling_window)
            X_full_scaled = scaler.fit_transform(features[feat_cols])
            X = X_full_scaled[valid_range]
            assert not np.isnan(X).any(), "rolling scaler 결과에 NaN 남음 (cold_start 계산 오류)"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        result[mode] = {
            'X': X,
            'cum_return': cum_full[valid_range].copy(),
            'valid_idx': valid_idx,
        }
    return result


# ════════════════════════════════════════════════════════════════
#  학습
# ════════════════════════════════════════════════════════════════

def train_all_modes(mode_data: dict, args) -> dict:
    """각 모드별로 HMMLabeler 학습."""
    results = {}
    for mode, d in mode_data.items():
        print(f"\n🎓 [{mode}] 모드 학습 시작 "
              f"(n_states={args.n_states}, restart={args.n_restart}, n_iter={args.n_iter})...")
        t0 = time.time()
        labeler = HMMLabeler(
            n_states=args.n_states,
            n_iter=args.n_iter,
            n_random_restart=args.n_restart,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            random_state=42,
        )
        labeler.fit(d['X'], d['cum_return'])
        elapsed = time.time() - t0

        labels = labeler.predict(d['X'])
        results[mode] = {
            'labeler': labeler,
            'labels': labels,
            'X': d['X'],
            'cum_return': d['cum_return'],
            'valid_idx': d['valid_idx'],
            'elapsed': elapsed,
        }
        print_mode_summary(mode, labeler, labels, elapsed)
    return results


def print_mode_summary(mode: str, labeler: HMMLabeler, labels: np.ndarray, elapsed: float):
    """모드별 학습 결과 콘솔 출력."""
    n_total = len(labeler.fit_history_)
    n_converged = sum(1 for h in labeler.fit_history_ if h['converged'])

    print(f"   ✅ 학습 완료 ({elapsed:.1f}초)")
    print(f"      수렴: {n_converged}/{n_total} restarts")
    print(f"      best log-likelihood: {labeler.best_score_:.2f}")

    # 국면 분포
    counts = np.bincount(labels, minlength=3)
    total = len(labels)
    print(f"      국면 분포:")
    for r in [BULL, SIDE, BEAR]:
        pct = counts[r] / total * 100
        print(f"         {REGIME_NAMES[r]:>4s}: {counts[r]:>6,} ({pct:>5.1f}%)")

    # 전이 행렬 (학습된 모델 기준, raw state ID 순서가 아니라 regime ID 순서로 재배열)
    transmat = labeler.model_.transmat_  # (n_states, n_states), 인덱스=raw_state_id
    # regime ID 순서로 재배열: [Bull, Side, Bear]
    regime_order = [BULL, SIDE, BEAR]
    raw_order = [labeler.regime_to_state_[r] for r in regime_order]
    transmat_reordered = transmat[np.ix_(raw_order, raw_order)]

    print(f"      전이 행렬 (Bull/Side/Bear 순):")
    print(f"             →Bull   →Side   →Bear")
    for i, r_from in enumerate(regime_order):
        row = "         " + REGIME_NAMES[r_from] + "  "
        for j in range(3):
            row += f"{transmat_reordered[i, j]:>6.3f}  "
        print(row)


# ════════════════════════════════════════════════════════════════
#  시각화 1: 가격 + 국면 오버레이 (모드별)
# ════════════════════════════════════════════════════════════════

def plot_regime_overlay(df: pd.DataFrame, features: pd.DataFrame, results: dict, args):
    """
    상단부터 모드별로 한 패널씩.
    각 패널: BTC 종가 (로그축) + 국면 색칠.
    """
    n_modes = len(results)
    fig, axes = plt.subplots(
        n_modes, 1,
        figsize=(15, 3.5 * n_modes),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    df_times = df['datetime'].values
    df_close = df['close'].values

    for ax, (mode, r) in zip(axes, results.items()):
        labels = r['labels']
        valid_idx = r['valid_idx']
        # 각 윈도우의 시간 = features의 window_end_time
        times = features['window_end_time'].iloc[valid_idx].values

        # 가격 라인 (전체)
        ax.plot(df_times, df_close, color='black', linewidth=0.5, alpha=0.85)
        ax.set_yscale('log')
        ax.set_ylabel(f'BTC ({mode})')
        ax.grid(True, alpha=0.3)

        # 국면별 색칠 — 같은 라벨이 연속된 구간을 axvspan으로 표시
        ymin, ymax = ax.get_ylim()
        prev_label = labels[0]
        seg_start_t = times[0]
        for i in range(1, len(labels)):
            if labels[i] != prev_label:
                ax.axvspan(seg_start_t, times[i],
                           color=REGIME_COLORS[prev_label], alpha=0.18, zorder=0)
                prev_label = labels[i]
                seg_start_t = times[i]
        # 마지막 세그먼트
        ax.axvspan(seg_start_t, times[-1],
                   color=REGIME_COLORS[prev_label], alpha=0.18, zorder=0)

        # 범례
        legend_handles = [
            Patch(facecolor=REGIME_COLORS[BULL], alpha=0.35, label='Bull'),
            Patch(facecolor=REGIME_COLORS[SIDE], alpha=0.35, label='Side'),
            Patch(facecolor=REGIME_COLORS[BEAR], alpha=0.35, label='Bear'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=0.9)

    axes[0].set_title(
        f"HMM Regime Labels — n_states={args.n_states}, "
        f"covariance='{config.HMM_COVARIANCE_TYPE}', "
        f"restart={args.n_restart}, rolling_window={args.rolling_window}"
    )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    if args.save_overlay:
        plt.savefig(args.save_overlay, dpi=120, bbox_inches='tight')
        print(f"\n📁 오버레이 저장: {args.save_overlay}")
    if not args.no_show:
        plt.show()


# ════════════════════════════════════════════════════════════════
#  시각화 2: 통계 종합 (boxplot + 모드 간 일치율)
# ════════════════════════════════════════════════════════════════

def plot_regime_stats(results: dict, args):
    """
    좌: 모드별 국면 cum_return 분포 (boxplot, 가로로 배치)
    우상: 모드별 국면 빈도 (stacked bar)
    우하: 모드 간 라벨 일치율 (히트맵)
    """
    n_modes = len(results)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # ── 좌: boxplot ───────────────────────────────────────────
    ax_box = fig.add_subplot(gs[:, 0])
    box_data = []
    box_labels = []
    box_colors = []
    for mode, r in results.items():
        labels = r['labels']
        cr = r['cum_return']
        for regime in [BULL, SIDE, BEAR]:
            mask = labels == regime
            if mask.sum() > 0:
                box_data.append(cr[mask])
                box_labels.append(f"{mode}\n{REGIME_NAMES[regime]}")
                box_colors.append(REGIME_COLORS[regime])

    # matplotlib 3.9+ uses 'tick_labels'; older versions use 'labels'.
    try:
        bp = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False)
    except TypeError:
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax_box.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_box.set_ylabel('cum_return')
    ax_box.set_title('국면별 cum_return 분포 (모드 비교)')
    ax_box.grid(True, axis='y', alpha=0.3)

    # ── 우상: 빈도 막대 ───────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 1])
    modes_list = list(results.keys())
    bull_pct = []
    side_pct = []
    bear_pct = []
    for m in modes_list:
        labels = results[m]['labels']
        total = len(labels)
        bull_pct.append((labels == BULL).sum() / total * 100)
        side_pct.append((labels == SIDE).sum() / total * 100)
        bear_pct.append((labels == BEAR).sum() / total * 100)

    x = np.arange(len(modes_list))
    ax_bar.bar(x, bull_pct, color=REGIME_COLORS[BULL], alpha=0.7, label='Bull')
    ax_bar.bar(x, side_pct, bottom=bull_pct, color=REGIME_COLORS[SIDE], alpha=0.7, label='Side')
    ax_bar.bar(x, bear_pct, bottom=np.array(bull_pct) + np.array(side_pct),
               color=REGIME_COLORS[BEAR], alpha=0.7, label='Bear')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(modes_list)
    ax_bar.set_ylabel('비율 (%)')
    ax_bar.set_title('국면 빈도')
    ax_bar.legend(fontsize=8, loc='center right')

    # ── 우하: 모드 간 일치율 히트맵 ─────────────────────────
    ax_agree = fig.add_subplot(gs[1, 1])
    if n_modes >= 2:
        agree_matrix = np.zeros((n_modes, n_modes))
        for i, m1 in enumerate(modes_list):
            for j, m2 in enumerate(modes_list):
                agree_matrix[i, j] = (results[m1]['labels'] == results[m2]['labels']).mean()
        im = ax_agree.imshow(agree_matrix, cmap='Greens', vmin=0, vmax=1, aspect='equal')
        ax_agree.set_xticks(np.arange(n_modes))
        ax_agree.set_yticks(np.arange(n_modes))
        ax_agree.set_xticklabels(modes_list)
        ax_agree.set_yticklabels(modes_list)
        for i in range(n_modes):
            for j in range(n_modes):
                color = 'white' if agree_matrix[i, j] > 0.6 else 'black'
                ax_agree.text(j, i, f"{agree_matrix[i, j]:.2f}",
                              ha='center', va='center', color=color, fontsize=10)
        ax_agree.set_title('모드 간 라벨 일치율')
        fig.colorbar(im, ax=ax_agree, fraction=0.046, pad=0.04)
    else:
        ax_agree.text(0.5, 0.5, '모드 1개 → 비교 생략',
                      ha='center', va='center', transform=ax_agree.transAxes)
        ax_agree.set_axis_off()

    fig.suptitle(
        f"HMM 라벨링 통계 — n_states={args.n_states}, "
        f"학습 행 수={len(next(iter(results.values()))['labels']):,}",
        fontsize=12,
    )
    if args.save_stats:
        plt.savefig(args.save_stats, dpi=120, bbox_inches='tight')
        print(f"📁 통계 저장: {args.save_stats}")
    if not args.no_show:
        plt.show()


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # --no-show일 때는 GUI 백엔드 초기화를 피하기 위해 Agg 강제 (헤드리스 환경 안전)
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg', force=True)

    # 1. 데이터 + 피처
    df, features = prepare_features(args)

    # 2. 모드별 X 준비
    mode_data = prepare_X_for_modes(features, args)

    # 3. 학습
    results = train_all_modes(mode_data, args)

    # 4. 시각화
    print("\n" + "=" * 70)
    print(" 시각화 생성 중...")
    print("=" * 70)
    plot_regime_overlay(df, features, results, args)
    plot_regime_stats(results, args)


if __name__ == '__main__':
    main()
