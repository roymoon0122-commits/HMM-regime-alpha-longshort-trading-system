"""
실제 BTC 데이터로 9개 피처가 직관적으로 말이 되는지 시각적으로 검증.

────────────────────────────────────────────────────────────────────
실행 방법
────────────────────────────────────────────────────────────────────
    cd Coin-trader-main
    python -m strategy.HMM_strategy.scripts.verify_features

또는 파라미터 변경:
    python -m strategy.HMM_strategy.scripts.verify_features \
        --timeframe 4H --window-size 60

────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일
────────────────────────────────────────────────────────────────────
1. config.py의 기본 파라미터로 BTC 1분봉 → 4시간봉 변환
2. 9개 윈도우 피처 계산
3. 콘솔에 통계 요약 출력 (min/max/mean/std/NaN 개수)
4. matplotlib으로 5단 그래프 출력:
   - BTC 종가
   - cum_return + slope
   - volatility
   - adx_mean / r2_mean
   - max_drawdown / up_candle_ratio

────────────────────────────────────────────────────────────────────
사용자가 봐야 할 점
────────────────────────────────────────────────────────────────────
- 큰 하락장 시점에 cum_return ↓, volatility ↑, max_drawdown 깊어짐 → ✅
- 큰 상승장 시점에 cum_return ↑, up_candle_ratio > 0.5 → ✅
- adx_mean이 추세장에서 높음 (>25), 횡보장에서 낮음 → ✅
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.resampler import load_and_resample
from strategy.HMM_strategy.features.window_features import (
    FEATURE_COLUMNS,
    compute_window_features,
)


# ════════════════════════════════════════════════════════════════
#  CLI 인자 파싱
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="HMM_strategy 윈도우 피처 시각화 검증"
    )
    parser.add_argument('--csv-path',     default=config.DATA_PATH,
                        help=f"1분봉 CSV 경로 (기본: {config.DATA_PATH})")
    parser.add_argument('--timeframe',    default=config.TIMEFRAME,
                        help=f"리샘플링 타임프레임 (기본: {config.TIMEFRAME})")
    parser.add_argument('--start',        default=config.VERIFY_START,
                        help=f"기간 시작 (기본: {config.VERIFY_START})")
    parser.add_argument('--end',          default=config.VERIFY_END,
                        help=f"기간 종료 (기본: {config.VERIFY_END})")
    parser.add_argument('--window-size',  type=int, default=config.WINDOW_SIZE,
                        help=f"윈도우 크기 (기본: {config.WINDOW_SIZE})")
    parser.add_argument('--adx-period',   type=int, default=config.ADX_PERIOD,
                        help=f"ADX 기간 (기본: {config.ADX_PERIOD})")
    parser.add_argument('--r2-period',    type=int, default=config.R2_PERIOD,
                        help=f"R² 기간 (기본: {config.R2_PERIOD})")
    parser.add_argument('--save',         default=None,
                        help="그래프 PNG 저장 경로 (선택)")
    parser.add_argument('--no-show',      action='store_true',
                        help="그래프 창 띄우지 않기 (저장만)")
    parser.add_argument('--show-correlation', action='store_true',
                        help="9개 피처 간 상관계수 히트맵을 별도 창으로 표시")
    parser.add_argument('--save-correlation', default=None,
                        help="히트맵 PNG 저장 경로 (선택, --show-correlation과 함께 사용)")
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
#  통계 요약 출력
# ════════════════════════════════════════════════════════════════

def print_summary(features_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(f" 윈도우 피처 통계 요약 (총 {len(features_df):,}개 윈도우)")
    print("=" * 70)
    print(f" 시간 범위: {features_df['window_end_time'].iloc[0]} "
          f"~ {features_df['window_end_time'].iloc[-1]}")
    print()

    summary = features_df[FEATURE_COLUMNS].agg(
        ['min', 'max', 'mean', 'std']
    ).T
    summary['nan_count'] = features_df[FEATURE_COLUMNS].isna().sum().values

    # 보기 좋게 출력
    pd.set_option('display.float_format', '{:>10.6f}'.format)
    print(summary.to_string())
    pd.reset_option('display.float_format')
    print()

    # NaN 경고
    total_nan = summary['nan_count'].sum()
    if total_nan > 0:
        print(f" ⚠️  NaN 값 {total_nan}개 발견 — dropna 처리 누락 의심")
    else:
        print(" ✅ NaN 없음")
    print("=" * 70 + "\n")


# ════════════════════════════════════════════════════════════════
#  5단 시각화
# ════════════════════════════════════════════════════════════════

def plot_features(df: pd.DataFrame, features: pd.DataFrame, args):
    """
    상단부터:
      1. BTC 종가 (참조)
      2. cum_return (양수=빨강, 음수=파랑) + slope
      3. volatility
      4. adx_mean + r2_mean
      5. max_drawdown + up_candle_ratio
    """
    fig, axes = plt.subplots(
        5, 1,
        figsize=(15, 14),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1.5, 1, 1.5, 1.5]},
    )

    times    = features['window_end_time']
    df_times = df['datetime']

    # ── (1) BTC 종가 ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(df_times, df['close'], color='black', linewidth=0.7)
    ax.set_ylabel('BTC Close (USDT)')
    ax.set_title(
        f"HMM Window Features — timeframe={args.timeframe}, "
        f"window_size={args.window_size}, "
        f"adx_period={args.adx_period}, r2_period={args.r2_period}"
    )
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # ── (2) cum_return + slope ──────────────────────────────
    ax = axes[1]
    cum = features['cum_return'].values
    pos = cum >= 0
    ax.fill_between(times, 0, cum, where=pos,  color='red',  alpha=0.4, label='cum_return ≥ 0')
    ax.fill_between(times, 0, cum, where=~pos, color='blue', alpha=0.4, label='cum_return < 0')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('cum_return')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times, features['slope'], color='purple', linewidth=0.7, alpha=0.6, label='slope')
    ax2.set_ylabel('slope', color='purple')
    ax2.legend(loc='upper right')

    # ── (3) volatility ───────────────────────────────────────
    ax = axes[2]
    ax.plot(times, features['volatility'], color='orange', linewidth=0.8)
    ax.set_ylabel('volatility')
    ax.grid(True, alpha=0.3)

    # ── (4) adx_mean + r2_mean ──────────────────────────────
    ax = axes[3]
    ax.plot(times, features['adx_mean'], color='blue',   linewidth=0.8, label='adx_mean')
    ax.axhline(25, color='blue', linestyle='--', linewidth=0.5, alpha=0.5, label='ADX=25')
    ax.set_ylabel('adx_mean', color='blue')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times, features['r2_mean'], color='darkorange', linewidth=0.8, label='r2_mean')
    ax2.axhline(0.55, color='darkorange', linestyle='--', linewidth=0.5, alpha=0.5, label='R²=0.55')
    ax2.set_ylabel('r2_mean', color='darkorange')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)

    # ── (5) max_drawdown + up_candle_ratio ─────────────────
    ax = axes[4]
    ax.fill_between(times, features['max_drawdown'], 0, color='red', alpha=0.4, label='max_drawdown')
    ax.set_ylabel('max_drawdown', color='red')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times, features['up_candle_ratio'], color='purple', linewidth=0.8, label='up_candle_ratio')
    ax2.axhline(0.5, color='purple', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel('up_candle_ratio', color='purple')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)

    # x축 날짜 포매팅
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=120, bbox_inches='tight')
        print(f" 📁 그래프 저장 완료: {args.save}")

    if not args.no_show:
        plt.show()


# ════════════════════════════════════════════════════════════════
#  상관계수 히트맵
# ════════════════════════════════════════════════════════════════

def compute_vif(corr_matrix: pd.DataFrame) -> pd.Series:
    """
    상관 행렬로부터 VIF (Variance Inflation Factor)를 계산.

    수학적 사실:
        VIF_i = (R^(-1))_ii   (R = 상관 행렬)

    이는 "피처 i를 나머지 모든 피처로 회귀했을 때의 R²"로 정의되는
    표준 VIF와 동일하다. 표준화된 데이터에서는 이 두 정의가 일치함.

    해석:
        VIF < 5    : 안전 — 다변량 중복 거의 없음
        5 ≤ VIF<10: 중등도 다중공선성, 주의
        VIF ≥ 10  : 심각한 다중공선성, 조치 권장

    Args:
        corr_matrix: 피처 간 상관 행렬 (n×n DataFrame).

    Returns:
        Series: 각 피처의 VIF 값.

    Note:
        상관 행렬이 특이행렬에 가까우면 역행렬 계산이 불안정해짐.
        그런 경우는 그 자체로 "심각한 다중공선성" 신호.
    """
    try:
        R_inv = np.linalg.inv(corr_matrix.values)
        vif = np.diag(R_inv)
    except np.linalg.LinAlgError:
        # 특이행렬 — 완벽히 동일한 피처가 있다는 뜻
        n = len(corr_matrix)
        vif = np.full(n, np.inf)
    return pd.Series(vif, index=corr_matrix.columns)


def print_vif_report(corr_matrix: pd.DataFrame):
    """VIF 결과를 콘솔에 보기 좋게 출력 + 자동 경고."""
    vif = compute_vif(corr_matrix)
    vif_sorted = vif.sort_values(ascending=False)

    print("\n" + "=" * 70)
    print(" VIF (Variance Inflation Factor) — 다변량 중복 측정")
    print("=" * 70)
    print(" 해석: VIF < 5 안전 / 5~10 주의 / ≥ 10 조치 권장\n")

    for feature, v in vif_sorted.items():
        if v >= 10:
            tag = "❌ 조치 권장"
        elif v >= 5:
            tag = "⚠️  주의"
        else:
            tag = "✅ 안전"
        print(f"   {feature:>17s} : VIF = {v:>7.3f}  {tag}")

    # 종합 진단
    n_severe = int((vif >= 10).sum())
    n_moderate = int(((vif >= 5) & (vif < 10)).sum())
    print()
    if n_severe > 0:
        print(f" → {n_severe}개 피처에서 심각한 다변량 중복 발견. HMM 입력에서 제외 권장.")
    elif n_moderate > 0:
        print(f" → {n_moderate}개 피처가 회색지대. covariance_type='diag' 시 주의.")
    else:
        print(" → 다변량 중복 양호 ✅")

    # 추가: 공분산 행렬 condition number
    eigvals = np.linalg.eigvalsh(corr_matrix.values)
    cond = float(eigvals.max() / max(eigvals.min(), 1e-12))
    print(f"\n 상관 행렬의 condition number: {cond:.1f}")
    if cond < 30:
        print("   → 매우 안전")
    elif cond < 100:
        print("   → 안전")
    elif cond < 1000:
        print("   → 주의 (covariance_type='full' 학습 시 흔들릴 수 있음)")
    else:
        print("   → 심각 (수치 불안정 위험)")
    print("=" * 70 + "\n")


def plot_correlation_heatmap(features: pd.DataFrame, args):
    """
    9개 피처 간 Pearson 상관계수 히트맵 + VIF 콘솔 출력.

    색상:
      파랑 (-1) → 흰색 (0) → 빨강 (+1)

    셀 안 숫자: 상관계수 (소수점 둘째 자리).
    절댓값 0.7 이상은 굵게 표시 — 한눈에 강한 상관 식별.

    부가: VIF + condition number를 콘솔에 출력 — 페어와이즈 상관계수가
    잡지 못하는 다변량 중복까지 진단.
    """
    # ── 1. 상관 행렬 계산 ─────────────────────────────────────
    corr = features[FEATURE_COLUMNS].corr(method='pearson')

    # ── 2. 콘솔에도 요약 출력 ─────────────────────────────────
    print("\n" + "=" * 70)
    print(" 피처 간 Pearson 상관계수")
    print("=" * 70)
    pd.set_option('display.float_format', '{:>+7.3f}'.format)
    print(corr.to_string())
    pd.reset_option('display.float_format')

    # 강한 상관 쌍 (|r| ≥ 0.7) 추출 — 자기 자신 제외, 중복 제거
    print("\n 강한 상관 쌍 (|r| ≥ 0.7):")
    strong_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.7:
                strong_pairs.append((cols[i], cols[j], r))
    if strong_pairs:
        for a, b, r in sorted(strong_pairs, key=lambda x: -abs(x[2])):
            print(f"   {a:>17s} ↔ {b:<17s}  r = {r:+.3f}")
        print("\n   → HMM에 둘 다 넣으면 거리 왜곡 위험. 한쪽만 넣거나 covariance_type='full' 사용.")
    else:
        print("   없음 ✅")
    print("=" * 70 + "\n")

    # ── 3. VIF + condition number 출력 ──────────────────────
    print_vif_report(corr)
    vif = compute_vif(corr)

    # ── 4. 히트맵 + VIF 막대그래프 (2패널) ──────────────────
    # gridspec으로 좌:우 = 3:1 비율 (히트맵이 메인, VIF는 보조)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_vif = fig.add_subplot(gs[0, 1], sharey=ax)

    n = len(FEATURE_COLUMNS)

    # ── 4-1. 왼쪽: 히트맵 ────────────────────────────────────
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha='right')
    ax.set_yticklabels(FEATURE_COLUMNS)

    # 셀 안에 상관계수 숫자 표시
    for i in range(n):
        for j in range(n):
            r = corr.iloc[i, j]
            color = 'white' if abs(r) > 0.6 else 'black'
            weight = 'bold' if abs(r) >= 0.7 and i != j else 'normal'
            ax.text(j, i, f"{r:+.2f}",
                    ha='center', va='center',
                    color=color, fontsize=9, fontweight=weight)

    # 컬러바 (히트맵 아래에)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, location='bottom')
    cbar.set_label('Pearson correlation', labelpad=8)

    ax.set_title("Pearson Correlation", pad=10)

    # ── 4-2. 오른쪽: VIF 가로 막대그래프 ────────────────────
    # 히트맵의 y축 순서와 동일하게 정렬 (sharey 덕분에 자동 정렬됨)
    vif_values = vif[FEATURE_COLUMNS].values

    # 색상: 안전(녹색) / 주의(주황) / 조치 권장(빨강)
    colors = []
    for v in vif_values:
        if v >= 10:
            colors.append('#d62728')   # red
        elif v >= 5:
            colors.append('#ff7f0e')   # orange
        else:
            colors.append('#2ca02c')   # green

    y_positions = np.arange(n)
    ax_vif.barh(y_positions, vif_values, color=colors, edgecolor='black', linewidth=0.5)

    # 위험도 기준선
    ax_vif.axvline(5,  color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='VIF=5 (주의)')
    ax_vif.axvline(10, color='red',    linestyle='--', linewidth=0.8, alpha=0.7, label='VIF=10 (조치)')

    # 막대 끝에 숫자 표시
    x_max = max(vif_values.max(), 12) * 1.15
    for i, v in enumerate(vif_values):
        # 값이 작으면 막대 안쪽에, 크면 막대 바깥쪽에 표시
        if v < x_max * 0.2:
            ax_vif.text(v + x_max * 0.01, i, f"{v:.2f}",
                        va='center', ha='left', fontsize=9)
        else:
            ax_vif.text(v - x_max * 0.01, i, f"{v:.2f}",
                        va='center', ha='right', fontsize=9, color='white', fontweight='bold')

    ax_vif.set_xlim(0, x_max)
    ax_vif.set_xlabel('VIF')
    ax_vif.set_title("VIF (다변량 중복)", pad=10)
    # y축 레이블은 왼쪽 히트맵과 공유 → 오른쪽에서는 숨김
    plt.setp(ax_vif.get_yticklabels(), visible=False)
    ax_vif.invert_yaxis()   # 히트맵의 y축 방향과 맞추기 (위→아래)
    ax_vif.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax_vif.grid(True, axis='x', alpha=0.3)

    # ── 4-3. 전체 제목 ──────────────────────────────────────
    fig.suptitle(
        f"Feature Correlation & VIF — "
        f"timeframe={args.timeframe}, window_size={args.window_size}, "
        f"n_windows={len(features):,}",
        fontsize=12,
        y=0.995,
    )

    plt.tight_layout()

    if args.save_correlation:
        plt.savefig(args.save_correlation, dpi=120, bbox_inches='tight')
        print(f" 📁 히트맵+VIF 저장 완료: {args.save_correlation}")

    if not args.no_show:
        plt.show()


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # CSV 경로 검증
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ CSV 파일을 찾을 수 없음: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📥 데이터 로드 + 리샘플링 ({args.timeframe})...")
    df = load_and_resample(
        csv_path=str(csv_path),
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
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

    print_summary(features)

    if args.show_correlation:
        plot_correlation_heatmap(features, args)

    plot_features(df, features, args)


if __name__ == '__main__':
    main()
