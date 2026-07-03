"""
Phase 4 통합 백테스트 시각화 (인터랙티브 HTML).

────────────────────────────────────────────────────────────────────
출력 구조 — 1개 HTML에 5개 그래프 (수직 배치)
────────────────────────────────────────────────────────────────────
  [그래프 1] 벤치마크 비교 — 모든 전략의 자산 곡선 한 그래프
             (Donchian / MA Cross / Buy&Hold + HMM 4 variant)
  [그래프 2] HMM Variant A: HMM✓ Smooth✓ (Donchian OFF)
  [그래프 3] HMM Variant B: HMM✓ Smooth✗ (Donchian OFF)
  [그래프 4] HMM Variant C: HMM✓ Smooth✓ + Donchian on SIDE
  [그래프 5] HMM Variant D: HMM✓ Smooth✗ + Donchian on SIDE

  - 그래프 2~5의 배경 음영 = Meta-model이 그 시점에 내린 거래 판단
    (Bull=초록 / Side=회색 / Bear=빨강). 즉 실제 포지션 라벨과 일치.
  - 별도의 regime_view 시각화는 더 이상 필요하지 않음.

────────────────────────────────────────────────────────────────────
Hover 정보
────────────────────────────────────────────────────────────────────
- 벤치마크 그래프: 각 전략별 자산값/수익률
- HMM variant 그래프 (각 시점):
    날짜 / 자산 / 누적수익률 /
    포지션 비중(이전→현재) / 비중 변화량 /
    국면(HMM Viterbi) vs 거래(Meta-model) — 두 단계 라벨 같이 표시

────────────────────────────────────────────────────────────────────
자산 일반화
────────────────────────────────────────────────────────────────────
result['asset_name']과 result['timeframe']을 활용해 BTC 외에도
ETH, SOL, AAPL 등 자산 무관하게 동작.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────
# 색상 정의
# ─────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    0: 'rgba(46, 204, 113, 0.18)',   # Bull — 초록
    1: 'rgba(149, 165, 166, 0.14)',  # Side — 회색
    2: 'rgba(231, 76, 60, 0.18)',    # Bear — 빨강
}
REGIME_NAMES = {0: 'Bull', 1: 'Side', 2: 'Bear'}

BENCHMARK_COLORS = {
    'donchian': '#2c3e50',
    'ma_cross': '#27ae60',
    'buy_hold': '#888888',
}

VARIANT_COLORS = {
    'variant_A': '#e74c3c',
    'variant_B': '#f39c12',
    'variant_C': '#3498db',
    'variant_D': '#9b59b6',
}

EQUITY_LINE_COLOR = '#2c3e50'
SIGNAL_LINE_COLOR = '#e67e22'


# ─────────────────────────────────────────────────────────────────
# 다운샘플링 (성능)
# ─────────────────────────────────────────────────────────────────

def _downsample_index(n: int, target: int = 1500) -> np.ndarray:
    """n개 포인트를 target개 이하로 줄이는 인덱스 (linspace 기반)."""
    if n <= target:
        return np.arange(n)
    step = max(1, n // target)
    return np.arange(0, n, step)


# ─────────────────────────────────────────────────────────────────
# 라벨 inference (variant 시각화용)
# ─────────────────────────────────────────────────────────────────
#
# 한 variant에서 두 가지 라벨이 필요하다:
#   1) HMM Viterbi argmax — HMM 1단계 원시 판단 (호버 표시·진단용)
#   2) Meta-model argmax  — 실제 거래에 쓰이는 2단계 판단 (음영용)
#
# 두 라벨을 따로 만들면 윈도우 피처/HMM proba 계산이 중복되므로
# 한 함수에서 둘 다 산출해 dict로 반환한다.
# (기존 _infer_regime_labels(strategy, df_test) → np.ndarray 함수의 후속.)
# ─────────────────────────────────────────────────────────────────

def _infer_labels(strategy, df_test: pd.DataFrame) -> dict:
    """
    학습 완료된 HMMStrategy로 df_test의 각 봉에 대해
    HMM Viterbi 라벨과 Meta-model 라벨을 동시에 산출.

    Returns:
        dict {
            'hmm':  np.ndarray shape (len(df_test),) — HMM Viterbi argmax (0/1/2)
            'meta': np.ndarray shape (len(df_test),) — Meta-model argmax (0/1/2)
        }
        둘 다 forward-fill 적용. 워밍업 등 -1 구간이 앞쪽에 남을 수 있음.
    """
    from strategy.HMM_strategy.regime.transition import TransitionPredictor

    n_bars = len(df_test)

    # ── 1. 윈도우 피처 + slope_norm (HMMStrategy._build_features와 동일) ──
    features = strategy._build_features(df_test)

    # ── 2. HMM 사후확률 → Viterbi 라벨 ───────────────────────────
    hmm_proba, _, _ = strategy._compute_hmm_proba(features)
    hmm_labels_win = np.argmax(hmm_proba, axis=1).astype(np.int64)

    # ── 3. 메타 입력 X_meta → Meta-model 라벨 ────────────────────
    # 학습 시 사용한 transition_predictor를 labeler에서 새로 구성
    transition_predictor = TransitionPredictor.from_labeler(strategy.labeler_)
    X_meta, _, nan_mask_meta = strategy._build_meta_input(
        features, hmm_proba, transition_predictor, df=df_test,
    )

    # cold-start (HMM proba가 1/3 균등) + X_meta에 NaN인 시점은 무효 윈도우
    cold_mask = np.isclose(hmm_proba, 1.0 / 3.0, atol=1e-9).all(axis=1)
    invalid_mask = nan_mask_meta | cold_mask

    X_meta_safe = np.where(np.isnan(X_meta), 0.0, X_meta)
    proba_meta = strategy.meta_model_.predict_proba(X_meta_safe)
    # 무효 윈도우는 균등분포로 마킹 → argmax는 의미 없지만 아래에서 -1로 덮음
    proba_meta[invalid_mask] = 1.0 / 3.0
    meta_labels_win = np.argmax(proba_meta, axis=1).astype(np.int64)
    meta_labels_win[invalid_mask] = -1  # 무효는 -1로 표시 → ffill 단계에서 처리

    # ── 4. 윈도우 라벨 → 봉 단위 라벨 매핑 (window_end_idx에 할당) ─
    def _windows_to_bars(win_labels: np.ndarray) -> np.ndarray:
        bar = np.full(n_bars, -1, dtype=np.int64)
        end_idx = features['window_end_idx'].astype(int).values
        for j, idx in enumerate(end_idx):
            if 0 <= idx < n_bars:
                bar[idx] = win_labels[j]
        # 빈 구간 forward-fill
        last = -1
        for i in range(n_bars):
            if bar[i] == -1:
                bar[i] = last
            else:
                last = bar[i]
        return bar

    return {
        'hmm':  _windows_to_bars(hmm_labels_win),
        'meta': _windows_to_bars(meta_labels_win),
    }


# ─────────────────────────────────────────────────────────────────
# 음영 처리 (Bull/Side/Bear)
# ─────────────────────────────────────────────────────────────────

def _add_regime_shading(fig, datetimes, labels, row, col):
    """봉 단위 라벨을 같은 라벨 연속 구간으로 묶어 패널 전체 높이의 음영 추가.

    Note (2026-05-27):
        plotly 6.7에서 add_vrect(row, col, ...) 가 호출 시점에는 에러 없이
        통과하지만 실제로는 fig.layout.shapes 에 사각형이 추가되지 않는
        호환성 이슈가 있어 음영이 전혀 렌더링되지 않음.
        → add_shape(type='rect', yref='y domain', y0=0, y1=1, row=row, col=col)
          패턴으로 직접 사각형을 그려서 우회. 이 방식은 패널의 도메인
          좌표계(0~1)에 묶이므로 secondary_y 가 켜진 subplot에서도
          주 y축과 충돌 없이 안정적으로 동작한다.
    """
    if labels is None or len(labels) == 0:
        return
    n = len(labels)
    seg_start = 0
    cur = labels[0]
    for i in range(1, n):
        if labels[i] != cur or i == n - 1:
            end = i if labels[i] != cur else i + 1
            if cur in REGIME_COLORS and cur >= 0:
                fig.add_shape(
                    type='rect',
                    x0=datetimes[seg_start], x1=datetimes[end - 1],
                    y0=0, y1=1,
                    yref='y domain',   # 패널 전체 높이로 칠하기
                    fillcolor=REGIME_COLORS[cur],
                    line=dict(width=0),
                    layer='below',
                    row=row, col=col,
                )
            seg_start = i
            cur = labels[i]


# ─────────────────────────────────────────────────────────────────
# 워밍업 음영 (모든 그래프 공통)
# ─────────────────────────────────────────────────────────────────

def _add_warmup_shading(fig, warmup_start, test_start, row, col):
    """워밍업 구간(test_start 이전)을 회색 음영 + 'Warmup' 텍스트."""
    fig.add_vrect(
        x0=warmup_start, x1=test_start,
        fillcolor='rgba(200, 200, 200, 0.25)',
        layer='below', line_width=0,
        annotation_text='Warmup', annotation_position='top left',
        annotation_font_size=10, annotation_font_color='#555',
        row=row, col=col,
    )


# ─────────────────────────────────────────────────────────────────
# 벤치마크 그래프 (모든 전략의 equity curve)
# ─────────────────────────────────────────────────────────────────

def _slice_oos(equity: np.ndarray, datetimes, test_start, test_end):
    """
    equity/datetime 배열에서 OOS 구간([test_start, test_end])만 추출.

    동적 슬라이싱 — test_start/test_end 변경 시 자동으로 따라감.
    Returns:
        eq_oos:  shape (m,)
        dt_oos:  pd.Series of pd.Timestamp, length m
        oos_idx: np.ndarray of int (원본 배열에서의 인덱스)
    """
    dt = pd.Series(pd.to_datetime(datetimes))
    mask = (dt.values >= np.datetime64(test_start)) & \
           (dt.values <= np.datetime64(test_end))
    oos_idx = np.where(mask)[0]
    if len(oos_idx) == 0:
        return equity[:0], dt.iloc[:0].reset_index(drop=True), oos_idx
    return equity[oos_idx], dt.iloc[oos_idx].reset_index(drop=True), oos_idx


def _plot_benchmark_panel(fig, result, row, col):
    initial_capital = result['initial_capital']
    test_start = result['test_start']
    test_end = result['test_end']

    # OOS 구간만 슬라이싱한 뒤, OOS 첫 봉을 baseline으로 normalize
    def normalized_oos(equity, datetimes):
        eq_oos, dt_oos, _ = _slice_oos(equity, datetimes, test_start, test_end)
        if len(eq_oos) == 0:
            return np.array([]), dt_oos
        baseline = float(eq_oos[0])
        if baseline <= 0:
            baseline = initial_capital
        return (eq_oos / baseline - 1) * 100, dt_oos

    # 벤치마크 (굵은 선)
    for bid, color in BENCHMARK_COLORS.items():
        b = result['benchmark'].get(bid)
        if b is None:
            continue
        norm, dt = normalized_oos(b['equity'], b['datetime'])
        if len(norm) == 0:
            continue
        idx = _downsample_index(len(norm))
        fig.add_trace(go.Scatter(
            x=dt.iloc[idx], y=norm[idx],
            name=b['label'], mode='lines',
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{b['label']}</b><br>%{{x|%Y-%m-%d %H:%M}}<br>"
                          f"수익률: %{{y:+.2f}}%<extra></extra>",
        ), row=row, col=col)

    # HMM variant (가는 선)
    for vid, color in VARIANT_COLORS.items():
        h = result['hmm_variants'].get(vid)
        if h is None:
            continue
        norm, dt = normalized_oos(h['equity'], h['datetime'])
        if len(norm) == 0:
            continue
        idx = _downsample_index(len(norm))
        fig.add_trace(go.Scatter(
            x=dt.iloc[idx], y=norm[idx],
            name=f"HMM {h['label']}", mode='lines',
            line=dict(color=color, width=1.4, dash='solid'),
            hovertemplate=f"<b>HMM {h['label']}</b><br>%{{x|%Y-%m-%d %H:%M}}<br>"
                          f"수익률: %{{y:+.2f}}%<extra></extra>",
        ), row=row, col=col)

    # 0% 기준선
    fig.add_hline(y=0, line=dict(color='#888', width=0.8, dash='dot'), row=row, col=col)
    # x축을 OOS 구간으로 강제 (혹시 남는 여백 제거)
    fig.update_xaxes(range=[test_start, test_end], row=row, col=col)


# ─────────────────────────────────────────────────────────────────
# HMM variant 그래프 (음영 + equity + signals)
# ─────────────────────────────────────────────────────────────────

def _plot_variant_panel(fig, variant_data, result, row, col,
                          variant_id, df_test):
    initial_capital = result['initial_capital']
    test_start = result['test_start']
    test_end = result['test_end']

    equity_full = variant_data['equity']
    datetimes_full = pd.Series(pd.to_datetime(variant_data['datetime']))
    signals_full = variant_data['signals']

    # ── 1. 라벨 추론 (전체 df_test 기준 — 워밍업 데이터로 윈도우 형성) ─
    # 라벨 inference는 그대로 전체로 하되, 시각화는 OOS 구간만 잘라서 사용.
    # (참고: 룩어헤드 영향 없음. 라벨[i]는 봉 i까지의 정보로만 결정됨.)
    #
    # 두 라벨을 둘 다 얻는다:
    #   - labels_hmm_full  : HMM Viterbi argmax (호버에 진단용)
    #   - labels_meta_full : Meta-model argmax (음영용 — 실제 거래 판단)
    strategy = variant_data.get('strategy')
    if strategy is not None:
        _labs = _infer_labels(strategy, df_test)
        labels_hmm_full  = _labs['hmm']
        labels_meta_full = _labs['meta']
    else:
        labels_hmm_full  = None
        labels_meta_full = None

    # ── 2. OOS 구간만 슬라이싱 ─────────────────────────────────
    eq_oos, dt_oos, oos_full_idx = _slice_oos(
        equity_full, datetimes_full.values, test_start, test_end,
    )
    if len(eq_oos) == 0:
        # OOS 데이터 없음 — 빈 패널
        fig.update_xaxes(range=[test_start, test_end], row=row, col=col)
        return

    sig_oos = signals_full[oos_full_idx]
    labels_hmm_oos  = labels_hmm_full[oos_full_idx]  if labels_hmm_full  is not None else None
    labels_meta_oos = labels_meta_full[oos_full_idx] if labels_meta_full is not None else None
    # 음영은 Meta 거래 판단 기준. (HMM 라벨은 호버에서만 보여줌)
    labels_oos = labels_meta_oos

    # ── 3. OOS baseline 잡고 0% 기준 normalized ──────────────
    baseline = float(eq_oos[0])
    if baseline <= 0:
        baseline = initial_capital
    norm = (eq_oos / baseline - 1) * 100

    # ── 4. 다운샘플링 인덱스 (슬라이싱된 OOS 데이터 기준) ─────
    idx = _downsample_index(len(eq_oos))
    dt_ds = dt_oos.iloc[idx].values

    # ── 5. 음영 (Bull/Side/Bear) — OOS 구간만 ────────────────
    if labels_oos is not None:
        labels_ds = labels_oos[idx]
        _add_regime_shading(fig, dt_ds, labels_ds, row, col)

    # ── 6. 포지션 비중 변화 계산 (hover에 표시) ────────────────
    prev_sig = np.zeros(len(sig_oos))
    prev_sig[1:] = sig_oos[:-1]
    delta_sig = sig_oos - prev_sig

    # ── 7. hover 텍스트 (각 시점) ────────────────────────────
    # 두 라벨을 같이 보여줌: HMM Viterbi(1단계) vs Meta-model(거래에 쓰이는 2단계)
    hover_texts = []
    for i_ds in idx:
        dt_str = dt_oos.iloc[i_ds].strftime('%Y-%m-%d %H:%M')
        eq_v = eq_oos[i_ds]
        ret = norm[i_ds]
        sig_v = sig_oos[i_ds]
        prev_v = prev_sig[i_ds]
        d_v = delta_sig[i_ds]

        if labels_hmm_oos is not None and labels_hmm_oos[i_ds] >= 0:
            hmm_str = REGIME_NAMES.get(int(labels_hmm_oos[i_ds]), '?')
        else:
            hmm_str = '-'
        if labels_meta_oos is not None and labels_meta_oos[i_ds] >= 0:
            meta_str = REGIME_NAMES.get(int(labels_meta_oos[i_ds]), '?')
        else:
            meta_str = '-'

        hover_texts.append(
            f"<b>{dt_str}</b><br>"
            f"자산: {eq_v:,.0f} ({ret:+.2f}%)<br>"
            f"포지션 비중: {prev_v:+.2f} → {sig_v:+.2f}<br>"
            f"변화량: {d_v:+.3f}<br>"
            f"국면(HMM): {hmm_str} / 거래(Meta): {meta_str}"
        )

    # ── 8. Buy & Hold 가격 곡선 (참조용 회색 선) — OOS 구간만 ──
    bh = result.get('benchmark', {}).get('buy_hold')
    if bh is not None:
        bh_eq_oos, bh_dt_oos, _ = _slice_oos(
            bh['equity'], bh['datetime'], test_start, test_end,
        )
        if len(bh_eq_oos) > 0:
            bh_baseline = float(bh_eq_oos[0])
            if bh_baseline <= 0:
                bh_baseline = initial_capital
            bh_norm = (bh_eq_oos / bh_baseline - 1) * 100

            show_bh_legend = (row == 2)
            _asset_label = result.get('asset_name', '자산')
            fig.add_trace(go.Scatter(
                x=bh_dt_oos.iloc[idx],
                y=bh_norm[idx],
                name=f"{_asset_label} 가격 (B&H)" if show_bh_legend else None,
                mode='lines',
                line=dict(color='#888888', width=1.0, dash='solid'),
                opacity=0.55,
                hovertemplate='B&H: %{y:+.2f}%<extra></extra>',
                showlegend=show_bh_legend,
            ), row=row, col=col)

    # ── 9. HMM variant equity curve (메인 굵은 선) ─────────────
    fig.add_trace(go.Scatter(
        x=dt_ds,
        y=norm[idx],
        name=f"수익률 ({variant_data['label']})",
        mode='lines',
        line=dict(color=EQUITY_LINE_COLOR, width=1.6),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False,
    ), row=row, col=col)

    # ── 10. 포지션 비중 (보조 y축, OOS 구간만) ───────────────
    fig.add_trace(go.Scatter(
        x=dt_ds,
        y=sig_oos[idx],
        name=f"포지션 ({variant_data['label']})",
        mode='lines',
        line=dict(color=SIGNAL_LINE_COLOR, width=1.0, dash='dot'),
        opacity=0.7,
        hoverinfo='skip',
        showlegend=False,
    ), row=row, col=col, secondary_y=True)

    # 0% 기준선
    fig.add_hline(y=0, line=dict(color='#888', width=0.8, dash='dot'),
                  row=row, col=col)
    # x축을 OOS 구간으로 강제 — 워밍업 영역 완전 제거
    fig.update_xaxes(range=[test_start, test_end], row=row, col=col)

    # 통계 요약 annotation (그래프 우상단)
    stats = variant_data.get('stats')
    if stats:
        ann_text = (f"CAGR {stats['cagr']:+.1f}% / "
                    f"Sharpe {stats['sharpe']:.2f} / "
                    f"MDD {stats['mdd']:+.1f}% / "
                    f"Trades {stats['total_trades']}")
        fig.add_annotation(
            xref=f'x{row} domain' if row > 1 else 'x domain',
            yref=f'y{2*row-1} domain' if row > 1 else 'y domain',
            x=0.99, y=0.97, xanchor='right', yanchor='top',
            text=ann_text,
            showarrow=False,
            font=dict(size=11, color='#222'),
            bgcolor='rgba(255,255,255,0.7)',
            row=row, col=col,
        )


# ─────────────────────────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────────────────────────

def plot_hmm_backtest(
    result: dict,
    df_test: pd.DataFrame,
    output_path: str = "backtest_hmm_result.html",
) -> None:
    """
    7개 백테스트 결과를 5개 그래프로 시각화하고 HTML로 저장.

    result 구조:
        result['asset_name']        str — 자산 이름 (예: 'BTC/USDT', 'AAPL')
        result['timeframe']          str — '4h', '1d' 등
        result['warmup_start']       pd.Timestamp
        result['test_start']         pd.Timestamp — 진짜 OOS 시작
        result['test_end']           pd.Timestamp
        result['initial_capital']    float
        result['benchmark']          dict — {'donchian'/'ma_cross'/'buy_hold': {...}}
        result['hmm_variants']       dict — {'variant_A'~'variant_D': {...}}
            각 entry: {'label', 'equity', 'datetime', 'signals',
                       'trades', 'strategy' (HMMStrategy), 'stats'}

    df_test:
        백테스트에 사용된 OHLCV df (워밍업 + OOS 구간 포함, datetime 정렬)
    """
    from plotly.subplots import make_subplots

    title = (f"HMM 백테스트 결과 — {result['asset_name']} ({result['timeframe']}) | "
             f"OOS {pd.Timestamp(result['test_start']).strftime('%Y-%m-%d')} ~ "
             f"{pd.Timestamp(result['test_end']).strftime('%Y-%m-%d')}")

    # 5개 행 (벤치마크 + 4 variant)
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.175, 0.175, 0.175, 0.175],
        vertical_spacing=0.05,
        subplot_titles=(
            "벤치마크 비교 (모든 전략 — OOS 시작 기준 0%)",
            "HMM Variant A: HMM✓ Smooth✓ (Donchian OFF)  ·  음영=Meta 거래 판단",
            "HMM Variant B: HMM✓ Smooth✗ (Donchian OFF)  ·  음영=Meta 거래 판단",
            "HMM Variant C: HMM✓ Smooth✓ + Donchian on SIDE  ·  음영=Meta 거래 판단",
            "HMM Variant D: HMM✓ Smooth✗ + Donchian on SIDE  ·  음영=Meta 거래 판단",
        ),
        specs=[[{'secondary_y': False}],
               [{'secondary_y': True}],
               [{'secondary_y': True}],
               [{'secondary_y': True}],
               [{'secondary_y': True}]],
    )

    # 1. 벤치마크
    _plot_benchmark_panel(fig, result, row=1, col=1)

    # 2~5. HMM variant
    variant_ids = ['variant_A', 'variant_B', 'variant_C', 'variant_D']
    for i, vid in enumerate(variant_ids):
        v = result['hmm_variants'].get(vid)
        if v is None:
            continue
        _plot_variant_panel(fig, v, result, row=i + 2, col=1, variant_id=vid,
                              df_test=df_test)

    # 레이아웃
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        hovermode='x unified',
        template='plotly_white',
        height=1500,
        margin=dict(l=70, r=30, t=80, b=40),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(size=10),
        ),
    )

    # y축 라벨
    fig.update_yaxes(title_text="수익률 %", row=1, col=1)
    for r in (2, 3, 4, 5):
        fig.update_yaxes(title_text="수익률 %", row=r, col=1, secondary_y=False)
        fig.update_yaxes(title_text="포지션", range=[-1.05, 1.05],
                          row=r, col=1, secondary_y=True,
                          showgrid=False)
    fig.update_xaxes(title_text="날짜", row=5, col=1)

    # 저장
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',  # 파일 크기 작게
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        },
    )
