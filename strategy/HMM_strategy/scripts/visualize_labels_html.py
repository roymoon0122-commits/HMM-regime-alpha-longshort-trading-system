"""
인터랙티브 HTML 시각화 — Original / Smoothed / Meta-model Prediction 비교.

────────────────────────────────────────────────────────────────────
이 스크립트가 만드는 것
────────────────────────────────────────────────────────────────────
줌/호버 가능한 Plotly HTML 파일. 4개 패널이 같은 X축 공유:

    [패널 1] BTC 가격 (로그 스케일) + backdate 시점 삼각형 마커
    [패널 2] Original HMM Viterbi 라벨        — Heatmap 음영
    [패널 3] Smoothed 라벨 (backdate 적용)    — Heatmap + 변경 시점 세로선
    [패널 4] Meta-model 예측 (out-of-sample)  — Heatmap 음영

색깔 규칙:
    Bull = 빨강 (#e74c3c)
    Side = 회색 (#95a5a6)
    Bear = 파랑 (#3498db)
    예측 없음 (cold-start, 첫 fold) = 투명/공백

────────────────────────────────────────────────────────────────────
구현 메모 (왜 Heatmap을 쓰는가)
────────────────────────────────────────────────────────────────────
이전 버전은 `add_shape`로 라벨 음영을 그리고 `update_layout(shapes=...)`
로 일괄 적용했는데, `shared_xaxes=True`인 subplot에서 xref='x2', 'x3'
참조가 plotly 버전에 따라 잘못 바인딩되어 음영이 아예 안 그려졌다.
Heatmap trace는 plotly가 자동으로 해당 row의 axis를 잡아주므로
shared_xaxes와도 안정적으로 동작한다.

────────────────────────────────────────────────────────────────────
실행 방법
────────────────────────────────────────────────────────────────────
    cd Coin-trader-main

    # 기본 실행
    python -m strategy.HMM_strategy.scripts.visualize_labels_html

    # 출력 경로 변경
    python -m strategy.HMM_strategy.scripts.visualize_labels_html \\
        --out outputs/label_compare.html

    # cold start 구간도 포함해서 보기
    python -m strategy.HMM_strategy.scripts.visualize_labels_html \\
        --include-coldstart

    # smoother 임계값/lookback 변경
    python -m strategy.HMM_strategy.scripts.visualize_labels_html \\
        --threshold 0.07 --lookback 15
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit

from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.resampler import load_and_resample
from strategy.HMM_strategy.features.window_features import compute_window_features
from strategy.HMM_strategy.features.scaling import RollingStandardScaler
from strategy.HMM_strategy.regime.hmm_labeler import (
    HMMLabeler, BULL, SIDE, BEAR, REGIME_NAMES,
)
from strategy.HMM_strategy.regime.label_smoother import RetrospectiveLabelSmoother
from strategy.HMM_strategy.regime.transition import TransitionPredictor
from strategy.HMM_strategy.classifiers.adx_classifier import ADXClassifier
from strategy.HMM_strategy.classifiers.r2_classifier import R2Classifier
from strategy.HMM_strategy.meta_model.logistic_meta_model import LogisticMetaModel


# ════════════════════════════════════════════════════════════════
#  색상 매핑
# ════════════════════════════════════════════════════════════════

REGIME_HEX = {
    BULL: '#e74c3c',   # 빨강
    SIDE: '#95a5a6',   # 회색
    BEAR: '#3498db',   # 파랑
}

# Heatmap discrete colorscale —
# z=0 (Bull) → 위치 0/3, z=1 (Side) → 위치 1/3~2/3, z=2 (Bear) → 위치 1.
# 같은 색을 두 stop에 반복해서 "끊김 없는 그라디언트"를 막는다.
DISCRETE_COLORSCALE = [
    [0.0,     REGIME_HEX[BULL]],
    [1 / 3,   REGIME_HEX[BULL]],
    [1 / 3,   REGIME_HEX[SIDE]],
    [2 / 3,   REGIME_HEX[SIDE]],
    [2 / 3,   REGIME_HEX[BEAR]],
    [1.0,     REGIME_HEX[BEAR]],
]


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="HTML 인터랙티브 라벨/예측 비교")
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
    # 메타 모델 OOS
    p.add_argument('--C', type=float, default=1.0,
                   help="LogisticMetaModel L2 정규화의 역수")
    p.add_argument('--ts-splits', type=int, default=config.TS_SPLIT_N,
                   help="TimeSeriesSplit fold 수 (out-of-sample 예측용)")
    # 출력
    p.add_argument('--include-coldstart', action='store_true',
                   help="cold start 구간도 시각화에 포함 (기본은 제외)")
    p.add_argument('--out', default='outputs/label_compare.html',
                   help="출력 HTML 경로")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  메타 모델 OOS 예측 — TimeSeriesSplit
# ════════════════════════════════════════════════════════════════

def compute_meta_oos_predictions(X, y, n_splits, gap, C, feature_names):
    """
    TimeSeriesSplit 기반 out-of-sample 예측.

    각 시점에 대해 그 시점이 test fold에 들어갔을 때의 예측만 모은다.
    train fold로만 쓰인 시점(=첫 train 구간)은 -1을 채워 "예측 없음"
    으로 표시한다. 이렇게 해야 룩어헤드 없이 "그 시점에서 모델이
    어떻게 판단했을까"를 그대로 보여줄 수 있다.

    Args:
        X: shape (n_filtered, n_features) — 메타 입력 (NaN/cold-start 제거됨)
        y: shape (n_filtered,) — smoothed 라벨의 t+1 시프트 (정답)
        n_splits: TimeSeriesSplit fold 수
        gap: train/test 사이 갭 (룩어헤드 추가 차단; window_size 권장)
        C: LogisticMetaModel 정규화
        feature_names: 메타 입력 피처 이름 리스트

    Returns:
        preds: shape (n_filtered,) — 각 인덱스의 OOS 예측 (없으면 -1)
    """
    n_filtered = len(X)
    preds = np.full(n_filtered, -1, dtype=np.int64)

    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        meta = LogisticMetaModel(
            C=C, class_weight='balanced', feature_names=feature_names,
        )
        meta.fit(X[train_idx], y[train_idx])
        preds[test_idx] = meta.predict(X[test_idx])

    return preds


# ════════════════════════════════════════════════════════════════
#  Heatmap 헬퍼 — 1×N 라벨/예측 색띠
# ════════════════════════════════════════════════════════════════

def make_regime_heatmap(times, labels, panel_name):
    """
    1×N 짜리 Heatmap trace 생성.

    Args:
        times: shape (n,) — datetime64 배열
        labels: shape (n,) — 정수 라벨, -1은 "예측 없음" (NaN 처리)
        panel_name: 호버 표시용 이름 ("Original" 등)

    Returns:
        go.Heatmap trace
    """
    # -1 → NaN (투명 처리). Heatmap은 NaN을 그리지 않음.
    z_row = labels.astype(float)
    z_row[labels < 0] = np.nan

    # 호버용 텍스트 (각 셀에 라벨 이름)
    text_row = [
        REGIME_NAMES[int(l)] if l >= 0 else 'N/A' for l in labels
    ]

    return go.Heatmap(
        x=times,
        y=[panel_name],          # y축 1칸짜리
        z=[z_row],                # 2D shape (1, n)
        text=[text_row],
        zmin=0, zmax=2,
        colorscale=DISCRETE_COLORSCALE,
        showscale=False,
        xgap=0, ygap=0,
        hovertemplate=(
            f'<b>{panel_name}</b><br>'
            '%{x|%Y-%m-%d %H:%M}<br>'
            'Regime: %{text}<extra></extra>'
        ),
        name=panel_name,
        showlegend=False,
    )


# ════════════════════════════════════════════════════════════════
#  Figure 빌드
# ════════════════════════════════════════════════════════════════

def build_figure(df, features, labels_orig, labels_smooth, meta_preds,
                 change_log, args):
    """4-패널 Plotly Figure."""

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=(
            f"BTC Price (window={args.window_size}, threshold={args.threshold:.0%})",
            "Original HMM Viterbi Labels",
            "Smoothed Labels (backdate applied)",
            "Meta-model Prediction (out-of-sample)",
        ),
    )

    # ── 패널 1: BTC 가격 ─────────────────────────────────────
    df_close = df['close'].values
    df_times = pd.to_datetime(df['datetime']).values
    fig.add_trace(
        go.Scatter(
            x=df_times, y=df_close,
            mode='lines',
            name='BTC Price',
            line=dict(color='black', width=1),
            hovertemplate='%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>',
        ),
        row=1, col=1,
    )
    fig.update_yaxes(type='log', title_text='BTC Price (log)', row=1, col=1)

    # ── 패널 2/3/4: 라벨/예측 Heatmap ───────────────────────
    times_dt = pd.to_datetime(features['window_end_time'].values)
    fig.add_trace(make_regime_heatmap(times_dt, labels_orig,  'Original'),
                  row=2, col=1)
    fig.add_trace(make_regime_heatmap(times_dt, labels_smooth, 'Smoothed'),
                  row=3, col=1)
    fig.add_trace(make_regime_heatmap(times_dt, meta_preds,   'Prediction'),
                  row=4, col=1)

    # 라벨 패널의 y축은 1칸이므로 tick 숨기고 zoom 막기
    for r in [2, 3, 4]:
        fig.update_yaxes(showticklabels=False, fixedrange=True,
                         showgrid=False, row=r, col=1)

    # ── Backdate 이벤트 마커 (가격 패널 위 + smoothed 패널 세로선) ──
    end_times_arr = times_dt.to_numpy()
    marker_x, marker_color, marker_symbol, marker_hover = [], [], [], []
    marker_y_const = float(np.nanmax(df_close)) * 1.10  # 가격 차트 위쪽

    for log in change_log:
        orig_t = end_times_arr[log['original_t']]
        bd_t = end_times_arr[log['backdated_to']]
        new_lbl = log['new_label']
        shift = log['shift']
        shock = log['shock_return']

        # smoothed 패널: backdate 시작점 (굵은 컬러 솔리드 라인)
        fig.add_vline(
            x=bd_t,
            line=dict(color=REGIME_HEX[new_lbl], width=2),
            opacity=0.85,
            row=3, col=1,
        )
        # smoothed 패널: HMM 원래 전환점 (회색 점선) — 얼마나 앞당겼는지 보이게
        fig.add_vline(
            x=orig_t,
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.6,
            row=3, col=1,
        )

        # 가격 패널: 삼각형 마커 (호버에 메타데이터)
        marker_x.append(bd_t)
        marker_color.append(REGIME_HEX[new_lbl])
        marker_symbol.append('triangle-down' if new_lbl == BEAR else 'triangle-up')
        marker_hover.append(
            f"<b>Backdate event</b><br>"
            f"New label: {REGIME_NAMES[new_lbl]}<br>"
            f"Shift: {shift} bars<br>"
            f"Shock return: {shock:+.2%}<br>"
            f"HMM transition: {pd.Timestamp(orig_t).strftime('%Y-%m-%d %H:%M')}<br>"
            f"Backdated to: {pd.Timestamp(bd_t).strftime('%Y-%m-%d %H:%M')}"
        )

    if marker_x:
        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=[marker_y_const] * len(marker_x),
                mode='markers',
                marker=dict(
                    size=12,
                    color=marker_color,
                    symbol=marker_symbol,
                    line=dict(color='white', width=1),
                ),
                name='Backdate event',
                hovertext=marker_hover,
                hovertemplate='%{hovertext}<extra></extra>',
                showlegend=True,
            ),
            row=1, col=1,
        )

    # ── 범례에 라벨 색 (가짜 trace) ────────────────────────
    for lbl in [BULL, SIDE, BEAR]:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=REGIME_HEX[lbl], symbol='square'),
                name=REGIME_NAMES[lbl],
                showlegend=True,
            ),
            row=1, col=1,
        )

    # ── 레이아웃 ─────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f'<b>HMM Label Comparison: Original vs Smoothed vs Prediction</b><br>'
                f'<sub>Backdate events: {len(change_log)} | '
                f'lookback={args.lookback} bars, threshold={args.threshold:.0%}, '
                f'persistence={args.persistence} | '
                f'OOS: TimeSeriesSplit={args.ts_splits}-fold, gap={args.window_size}</sub>'
            ),
            x=0.5, xanchor='center',
        ),
        height=900,
        hovermode='closest',
        legend=dict(orientation='h', y=-0.05),
        margin=dict(l=60, r=30, t=90, b=40),
        plot_bgcolor='white',
    )
    fig.update_xaxes(rangeslider_visible=False, showgrid=True,
                     gridcolor='rgba(200,200,200,0.3)')
    fig.update_xaxes(title_text='Date', row=4, col=1)

    return fig


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── [1] 데이터 + 피처 ───────────────────────────────────
    print(f"[1/5] 데이터 로드 + 윈도우 피처")
    df = load_and_resample(args.csv_path, timeframe=args.timeframe)
    features = compute_window_features(
        df, window_size=args.window_size,
        adx_period=config.ADX_PERIOD, r2_period=config.R2_PERIOD,
    )
    # slope_norm — 다른 검증 스크립트와 일관성 위해 계산만 (메타 X엔 미사용)
    slope_scaler = RollingStandardScaler(window=args.rolling_window)
    features = features.copy()
    features['slope_norm'] = slope_scaler.fit_transform(
        features[['slope']].values
    ).flatten()

    bar_returns = df['close'].pct_change().values
    end_indices = features['window_end_idx'].astype(int).values
    last_bar_returns = bar_returns[end_indices]
    print(f"      → 윈도우: {len(features):,}")

    # ── [2] HMM 라벨 + smoother ────────────────────────────
    print(f"[2/5] HMM 라벨 로드 + smoother 적용")
    labeler = HMMLabeler()
    labeler.load(args.hmm_cache)
    transition_predictor = TransitionPredictor.from_labeler(labeler)

    X_hmm_raw = features[config.HMM_FEATURE_COLS].values
    hmm_scaler = RollingStandardScaler(window=args.rolling_window)
    X_hmm_scaled = hmm_scaler.fit_transform(X_hmm_raw)
    nan_mask_hmm = np.isnan(X_hmm_scaled).any(axis=1)
    X_hmm_safe = np.where(np.isnan(X_hmm_scaled), 0.0, X_hmm_scaled)
    hmm_proba = labeler.predict_proba(X_hmm_safe)
    hmm_proba[nan_mask_hmm] = 1 / 3

    labels_orig = np.argmax(hmm_proba, axis=1).astype(np.int64)

    smoother = RetrospectiveLabelSmoother(
        lookback=args.lookback,
        threshold=args.threshold,
        persistence_check=args.persistence,
        include_side=args.include_side,
    )
    labels_smooth, change_log = smoother.smooth(labels_orig, last_bar_returns)
    print(f"      → Backdate 이벤트: {len(change_log)}건")

    # ── [3] 메타 입력 X 구성 + OOS 예측 ────────────────────
    print(f"[3/5] 메타 입력 X (16 피처) + OOS 예측 (TimeSeriesSplit)")
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

    n = len(features)
    nan_mask_meta = np.isnan(X_meta).any(axis=1)
    hmm_cold = np.isclose(hmm_proba, 1 / 3, atol=1e-9).all(axis=1)

    # y[t] = labels_smooth[t+1] — 다음 시점의 (smoothed) 정답을 학습
    y_smooth_full = np.full(n, -1, dtype=np.int64)
    y_smooth_full[:-1] = labels_smooth[1:]
    y_nan = (y_smooth_full == -1)

    final_mask = ~(nan_mask_meta | y_nan | hmm_cold)
    X_filtered = X_meta[final_mask]
    y_filtered = y_smooth_full[final_mask]
    filtered_indices = np.where(final_mask)[0]
    print(f"      → 학습 가능: {X_filtered.shape}")

    pred_filtered = compute_meta_oos_predictions(
        X_filtered, y_filtered,
        n_splits=args.ts_splits, gap=args.window_size,
        C=args.C, feature_names=feature_names,
    )
    n_oos = (pred_filtered >= 0).sum()
    print(f"      → OOS 예측 가능: {n_oos:,} / {len(pred_filtered):,}")

    # 전체 timeline에 매핑 — pred_filtered[i]는 X_filtered[i]에서 만든 예측.
    # X_filtered[i]의 타깃은 labels_smooth[filtered_indices[i] + 1]이므로,
    # 시각화는 (filtered_indices[i] + 1) 위치에 그려서 "예측 vs 실제 라벨"이
    # 같은 시점에 정렬되도록 한다.
    meta_pred_full = np.full(n, -1, dtype=np.int64)
    for i, fi in enumerate(filtered_indices):
        if pred_filtered[i] < 0:
            continue
        target_idx = fi + 1
        if 0 <= target_idx < n:
            meta_pred_full[target_idx] = pred_filtered[i]

    # ── [4] Cold start 처리 ─────────────────────────────────
    if not args.include_coldstart:
        valid_idx = np.where(~nan_mask_hmm)[0]
        first_valid = int(valid_idx[0])
        last_valid = int(valid_idx[-1]) + 1
        features_use = features.iloc[first_valid:last_valid].reset_index(drop=True)
        labels_orig = labels_orig[first_valid:last_valid]
        labels_smooth = labels_smooth[first_valid:last_valid]
        meta_pred_full = meta_pred_full[first_valid:last_valid]
        # change_log 인덱스 보정
        adjusted_log = []
        n_use = len(features_use)
        for log in change_log:
            new = log.copy()
            new['original_t'] = log['original_t'] - first_valid
            new['backdated_to'] = log['backdated_to'] - first_valid
            if 0 <= new['backdated_to'] < n_use and 0 <= new['original_t'] < n_use:
                adjusted_log.append(new)
        change_log = adjusted_log
        first_time = pd.to_datetime(features_use['window_end_time'].iloc[0])
        df_use = df[pd.to_datetime(df['datetime']) >= first_time].reset_index(drop=True)
        print(f"      → Cold start ({first_valid}봉) 제외 후 표시: {n_use:,}봉")
    else:
        features_use = features
        df_use = df

    # ── [5] HTML 생성 ───────────────────────────────────────
    print(f"[5/5] HTML 생성")
    fig = build_figure(df_use, features_use, labels_orig, labels_smooth,
                       meta_pred_full, change_log, args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        include_plotlyjs='cdn',     # CDN으로 plotly.js 로드 (파일 경량화)
        full_html=True,
        config={'displaylogo': False, 'scrollZoom': True},
    )
    file_size_kb = out_path.stat().st_size / 1024
    print(f"      → 저장: {out_path} ({file_size_kb:.1f} KB)")
    print(f"\n✅ 브라우저에서 열어보세요: {out_path.resolve()}")


if __name__ == '__main__':
    main()
