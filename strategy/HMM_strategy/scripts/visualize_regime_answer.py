"""
HMM regime 분류 인터랙티브 시각화 — 주식 8종목용.

════════════════════════════════════════════════════════════════════
[TODO 2026-05-27] 이 시각화의 방향성을 전환할 예정.
════════════════════════════════════════════════════════════════════

목적 전환:
    변경 전(현재) — OOS Meta-model 예측 + HMM Viterbi 배경 음영 시각화.
    변경 후(예정) — Meta-model이 "학습하는 정답지(labels)"가 어떻게
                  만들어졌는지를 시각적으로 보여주는 도구.

배경:
    Meta-model의 학습 라벨은 RetrospectiveLabelSmoother의 backdate
    처리를 거친 Smoothed Labels이다. backdate 처리가 라벨에 어떤
    변화를 주는지(즉 HMM Viterbi 원본 라벨과 어디서·왜 달라지는지)
    가격선 위에 직접 겹쳐 보여줘서, 정답지의 질을 직관적으로 검증
    가능하게 한다.

변경 후 패널 구성 (총 2개 패널, 둘 다 가격선 포함):
    패널 1 — Backdate 미적용
            · 가격 그래프 + HMM Viterbi 원본 라벨 음영
            · 호버: 시각·가격·Viterbi 라벨·HMM 사후확률
    패널 2 — Backdate 적용
            · 가격 그래프 + Smoothed Labels 음영
            · 호버: 시각·가격·Smoothed 라벨·변경 이력(Backdate 이벤트 여부)

핵심 비교 포인트:
    두 패널의 음영을 위·아래로 나란히 보면 backdate가
    "어느 구간을 어떤 방향으로 보정했는지"가 한눈에 드러나야 한다.

실제 수정이 반영되면 이 TODO 블록은 삭제한다.
════════════════════════════════════════════════════════════════════


────────────────────────────────────────────────────────────────────
이 스크립트가 만드는 것 (현재 — 전환 전)
────────────────────────────────────────────────────────────────────
종목별 2-패널 Plotly HTML:

    [패널 1] 종가 라인 그래프
             - 배경: HMM 원본 Viterbi 라벨에 따라 음영 (Bull/Side/Bear)
             - 가격 선 위에 호버 → HMM proba, Original, Smoothed, Meta 표시
    [패널 2] Meta-model out-of-sample 예측 heatmap (얇은 띠)

색깔 매핑 (사용자 지정):
    Bull = 초록 (#2ecc71)
    Side = 회색 (#95a5a6)
    Bear = 빨강 (#e74c3c)
    예측 없음 (cold-start, 첫 fold) = 투명/공백

────────────────────────────────────────────────────────────────────
기존 visualize_labels_html.py 와의 차이
────────────────────────────────────────────────────────────────────
1. 데이터 로더: load_and_resample(코인 CSV) → load_resampled_bars(주식 parquet)
2. 4패널 → 2패널 (가격+배경음영 / Meta 예측)
3. Bull/Bear 색 매핑 반전 (코인은 Bull=빨강, 사용자가 Bull=초록 요구)
4. 가격 선 호버에 customdata로 4가지 라벨/확률 한 번에 표시
5. --symbol AAPL / --all 옵션으로 종목 선택

────────────────────────────────────────────────────────────────────
실행 방법
────────────────────────────────────────────────────────────────────
    # 프로젝트 루트에서
    python -m strategy.HMM_strategy.scripts.visualize_regime_answer --symbol AAPL

    # 8종목 일괄 생성
    python -m strategy.HMM_strategy.scripts.visualize_regime_answer --all

    # 출력 경로 변경
    python -m strategy.HMM_strategy.scripts.visualize_regime_answer \\
        --symbol NVDA --out results/regime_view_NVDA.html

    # cold start 구간도 포함해서 보기
    python -m strategy.HMM_strategy.scripts.visualize_regime_answer \\
        --symbol AAPL --include-coldstart
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit

from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.stock_loader import load_resampled_bars
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
#  색상 매핑 — 사용자 지정 (Bull=초록 / Side=회색 / Bear=빨강)
# ════════════════════════════════════════════════════════════════

REGIME_HEX = {
    BULL: '#2ecc71',   # 초록
    SIDE: '#95a5a6',   # 회색
    BEAR: '#e74c3c',   # 빨강
}

# Heatmap discrete colorscale —
# z=0 (Bull) → 0/3, z=1 (Side) → 1/3~2/3, z=2 (Bear) → 1
DISCRETE_COLORSCALE = [
    [0.0,     REGIME_HEX[BULL]],
    [1 / 3,   REGIME_HEX[BULL]],
    [1 / 3,   REGIME_HEX[SIDE]],
    [2 / 3,   REGIME_HEX[SIDE]],
    [2 / 3,   REGIME_HEX[BEAR]],
    [1.0,     REGIME_HEX[BEAR]],
]

# 종목 선택용 — config 와 data/30min 디렉토리 명명규칙 따름
DEFAULT_SYMBOLS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'SPY', 'TSLA']
DATA_DIR = 'data/30min'
DATA_SUFFIX = '_20210101_20260523_30min.parquet'


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="HMM regime 분류 인터랙티브 시각화 (주식)")

    # 종목 선택 — --symbol 또는 --all (둘 중 하나 필수)
    sym_group = p.add_mutually_exclusive_group(required=True)
    sym_group.add_argument('--symbol', type=str, default=None,
                           help="단일 종목 심볼 (예: AAPL)")
    sym_group.add_argument('--all', action='store_true',
                           help=f"8종목 일괄 처리: {DEFAULT_SYMBOLS}")

    p.add_argument('--data-dir', default=DATA_DIR,
                   help="리샘플된 30분봉 parquet 디렉토리")
    p.add_argument('--timeframe', default=config.TIMEFRAME)
    p.add_argument('--window-size', type=int, default=config.WINDOW_SIZE)
    p.add_argument('--rolling-window', type=int, default=config.ROLLING_SCALER_WINDOW)

    # HMM 모델 캐시 — 종목별로 동적 생성, --hmm-cache로 명시 가능
    p.add_argument('--hmm-cache', default=None,
                   help="HMM 모델 .joblib (기본: models/hmm_{symbol_lower}.joblib)")

    # Smoother
    p.add_argument('--lookback', type=int, default=config.LABEL_SMOOTHER_LOOKBACK)
    p.add_argument('--threshold', type=float, default=config.LABEL_SMOOTHER_THRESHOLD)
    p.add_argument('--persistence', type=int, default=config.LABEL_SMOOTHER_PERSISTENCE)
    p.add_argument('--include-side', action='store_true')

    # 메타 OOS
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--ts-splits', type=int, default=config.TS_SPLIT_N)

    # 출력
    p.add_argument('--include-coldstart', action='store_true',
                   help="cold start 구간도 표시 (기본 제외)")
    p.add_argument('--out', default=None,
                   help="단일 종목 출력 경로 (기본: results/regime_view_{SYMBOL}.html)")
    p.add_argument('--out-dir', default='results',
                   help="--all 일괄 모드 시 출력 디렉토리")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  메타 모델 OOS 예측 — TimeSeriesSplit
# ════════════════════════════════════════════════════════════════

def compute_meta_oos_predictions(X, y, n_splits, gap, C, feature_names):
    """
    TimeSeriesSplit 기반 out-of-sample 예측.
    visualize_labels_html.py 와 동일 로직.
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
#  배경 음영 — 같은 regime 연속 구간을 하나의 rectangle 로 병합
# ════════════════════════════════════════════════════════════════

def make_regime_segments(times, labels):
    """
    같은 라벨이 연속되는 구간을 (start_time, end_time, label) 로 묶는다.

    예: labels = [0, 0, 0, 1, 1, 2, 2, 2]
        → [(t0, t3, 0), (t3, t5, 1), (t5, t7, 2)]
        (끝 시점은 다음 세그먼트의 시작 시점, 마지막은 times[-1])

    Args:
        times: shape (n,) — datetime64 배열 (윈도우 끝 시각).
        labels: shape (n,) — 정수 라벨 (Bull=0, Side=1, Bear=2).

    Returns:
        list of (x0, x1, lbl) — 그릴 사각형 정보.
    """
    if len(labels) == 0:
        return []
    segments = []
    seg_start = 0
    cur_lbl = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != cur_lbl:
            # 세그먼트 종료 — 이전~현재 직전까지
            segments.append((times[seg_start], times[i], cur_lbl))
            seg_start = i
            cur_lbl = int(labels[i])
    # 마지막 세그먼트
    segments.append((times[seg_start], times[-1], cur_lbl))
    return segments


# ════════════════════════════════════════════════════════════════
#  Meta 예측 heatmap (패널 2)
# ════════════════════════════════════════════════════════════════

def make_meta_heatmap(times, labels, panel_name="Meta Prediction"):
    """1×N 짜리 Heatmap trace 생성 — Meta 예측용 패널."""
    z_row = labels.astype(float)
    z_row[labels < 0] = np.nan
    text_row = [
        REGIME_NAMES[int(l)] if l >= 0 else 'N/A' for l in labels
    ]
    return go.Heatmap(
        x=times,
        y=[panel_name],
        z=[z_row],
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
#  Figure 빌드 — 2-패널
# ════════════════════════════════════════════════════════════════

def build_figure(symbol, df, features, hmm_proba, labels_orig, labels_smooth,
                 meta_preds, args):
    """
    2-패널 Plotly Figure.

    Args:
        symbol: 종목 심볼 (제목용).
        df: 가격 DataFrame (datetime, open, high, low, close, volume).
        features: 윈도우 피처 DataFrame (window_end_time 포함).
        hmm_proba: shape (n_windows, 3) — HMM 사후확률.
        labels_orig: shape (n_windows,) — HMM Viterbi 원본.
        labels_smooth: shape (n_windows,) — smoother 보정.
        meta_preds: shape (n_windows,) — Meta OOS 예측 (-1 = 없음).
        args: argparse 결과.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.85, 0.15],
        subplot_titles=(
            f"{symbol} 가격 + HMM regime 분류 (배경 음영)",
            "Meta-model out-of-sample 예측",
        ),
    )

    # ── [1] 가격 데이터 + 윈도우 라벨 매핑 ────────────────────────
    df_close = df['close'].values
    df_times = pd.to_datetime(df['datetime']).values
    win_times = pd.to_datetime(features['window_end_time']).values

    # 가격 봉(전체)과 윈도우(워밍업 제외)는 행 수가 다르다.
    # 각 가격 봉 시점에 대해 "그 시점까지의 가장 최근 윈도우 끝 시각"의
    # 라벨/확률을 가져온다 (forward-fill 효과).
    # → pd.Series.reindex(method='ffill') 가 가장 간결.
    feat_idx = pd.DatetimeIndex(win_times)

    def _ffill_to_price(values, dtype=float, fill='N/A'):
        """윈도우 단위 배열을 가격 봉 시점에 맞춰 ffill."""
        s = pd.Series(values, index=feat_idx)
        # df_times 의 가장 가까운 과거 윈도우 끝 시각의 값을 가져옴
        aligned = s.reindex(pd.DatetimeIndex(df_times), method='ffill')
        if dtype == str:
            return aligned.fillna(fill).values
        return aligned.values  # NaN 그대로

    # 라벨/확률을 가격 봉 시점에 맞춰 정렬
    orig_names = np.array([REGIME_NAMES[int(l)] for l in labels_orig])
    smooth_names = np.array([REGIME_NAMES[int(l)] for l in labels_smooth])
    meta_names = np.array([
        REGIME_NAMES[int(l)] if l >= 0 else 'N/A' for l in meta_preds
    ])

    price_orig = _ffill_to_price(orig_names, dtype=str)
    price_smooth = _ffill_to_price(smooth_names, dtype=str)
    price_meta = _ffill_to_price(meta_names, dtype=str)
    price_p_bull = _ffill_to_price(hmm_proba[:, BULL].astype(float))
    price_p_side = _ffill_to_price(hmm_proba[:, SIDE].astype(float))
    price_p_bear = _ffill_to_price(hmm_proba[:, BEAR].astype(float))

    # customdata 로 묶기 — hovertemplate 에서 %{customdata[i]} 로 참조
    customdata = np.stack([
        price_orig, price_smooth, price_meta,
        price_p_bull, price_p_side, price_p_bear,
    ], axis=1)

    # ── [2] 패널 1 배경 음영 (regime 세그먼트) ──────────────────
    # 같은 regime 연속 구간을 하나의 rectangle 로 합쳐서 그린다.
    # labels_orig 기준 — HMM 의 "raw 분류 결과"를 시각화.
    segments = make_regime_segments(win_times, labels_orig)

    # 마지막 segment 가 단일 봉이면 zero-width 가 되므로 가격 마지막 시점까지 연장.
    if segments and len(df_times) > 0:
        last = segments[-1]
        segments[-1] = (last[0], df_times[-1], last[2])

    # 가격 y 범위 (rectangle 높이용)
    y_min = float(np.nanmin(df_close)) * 0.95
    y_max = float(np.nanmax(df_close)) * 1.05

    for x0, x1, lbl in segments:
        fig.add_shape(
            type='rect',
            xref='x', yref='y',
            x0=x0, x1=x1,
            y0=y_min, y1=y_max,
            fillcolor=REGIME_HEX[lbl],
            opacity=0.18,           # 가격 선 가독성 유지
            line=dict(width=0),
            layer='below',
            row=1, col=1,
        )

    # ── [3] 가격 라인 (호버에 모든 정보) ─────────────────────────
    hover_tmpl = (
        '<b>%{x|%Y-%m-%d %H:%M}</b><br>'
        '가격: $%{y:,.2f}<br>'
        '─────────────<br>'
        '<b>Original HMM:</b> %{customdata[0]}<br>'
        '<b>Smoothed:</b> %{customdata[1]}<br>'
        '<b>Meta 예측:</b> %{customdata[2]}<br>'
        '<i>HMM proba</i> Bull=%{customdata[3]:.0%} '
        'Side=%{customdata[4]:.0%} Bear=%{customdata[5]:.0%}'
        '<extra></extra>'
    )

    fig.add_trace(
        go.Scatter(
            x=df_times, y=df_close,
            mode='lines',
            name=f'{symbol} Close',
            line=dict(color='black', width=1.2),
            customdata=customdata,
            hovertemplate=hover_tmpl,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title_text=f'{symbol} 가격 ($)', row=1, col=1)

    # ── [4] 패널 2 — Meta 예측 heatmap ──────────────────────────
    fig.add_trace(make_meta_heatmap(win_times, meta_preds), row=2, col=1)
    fig.update_yaxes(showticklabels=False, fixedrange=True,
                     showgrid=False, row=2, col=1)

    # ── [5] 범례 (가짜 trace 로 색깔만) ─────────────────────────
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

    # ── [6] 레이아웃 ─────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f'<b>{symbol} — HMM regime 분류 시각화</b><br>'
                f'<sub>가격 그래프 배경 = HMM Viterbi 원본 라벨 | '
                f'하단 = Meta-model OOS 예측 | '
                f'window={args.window_size}, threshold={args.threshold:.0%}, '
                f'OOS={args.ts_splits}-fold</sub>'
            ),
            x=0.5, xanchor='center',
        ),
        height=820,
        hovermode='x unified',     # 같은 x에서 모든 정보 한 번에
        legend=dict(orientation='h', y=-0.05),
        margin=dict(l=60, r=30, t=90, b=40),
        plot_bgcolor='white',
    )
    fig.update_xaxes(rangeslider_visible=False, showgrid=True,
                     gridcolor='rgba(200,200,200,0.3)')
    fig.update_xaxes(title_text='Date', row=2, col=1)

    return fig


# ════════════════════════════════════════════════════════════════
#  종목 1개 처리 — 데이터 → 라벨 → 메타 OOS → HTML
# ════════════════════════════════════════════════════════════════

def process_symbol(symbol, args, out_path):
    """단일 종목 처리. 데이터 로드부터 HTML 저장까지."""

    # 종목별 경로 결정
    data_path = Path(args.data_dir) / f"{symbol}{DATA_SUFFIX}"
    if not data_path.exists():
        print(f"      ⚠️ {data_path} 없음 — 건너뜀")
        return False
    hmm_cache = args.hmm_cache or f"models/hmm_{symbol.lower()}.joblib"
    if not Path(hmm_cache).exists():
        print(f"      ⚠️ HMM 캐시 {hmm_cache} 없음 — 건너뜀")
        return False

    # ── [1] 데이터 로드 + 윈도우 피처 ────────────────────────────
    print(f"  [1/5] 데이터 로드 + 윈도우 피처 ({symbol})")
    df = load_resampled_bars(str(data_path))
    features = compute_window_features(
        df, window_size=args.window_size,
        adx_period=config.ADX_PERIOD, r2_period=config.R2_PERIOD,
    )
    slope_scaler = RollingStandardScaler(window=args.rolling_window)
    features = features.copy()
    features['slope_norm'] = slope_scaler.fit_transform(
        features[['slope']].values
    ).flatten()

    bar_returns = df['close'].pct_change().values
    end_indices = features['window_end_idx'].astype(int).values
    last_bar_returns = bar_returns[end_indices]
    print(f"        → 가격 {len(df):,}봉 / 윈도우 {len(features):,}개")

    # ── [2] HMM 라벨 + smoother ─────────────────────────────────
    print(f"  [2/5] HMM 라벨 로드 + smoother 적용")
    labeler = HMMLabeler()
    labeler.load(hmm_cache)
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
    print(f"        → Backdate 이벤트 {len(change_log)}건")

    # ── [3] 메타 입력 X + OOS 예측 ──────────────────────────────
    print(f"  [3/5] 메타 입력 X (16 피처) + OOS 예측")
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

    y_smooth_full = np.full(n, -1, dtype=np.int64)
    y_smooth_full[:-1] = labels_smooth[1:]
    y_nan = (y_smooth_full == -1)

    final_mask = ~(nan_mask_meta | y_nan | hmm_cold)
    X_filtered = X_meta[final_mask]
    y_filtered = y_smooth_full[final_mask]
    filtered_indices = np.where(final_mask)[0]

    pred_filtered = compute_meta_oos_predictions(
        X_filtered, y_filtered,
        n_splits=args.ts_splits, gap=args.window_size,
        C=args.C, feature_names=feature_names,
    )
    n_oos = (pred_filtered >= 0).sum()
    print(f"        → OOS 예측 {n_oos:,} / {len(pred_filtered):,}")

    # 전체 timeline 에 매핑 (예측 위치는 target_idx = filtered_indices + 1)
    meta_pred_full = np.full(n, -1, dtype=np.int64)
    for i, fi in enumerate(filtered_indices):
        if pred_filtered[i] < 0:
            continue
        target_idx = fi + 1
        if 0 <= target_idx < n:
            meta_pred_full[target_idx] = pred_filtered[i]

    # ── [4] Cold start 제외 ─────────────────────────────────────
    if not args.include_coldstart:
        valid_idx = np.where(~nan_mask_hmm)[0]
        if len(valid_idx) == 0:
            print(f"      ⚠️ {symbol} valid 윈도우 없음 — 건너뜀")
            return False
        first_valid = int(valid_idx[0])
        last_valid = int(valid_idx[-1]) + 1
        features_use = features.iloc[first_valid:last_valid].reset_index(drop=True)
        labels_orig = labels_orig[first_valid:last_valid]
        labels_smooth = labels_smooth[first_valid:last_valid]
        meta_pred_full = meta_pred_full[first_valid:last_valid]
        hmm_proba_use = hmm_proba[first_valid:last_valid]
        first_time = pd.to_datetime(features_use['window_end_time'].iloc[0])
        df_use = df[pd.to_datetime(df['datetime']) >= first_time].reset_index(drop=True)
        print(f"        → Cold start ({first_valid}봉) 제외, 표시 {len(features_use):,}윈도우")
    else:
        features_use = features
        df_use = df
        hmm_proba_use = hmm_proba

    # ── [5] HTML 생성 ───────────────────────────────────────────
    print(f"  [4/5] Figure 빌드")
    fig = build_figure(
        symbol=symbol,
        df=df_use,
        features=features_use,
        hmm_proba=hmm_proba_use,
        labels_orig=labels_orig,
        labels_smooth=labels_smooth,
        meta_preds=meta_pred_full,
        args=args,
    )

    print(f"  [5/5] HTML 저장: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        include_plotlyjs='cdn',
        full_html=True,
        config={'displaylogo': False, 'scrollZoom': True},
    )
    file_size_kb = out_path.stat().st_size / 1024
    print(f"        → {out_path} ({file_size_kb:,.1f} KB)")
    return True


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # 종목 리스트 결정
    if args.all:
        symbols = DEFAULT_SYMBOLS
    else:
        symbols = [args.symbol.upper()]

    ok_count = 0
    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] === {sym} ===")
        # 출력 경로 결정
        if args.out and len(symbols) == 1:
            out_path = Path(args.out)
        else:
            out_path = Path(args.out_dir) / f"regime_view_{sym}.html"

        try:
            if process_symbol(sym, args, out_path):
                ok_count += 1
        except Exception as e:
            print(f"  ❌ {sym} 처리 실패: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ 완료: {ok_count}/{len(symbols)} 종목 HTML 생성됨")
    if ok_count > 0:
        print(f"   결과 위치: {args.out_dir}/regime_view_*.html")


if __name__ == '__main__':
    main()
