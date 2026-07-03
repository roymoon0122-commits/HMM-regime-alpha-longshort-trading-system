"""
주식 HMM 통합 백테스트 스크립트.

────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일
────────────────────────────────────────────────────────────────────
1. 정규장 30분봉 parquet 로드 → train / test(워밍업 포함) 분리 (OOS)
2. 4개 HMM variant 학습 + EngineHMM 백테스트
   - variant_A: include_hmm_proba=True,  use_smoothed_labels=True
   - variant_B: include_hmm_proba=True,  use_smoothed_labels=False
   - variant_C: include_hmm_proba=False, use_smoothed_labels=True
   - variant_D: include_hmm_proba=False, use_smoothed_labels=False
3. 비교군 백테스트 — Donchian+ADX/R², MA Cross, Buy & Hold
4. 결과 표 출력 + 인터랙티브 HTML 차트

────────────────────────────────────────────────────────────────────
실행 예시 (프로젝트 루트에서)
────────────────────────────────────────────────────────────────────
  # 기본 (config.py 기본 종목 AAPL, OOS 2025-01-01 ~ 2026-05-22)
  python run_backtest_hmm.py

  # 다른 종목
  python run_backtest_hmm.py \\
    --csv-path data/30min/NVDA_20210101_20260523_30min.parquet \\
    --hmm-cache models/hmm_nvda.joblib \\
    --output-html backtest_hmm_NVDA.html

  # 빠른 확인 (HTML 없이 콘솔 표만)
  python run_backtest_hmm.py --no-visualize

주요 옵션:
  --csv-path        리샘플 완료된 parquet 경로 (기본: config.DATA_PATH)
  --train-start/end 학습 기간 (train-end 미지정 시 test-start 하루 전 자동)
  --test-start/end  OOS 백테스트 기간
  --warmup-bars     test_start 이전 워밍업 봉 수 (기본: config.WARMUP_BARS)
  --fee-rate        거래 비용 (기본 0.0003; Alpaca 주식 수수료 무료)
  --retrain-hmm     HMM 캐시 무시하고 재학습

────────────────────────────────────────────────────────────────────
OOS (Out-of-Sample) 흐름
────────────────────────────────────────────────────────────────────
[train_start ─ train_end]   학습 (HMMStrategy.fit)
[워밍업 WARMUP_BARS봉]       백테스트 데이터 앞쪽 (윈도우/scaler 형성용)
[test_start ─ test_end]     진짜 OOS 백테스트 (성능 측정)

워밍업 구간을 백테스트 데이터에 포함시키되, 통계/시각화는
test_start 이후만 사용한다.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.stock_loader import load_resampled_bars
from strategy.HMM_strategy.strategy import HMMStrategy

from strategy.donchian_adx_r2_B import DonchianADXR2Strategy
from strategy.ma_cross import MACrossStrategy

from backtester.engine import Engine
from backtester.backtester_hmm import EngineHMM
from backtester.report import Report

# 시각화
from backtester.visualizer_run_backtest_hmm import plot_hmm_backtest


# ════════════════════════════════════════════════════════════════
#  유틸리티
# ════════════════════════════════════════════════════════════════

def _asset_name_from_path(csv_path: str) -> str:
    """파일명 첫 토큰(첫 '_' 앞)을 차트 표시용 종목 심볼로 추출한다.

    예시:
      data/30min/AAPL_20210101_20260523_30min.parquet  →  'AAPL'
      data/30min/NVDA_20210101_20260523_30min.parquet  →  'NVDA'
    """
    return Path(csv_path).stem.split('_')[0]


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 통합 HMM 백테스트")

    # 자산 / 데이터
    p.add_argument('--csv-path', default=config.DATA_PATH,
                   help=f"리샘플 완료된 parquet 경로 (기본: {config.DATA_PATH})")
    p.add_argument('--asset-name', default=None,
                   help="차트 제목용 자산 이름 (기본: --csv-path 파일명에서 자동 추출)")
    p.add_argument('--timeframe', default=config.TIMEFRAME,
                   help=f"타임프레임 라벨 — 차트 표시·연율화용 (기본 {config.TIMEFRAME})")

    # 기간 (OOS 분리) — config.py에서 기본값 참조
    p.add_argument('--train-start', default=config.TRAIN_START)
    p.add_argument('--train-end',   default=None,
                   help="학습 종료일 (기본: test-start 하루 전으로 자동 설정). "
                        "명시하지 않으면 test-start 기준으로 자동 계산됨.")
    p.add_argument('--test-start',  default=config.TEST_START)
    p.add_argument('--test-end',    default=config.TEST_END)
    p.add_argument('--warmup-bars', type=int, default=config.WARMUP_BARS,
                   help=f"test_start 이전 워밍업 봉 수 "
                        f"(기본값 config.WARMUP_BARS={config.WARMUP_BARS}봉, 자동 계산). "
                        f"ROLLING_SCALER_WINDOW·WINDOW_SIZE·ADX/R2_PERIOD 변경 시 "
                        f"config.py에서 자동으로 재계산됨.")

    # HMM 캐시
    p.add_argument('--hmm-cache', default=config.HMM_MODEL_PATH,
                   help="HMM 라벨러 캐시 경로 (자산별로 분리)")
    p.add_argument('--retrain-hmm', action='store_true', default=config.FORCE_RETRAIN,
                   help="HMM 캐시 무시하고 새로 학습 (기본값: config.FORCE_RETRAIN)")

    # 백테스트 파라미터
    p.add_argument('--initial-capital', type=float, default=10_000.0)
    p.add_argument('--fee-rate', type=float, default=0.0003,
                   help="거래 비용 (기본 0.0003 = 슬리피지 0.03%; "
                        "Alpaca 미국 주식은 거래 수수료 무료)")
    p.add_argument('--rebalance-threshold', type=float,
                   default=config.REBALANCE_THRESHOLD)

    # 출력
    p.add_argument('--output-html', default=None,
                   help="시각화 HTML 출력 경로 (기본: 'backtest_hmm_{종목심볼}.html')")
    p.add_argument('--no-visualize', action='store_true',
                   help="HTML 시각화 생략 (콘솔 표만 출력)")

    args = p.parse_args()

    # --asset-name / --output-html 미지정 시 실제 --csv-path 기준으로 자동 설정
    # (config.DATA_PATH 하드코딩 방지 — TSLA·NVDA 등 다른 종목도 올바른 이름 표시)
    if args.asset_name is None:
        args.asset_name = _asset_name_from_path(args.csv_path)
    if args.output_html is None:
        _slug = args.asset_name.split('/')[0]  # 'BTC/USDT' → 'BTC'
        args.output_html = f'backtest_hmm_{_slug}.html'

    return args


# ════════════════════════════════════════════════════════════════
#  데이터 로드 & 분할
# ════════════════════════════════════════════════════════════════

def prepare_data(args):
    """
    리샘플 완료된 parquet 로드 → train / test(워밍업 포함) 분리.

    워밍업은 봉 개수(args.warmup_bars) 기준으로 자른다 — 주식은 거래일
    밀도가 캘린더 일수와 다르므로 날짜보다 봉 수가 정확하다.

    Returns:
        df_full:  train_start ~ test_end 전체 데이터
        df_train: train_start ~ train_end (학습용)
        df_test:  (test_start 직전 warmup_bars봉) ~ test_end (백테스트용)
        test_start_dt: pd.Timestamp — 진짜 OOS 시작
    """
    print(f"[데이터] 로드 ({args.csv_path})")

    df_full = load_resampled_bars(
        args.csv_path,
        start=pd.Timestamp(args.train_start),
        end=pd.Timestamp(args.test_end),
    )
    print(f"      → 전체 봉: {len(df_full):,} "
          f"({df_full['datetime'].iloc[0]} ~ {df_full['datetime'].iloc[-1]})")

    train_end_dt = pd.Timestamp(args.train_end)
    test_start_dt = pd.Timestamp(args.test_start)

    # 학습 데이터
    df_train = (df_full[df_full['datetime'] <= train_end_dt]
                .copy().reset_index(drop=True))

    # 백테스트 데이터 — test_start 직전 warmup_bars봉부터 test_end까지.
    # df_full은 test_end에서 끝나므로 위쪽만 잘라내면 된다.
    test_mask = df_full['datetime'] >= test_start_dt
    if not test_mask.any():
        raise ValueError(f"test_start({test_start_dt}) 이후 데이터가 없습니다.")
    test_first_idx = int(test_mask.idxmax())          # OOS 첫 봉 위치
    warmup_start_idx = test_first_idx - args.warmup_bars
    if warmup_start_idx < 0:
        print(f"      ⚠ 워밍업 봉 부족: 필요 {args.warmup_bars:,}, "
              f"확보 가능 {test_first_idx:,}. 앞쪽 데이터부터 사용.")
        warmup_start_idx = 0
    df_test = df_full.iloc[warmup_start_idx:].copy().reset_index(drop=True)

    print(f"      → 학습 (train): {len(df_train):,}봉 "
          f"({df_train['datetime'].iloc[0]} ~ {df_train['datetime'].iloc[-1]})")
    print(f"      → 백테스트 (워밍업 {test_first_idx - warmup_start_idx:,}봉 포함): "
          f"{len(df_test):,}봉 "
          f"({df_test['datetime'].iloc[0]} ~ {df_test['datetime'].iloc[-1]})")
    print(f"      → 진짜 OOS 시작: {test_start_dt}")

    return df_full, df_train, df_test, test_start_dt


# ════════════════════════════════════════════════════════════════
#  HMM variant 학습 + 백테스트
# ════════════════════════════════════════════════════════════════

# VARIANT_LIST 구조: (id, 라벨, include_hmm_proba, use_smoothed_labels, use_donchian_on_side)
#
# 위 2개 (A, B): HMM✓ + Donchian OFF (기존 알파 그대로)
# 아래 2개 (C, D): HMM✓ + Donchian ON  (SIDE 시점에 돈치안 시그널 × P(Side))
#
# 위↔아래 직접 A/B 비교 가능:
#   variant_A (Smooth✓)  ↔  variant_C (Smooth✓ + Donchian)
#   variant_B (Smooth✗)  ↔  variant_D (Smooth✗ + Donchian)
VARIANT_LIST = [
    ('variant_A', "HMM✓ Smooth✓",            True, True,  False),
    ('variant_B', "HMM✓ Smooth✗",            True, False, False),
    ('variant_C', "HMM✓ Smooth✓ + Donchian", True, True,  True),
    ('variant_D', "HMM✓ Smooth✗ + Donchian", True, False, True),
]


def run_hmm_variants(df_train, df_test, args):
    """4개 HMM variant 학습 + EngineHMM 백테스트."""
    results = {}

    for variant_id, variant_label, inc_hmm, use_smooth, use_donch in VARIANT_LIST:
        print(f"\n[HMM {variant_id}] {variant_label}")
        t0 = time.time()

        # variant마다 별도 인스턴스 (메타 모델 따로 학습)
        # 첫 번째 variant만 HMM 학습 (또는 캐시 로드), 이후는 같은 캐시 재사용
        # → from_config(overrides) 패턴으로 모든 변수 자동 적용
        strategy = HMMStrategy.from_config(
            include_hmm_proba=inc_hmm,
            use_smoothed_labels=use_smooth,
            use_donchian_on_side=use_donch,
            hmm_model_path=args.hmm_cache,
            verbose=False,
        )

        # 첫 variant이고 retrain 옵션이면 캐시 삭제 (강제 재학습)
        if variant_id == 'variant_A' and args.retrain_hmm:
            cache_path = Path(args.hmm_cache)
            if cache_path.exists():
                cache_path.unlink()
                print(f"      → 캐시 삭제 (--retrain-hmm): {cache_path}")

        # 학습 (df_train만 사용 — OOS 보장)
        print(f"      → 학습 중...")
        strategy.fit(df_train)
        diag = strategy._fit_diagnostics
        print(f"      → 학습 완료 ({time.time()-t0:.1f}초): "
              f"X_meta={diag['X_meta_shape']}, n_train={diag['n_train']:,}, "
              f"smoother_changes={diag['smoother_changes']}")

        # 백테스트 (df_test = 워밍업 + OOS 구간)
        print(f"      → 백테스트 중 (df_test {len(df_test):,}봉)...")
        t1 = time.time()
        engine = EngineHMM(
            strategy=strategy,
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            cooldown=0,
            rebalance_threshold=args.rebalance_threshold,
        )
        result = engine.run(df_test)
        print(f"      → 백테스트 완료 ({time.time()-t1:.1f}초)")

        # variant 정보 저장 (시각화에서 활용)
        # signals: float (-1~+1), strategy._fit_diagnostics에 진단 정보
        results[variant_id] = {
            'label':   variant_label,
            'equity':  result['equity_curve'],
            'datetime': df_test['datetime'].values,
            'signals': result['signals'],
            'trades':  result['trades'],
            'strategy': strategy,
            'diag':    diag,
        }

    return results


# ════════════════════════════════════════════════════════════════
#  비교군 백테스트
# ════════════════════════════════════════════════════════════════

def run_donchian(df_test, args):
    """DonchianADXR2Strategy 백테스트 (기존 Engine, int8 시그널)."""
    print(f"\n[비교군] Donchian + ADX/R²")
    t0 = time.time()
    strategy = DonchianADXR2Strategy(entry_period=120, exit_period=60)
    engine = Engine(
        strategy,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        cooldown=6,
    )
    result = engine.run(df_test)
    print(f"      → 백테스트 완료 ({time.time()-t0:.1f}초)")
    return {
        'label':    'Donchian + ADX/R²',
        'equity':   result['equity_curve'],
        'datetime': df_test['datetime'].values,
        'signals':  result['signals'].astype(np.float64),
        'trades':   result['trades'],
    }


def run_ma_cross(df_test, args):
    """MACrossStrategy 백테스트."""
    print(f"\n[비교군] MA Cross (20/60)")
    t0 = time.time()
    strategy = MACrossStrategy(fast_period=20, slow_period=60)
    engine = Engine(
        strategy,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        cooldown=0,
    )
    result = engine.run(df_test)
    print(f"      → 백테스트 완료 ({time.time()-t0:.1f}초)")
    return {
        'label':    'MA Cross (20/60)',
        'equity':   result['equity_curve'],
        'datetime': df_test['datetime'].values,
        'signals':  result['signals'].astype(np.float64),
        'trades':   result['trades'],
    }


def run_buy_hold(df_test, args):
    """Buy & Hold (수동 계산 — 첫 봉 시가 매수, 마지막 봉 종가 청산)."""
    print(f"\n[비교군] Buy & Hold")
    initial = args.initial_capital
    fee = args.fee_rate

    open_arr = df_test['open'].to_numpy(dtype=np.float64)
    close_arr = df_test['close'].to_numpy(dtype=np.float64)

    entry_price = open_arr[0]
    coin_qty = (initial * (1 - fee)) / entry_price
    equity = coin_qty * close_arr     # 매 봉 평가액
    # 마지막 봉에서 청산 수수료
    equity = equity.copy()
    equity[-1] = equity[-1] * (1 - fee)

    return {
        'label':    'Buy & Hold',
        'equity':   equity,
        'datetime': df_test['datetime'].values,
        'signals':  np.ones(len(df_test), dtype=np.float64),
        'trades':   [],
    }


# ════════════════════════════════════════════════════════════════
#  통계 계산 (OOS 구간만)
# ════════════════════════════════════════════════════════════════

def compute_oos_stats(equity, datetime_arr, trades, test_start_dt,
                      timeframe, initial_capital):
    """
    백테스트 결과의 OOS 구간 통계 (워밍업 제외).

    - test_start_dt 이전 봉 무시
    - test_start 시점의 자본을 새 baseline으로 잡고 stats 계산
    """
    dt_series = pd.Series(pd.to_datetime(datetime_arr))
    oos_mask = dt_series >= pd.Timestamp(test_start_dt)
    oos_idx = np.where(oos_mask.values)[0]
    if len(oos_idx) == 0:
        return None

    oos_start_idx = oos_idx[0]
    eq = equity[oos_start_idx:]
    dt = dt_series.iloc[oos_start_idx:].reset_index(drop=True)

    # OOS baseline (initial_capital 보정)
    baseline = float(eq[0])
    if baseline <= 0:
        # 워밍업에서 파산한 경우 (드물지만 가능)
        return {
            'oos_start_equity': 0.0,
            'oos_end_equity':   0.0,
            'total_return':     -100.0,
            'cagr':             -100.0,
            'sharpe':           0.0,
            'mdd':              -100.0,
            'total_trades':     0,
            'win_rate':         0.0,
            'avg_pnl_pct':      0.0,
        }

    final = float(eq[-1])
    total_return_pct = (final / baseline - 1) * 100

    days = (dt.iloc[-1] - dt.iloc[0]).days
    years = max(days / 365.25, 1e-6)
    cagr = ((final / baseline) ** (1 / years) - 1) * 100

    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    mdd = float(drawdown.min()) * 100

    rets = np.diff(eq) / eq[:-1]
    # 미국 주식 정규장 기준 연간 봉 수 (252거래일).
    #   30min → 13봉/일 × 252 = 3276,  1d → 252
    annual_bars_lookup = {'30min': 13 * 252, '1h': 7 * 252, '1d': 252,
                          '1m': 525_600, '5m': 105_120, '15m': 35_040,
                          '4h': 2_190}
    annual_bars = annual_bars_lookup.get(timeframe.lower(), 13 * 252)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(annual_bars)) if rets.std() > 0 else 0.0

    # 거래 통계 — OOS 구간 진입 시점만
    oos_trades = []
    for t in trades:
        if t.get('entry_dt') is None:
            continue
        if pd.Timestamp(t['entry_dt']) >= pd.Timestamp(test_start_dt):
            oos_trades.append(t)
    n = len(oos_trades)
    if n > 0:
        wins = sum(1 for t in oos_trades if t['pnl'] > 0)
        win_rate = wins / n * 100
        avg_pnl_pct = sum(t['pnl_pct'] for t in oos_trades) / n
    else:
        win_rate = 0.0
        avg_pnl_pct = 0.0

    return {
        'oos_start_equity': baseline,
        'oos_end_equity':   final,
        'total_return':     total_return_pct,
        'cagr':             cagr,
        'sharpe':           sharpe,
        'mdd':              mdd,
        'total_trades':     n,
        'win_rate':         win_rate,
        'avg_pnl_pct':      avg_pnl_pct,
    }


# ════════════════════════════════════════════════════════════════
#  콘솔 표 출력
# ════════════════════════════════════════════════════════════════

def print_summary_table(all_results, asset_name, timeframe, test_start, test_end):
    print()
    print("=" * 90)
    print(f"  백테스트 결과 비교 — {asset_name} ({timeframe}, OOS {test_start} ~ {test_end})")
    print("=" * 90)
    print(f"  {'전략':<32} {'CAGR':>8} {'Sharpe':>8} {'MDD':>9} {'Trades':>8} {'Win%':>8}")
    print("  " + "-" * 86)
    for label, stats in all_results:
        if stats is None:
            print(f"  {label:<32} {'(N/A — OOS 데이터 없음)':>50}")
            continue
        print(f"  {label:<32} "
              f"{stats['cagr']:>+7.2f}% "
              f"{stats['sharpe']:>8.2f} "
              f"{stats['mdd']:>+8.2f}% "
              f"{stats['total_trades']:>8,} "
              f"{stats['win_rate']:>7.2f}%")
    print("=" * 90)


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # train-end 자동 연동: 명시하지 않으면 test-start 하루 전으로 자동 설정
    if args.train_end is None:
        args.train_end = (
            pd.Timestamp(args.test_start) - pd.Timedelta(days=1)
        ).strftime('%Y-%m-%d')
        print(f"[기간] --train-end 미지정 → test-start({args.test_start}) 기준으로 "
              f"자동 설정: {args.train_end}")

    # 데이터 누수 방지 검증
    if pd.Timestamp(args.train_end) >= pd.Timestamp(args.test_start):
        raise ValueError(
            f"[오류] train_end({args.train_end})가 test_start({args.test_start}) 이상입니다.\n"
            f"       학습 기간과 테스트 기간이 겹치면 데이터 누수(look-ahead bias)가 발생합니다.\n"
            f"       train_end < test_start 가 되도록 설정해주세요."
        )

    # 1. 데이터 로드 & 분할
    df_full, df_train, df_test, test_start_dt = prepare_data(args)

    # 2. 4개 HMM variant
    print("\n" + "=" * 60)
    print("  HMM 4 variant 학습 + 백테스트")
    print("=" * 60)
    hmm_results = run_hmm_variants(df_train, df_test, args)

    # 3. 비교군
    print("\n" + "=" * 60)
    print("  비교군 백테스트 (Donchian / MA Cross / Buy&Hold)")
    print("=" * 60)
    benchmark_results = {
        'donchian': run_donchian(df_test, args),
        'ma_cross': run_ma_cross(df_test, args),
        'buy_hold': run_buy_hold(df_test, args),
    }

    # 4. OOS 통계 계산
    summary_for_print = []
    for vid, vlabel, _, _, _ in VARIANT_LIST:
        r = hmm_results[vid]
        stats = compute_oos_stats(
            r['equity'], r['datetime'], r['trades'],
            test_start_dt, args.timeframe, args.initial_capital,
        )
        r['stats'] = stats
        summary_for_print.append((f"HMM ({vlabel})", stats))

    for bid, blabel in [('donchian', 'Donchian + ADX/R²'),
                          ('ma_cross', 'MA Cross (20/60)'),
                          ('buy_hold', 'Buy & Hold')]:
        r = benchmark_results[bid]
        stats = compute_oos_stats(
            r['equity'], r['datetime'], r['trades'],
            test_start_dt, args.timeframe, args.initial_capital,
        )
        r['stats'] = stats
        summary_for_print.append((blabel, stats))

    # 5. 콘솔 표
    print_summary_table(
        summary_for_print,
        asset_name=args.asset_name,
        timeframe=args.timeframe,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    # 6. 시각화
    if not args.no_visualize:
        result_dict = {
            'asset_name':   args.asset_name,
            'timeframe':    args.timeframe,
            'warmup_start': df_test['datetime'].iloc[0],
            'test_start':   test_start_dt,
            'test_end':     pd.Timestamp(args.test_end),
            'initial_capital': args.initial_capital,
            'hmm_variants': hmm_results,
            'benchmark':    benchmark_results,
        }
        print(f"\n[시각화] HTML 생성 중: {args.output_html}")
        plot_hmm_backtest(result_dict, df_test, output_path=args.output_html)
        print(f"      → 완료. 브라우저에서 {args.output_html} 파일을 여세요.")


if __name__ == '__main__':
    main()
