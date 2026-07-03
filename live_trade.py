"""
live_trade.py — HMM 알파 Alpaca 페이퍼 트레이딩 라이브 실행

────────────────────────────────────────────────────────────────────
하는 일 (한 사이클)
────────────────────────────────────────────────────────────────────
1. Alpaca 페이퍼 계좌 연결 (.env 의 API 키 사용)
2. LIVE_SYMBOLS 각 종목:
   - 저장된 30분봉 parquet(과거) + Alpaca 최신 분봉(갭) → 전체 30분봉
   - HMM 전략으로 현재 시점 시그널(비중 -1.0 ~ +1.0) 계산
3. allocation × signal 로 raw portfolio weight 계산
4. 오늘 이전 일별 수익률만 써서 종목별 SPY beta 추정 후 순베타 캡 적용
5. 조정 weight → 목표 포지션(주식 수)
6. 현재 Alpaca 포지션과 비교 → 차이만큼 시장가 주문
7. 사이클 결과(시그널·주문·자산)를 logs/live_log.csv 에 기록

────────────────────────────────────────────────────────────────────
실행 모드
────────────────────────────────────────────────────────────────────
  python live_trade.py --once              # 1회 실행 후 종료 (기본, dry-run)
  python live_trade.py --loop              # 30분마다 반복 (dry-run)
  python live_trade.py --loop --execute    # 30분마다 반복 + 실제 주문 제출

  --dry-run (기본 ON): 주문을 제출하지 않고 "낼 주문"만 출력해 검증.
  --execute          : dry-run 해제 → 실제 페이퍼 계좌에 주문 제출.

────────────────────────────────────────────────────────────────────
안전장치 / 설계 메모
────────────────────────────────────────────────────────────────────
- 페이퍼 계좌(paper=True)에만 연결한다. 실제 돈 아님.
- 기본이 dry-run 이라, 먼저 출력만 보고 검증한 뒤 --execute 로 전환.
- HMM 은 매 실행 시작 시 전체 과거 데이터로 새로 학습한다(캐시 미사용).
- 시그널은 "마지막으로 완성된 30분봉" 기준 — 미완성 봉은 제외.
- 리밸런싱 임계값(config.REBALANCE_THRESHOLD) 미만 변화는 거래 스킵.
- 포지션 부호가 바뀌면(롱↔숏) 청산 주문 + 신규 주문 2건으로 분리.
- 최신 분봉은 Alpaca IEX 피드(실시간·무료)로 받는다.
"""

import argparse
import atexit
import csv
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderStatus
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

from strategy.HMM_strategy import config
from strategy.HMM_strategy import allocations
from strategy.HMM_strategy.strategy import HMMStrategy
from strategy.HMM_strategy.features.stock_loader import (
    load_resampled_bars, resample_bars_df,
)

# ════════════════════════════════════════════════════════════════
#  설정 — 기본값은 config.py 에서 가져온다 (한 곳에서 튜닝).
# ════════════════════════════════════════════════════════════════
SYMBOLS   = config.LIVE_SYMBOLS
MARKET_TZ = config.MARKET_TZ
DATA_DIR  = config.LIVE_DATA_DIR
SETTLE_BUFFER_SEC = config.SETTLE_BUFFER_SEC   # 30분봉 마감 후 데이터 정착 대기 (IEX)
BAR_MINUTES = config.BAR_MINUTES
BETA_BENCHMARK_SYMBOL = getattr(config, "LIVE_BETA_BENCHMARK_SYMBOL", "SPY")
LIVE_MODEL_DIR = Path(getattr(config, "LIVE_MODEL_DIR", "models/live"))

# 거래·시그널 로그 (매 사이클 결과를 CSV로 누적)
LOG_PATH   = "logs/live_log.csv"
LOG_FIELDS = ["timestamp", "symbol", "signal", "price", "pos_before",
              "target", "action", "order_delta", "equity", "mode", "note"]

# 순베타 캡 사이클 로그. 기존 live_log.csv 헤더를 깨지 않기 위해 별도 파일로 둔다.
BETA_LOG_PATH = "logs/live_beta_log.csv"
BETA_LOG_FIELDS = [
    "timestamp", "raw_net_beta", "adjusted_net_beta", "cap", "capped",
    "scale", "beta_symbols", "missing_beta_symbols", "mode", "note",
]

# 주문 전 guard 결과. no-trade도 정상 의사결정이므로 별도 로그로 남긴다.
GUARD_LOG_PATH = "logs/live_guard_log.csv"
GUARD_LOG_FIELDS = [
    "cycle_id", "timestamp", "mode", "guard", "passed", "reason", "details",
]

RETRAIN_LOG_PATH = "logs/live_retrain_log.csv"
RETRAIN_LOG_FIELDS = [
    "timestamp", "run_id", "status", "symbols", "trained_through_date", "note",
]


@dataclass
class LiveDataStatus:
    """한 심볼의 최신 데이터 fetch/freshness 진단 정보."""

    symbol: str
    request_needed: bool = False
    request_attempted: bool = False
    request_succeeded: bool = False
    fetch_failed: bool = False
    fetch_error: str = ""
    bars_empty: bool = False
    resample_empty: bool = False
    recent_rows: int = 0
    completed_recent_rows: int = 0
    last_bar: object = None
    request_start_utc: object = None
    request_end_utc: object = None


# ════════════════════════════════════════════════════════════════
#  연결 / 준비
# ════════════════════════════════════════════════════════════════

def connect():
    """Alpaca 페이퍼 트레이딩 + 데이터 클라이언트 생성."""
    load_dotenv()
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("[오류] .env 에 ALPACA_API_KEY / ALPACA_SECRET_KEY 가 없습니다.")
    trading = TradingClient(key, sec, paper=True)
    data = StockHistoricalDataClient(key, sec)
    return trading, data


def _parquet_path(symbol: str) -> str:
    """data/30min/ 에서 종목 parquet 경로 찾기."""
    from pathlib import Path
    files = sorted(Path(DATA_DIR).glob(f"{symbol}_*_30min.parquet"))
    if not files:
        raise FileNotFoundError(f"{DATA_DIR}/{symbol}_*_30min.parquet 없음")
    # 파일명 끝의 종료일(YYYYMMDD)이 사전식=시간순이므로 마지막 = 가장 최신.
    return str(files[-1])


def build_strategies(lookback_years: int = 5):
    """LIVE_SYMBOLS 각각 HMM 전략을 최근 lookback_years년 데이터로 학습.

    Args:
        lookback_years: HMM 학습에 쓸 최근 연수. 0 이하이면 가용한 전체 과거.

    Returns:
        strategies: {symbol: 학습된 HMMStrategy}
        histories:  {symbol: 학습/워밍업에 쓸 30분봉 DataFrame}
    """
    cutoff = None
    if lookback_years and lookback_years > 0:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)

    strategies, histories = {}, {}
    for sym in SYMBOLS:
        hist = load_resampled_bars(_parquet_path(sym))
        if cutoff is not None:
            hist = hist[hist["datetime"] >= cutoff].reset_index(drop=True)
        span = (f"{hist['datetime'].iloc[0].date()} ~ "
                f"{hist['datetime'].iloc[-1].date()}")
        print(f"  [{sym}] {len(hist):,}봉 ({span}) — HMM 학습 ...", flush=True)
        # hmm_model_path=None → 캐시 없이 항상 최신 데이터로 새로 학습
        strat = HMMStrategy.from_config(hmm_model_path=None, verbose=False)
        strat.fit(hist)
        strategies[sym] = strat
        histories[sym] = hist
    print(f"  → {len(strategies)}개 종목 전략 준비 완료\n")
    return strategies, histories


def retrain_strategies(strategies, histories, data_client, lookback_years=5):
    """루프 도중 호출되는 재학습. 최신 분봉까지 포함해 HMM을 새로 fit 한다.

    build_strategies()는 프로그램 시작 시 parquet 기준으로 1회 학습하지만,
    이 함수는 get_full_df()로 Alpaca 최신 완성봉을 붙여 '오늘까지의 데이터'로
    다시 학습한다. strategies/histories 딕셔너리를 제자리(in-place)로 갱신한다.

    Args:
        lookback_years: 학습에 쓸 최근 연수. 0 이하이면 가용한 전체 과거.
    """
    cutoff = None
    if lookback_years and lookback_years > 0:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)

    print("  [재학습] 종목별 HMM을 최신 데이터로 다시 학습 ...", flush=True)
    for sym in SYMBOLS:
        try:
            # 최신 완성봉까지 붙인 전체 df. histories[sym]도 갱신됨.
            full, histories[sym] = get_full_df(sym, data_client, histories[sym])
            train_df = full
            if cutoff is not None:
                train_df = full[full["datetime"] >= cutoff].reset_index(drop=True)
            span = (f"{train_df['datetime'].iloc[0].date()} ~ "
                    f"{train_df['datetime'].iloc[-1].date()}")
            strat = HMMStrategy.from_config(hmm_model_path=None, verbose=False)
            strat.fit(train_df)
            strategies[sym] = strat
            print(f"    [{sym}] {len(train_df):,}봉 ({span}) 재학습 완료", flush=True)
        except Exception as exc:
            # 한 종목 실패해도 기존 모델을 유지하고 계속 — 거래 중단 방지
            print(f"    [{sym}] 재학습 실패: {exc} → 기존 모델 유지")
    print("  [재학습] 완료\n", flush=True)


# ════════════════════════════════════════════════════════════════
#  데이터: 과거 parquet + 최신 분봉 → 전체 30분봉
# ════════════════════════════════════════════════════════════════

def _full_df_return(full_df, adjusted_history_df, status, return_status):
    if return_status:
        return full_df, adjusted_history_df, status
    return full_df, adjusted_history_df


def get_full_df(symbol, data_client, history_df, return_status=False):
    """과거 30분봉 + Alpaca 최신 분봉(갭) → 마지막 완성봉까지의 전체 df.

    Returns:
        (full_df, adjusted_history_df)
        또는 return_status=True 이면 (full_df, adjusted_history_df, status)
        - full_df          : 신호 계산에 쓸 전체 DataFrame
        - adjusted_history : 분할이 감지되면 소급 조정된 history_df,
                             아니면 입력받은 history_df 그대로.
                             호출자(run_cycle)가 histories[sym]을 갱신하는 데 사용.
        - status           : 최신 데이터 fetch/freshness 진단 정보.
    """
    last_hist = pd.Timestamp(history_df["datetime"].iloc[-1])      # naive ET
    start_utc = (pd.Timestamp(last_hist, tz=MARKET_TZ).tz_convert("UTC")
                 + pd.Timedelta(minutes=BAR_MINUTES))
    now_utc = pd.Timestamp.now(tz="UTC")
    status = LiveDataStatus(
        symbol=symbol,
        last_bar=last_hist,
        request_start_utc=start_utc,
        request_end_utc=now_utc,
    )

    if start_utc >= now_utc:
        return _full_df_return(
            history_df, history_df, status, return_status)   # 갱신할 새 데이터 없음

    # Alpaca 최신 분봉 요청 (IEX 실시간 피드)
    status.request_needed = True
    status.request_attempted = True
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_utc.to_pydatetime(),
            end=now_utc.to_pydatetime(),
            feed="iex",
            adjustment=Adjustment.SPLIT,   # ★ 역사 parquet과 가격 연속성 유지
        )
        bars = data_client.get_stock_bars(req)
        status.request_succeeded = True
    except Exception as exc:
        status.fetch_failed = True
        status.fetch_error = str(exc)
        print(f"    [{symbol}] 최신 분봉 요청 실패: {exc} → 과거 데이터만 사용")
        return _full_df_return(history_df, history_df, status, return_status)

    if bars.df is None or len(bars.df) == 0:
        status.bars_empty = True
        return _full_df_return(history_df, history_df, status, return_status)

    try:
        recent = resample_bars_df(bars.df, symbol=symbol, timeframe="30min",
                                  rth_only=True)
    except ValueError:
        # 받은 분봉이 전부 장외(프리/애프터장)이거나 비어 있음
        # (휴장일·주말 직후·장 시작 전 실행 등) → 새 완성봉 없음 → 과거만 사용
        status.resample_empty = True
        return _full_df_return(history_df, history_df, status, return_status)

    # 미완성(진행 중) 마지막 봉 제외 — 마감 30분이 지난 봉만 사용
    now_et = pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None)
    recent = recent[
        recent["datetime"] + pd.Timedelta(minutes=BAR_MINUTES) <= now_et
    ]
    status.recent_rows = len(recent)
    if len(recent) == 0:
        return _full_df_return(history_df, history_df, status, return_status)

    # ── 분할/역분할 경계 감지 및 소급 보정 ───────────────────────────
    adjusted_history, split_detected, _ = detect_split_adjust(
        history_df, recent, symbol
    )

    full = pd.concat([adjusted_history, recent], ignore_index=True)
    full = (full.drop_duplicates(subset="datetime", keep="last")
                .sort_values("datetime").reset_index(drop=True))
    status.completed_recent_rows = len(recent)
    status.last_bar = pd.Timestamp(full["datetime"].iloc[-1])
    return _full_df_return(full, adjusted_history, status, return_status)


# ════════════════════════════════════════════════════════════════
#  액면분할 / 역분할 런타임 감지 및 소급 보정
# ════════════════════════════════════════════════════════════════

def detect_split_adjust(
    history_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    symbol: str,
) -> tuple:
    """parquet 마지막 봉 vs live 첫 봉 가격 비율로 분할/역분할 자동 감지.

    adjustment=Adjustment.SPLIT을 써도, parquet 저장 이후에 새로운 분할이
    생기면 parquet(구 조정 기준) vs live(최신 조정 기준) 사이에 불연속이
    발생한다. 이 함수는 그 경계 비율로 분할 배수를 추정하고, history_df를
    소급 조정한 복사본을 반환한다.

    반환값:
        (adjusted_history_df, split_occurred: bool, split_ratio: float)
        - split_occurred=False 면 history_df 원본 그대로 반환
        - split_ratio: 분할이면 <1 (예: 10:1 분할 → 0.1),
                       역분할이면 >1 (예: 1:10 역분할 → 10)
    """
    if recent_df.empty:
        return history_df, False, 1.0

    last_close  = float(history_df["close"].iloc[-1])
    first_open  = float(recent_df["open"].iloc[0])

    if last_close <= 0 or first_open <= 0:
        return history_df, False, 1.0

    ratio = first_open / last_close   # 1.0 = 정상, <0.5 = 분할 의심, >2 = 역분할 의심

    SPLIT_THRESHOLD         = 0.5    # 50% 이상 하락 → 분할로 판단
    REVERSE_SPLIT_THRESHOLD = 2.0   # 2배 이상 상승 → 역분할로 판단

    if ratio < SPLIT_THRESHOLD:
        # 분할: 역사 가격을 split_ratio 로 나눠서 현재 기준으로 낮춤
        estimated = round(1.0 / ratio)        # 예: ratio=0.1 → 10:1 분할
        actual_ratio = 1.0 / estimated        # 0.1
        print(f"  ★ [분할 감지] {symbol}: {last_close:.2f} → {first_open:.2f} "
              f"(약 {estimated}:1 분할) — 역사 {len(history_df):,}봉 소급 조정")
        print(f"    → parquet 재다운로드 권장: python data/1_fetch_minute_bars.py")
        adjusted = history_df.copy()
        for col in ("open", "high", "low", "close"):
            adjusted[col] = adjusted[col] * actual_ratio   # ÷ estimated
        return adjusted, True, actual_ratio

    elif ratio > REVERSE_SPLIT_THRESHOLD:
        # 역분할: 역사 가격을 reverse_ratio 로 곱해서 현재 기준으로 높임
        estimated = round(ratio)              # 예: ratio=10 → 1:10 역분할
        print(f"  ★ [역분할 감지] {symbol}: {last_close:.2f} → {first_open:.2f} "
              f"(약 1:{estimated} 역분할) — 역사 {len(history_df):,}봉 소급 조정")
        print(f"    → parquet 재다운로드 권장: python data/1_fetch_minute_bars.py")
        adjusted = history_df.copy()
        for col in ("open", "high", "low", "close"):
            adjusted[col] = adjusted[col] * float(estimated)
        return adjusted, True, float(estimated)

    return history_df, False, 1.0


# ════════════════════════════════════════════════════════════════
#  주문 계획
# ════════════════════════════════════════════════════════════════

def plan_orders(current_shares: int, target_shares: int):
    """현재→목표 포지션 전환 주문 목록. 부호 반전 시 청산+신규 2건.

    Returns:
        [(signed_qty, kind), ...]
          - signed_qty : 부호 있는 정수 (양수=매수, 음수=매도)
          - kind       : 'close'(청산) | 'entry'(신규 진입) | 'adjust'(증감)
        부호 반전(롱↔숏)일 때만 ['close', 'entry'] 2건으로 나뉜다.
        호출자는 'close' 가 체결된 뒤에야 'entry' 를 제출해야 한다
        (wait_for_fill 사용). 같은 부호 증감은 'adjust' 1건.
    """
    if current_shares == target_shares:
        return []
    crosses_zero = (current_shares != 0 and target_shares != 0
                    and (current_shares > 0) != (target_shares > 0))
    if crosses_zero:
        # 1) 현재 포지션 청산  2) 목표 포지션 신규 진입
        return [(-current_shares, "close"), (target_shares, "entry")]
    return [(target_shares - current_shares, "adjust")]


# 재시도 대상으로 삼을 '전송 계층' 예외만 모은다.
# Alpaca 의 API 거부(APIError: wash trade·수량부족 등)는 여기 포함되지 않으므로
# 재시도 없이 즉시 전파된다. ConnectionResetError/Timeout 등은 requests 가
# RequestException 으로 감싸지만, 혹시 모를 raw 소켓 예외도 함께 잡는다.
_TRANSIENT_NET_ERRORS = (
    requests.exceptions.RequestException,   # ConnectionError, ReadTimeout 등 포함
    ConnectionError,
    ConnectionResetError,
    TimeoutError,
)


def _find_existing_order(trading_client, client_order_id):
    """client_order_id 로 이미 접수된 주문을 조회. 없거나 조회 실패면 None."""
    try:
        return trading_client.get_order_by_client_id(client_order_id)
    except Exception:
        return None


def submit_order(trading_client, symbol, signed_qty, dry_run,
                 client_order_id=None):
    """부호 있는 수량으로 시장가 주문 (dry_run 이면 출력만).

    연결오류(전송 계층) 발생 시 지수 백오프로 최대 ORDER_MAX_RETRIES 회 재시도.
    멱등성은 client_order_id 로 보장한다 — 첫 시도가 서버에 접수됐다면 같은
    id 의 재제출은 거부되고, 재시도 직전 조회로도 이중 체결을 막는다.
    wash trade·수량부족 같은 APIError 는 재시도하지 않고 즉시 전파(상위에서
    '실패' 로 기록).

    Returns:
        제출된 Order 객체. dry_run 이거나 qty==0 이면 None.
    """
    qty = abs(int(signed_qty))
    if qty == 0:
        return None
    side = OrderSide.BUY if signed_qty > 0 else OrderSide.SELL
    if dry_run:
        print(f"      [DRY-RUN] {symbol:6s} {side.value:4s} {qty}주")
        return None

    # 이 논리적 주문 1건에 고유 멱등 키 부여(재시도 간 공유, 사이클 간 고유).
    if client_order_id is None:
        client_order_id = f"hmm-{symbol}-{side.value}-{uuid.uuid4().hex[:12]}"
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=side,
                             time_in_force=TimeInForce.DAY,
                             client_order_id=client_order_id)

    max_tries = getattr(config, "ORDER_MAX_RETRIES", 3)
    backoff0 = getattr(config, "ORDER_RETRY_BACKOFF_SEC", 0.5)
    last_exc = None
    for attempt in range(1, max_tries + 1):
        # 2번째 시도부터는, 직전 시도가 실제로 접수됐는지 먼저 확인(이중 안전장치).
        if attempt > 1:
            existing = _find_existing_order(trading_client, client_order_id)
            if existing is not None:
                print(f"      [재시도] {symbol:6s} 직전 주문 접수 확인 "
                      f"→ 재제출 생략 ({existing.status})")
                return existing
        try:
            order = trading_client.submit_order(req)
            print(f"      [주문제출] {symbol:6s} {side.value:4s} {qty}주 "
                  f"→ {order.status}")
            return order
        except _TRANSIENT_NET_ERRORS as exc:
            last_exc = exc
            if attempt < max_tries:
                backoff = backoff0 * (2 ** (attempt - 1))
                print(f"      [재시도] {symbol:6s} 연결오류 "
                      f"{attempt}/{max_tries} → {backoff:.1f}s 후 재시도")
                time.sleep(backoff)
            else:
                print(f"      [재시도] {symbol:6s} 연결오류 "
                      f"{max_tries}회 모두 실패 → 이번 사이클 주문 누락")
    # 모든 재시도 실패: 상위 run_cycle 의 except 가 '실패' 로 기록(현행 유지).
    raise last_exc


def cancel_open_orders_for_symbols(trading_client, symbols,
                                   timeout_sec, poll_sec):
    """우리 심볼의 미체결 주문만 취소하고, 모두 정리될 때까지 폴링한다.

    계좌 전체 취소(cancel_orders)를 쓰지 않는 이유: 같은 페이퍼 계좌에서
    다른 알파/전략을 돌릴 미래를 대비해, 이 전략의 종목 주문만 건드린다.

    Returns:
        True  : 조회/취소/정리 완료
        False : 조회 실패, 취소 실패, timeout 내 정리 실패
    """
    sym_list = list(symbols)
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=sym_list)
        open_orders = trading_client.get_orders(filter=req)
    except Exception as exc:
        print(f"  [취소] 미체결 조회 실패: {exc} → 취소 생략")
        return False

    if not open_orders:
        return True

    print(f"  [취소] 우리 심볼 미체결 {len(open_orders)}건 취소 시도 ...")
    cancel_failed = False
    for o in open_orders:
        try:
            trading_client.cancel_order_by_id(o.id)
        except Exception as exc:
            cancel_failed = True
            print(f"    주문 {o.id} 취소 실패: {exc}")

    if cancel_failed and getattr(config, "LIVE_CANCEL_MUST_SETTLE", True):
        return False

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=sym_list)
            remaining = trading_client.get_orders(filter=req)
        except Exception as exc:
            print(f"  [취소] 미체결 재조회 실패: {exc}")
            return False
        if not remaining:
            print("  [취소] 미체결 정리 완료")
            return True
        time.sleep(poll_sec)
    print(f"  [취소] 경고: {timeout_sec:.0f}초 내 정리 미완 — 이번 cycle 주문 중단")
    return False


def wait_for_fill(trading_client, order_id, timeout_sec, poll_sec):
    """order_id 가 체결(FILLED)될 때까지 폴링.

    Returns:
        True  : 완전 체결됨
        False : 타임아웃 또는 취소/거부/만료 등 비체결 종료
    """
    dead = (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            o = trading_client.get_order_by_id(order_id)
        except Exception:
            time.sleep(poll_sec)
            continue
        if o.status == OrderStatus.FILLED:
            return True
        if o.status in dead:
            return False
        time.sleep(poll_sec)
    return False


# ════════════════════════════════════════════════════════════════
#  한 사이클
# ════════════════════════════════════════════════════════════════

def _append_log(rows):
    """사이클 결과 행들을 logs/live_log.csv 에 추가한다 (헤더 자동 생성)."""
    from pathlib import Path
    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _append_beta_log(row):
    """사이클 순베타 요약을 logs/live_beta_log.csv 에 추가한다."""
    from pathlib import Path
    path = Path(BETA_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BETA_LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _append_guard_log(rows):
    """주문 전 guard 결과를 logs/live_guard_log.csv 에 추가한다."""
    if not rows:
        return
    from pathlib import Path
    path = Path(GUARD_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GUARD_LOG_FIELDS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_retrain_log(row):
    """EOD 학습 결과를 logs/live_retrain_log.csv 에 추가한다."""
    path = Path(RETRAIN_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RETRAIN_LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _base_log_row(stamp, sym, equity, mode):
    return {
        "timestamp": stamp,
        "symbol": sym,
        "signal": "",
        "price": "",
        "pos_before": "",
        "target": "",
        "action": "",
        "order_delta": 0,
        "equity": "" if equity is None else round(float(equity), 2),
        "mode": mode,
        "note": "",
    }


def _mark_cycle_no_trade(log_rows, stamp, mode, equity, reason):
    """아직 row가 없는 심볼을 no-trade로 채워 cycle 전체 중단을 기록한다."""
    for sym in SYMBOLS:
        if sym not in log_rows:
            row = _base_log_row(stamp, sym, equity, mode)
            row["action"] = "거래금지"
            row["note"] = reason[:200]
            log_rows[sym] = row
        elif log_rows[sym].get("action") not in ("실패",):
            log_rows[sym]["action"] = "거래금지"
            note = log_rows[sym].get("note", "")
            combined = f"{note}; {reason}" if note else reason
            log_rows[sym]["note"] = combined[:200]


def _finish_cycle_log(log_rows, beta_logged=False):
    ordered_rows = [log_rows[sym] for sym in SYMBOLS if sym in log_rows]
    _append_log(ordered_rows)
    print(f"\n  → logs/live_log.csv 에 {len(ordered_rows)}건 기록")
    if beta_logged:
        print(f"  → logs/live_beta_log.csv 에 순베타 요약 1건 기록")
    print("=" * 78 + "\n")


def _fmt_optional(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def _to_et_aware(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(MARKET_TZ)
    return t.tz_convert(MARKET_TZ)


def _to_et_naive(ts):
    if ts is None:
        return None
    return _to_et_aware(ts).tz_localize(None)


def _clock_timestamp_et(clock):
    ts = getattr(clock, "timestamp", None)
    if ts is None:
        return pd.Timestamp.now(tz=MARKET_TZ)
    return _to_et_aware(ts)


def _expected_completed_bar_start(now_et):
    """현재 ET 기준 최신 완성 30분봉의 시작 시각(naive ET)."""
    now = _to_et_naive(now_et)
    session_open = now.normalize() + pd.Timedelta(hours=9, minutes=30)
    session_close = now.normalize() + pd.Timedelta(hours=16)
    latest_end = min(now, session_close)
    minute = (latest_end.minute // BAR_MINUTES) * BAR_MINUTES
    boundary = latest_end.replace(minute=minute, second=0, microsecond=0)
    bar_start = boundary - pd.Timedelta(minutes=BAR_MINUTES)
    if bar_start < session_open or boundary <= session_open:
        return None
    return bar_start


def _format_bar(ts):
    if ts is None:
        return "n/a"
    return _to_et_naive(ts).strftime("%Y-%m-%d %H:%M")


def _record_guard(rows, cycle_id, stamp, mode, guard, passed, reason, details=""):
    rows.append({
        "cycle_id": cycle_id,
        "timestamp": stamp,
        "mode": mode,
        "guard": guard,
        "passed": bool(passed),
        "reason": reason[:300],
        "details": details[:1000],
    })
    status = "통과" if passed else "차단"
    print(f"  [guard:{guard}] {status} — {reason}")


def _check_market_open_for_submit(trading_client):
    try:
        clock = trading_client.get_clock()
    except Exception as exc:
        return False, f"Alpaca clock 조회 실패: {exc}"

    now_et = _clock_timestamp_et(clock)
    next_close = _to_et_aware(clock.next_close)
    if not clock.is_open:
        return False, f"market closed at {_format_bar(now_et)} ET"

    seconds_to_close = (next_close - now_et).total_seconds()
    if seconds_to_close <= 0:
        return False, f"next_close 경과: {_format_bar(next_close)} ET"

    cutoff_min = getattr(config, "LIVE_NO_NEW_ORDERS_BEFORE_CLOSE_MIN", 0)
    if cutoff_min and seconds_to_close < cutoff_min * 60:
        return False, f"장 마감 {seconds_to_close / 60:.1f}분 전 — 신규 주문 금지"

    return True, f"market open, close까지 {seconds_to_close / 60:.1f}분"


def _check_cycle_freshness(cycle_start_et, cycle_start_monotonic):
    now_et = pd.Timestamp.now(tz=MARKET_TZ)
    wall_age = (now_et - cycle_start_et).total_seconds()
    mono_age = time.monotonic() - cycle_start_monotonic
    max_age = getattr(config, "LIVE_MAX_CYCLE_AGE_SEC", 300)

    if now_et.date() != cycle_start_et.date():
        return False, "cycle 시작일과 주문 직전 날짜가 다름"
    if max(wall_age, mono_age) > BAR_MINUTES * 60:
        return False, (
            f"cycle age가 30분봉 1개 초과 "
            f"(wall={wall_age:.1f}s, mono={mono_age:.1f}s)"
        )
    if max(wall_age, mono_age) > max_age:
        return False, (
            f"cycle age {max(wall_age, mono_age):.1f}s > "
            f"limit {max_age:.1f}s"
        )
    return True, f"cycle age wall={wall_age:.1f}s, mono={mono_age:.1f}s"


def _check_data_freshness(status_by_symbol, symbols, now_et):
    expected_bar = _expected_completed_bar_start(now_et)
    tolerance = pd.Timedelta(
        minutes=getattr(config, "LIVE_EXPECTED_BAR_TOLERANCE_MIN", 5))
    reasons = []

    if expected_bar is None:
        reasons.append("현재 정규장 세션의 완성 30분봉이 아직 없음")

    for sym in symbols:
        status = status_by_symbol.get(sym)
        if status is None:
            reasons.append(f"{sym}: data status missing")
            continue
        if (
            getattr(config, "LIVE_REQUIRE_FRESH_ALPACA_DATA", True)
            and status.fetch_failed
        ):
            reasons.append(f"{sym}: latest Alpaca fetch failed ({status.fetch_error})")
        last_bar = _to_et_naive(status.last_bar)
        if expected_bar is not None:
            if last_bar is None:
                reasons.append(f"{sym}: last bar missing")
            elif abs(last_bar - expected_bar) > tolerance:
                reasons.append(
                    f"{sym}: last_bar {_format_bar(last_bar)} "
                    f"!= expected {_format_bar(expected_bar)}"
                )

    if reasons:
        detail = "; ".join(reasons[:8])
        if len(reasons) > 8:
            detail += f"; ... +{len(reasons) - 8} more"
        return False, detail
    return True, f"expected completed bar {_format_bar(expected_bar)} ET"


def _check_beta_coverage(cap_result, raw_weights):
    required = getattr(config, "LIVE_MIN_BETA_COVERAGE", 0.0)
    total = len(raw_weights)
    have = len(cap_result.beta_symbols)
    coverage = have / total if total else 0.0
    if coverage + 1e-12 < required:
        missing = ", ".join(cap_result.missing_beta_symbols[:10])
        if len(cap_result.missing_beta_symbols) > 10:
            missing += f", ... +{len(cap_result.missing_beta_symbols) - 10} more"
        return False, (
            f"beta coverage {have}/{total}={coverage:.1%} "
            f"< required {required:.1%}; missing: {missing}"
        )
    return True, f"beta coverage {have}/{total}={coverage:.1%}"


def _symbols_requiring_shortability(targets_by_symbol, positions):
    symbols = []
    for sym, target in targets_by_symbol.items():
        cur = int(round(positions.get(sym, 0.0)))
        if target < 0 and target < cur:
            symbols.append(sym)
    return symbols


def _check_shortability(trading_client, targets_by_symbol, positions):
    if not getattr(config, "LIVE_REQUIRE_SHORTABLE", True):
        return True, "shortability guard disabled"

    symbols = _symbols_requiring_shortability(targets_by_symbol, positions)
    if not symbols:
        return True, "신규/증가 숏 없음"

    require_easy = getattr(config, "LIVE_REQUIRE_EASY_TO_BORROW", True)
    failures = []
    for sym in symbols:
        try:
            asset = trading_client.get_asset(sym)
        except Exception as exc:
            failures.append(f"{sym}: asset 조회 실패 ({exc})")
            continue

        tradable = bool(getattr(asset, "tradable", False))
        shortable = bool(getattr(asset, "shortable", False))
        easy = bool(getattr(asset, "easy_to_borrow", False))
        if not tradable:
            failures.append(f"{sym}: not tradable")
        elif not shortable:
            failures.append(f"{sym}: not shortable")
        elif require_easy and not easy:
            failures.append(f"{sym}: not easy_to_borrow")

    if failures:
        return False, "; ".join(failures[:10])
    return True, f"shortability OK: {', '.join(symbols)}"


def _daily_close_frame(full_by_symbol):
    """30분봉 full df dict → 일별 종가 wide DataFrame."""
    closes = {}
    for symbol, df in full_by_symbol.items():
        if df is None or df.empty or "datetime" not in df or "close" not in df:
            continue
        daily = (
            df.assign(date=lambda x: pd.to_datetime(x["datetime"]).dt.normalize())
              .groupby("date")["close"]
              .last()
              .astype(float)
        )
        closes[symbol] = daily
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index()


def _ensure_benchmark_full_df(data_client, histories, full_by_symbol):
    """beta benchmark(SPY) full df를 full_by_symbol에 추가한다."""
    benchmark = BETA_BENCHMARK_SYMBOL
    if benchmark in full_by_symbol:
        return "", None
    try:
        if benchmark not in histories:
            histories[benchmark] = load_resampled_bars(_parquet_path(benchmark))
        full, histories[benchmark], status = get_full_df(
            benchmark, data_client, histories[benchmark], return_status=True)
        full_by_symbol[benchmark] = full
        return "", status
    except Exception as exc:
        status = LiveDataStatus(
            symbol=benchmark,
            fetch_failed=True,
            fetch_error=str(exc),
        )
        return f"{benchmark} beta benchmark load failed: {exc}", status


def _build_beta_cap_result(raw_weights, full_by_symbol, cycle_date):
    """raw weight와 full df들로 live 순베타 캡 결과를 만든다."""
    close_daily = _daily_close_frame(full_by_symbol)
    beta_map, missing_reasons = allocations.estimate_capm_betas_from_daily_closes(
        close_daily=close_daily,
        symbols=list(raw_weights.keys()),
        benchmark_symbol=BETA_BENCHMARK_SYMBOL,
        as_of_date=cycle_date,
        lookback_days=config.LIVE_BETA_LOOKBACK_DAYS,
        min_obs=config.LIVE_BETA_MIN_OBS,
    )
    cap_result = allocations.apply_net_beta_cap(
        raw_weights,
        beta_map,
        config.NET_BETA_CAP,
    )
    return cap_result, beta_map, missing_reasons


def _latest_manifest_path():
    return LIVE_MODEL_DIR / "latest" / "manifest.json"


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _previous_weekday(date_value):
    day = pd.Timestamp(date_value).normalize() - pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day.date()


def _manifest_date(manifest, key):
    value = manifest.get(key)
    if not value:
        return None
    return pd.Timestamp(value).date()


def _check_manifest_freshness(manifest, as_of_date):
    required_date = _previous_weekday(as_of_date)
    trained_date = _manifest_date(manifest, "trained_through_date")
    beta_date = _manifest_date(manifest, "beta_asof_date")
    reasons = []

    if not manifest.get("approved", False):
        reasons.append("manifest is not approved")
    if manifest.get("schema_version") != 1:
        reasons.append(f"unsupported manifest schema_version={manifest.get('schema_version')}")

    if trained_date is None or trained_date < required_date:
        reasons.append(
            f"model trained_through_date={trained_date} < required {required_date}"
        )
    if beta_date is None or beta_date < required_date:
        reasons.append(f"beta_asof_date={beta_date} < required {required_date}")

    manifest_symbols = manifest.get("universe", [])
    if list(manifest_symbols) != list(SYMBOLS):
        reasons.append("manifest universe does not match config.LIVE_SYMBOLS")

    beta_by_symbol = manifest.get("beta_by_symbol", {})
    missing_beta = [sym for sym in SYMBOLS if sym not in beta_by_symbol]
    if missing_beta:
        reasons.append(f"beta missing symbols: {', '.join(missing_beta[:10])}")
    if len(beta_by_symbol) != len(SYMBOLS):
        reasons.append(
            f"beta count {len(beta_by_symbol)} != universe count {len(SYMBOLS)}"
        )

    strategies = manifest.get("strategies", {})
    missing_models = [sym for sym in SYMBOLS if sym not in strategies]
    if missing_models:
        reasons.append(f"model missing symbols: {', '.join(missing_models[:10])}")
    if len(strategies) != len(SYMBOLS):
        reasons.append(
            f"model count {len(strategies)} != universe count {len(SYMBOLS)}"
        )

    if reasons:
        return False, "; ".join(reasons)
    return True, f"approved artifacts fresh through {required_date}"


def _load_histories_for_symbols(symbols):
    histories = {}
    for sym in symbols:
        histories[sym] = load_resampled_bars(_parquet_path(sym))
    return histories


def load_approved_live_artifacts(require_fresh=True):
    """latest approved model/beta manifest를 로드한다."""
    manifest_path = _latest_manifest_path()
    if not manifest_path.exists():
        raise FileNotFoundError(f"approved manifest 없음: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_fresh:
        ok, reason = _check_manifest_freshness(
            manifest,
            pd.Timestamp.now(tz=MARKET_TZ).date(),
        )
        if not ok:
            raise RuntimeError(f"approved artifact freshness 실패: {reason}")

    strategies = {}
    for sym in SYMBOLS:
        model_path = Path(manifest["strategies"][sym])
        if not model_path.exists():
            raise FileNotFoundError(f"{sym} approved model 없음: {model_path}")
        strategies[sym] = joblib.load(model_path)

    histories = _load_histories_for_symbols(SYMBOLS)
    beta_map = {
        sym: float(manifest["beta_by_symbol"][sym])
        for sym in SYMBOLS
    }
    return strategies, histories, beta_map, manifest


def _config_snapshot_for_manifest():
    keys = [
        "HMM_RANDOM_RESTART", "HMM_N_ITER", "HMM_COVARIANCE_TYPE",
        "ROLLING_SCALER_WINDOW", "WINDOW_SIZE", "NET_BETA_CAP",
        "LIVE_BETA_LOOKBACK_DAYS", "LIVE_BETA_MIN_OBS",
        "LIVE_LOOKBACK_YEARS", "LIVE_DISABLE_INTRADAY_RETRAIN",
    ]
    return {key: getattr(config, key, None) for key in keys}


def run_eod_training(data_client, lookback_years=5):
    """장 마감 후 49종목 모델 + SPY beta를 만들고 latest approved manifest를 교체."""
    started = pd.Timestamp.now(tz=MARKET_TZ)
    run_id = started.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = LIVE_MODEL_DIR / "runs" / run_id
    model_dir = run_dir / "strategies"
    stamp = started.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 78)
    print(f"  EOD 학습 실행 — {stamp} ET")
    print(f"  run_id: {run_id}")
    print("=" * 78)

    histories = _load_histories_for_symbols(
        list(SYMBOLS) + [BETA_BENCHMARK_SYMBOL])
    full_by_symbol = {}
    status_by_symbol = {}
    train_failures = {}

    cutoff = None
    if lookback_years and lookback_years > 0:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)

    for sym in list(SYMBOLS) + [BETA_BENCHMARK_SYMBOL]:
        try:
            print(f"  [{sym}] EOD 최신 데이터 수집 ...", flush=True)
            full, histories[sym], status = get_full_df(
                sym, data_client, histories[sym], return_status=True)
            full_by_symbol[sym] = full
            status_by_symbol[sym] = status
            print(f"    last_bar={_format_bar(status.last_bar)}")
        except Exception as exc:
            train_failures[sym] = str(exc)
            print(f"    [{sym}] 데이터 수집 실패: {exc}")

    freshness_symbols = list(SYMBOLS)
    if BETA_BENCHMARK_SYMBOL not in freshness_symbols:
        freshness_symbols.append(BETA_BENCHMARK_SYMBOL)
    fresh_ok, fresh_reason = _check_data_freshness(
        status_by_symbol,
        freshness_symbols,
        pd.Timestamp.now(tz=MARKET_TZ),
    )
    if not fresh_ok:
        train_failures["freshness"] = fresh_reason

    strategies_paths = {}
    if not train_failures:
        model_dir.mkdir(parents=True, exist_ok=True)
        for sym in SYMBOLS:
            try:
                train_df = full_by_symbol[sym]
                if cutoff is not None:
                    train_df = train_df[
                        train_df["datetime"] >= cutoff
                    ].reset_index(drop=True)
                span = (
                    f"{train_df['datetime'].iloc[0].date()} ~ "
                    f"{train_df['datetime'].iloc[-1].date()}"
                )
                print(f"  [{sym}] {len(train_df):,}봉 ({span}) EOD 학습 ...")
                strat = HMMStrategy.from_config(hmm_model_path=None, verbose=False)
                strat.fit(train_df)
                model_path = model_dir / f"{sym.replace('.', '_')}.joblib"
                joblib.dump(strat, model_path)
                strategies_paths[sym] = str(model_path)
            except Exception as exc:
                train_failures[sym] = str(exc)
                print(f"    [{sym}] EOD 학습 실패: {exc}")
                break

    beta_map = {}
    beta_missing = {}
    trained_through_bar = None
    trained_through_date = None
    if not train_failures:
        symbol_last_bars = [
            _to_et_naive(status_by_symbol[sym].last_bar)
            for sym in SYMBOLS
        ]
        trained_through_bar = min(symbol_last_bars)
        trained_through_date = trained_through_bar.date()
        beta_asof = pd.Timestamp(trained_through_date) + pd.Timedelta(days=1)
        beta_map, beta_missing = allocations.estimate_capm_betas_from_daily_closes(
            close_daily=_daily_close_frame(full_by_symbol),
            symbols=list(SYMBOLS),
            benchmark_symbol=BETA_BENCHMARK_SYMBOL,
            as_of_date=beta_asof,
            lookback_days=config.LIVE_BETA_LOOKBACK_DAYS,
            min_obs=config.LIVE_BETA_MIN_OBS,
        )
        missing_beta = [sym for sym in SYMBOLS if sym not in beta_map]
        if missing_beta:
            train_failures["beta"] = (
                f"beta missing {len(missing_beta)}/{len(SYMBOLS)}: "
                f"{', '.join(missing_beta[:10])}"
            )

    if train_failures:
        note = "; ".join(f"{k}: {v}" for k, v in train_failures.items())
        _append_retrain_log({
            "timestamp": stamp,
            "run_id": run_id,
            "status": "failed",
            "symbols": len(strategies_paths),
            "trained_through_date": "" if trained_through_date is None else trained_through_date,
            "note": note[:500],
        })
        print(f"  [EOD 실패] latest approved manifest는 교체하지 않습니다: {note}")
        return False

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": pd.Timestamp.now(tz=MARKET_TZ).isoformat(),
        "approved": True,
        "universe": list(SYMBOLS),
        "benchmark": BETA_BENCHMARK_SYMBOL,
        "trained_through_date": str(trained_through_date),
        "trained_through_bar": str(trained_through_bar),
        "beta_asof_date": str(trained_through_date),
        "strategies": strategies_paths,
        "beta_by_symbol": {sym: float(beta_map[sym]) for sym in SYMBOLS},
        "beta_missing_reasons": beta_missing,
        "config": _config_snapshot_for_manifest(),
    }
    _write_json_atomic(run_dir / "manifest.json", manifest)
    _write_json_atomic(_latest_manifest_path(), manifest)
    _append_retrain_log({
        "timestamp": stamp,
        "run_id": run_id,
        "status": "approved",
        "symbols": len(strategies_paths),
        "trained_through_date": trained_through_date,
        "note": f"latest manifest updated: {_latest_manifest_path()}",
    })
    print(f"  [EOD 완료] latest approved manifest 갱신: {_latest_manifest_path()}")
    print("=" * 78 + "\n")
    return True


def run_cycle(trading_client, data_client, strategies, histories, dry_run,
              approved_beta_map=None):
    """전 종목 1회 의사결정 + 주문. 결과를 logs/live_log.csv 에 기록."""
    cycle_ts = pd.Timestamp.now(tz=MARKET_TZ)
    cycle_start_monotonic = time.monotonic()
    cycle_date = cycle_ts.tz_localize(None).normalize()
    cycle_id = f"{cycle_ts.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    stamp = cycle_ts.strftime("%Y-%m-%d %H:%M:%S")
    mode = "DRY-RUN" if dry_run else "EXECUTE"
    guard_rows = []
    log_rows = {}
    beta_logged = False
    print("=" * 78)
    print(f"  사이클 실행 — {stamp} ET  {'[DRY-RUN]' if dry_run else '[실제 주문]'}")
    print(f"  cycle_id: {cycle_id}")
    print("=" * 78)

    # ── 0) 우리 심볼 미체결 정리 (실제 모드만) ──────────────────────
    # 직전 사이클의 미체결 주문이 수량을 잡아두면(held_for_orders) 이번
    # 주문이 거부된다. 먼저 우리 심볼 미체결을 취소·정리한 뒤 포지션을
    # '새로' 읽어 available == qty 상태에서 의사결정한다.
    if not dry_run:
        cancel_ok = cancel_open_orders_for_symbols(
            trading_client, SYMBOLS,
            config.CANCEL_SETTLE_TIMEOUT_SEC, config.POLL_INTERVAL_SEC)
        _record_guard(
            guard_rows, cycle_id, stamp, mode, "open_order_cancel",
            cancel_ok,
            "open orders 정리 완료" if cancel_ok else "open orders 정리 실패/timeout",
        )
        if not cancel_ok and getattr(config, "LIVE_CANCEL_MUST_SETTLE", True):
            reason = "open order cancel guard failed"
            _append_guard_log(guard_rows)
            _mark_cycle_no_trade(log_rows, stamp, mode, None, reason)
            _finish_cycle_log(log_rows, beta_logged=False)
            return

    # ── 1) 계좌/positions/budgets 먼저 로드 ───────────────────────
    account = trading_client.get_account()
    equity = float(account.equity)
    allocation_ratios = allocations.get_allocations(
        SYMBOLS,
        allocations.SYMBOL_ALLOCATIONS,
        allocations.NORMALIZE_ALLOCATIONS,
    )
    budgets = {sym: equity * allocation_ratios.get(sym, 0.0) for sym in SYMBOLS}
    print(f"  계좌 자산: ${equity:,.2f}   |   배분: "
          f"{'등가중' if allocations.SYMBOL_ALLOCATIONS is None else '차등비중'}")

    positions = {p.symbol: float(p.qty)
                 for p in trading_client.get_all_positions()}

    rebal = config.REBALANCE_THRESHOLD

    # ── 2) 모든 심볼의 최신 full df / signal / price 먼저 수집 ────
    symbol_state = {}
    full_by_symbol = {}
    data_status_by_symbol = {}
    for sym in SYMBOLS:
        row = _base_log_row(stamp, sym, equity, mode)
        try:
            print(f"    [{sym}] 최신 데이터/시그널 수집 ...", flush=True)
            full, histories[sym], data_status = get_full_df(
                sym, data_client, histories[sym], return_status=True)
            data_status_by_symbol[sym] = data_status
            signals = strategies[sym].generate_signals(full)
            signal = float(signals[-1])                       # -1.0 ~ +1.0
            price = float(full["close"].iloc[-1])
            cur = int(round(positions.get(sym, 0.0)))
            allocation = allocation_ratios.get(sym, 0.0)
            raw_weight = allocation * signal
            row.update(signal=round(signal, 4), price=round(price, 2),
                       pos_before=cur)
            symbol_state[sym] = {
                "row": row,
                "signal": signal,
                "price": price,
                "cur": cur,
                "allocation": allocation,
                "budget": budgets.get(sym, 0.0),
                "raw_weight": raw_weight,
                "last_bar": data_status.last_bar,
            }
            full_by_symbol[sym] = full
            print(
                f"    [{sym}] 수집 완료: signal={signal:+.3f}, "
                f"price={price:.2f}, pos={cur}, "
                f"last_bar={_format_bar(data_status.last_bar)}",
                flush=True,
            )

        except Exception as exc:
            row["action"] = "실패"
            row["note"] = str(exc)[:200]
            print(f"  {sym:6s} 처리 실패: {exc}")
            log_rows[sym] = row

    if getattr(config, "LIVE_REQUIRE_ALL_SYMBOLS", True):
        missing_symbols = [sym for sym in SYMBOLS if sym not in symbol_state]
        all_symbols_ok = not missing_symbols
        reason = (
            f"{len(symbol_state)}/{len(SYMBOLS)} symbols ready"
            if all_symbols_ok
            else f"symbol failure/missing: {', '.join(missing_symbols[:10])}"
        )
        _record_guard(
            guard_rows, cycle_id, stamp, mode, "all_symbols",
            all_symbols_ok, reason,
        )
        if not dry_run and not all_symbols_ok:
            reason = "49종목 all-or-nothing guard failed"
            _append_guard_log(guard_rows)
            _mark_cycle_no_trade(log_rows, stamp, mode, equity, reason)
            _finish_cycle_log(log_rows, beta_logged=False)
            return

    # ── 3~7) raw weight → rolling beta → 순베타 cap ───────────────
    raw_weights = {
        sym: state["raw_weight"]
        for sym, state in symbol_state.items()
    }
    freshness_symbols = list(SYMBOLS)
    benchmark_note = ""
    if approved_beta_map is None and BETA_BENCHMARK_SYMBOL not in freshness_symbols:
        print(f"  beta benchmark({BETA_BENCHMARK_SYMBOL}) 데이터 준비 ...", flush=True)
        benchmark_note, benchmark_status = _ensure_benchmark_full_df(
            data_client, histories, full_by_symbol)
        if benchmark_status is not None:
            data_status_by_symbol[BETA_BENCHMARK_SYMBOL] = benchmark_status
        freshness_symbols.append(BETA_BENCHMARK_SYMBOL)
    fresh_ok, fresh_reason = _check_data_freshness(
        data_status_by_symbol,
        freshness_symbols,
        pd.Timestamp.now(tz=MARKET_TZ),
    )
    _record_guard(
        guard_rows, cycle_id, stamp, mode, "signal_bar_freshness",
        fresh_ok, fresh_reason,
    )
    if not dry_run and not fresh_ok:
        reason = "signal/bar freshness guard failed"
        _append_guard_log(guard_rows)
        _mark_cycle_no_trade(log_rows, stamp, mode, equity, reason)
        _finish_cycle_log(log_rows, beta_logged=False)
        return

    if approved_beta_map is not None:
        print("  approved beta로 순베타 cap 계산 ...", flush=True)
        beta_map = {
            sym: float(approved_beta_map[sym])
            for sym in raw_weights
            if sym in approved_beta_map
        }
        missing_reasons = {
            sym: "approved beta missing"
            for sym in raw_weights
            if sym not in beta_map
        }
        cap_result = allocations.apply_net_beta_cap(
            raw_weights,
            beta_map,
            config.NET_BETA_CAP,
        )
        beta_note = "approved beta loaded from latest manifest"
    else:
        print("  rolling beta 및 순베타 cap 계산 ...", flush=True)
        try:
            cap_result, beta_map, missing_reasons = _build_beta_cap_result(
                raw_weights,
                full_by_symbol,
                cycle_date,
            )
            beta_note = benchmark_note
        except Exception as exc:
            beta_map = {}
            missing_reasons = {
                sym: f"beta calculation failed: {exc}"
                for sym in raw_weights
            }
            cap_result = allocations.apply_net_beta_cap(
                raw_weights,
                beta_map,
                config.NET_BETA_CAP,
            )
            beta_note = f"{benchmark_note}; beta calculation failed: {exc}".strip("; ")

    print(
        "  순베타:"
        f" raw {_fmt_optional(cap_result.raw_net_beta)}"
        f" → adjusted {_fmt_optional(cap_result.adjusted_net_beta)}"
        f" | cap ±{config.NET_BETA_CAP:.2f}"
        f" | beta {len(cap_result.beta_symbols)}/{len(raw_weights)}"
        f" | {'cap 적용' if cap_result.capped else 'cap 미적용'}"
    )
    if cap_result.missing_beta_symbols:
        missing_text = ", ".join(cap_result.missing_beta_symbols)
        print(f"  beta 부족/제외: {missing_text}")
    if beta_note:
        print(f"  beta 참고: {beta_note}")

    _append_beta_log({
        "timestamp": stamp,
        "raw_net_beta": (
            "" if cap_result.raw_net_beta is None
            else round(cap_result.raw_net_beta, 6)
        ),
        "adjusted_net_beta": (
            "" if cap_result.adjusted_net_beta is None
            else round(cap_result.adjusted_net_beta, 6)
        ),
        "cap": config.NET_BETA_CAP,
        "capped": cap_result.capped,
        "scale": round(cap_result.scale, 6),
        "beta_symbols": ";".join(cap_result.beta_symbols),
        "missing_beta_symbols": ";".join(cap_result.missing_beta_symbols),
        "mode": mode,
        "note": beta_note[:300],
    })
    beta_logged = True

    beta_ok, beta_reason = _check_beta_coverage(cap_result, raw_weights)
    _record_guard(
        guard_rows, cycle_id, stamp, mode, "beta_coverage",
        beta_ok, beta_reason,
    )
    if not dry_run and not beta_ok:
        reason = "beta coverage guard failed"
        _append_guard_log(guard_rows)
        _mark_cycle_no_trade(log_rows, stamp, mode, equity, reason)
        _finish_cycle_log(log_rows, beta_logged=beta_logged)
        return

    print(f"  {'종목':6s} {'시그널':>8s} {'Beta':>7s} {'RawW':>8s} "
          f"{'AdjW':>8s} {'현재가':>10s} {'보유':>8s} {'목표':>8s} {'동작':>8s}")
    print("  " + "-" * 92)

    orders_by_symbol = {}
    targets_by_symbol = {}
    for sym in SYMBOLS:
        if sym not in symbol_state:
            continue
        state = symbol_state[sym]
        row = state["row"]
        try:
            price = state["price"]
            cur = state["cur"]
            allocation = state["allocation"]
            budget = state["budget"]
            raw_weight = state["raw_weight"]
            adjusted_weight = cap_result.adjusted_weights.get(sym, raw_weight)
            target = int(round(equity * adjusted_weight / price)) if price > 0 else 0
            row["target"] = target
            targets_by_symbol[sym] = target

            target_signal = (
                adjusted_weight / allocation if allocation > 0 else 0.0
            )
            cur_signal = (cur * price / budget) if budget > 0 else 0.0
            cap_adjusted_symbol = abs(adjusted_weight - raw_weight) > 1e-12

            beta_value = beta_map.get(sym)
            beta_text = _fmt_optional(beta_value, digits=2)
            note_parts = [
                f"raw_w={raw_weight:+.4f}",
                f"adj_w={adjusted_weight:+.4f}",
            ]
            if beta_value is None:
                reason = missing_reasons.get(sym, "beta unavailable")
                note_parts.append(f"beta_missing={reason}")
            else:
                note_parts.append(f"beta={beta_value:+.4f}")

            # 리밸런싱 임계값: beta cap으로 바뀐 종목은 risk cap을 우선 반영하고,
            # 그 외에는 기존처럼 allocation 단위 signal 변화가 작으면 스킵한다.
            if (
                target != cur
                and allocation > 0
                and not cap_adjusted_symbol
                and abs(target_signal - cur_signal) < rebal
            ):
                row["action"] = "스킵"
                row["note"] = "; ".join(note_parts)[:200]
                print(f"  {sym:6s} {state['signal']:>+8.2f} {beta_text:>7s} "
                      f"{raw_weight:>+8.3f} {adjusted_weight:>+8.3f} "
                      f"{price:>10.2f} {cur:>8d} {target:>8d} {'스킵':>8s}")
                log_rows[sym] = row
                continue

            orders = plan_orders(cur, target)
            orders_by_symbol[sym] = orders
            row["action"] = "거래" if orders else "유지"
            row["order_delta"] = (target - cur) if orders else 0
            row["note"] = "; ".join(note_parts)[:200]
            print(f"  {sym:6s} {state['signal']:>+8.2f} {beta_text:>7s} "
                  f"{raw_weight:>+8.3f} {adjusted_weight:>+8.3f} "
                  f"{price:>10.2f} {cur:>8d} {target:>8d} {row['action']:>8s}")

        except Exception as exc:
            row["action"] = "실패"
            row["note"] = str(exc)[:200]
            print(f"  {sym:6s} 처리 실패: {exc}")

        log_rows[sym] = row

    has_orders = any(bool(orders) for orders in orders_by_symbol.values())
    if has_orders:
        market_ok, market_reason = _check_market_open_for_submit(trading_client)
        _record_guard(
            guard_rows, cycle_id, stamp, mode, "market_open_submit",
            market_ok, market_reason,
        )

        cycle_ok, cycle_reason = _check_cycle_freshness(
            cycle_ts, cycle_start_monotonic)
        _record_guard(
            guard_rows, cycle_id, stamp, mode, "cycle_freshness",
            cycle_ok, cycle_reason,
        )

        final_fresh_ok, final_fresh_reason = _check_data_freshness(
            data_status_by_symbol,
            freshness_symbols,
            pd.Timestamp.now(tz=MARKET_TZ),
        )
        _record_guard(
            guard_rows, cycle_id, stamp, mode, "final_signal_bar_freshness",
            final_fresh_ok, final_fresh_reason,
        )

        short_ok, short_reason = _check_shortability(
            trading_client, targets_by_symbol, positions)
        _record_guard(
            guard_rows, cycle_id, stamp, mode, "shortability",
            short_ok, short_reason,
        )

        final_ok = market_ok and cycle_ok and final_fresh_ok and short_ok
        if not final_ok:
            reason = "pre-submit guard failed: " + "; ".join(
                r for ok, r in [
                    (market_ok, market_reason),
                    (cycle_ok, cycle_reason),
                    (final_fresh_ok, final_fresh_reason),
                    (short_ok, short_reason),
                ]
                if not ok
            )
            _append_guard_log(guard_rows)
            _mark_cycle_no_trade(log_rows, stamp, mode, equity, reason)
            _finish_cycle_log(log_rows, beta_logged=beta_logged)
            return

        _append_guard_log(guard_rows)

        # ── 순차 집행: 부호 반전이면 'close' 체결 확인 후에만 'entry' ──
        order_seq = 0
        abort_remaining = False
        for sym in SYMBOLS:
            row = log_rows.get(sym)
            if row is None:
                continue
            for signed_qty, kind in orders_by_symbol.get(sym, []):
                side_code = "b" if signed_qty > 0 else "s"
                sym_slug = "".join(ch for ch in sym if ch.isalnum())
                client_order_id = (
                    f"hmm-{cycle_ts.strftime('%y%m%d%H%M')}-"
                    f"{cycle_id[-6:]}-{sym_slug}-{side_code}{order_seq:02d}"
                )
                order_seq += 1
                try:
                    order = submit_order(
                        trading_client, sym, signed_qty, dry_run,
                        client_order_id=client_order_id,
                    )
                except Exception as exc:
                    row["action"] = "실패"
                    row["note"] = str(exc)[:200]
                    print(f"  {sym:6s} 주문 실패: {exc}")
                    if not dry_run:
                        abort_remaining = True
                    break

                if kind == "close" and not dry_run and order is not None:
                    filled = wait_for_fill(
                        trading_client, order.id,
                        config.FILL_WAIT_TIMEOUT_SEC, config.POLL_INTERVAL_SEC)
                    if not filled:
                        # 청산 미체결 → 반대 포지션 진입 보류(잘못된 노출 방지).
                        # 다음 사이클에서 재조정된다.
                        row["action"] = "청산대기"
                        row["note"] = (
                            row.get("note", "")
                            + "; 청산 미체결 → 진입 보류(다음 사이클 재조정)"
                        )[:200]
                        print(f"      [보류] {sym} 청산 미체결 → 진입 생략")
                        break
            if abort_remaining:
                print("  [중단] 주문 실패로 남은 신규 주문 제출을 중단합니다.")
                break
    else:
        _append_guard_log(guard_rows)

    _finish_cycle_log(log_rows, beta_logged=beta_logged)


# ════════════════════════════════════════════════════════════════
#  반복 실행 (--loop)
# ════════════════════════════════════════════════════════════════

def _sleep_until(target_utc, label):
    """target_utc 까지 60초 단위로 대기 (Ctrl+C 로 중단 가능)."""
    while True:
        remain = (target_utc - pd.Timestamp.now(tz="UTC")).total_seconds()
        if remain <= 0:
            return
        print(f"  ...{label} — {remain/60:.1f}분 대기", flush=True)
        time.sleep(min(remain, 60))


def _next_boundary_utc():
    """다음 30분 경계(:00 / :30) + 정착 버퍼의 UTC 시각."""
    now_et = pd.Timestamp.now(tz=MARKET_TZ)
    if now_et.minute < 30:
        nb = now_et.replace(minute=30, second=0, microsecond=0)
    else:
        nb = (now_et + pd.Timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0)
    nb = nb + pd.Timedelta(seconds=SETTLE_BUFFER_SEC)
    return nb.tz_convert("UTC")


def run_loop(trading_client, data_client, strategies, histories, dry_run,
             retrain_every_days=1, lookback_years=5, approved_beta_map=None):
    """장중 30분마다 사이클 실행. 장 마감 시 다음 개장까지 대기.

    재학습: 거래일(ET 날짜)이 last_train_date 대비 retrain_every_days 이상
    지나면, 그 날 첫 사이클 직전에 retrain_strategies()를 호출한다.
    retrain_every_days <= 0 이면 재학습하지 않는다(시작 시 학습 그대로 사용).
    """
    print("[루프 모드] Ctrl+C 로 종료.")
    if getattr(config, "LIVE_DISABLE_INTRADAY_RETRAIN", True):
        retrain_every_days = 0
    if retrain_every_days and retrain_every_days > 0:
        print(f"[재학습] {retrain_every_days}거래일마다 최신 데이터로 재학습\n")
    else:
        print("[재학습] 비활성 — 시작 시 학습한 모델 유지\n")

    # 시작 시 build_strategies()로 이미 학습됨 → 오늘을 기준일로 설정.
    last_train_date = pd.Timestamp.now(tz=MARKET_TZ).date()

    while True:
        try:
            clock = trading_client.get_clock()
            if not clock.is_open:
                nxt = pd.Timestamp(clock.next_open)
                print(f"  장 마감 상태 — 다음 개장: {nxt}")
                _sleep_until(nxt + pd.Timedelta(seconds=SETTLE_BUFFER_SEC),
                             "개장 대기")
                continue

            # ── 재학습 트리거: 거래일이 바뀌었고 주기가 찼으면 사이클 전에 재학습 ──
            today = pd.Timestamp.now(tz=MARKET_TZ).date()
            if (retrain_every_days and retrain_every_days > 0
                    and (today - last_train_date).days >= retrain_every_days):
                retrain_strategies(strategies, histories, data_client,
                                   lookback_years)
                last_train_date = today

            run_cycle(trading_client, data_client, strategies, histories,
                      dry_run, approved_beta_map=approved_beta_map)

            # 다음 30분 경계까지 대기 (오늘 장 마감 넘어가면 다음 개장까지)
            nb = _next_boundary_utc()
            next_close = pd.Timestamp(clock.next_close)
            if nb >= next_close:
                clock = trading_client.get_clock()
                _sleep_until(pd.Timestamp(clock.next_open)
                             + pd.Timedelta(seconds=SETTLE_BUFFER_SEC),
                             "장 마감 — 개장 대기")
            else:
                _sleep_until(nb, "다음 봉")
        except KeyboardInterrupt:
            print("\n[종료] 사용자 중단.")
            return
        except Exception as exc:
            print(f"  [루프 오류] {exc} — 60초 후 재시도")
            time.sleep(60)


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="HMM 알파 Alpaca 페이퍼 라이브 트레이딩")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true",
                      help="1회 실행 후 종료 (기본)")
    mode.add_argument("--loop", action="store_true",
                      help="장중 30분마다 반복 실행")
    mode.add_argument("--train-eod", action="store_true",
                      help="장 마감 후 49종목 모델/beta를 학습해 latest approved manifest 갱신")
    p.add_argument("--execute", action="store_true",
                   help="실제 주문 제출 (미지정 시 dry-run — 출력만)")
    p.add_argument("--lookback-years", type=int, default=config.LIVE_LOOKBACK_YEARS,
                   help="HMM 학습에 쓸 최근 연수 (0이면 전체 과거). "
                        "기본값은 config.LIVE_LOOKBACK_YEARS")
    p.add_argument("--retrain-every-days", type=int,
                   default=config.RETRAIN_EVERY_DAYS,
                   help="루프에서 재학습 주기(거래일). 1=매일, 0이면 재학습 안 함. "
                        "기본값은 config.RETRAIN_EVERY_DAYS (--loop 에서만 적용)")
    return p.parse_args()


def _pid_is_alive(pid: int) -> bool:
    """주어진 PID 프로세스가 살아 있는지 확인."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # 존재하지만 시그널 권한 없음 → 살아있음
    return True


def acquire_single_instance_lock(pid_path: str):
    """PID 락 획득. 이미 살아있는 인스턴스가 있으면 종료(중복 실행 방지).

    종료 시 atexit 으로 자기 PID 파일만 정리한다.
    """
    from pathlib import Path
    p = Path(pid_path)
    if p.exists():
        try:
            old = int(p.read_text().strip())
        except (ValueError, OSError):
            old = None
        if old and old != os.getpid() and _pid_is_alive(old):
            sys.exit(f"[중단] 이미 실행 중입니다 (PID {old}). "
                     f"중복 실행을 막았습니다 ({pid_path}).\n"
                     f"  강제 실행하려면 그 프로세스를 종료하거나 {pid_path} 를 지우세요.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(os.getpid()))

    def _release():
        try:
            if p.exists() and p.read_text().strip() == str(os.getpid()):
                p.unlink()
        except OSError:
            pass
    atexit.register(_release)


def main():
    args = parse_args()
    if args.train_eod and args.execute:
        sys.exit("[중단] --train-eod는 주문 모드가 아니므로 --execute와 함께 쓰지 않습니다.")
    dry_run = not args.execute

    # 단일 인스턴스 가드: 실제 주문/루프처럼 오래 도는 모드에서만 중복 차단.
    # (--once dry-run 은 점검용이라 언제든 돌 수 있게 둔다)
    if config.SINGLE_INSTANCE_LOCK and (args.loop or args.execute):
        acquire_single_instance_lock(config.LIVE_PID_PATH)

    mode_label = "TRAIN-EOD" if args.train_eod else ("LOOP" if args.loop else "ONCE")
    print("\n" + "=" * 78)
    print("  HMM 알파 — Alpaca 페이퍼 라이브 트레이딩")
    print(f"  모드: {mode_label}   "
          f"주문: {'DRY-RUN (제출 안 함)' if dry_run else '★ 실제 제출 ★'}")
    print("=" * 78 + "\n")

    trading_client, data_client = connect()
    account = trading_client.get_account()
    print(f"  계좌 상태: {account.status}   자산: ${float(account.equity):,.2f}\n")

    lb = args.lookback_years
    if args.train_eod:
        ok = run_eod_training(data_client, lookback_years=lb)
        if not ok:
            sys.exit(2)
        return

    approved_beta_map = None
    strategies = histories = None
    try:
        print("[준비] latest approved model/beta 로드 시도")
        strategies, histories, approved_beta_map, manifest = load_approved_live_artifacts(
            require_fresh=not dry_run or getattr(config, "LIVE_REQUIRE_APPROVED_MODEL", True)
        )
        print(
            "  approved artifact 로드 완료: "
            f"trained_through={manifest.get('trained_through_date')}, "
            f"beta_asof={manifest.get('beta_asof_date')}"
        )
    except Exception as exc:
        if not dry_run and getattr(config, "LIVE_REQUIRE_APPROVED_MODEL", True):
            sys.exit(f"[중단] execute 모드는 fresh approved model/beta가 필요합니다: {exc}")
        print(f"  [경고] approved artifact 로드 실패: {exc}")
        print("  dry-run 진단용으로 즉석 학습/rolling beta fallback을 사용합니다.")
        print(f"[준비] 종목별 HMM 전략 학습 "
              f"(학습 기간: {'전체 과거' if lb <= 0 else f'최근 {lb}년'})")
        strategies, histories = build_strategies(lb)

    if args.loop:
        run_loop(trading_client, data_client, strategies, histories, dry_run,
                 retrain_every_days=args.retrain_every_days,
                 lookback_years=lb,
                 approved_beta_map=approved_beta_map)
    else:
        run_cycle(trading_client, data_client, strategies, histories, dry_run,
                  approved_beta_map=approved_beta_map)
        if dry_run:
            print("dry-run 완료. 실제 주문을 내려면 --execute 를 붙이세요.")


if __name__ == "__main__":
    main()
