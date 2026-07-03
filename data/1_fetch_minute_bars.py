# 분봉 데이터 다운로더
#
# ★ 아래 설정 블록만 수정하면 종목/기간을 자유롭게 바꿀 수 있습니다.
#
# 출력 경로: data/minute/{SYMBOL}_{START}_{END}_1min.parquet
# ─────────────────────────────────────────────────────────────
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment


# ══════════════════════════════════════════════════
# ★ 설정 — 여기만 바꾸세요
# ══════════════════════════════════════════════════

# ── [2026-06-25] 하락장 walk-forward용 2016~ 백필 ──────────────────────────
# STEP 1 (현재 설정): 검증 fetch — Alpaca가 2016 분봉을 주는지 소수로 먼저 확인.
#   기대: AAPL/JPM/XOM/SPY 는 2016-01-04부터, UBER 는 2019-05부터만 채워짐.
SYMBOLS      = ['AAPL', 'JPM', 'XOM', 'SPY', 'UBER']
#
# STEP 2 (검증 OK 후, 아래 리스트로 교체해 전체 백필): 49 유니버스 + SPY = 50종목.
#   상장 늦은 종목(CVNA 2017-04, UBER 2019-05, COIN 2021-04, HOOD 2021-07,
#   CEG 2022-02 등)은 상장일부터만 채워짐 → walk-forward 가변유니버스로 자동 처리.
#   GEV(2024-04 스핀오프)는 기존 방침대로 제외.
# SYMBOLS = ['META','GOOGL','NFLX','TMUS','TSLA','AMZN','HD','BKNG','CVNA','WMT',
#            'COST','PG','KO','PEP','XOM','CVX','COP','SLB','HOOD','COIN','JPM',
#            'BRK.B','V','UNH','LLY','JNJ','ABBV','PFE','BA','UBER','CAT','GE',
#            'NVDA','AAPL','MSFT','MU','AMD','LIN','NEM','FCX','SHW','WELL','AMT',
#            'EQIX','PLD','CEG','VST','NEE','SO','SPY']
# 주의: BRK.B 가 빈 응답이면 "BRK/B" 로 바꿔 재시도.
# 이전 기본값(참고): ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "SPY"]
START_DATE   = "2016-01-01"       # 시작일 (하락장 walk-forward용 백필)
END_DATE     = "2026-05-27"       # 종료일 (기보유 종목과 동일 구간 정렬)
CHUNK_MONTHS = 3                  # 1회 API 요청 단위 (월). 크게 잡으면 빠르지만 실패 리스크↑
OUTPUT_DIR   = "data/minute"      # 저장 폴더

# ══════════════════════════════════════════════════


def make_chunks(start: str, end: str, months: int) -> list:
    """start~end 구간을 months 단위 청크 리스트로 분할."""
    chunks = []
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC")
    while s < e:
        chunk_end = min(s + pd.DateOffset(months=months), e)
        chunks.append((s, chunk_end))
        s = chunk_end
    return chunks


def fetch_symbol(client: StockHistoricalDataClient, symbol: str) -> pd.DataFrame:
    """한 종목의 분봉 전체를 청크 단위로 받아 하나의 DataFrame으로 반환."""
    chunks = make_chunks(START_DATE, END_DATE, CHUNK_MONTHS)
    collected = []

    for i, (s, e) in enumerate(chunks, 1):
        label = f"{s.date()} ~ {e.date()}"
        print(f"    [{i:>2}/{len(chunks)}] {label} ... ", end="", flush=True)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=s,
            end=e,
            adjustment=Adjustment.SPLIT,   # ★ 액면분할 역방향 조정 (필수)
        )

        try:
            bars = client.get_stock_bars(req)
            df   = bars.df
            if df.empty:
                print("빈 응답")
            else:
                collected.append(df)
                print(f"OK  ({len(df):,}행)")
        except Exception as exc:
            print(f"실패: {exc}")

        time.sleep(0.4)   # rate-limit 여유

    if not collected:
        return pd.DataFrame()
    return pd.concat(collected).sort_index()


def save(df: pd.DataFrame, symbol: str) -> str:
    """DataFrame을 parquet으로 저장하고 경로를 반환."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    s = START_DATE.replace("-", "")
    e = END_DATE.replace("-", "")
    path = f"{OUTPUT_DIR}/{symbol}_{s}_{e}_1min.parquet"
    df.to_parquet(path)
    return path


if __name__ == "__main__":
    load_dotenv()
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
    )

    print(f"기간: {START_DATE} ~ {END_DATE}  |  청크: {CHUNK_MONTHS}개월 단위")
    print(f"종목: {SYMBOLS}\n")

    results = {}
    for symbol in SYMBOLS:
        print(f"▶ {symbol}")
        df = fetch_symbol(client, symbol)

        if df.empty:
            print("  → 데이터 없음, 건너뜀\n")
            results[symbol] = None
            continue

        path = save(df, symbol)
        results[symbol] = path

        ts = df.index.get_level_values("timestamp")
        print(f"  → 저장: {path}")
        print(f"     행 수: {len(df):,}  |  "
              f"날짜 범위: {ts.min().date()} ~ {ts.max().date()}\n")

    # ── 요약 ──────────────────────────────────────
    print("=" * 55)
    print("완료 요약")
    print("=" * 55)
    for sym, path in results.items():
        status = path if path else "실패/데이터 없음"
        print(f"  {sym:<8} → {status}")
