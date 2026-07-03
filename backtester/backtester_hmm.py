"""
HMM 알파용 백테스트 엔진

기존 Engine과의 차이점:
- PortfolioContinuous 사용 → float 포지션 비중 지원
  예) signal = 0.7 → 자본의 70% 롱
- rebalance_threshold: 포지션 변화량이 이 값 이하면 거래 생략 (수수료 절감)
- 시그널 타입: float (-1.0 ~ +1.0)

look-ahead bias 제거 원칙 (기존 Engine과 동일):
- signals[i]: 바 i 종가 기준으로 계산된 시그널
- 체결 시점: 바 i+1의 시가 (open[i+1])
"""

import numpy as np
import pandas as pd
from backtester.portfolio_continuous import PortfolioContinuous


class EngineHMM:

    def __init__(
        self,
        strategy,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        cooldown: int = 0,
        rebalance_threshold: float = 0.05,
    ):
        """
        strategy:             generate_signals()를 구현한 전략 객체
        initial_capital:      초기 자본
        fee_rate:             거래 수수료율 (0.001 = 0.1%)
        cooldown:             거래 체결 후 다음 거래까지 최소 대기 바 수
        rebalance_threshold:  포지션 변화량이 이 값 이하면 리밸런싱 생략
                              (예: 0.05 → 5% 미만 변화는 무시)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.cooldown = cooldown
        self.rebalance_threshold = rebalance_threshold

    def run(self, df: pd.DataFrame) -> dict:
        df = df.reset_index(drop=True)

        # 전체 시그널 한번에 계산 (벡터화)
        # generate_signals()는 float 배열(-1.0 ~ +1.0)을 반환해야 함
        signals = self.strategy.generate_signals(df)

        # numpy 배열로 추출 (루프 내 pandas 접근 제거)
        open_arr  = df['open'].to_numpy(dtype=np.float64)
        close_arr = df['close'].to_numpy(dtype=np.float64)
        dt_arr    = df['datetime'].values

        portfolio = PortfolioContinuous(self.initial_capital, self.fee_rate)
        n = len(df)
        equity_arr = np.empty(n, dtype=np.float64)
        pending = None
        cooldown_remaining = 0

        print(f"[HMM 백테스터] 시뮬레이션 시작: {n:,}개 바 "
              f"(쿨다운: {self.cooldown}바, 리밸런싱 임계값: {self.rebalance_threshold})")

        for i in range(n):
            if i % 500_000 == 0 and i > 0:
                print(f"  진행 중... {i:,} / {n:,} ({i/n*100:.0f}%)")

            # ── 1. 이전 바 시그널 → 현재 바 시가에 체결 ──────────────
            if pending is not None:
                sig_to_execute = pending
                portfolio.execute(
                    sig_to_execute,
                    open_arr[i],
                    dt_arr[i],
                    rebalance_threshold=self.rebalance_threshold,
                )
                pending = None
                cooldown_remaining = self.cooldown

            # ── 2. 현재 바 종가 기준 자산 평가 기록 ──────────────────
            current_equity = portfolio.get_equity(close_arr[i])

            # ── 3. 강제 청산: 자산 0 이하 → 파산 ────────────────────
            if current_equity <= 0 and portfolio.position != 0.0:
                portfolio.execute(0.0, close_arr[i], dt_arr[i])
                current_equity = max(portfolio.cash, 0.0)
                pending = None
                cooldown_remaining = 0

            equity_arr[i] = current_equity

            # 파산 이후 루프 종료
            if equity_arr[i] <= 0:
                equity_arr[i:] = 0.0
                break

            # ── 4. 쿨다운 중이면 시그널 무시 ─────────────────────────
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            # ── 5. 시그널 변경 시 다음 바에 체결 예약 ────────────────
            # float 시그널: 임계값 이상의 변화가 있을 때만 예약
            sig = float(signals[i])
            if abs(sig - portfolio.position) > self.rebalance_threshold:
                pending = sig

        # ── 마지막 포지션 종가에 청산 ─────────────────────────────────
        if portfolio.position != 0.0:
            portfolio.execute(0.0, close_arr[n - 1], dt_arr[n - 1])

        return {
            'equity_curve':    equity_arr,
            'trades':          portfolio.trades,
            'signals':         signals,
            'initial_capital': self.initial_capital,
            'start':           df['datetime'].iloc[0],
            'end':             df['datetime'].iloc[-1],
        }
