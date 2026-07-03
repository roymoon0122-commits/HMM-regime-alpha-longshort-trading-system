"""
연속적 비중 포지셔닝을 지원하는 포트폴리오

기존 Portfolio와의 차이점:
- position: int(0/1/-1) → float(-1.0 ~ +1.0)
  예) 0.5 = 자본의 50%로 롱, -0.3 = 자본의 30%로 숏
- 전체 자본이 아닌 signal 비중만큼만 투자
- 나머지 자본은 현금(unallocated cash)으로 보유
- execute()에 rebalance_threshold 파라미터 추가
  → 포지션 변화량이 임계값 미만이면 거래 생략 (수수료 절감)

수수료 계산 (기존과 동일):
- 진입: 투입 현금 * fee_rate
- 청산 (롱): 코인 가치 * fee_rate
- 청산 (숏): 코인 수량 * 청산가 * fee_rate
"""


class PortfolioContinuous:

    def __init__(self, initial_capital: float, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

        self.cash = initial_capital      # 현재 보유 현금 (미투자 + 청산 후 회수분)
        self.position = 0.0              # -1.0 ~ +1.0 (양수=롱, 음수=숏)
        self.coin_qty = 0.0
        self.entry_price = 0.0
        self.capital_at_entry = 0.0      # 실제 포지션 가치 (진입 수수료 차감 후)
        self._entry_invested = 0.0       # 투입한 현금 (수수료 포함, PnL 기준점)
        self._entry_cash = 0.0           # 진입 직전 전체 자산 (equity 기록용)
        self._entry_dt = None

        self.trades = []

    # ── 외부 인터페이스 ────────────────────────────────────────────────

    def execute(self, signal: float, price: float, dt=None,
                rebalance_threshold: float = 0.0):
        """
        signal: -1.0 ~ +1.0 사이의 float 포지션 비중
                0.0 = 포지션 없음
                0.5 = 자본의 50% 롱
               -0.7 = 자본의 70% 숏

        rebalance_threshold: 이 값 이하의 포지션 변화는 무시 (수수료 절감)
                             기본값 0.0 → 항상 리밸런싱
        """
        # 포지션 변화량이 임계값 미만이면 거래 생략
        if abs(signal - self.position) <= rebalance_threshold:
            return

        # 기존 포지션 청산
        if self.position != 0.0:
            self._close(price, dt)

        # 새 포지션 진입 (signal이 의미 있는 크기일 때만)
        if abs(signal) > rebalance_threshold:
            self._open(signal, price, dt)

    def get_equity(self, current_price: float) -> float:
        """현재 가격 기준 총 자산 = 미투자 현금 + 포지션 평가액"""
        if self.position == 0.0:
            return self.cash

        if self.position > 0:  # 롱
            position_value = self.coin_qty * current_price
        else:  # 숏
            pnl = self.coin_qty * (self.entry_price - current_price)
            position_value = max(self.capital_at_entry + pnl, 0.0)

        return self.cash + position_value

    # ── 내부 구현 ─────────────────────────────────────────────────────

    def _open(self, signal: float, price: float, dt=None):
        """
        signal 비중만큼 현금을 투자해서 포지션 진입.
        나머지 현금은 self.cash에 그대로 보유.
        """
        direction = 1 if signal > 0 else -1
        fraction = abs(signal)               # 투자 비율 (0 ~ 1)

        invest_cash = self.cash * fraction   # 실제 투입할 현금
        entry_fee = invest_cash * self.fee_rate
        self.capital_at_entry = invest_cash - entry_fee   # 수수료 차감 후 포지션 가치
        self.coin_qty = self.capital_at_entry / price

        self._entry_invested = invest_cash   # PnL 계산 기준점
        self._entry_cash = self.cash         # 진입 전 전체 현금 (로깅용)
        self._entry_dt = dt

        self.position = signal               # float 그대로 저장
        self.entry_price = price
        self.cash -= invest_cash             # 투입한 만큼 현금 차감

    def _close(self, price: float, dt=None):
        """포지션 전체 청산. 청산 금액을 self.cash에 복귀."""
        direction = 1 if self.position > 0 else -1

        if direction == 1:  # 롱 청산
            gross = self.coin_qty * price
            close_fee = gross * self.fee_rate
            final_equity = gross - close_fee
        else:  # 숏 청산
            pnl = self.coin_qty * (self.entry_price - price)
            close_fee = self.coin_qty * price * self.fee_rate
            final_equity = self.capital_at_entry + pnl - close_fee

        # 파산 시 최대 손실은 투입 자본 전체 (-100%)
        final_equity = max(final_equity, 0.0)

        self.trades.append({
            'direction':   direction,
            'fraction':    abs(self.position),
            'entry_dt':    self._entry_dt,
            'exit_dt':     dt,
            'entry_price': self.entry_price,
            'exit_price':  price,
            'coin_qty':    self.coin_qty,
            'pnl':         final_equity - self._entry_invested,
            'pnl_pct':     (final_equity / self._entry_invested - 1) * 100
                           if self._entry_invested > 0 else -100.0,
        })

        self.cash += final_equity            # 청산 금액을 현금으로 복귀
        self.position = 0.0
        self.coin_qty = 0.0
        self.entry_price = 0.0
        self.capital_at_entry = 0.0
        self._entry_invested = 0.0
        self._entry_dt = None
