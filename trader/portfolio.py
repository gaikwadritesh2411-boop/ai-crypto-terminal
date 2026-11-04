"""Paper portfolio: simulated order execution, fees, slippage, stops, P&L, risk gates.

Nothing here talks to an exchange. `store` (optional) persists every change to SQLite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import Config
from .risk import LONG, SHORT, build_plan, daily_loss_breached, size_position
from .utils import utc


@dataclass
class Position:
    symbol: str
    side: str
    timeframe: str
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    entry_fee: float
    risk_amount: float
    id: Optional[int] = None

    @property
    def direction(self) -> int:
        return 1 if self.side == LONG else -1

    @property
    def notional(self) -> float:
        return self.entry_price * self.qty

    def unrealized(self, mark: float) -> float:
        return (mark - self.entry_price) * self.qty * self.direction


@dataclass
class Trade:
    symbol: str
    side: str
    timeframe: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    risk_amount: float
    r_multiple: float


def check_exit(pos: Position, open_: float, high: float, low: float, cfg: Config):
    """Return (exit_price, reason) if this candle hits the stop or target, else None.

    Conservative: if one candle touches both levels, the STOP is assumed to hit first.
    Stops are market orders (slippage applied, gap-aware); targets are limit orders (no slippage).
    """
    s = cfg.slippage_rate
    if pos.side == LONG:
        if low <= pos.stop_loss:
            return min(pos.stop_loss, open_) * (1 - s), "STOP_LOSS"
        if high >= pos.take_profit:
            return pos.take_profit, "TAKE_PROFIT"
    else:
        if high >= pos.stop_loss:
            return max(pos.stop_loss, open_) * (1 + s), "STOP_LOSS"
        if low <= pos.take_profit:
            return pos.take_profit, "TAKE_PROFIT"
    return None


class Portfolio:
    def __init__(self, cfg: Config, store=None):
        self.cfg = cfg
        self.store = store
        self.balance: float = cfg.initial_balance
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.day_key: Optional[str] = None
        self.day_start_equity: Optional[float] = None

    # ---- persistence ----
    @classmethod
    def load(cls, cfg: Config, store) -> "Portfolio":
        pf = cls(cfg, store)
        bal = store.get_meta("balance")
        if bal is None:
            store.set_meta("balance", pf.balance)
        else:
            pf.balance = float(bal)
        for r in store.positions():
            pf.positions.append(Position(
                symbol=r["symbol"], side=r["side"], timeframe=r["timeframe"], qty=r["qty"],
                entry_price=r["entry_price"], stop_loss=r["stop_loss"], take_profit=r["take_profit"],
                entry_time=utc(r["entry_time"]), entry_fee=r["entry_fee"], risk_amount=r["risk_amount"],
                id=r["id"]))
        pf.day_key = store.get_meta("day_key")
        dse = store.get_meta("day_start_equity")
        pf.day_start_equity = float(dse) if dse is not None else None
        return pf

    def _save_balance(self) -> None:
        if self.store:
            self.store.set_meta("balance", self.balance)

    # ---- accounting ----
    def unrealized(self, marks: Optional[dict] = None) -> float:
        marks = marks or {}
        return sum(p.unrealized(marks.get(p.symbol, p.entry_price)) for p in self.positions)

    def equity(self, marks: Optional[dict] = None) -> float:
        return self.balance + self.unrealized(marks)

    def open_notional(self) -> float:
        return sum(p.notional for p in self.positions)

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.positions)

    def roll_day(self, now, marks: Optional[dict] = None) -> None:
        key = utc(now).strftime("%Y-%m-%d")
        if self.day_key != key or self.day_start_equity is None:
            self.day_key = key
            self.day_start_equity = self.equity(marks)
            if self.store:
                self.store.set_meta("day_key", key)
                self.store.set_meta("day_start_equity", self.day_start_equity)

    def daily_pnl(self, marks: Optional[dict] = None) -> float:
        if self.day_start_equity is None:
            return 0.0
        return self.equity(marks) - self.day_start_equity

    # ---- risk gate ----
    def can_open(self, symbol: str, now, marks: Optional[dict] = None):
        self.roll_day(now, marks)
        eq = self.equity(marks)
        if eq <= 0:
            return False, "equity depleted"
        if self.has_position(symbol):
            return False, f"position already open in {symbol}"
        if len(self.positions) >= self.cfg.max_open_positions:
            return False, f"max open positions ({self.cfg.max_open_positions}) reached"
        if daily_loss_breached(eq, self.day_start_equity, self.cfg):
            return False, f"daily loss limit ({self.cfg.daily_loss_limit:.0%}) reached"
        return True, "ok"

    # ---- execution ----
    def open_position(self, symbol: str, side: str, timeframe: str, price: float, atr: float,
                      entry_time, now, marks: Optional[dict] = None):
        """Simulated market entry at `price` (+ slippage). Returns (Position|None, reason)."""
        if side == SHORT and not self.cfg.allow_short:
            return None, "short selling disabled"
        ok, why = self.can_open(symbol, now, marks)
        if not ok:
            return None, why
        plan = build_plan(side, price, atr, self.cfg)
        if plan is None:
            return None, "invalid price/ATR for trade plan"
        eq = self.equity(marks)
        qty, capped = size_position(eq, self.open_notional(), plan, self.cfg)
        if qty * plan.entry_price < self.cfg.min_notional:
            return None, "no capital available under leverage cap / below minimum notional"
        fee = self.cfg.fee_rate * plan.entry_price * qty
        self.balance -= fee
        pos = Position(symbol, side, timeframe, qty, plan.entry_price, plan.stop_loss, plan.take_profit,
                       utc(entry_time), fee, qty * plan.stop_distance)
        if self.store:
            pos.id = self.store.insert_position(pos)
            self._save_balance()
        self.positions.append(pos)
        return pos, "capped by leverage limit" if capped else "ok"

    def close_position(self, pos: Position, exit_price: float, exit_time, reason: str) -> Trade:
        exit_fee = self.cfg.fee_rate * exit_price * pos.qty
        gross = (exit_price - pos.entry_price) * pos.qty * pos.direction
        self.balance += gross - exit_fee
        net = gross - pos.entry_fee - exit_fee
        trade = Trade(pos.symbol, pos.side, pos.timeframe, pos.qty, pos.entry_price, exit_price,
                      pos.entry_time, utc(exit_time), gross, pos.entry_fee + exit_fee, net, reason,
                      pos.risk_amount, net / pos.risk_amount if pos.risk_amount > 0 else 0.0)
        self.positions.remove(pos)
        self.trades.append(trade)
        if self.store:
            if pos.id is not None:
                self.store.delete_position(pos.id)
            self.store.insert_trade(trade)
            self._save_balance()
        return trade

    def process_candles(self, symbol: str, candles: pd.DataFrame, candle_seconds: int) -> list:
        """Check open positions in `symbol` against COMPLETED candles after entry."""
        closed: list = []
        for pos in [p for p in self.positions if p.symbol == symbol]:
            sub = candles[candles["timestamp"] >= pos.entry_time]
            for row in sub.itertuples(index=False):
                res = check_exit(pos, row.open, row.high, row.low, self.cfg)
                if res:
                    price, reason = res
                    exit_time = row.timestamp + pd.Timedelta(seconds=candle_seconds)
                    closed.append(self.close_position(pos, price, exit_time, reason))
                    break
        return closed
