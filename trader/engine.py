"""One trading cycle: fetch completed candles -> manage exits -> signals -> risk gate -> paper entry.

The cycle is idempotent: running it twice on the same candles never duplicates a signal or trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .config import EXIT_TIMEFRAME, HTF, SYMBOLS, TF_SECONDS, TIMEFRAMES, Config
from .data import DataError
from .indicators import ensure_indicators
from .portfolio import Portfolio, Position, Trade
from .risk import LONG, SHORT
from .signals import BUY, HOLD, SELL, Signal, generate_signal
from .store import Store
from .utils import now_utc, utc

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    ran_at: pd.Timestamp
    signals: list = field(default_factory=list)  # every Signal evaluated (incl. HOLD)
    opened: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    marks: dict = field(default_factory=dict)
    equity: float = 0.0


class TradingEngine:
    def __init__(self, cfg: Config, store: Store, data, symbols=SYMBOLS):
        self.cfg, self.store, self.data, self.symbols = cfg, store, data, tuple(symbols)

    def _is_duplicate(self, sig: Signal) -> bool:
        if self.store.signal_exists(sig.symbol, sig.timeframe, sig.candle_time):
            return True
        last = self.store.last_signal(sig.symbol, sig.timeframe)
        if last and last["action"] == sig.action:
            gap = sig.candle_time - utc(last["candle_time"])
            cooldown = pd.Timedelta(seconds=TF_SECONDS[sig.timeframe] * self.cfg.signal_cooldown_candles)
            if abs(gap) < cooldown:
                return True
        return False

    def run_cycle(self, now=None) -> CycleResult:
        cfg = self.cfg
        now = utc(now) if now is not None else now_utc()
        res = CycleResult(ran_at=now)
        pf = Portfolio.load(cfg, self.store)

        frames: dict = {}
        for sym in self.symbols:
            for tf in TIMEFRAMES:
                try:
                    df = self.data.fetch_ohlcv(sym, tf, limit=cfg.candle_limit, now=now)
                    frames[(sym, tf)] = ensure_indicators(df, cfg)
                except DataError as e:
                    res.errors.append(str(e))
                except Exception as e:  # never let one symbol kill the cycle
                    res.errors.append(f"{sym} {tf}: unexpected {type(e).__name__}: {e}")

        # Mark prices: last completed 5m close, overridden by live ticker when available.
        marks = {s: float(frames[(s, EXIT_TIMEFRAME)]["close"].iloc[-1])
                 for s in self.symbols if (s, EXIT_TIMEFRAME) in frames}
        try:
            marks.update(self.data.fetch_last_prices(self.symbols) or {})
        except Exception as e:
            res.errors.append(f"ticker fetch failed: {e}")
        res.marks = marks

        pf.roll_day(now, marks)

        # 1) exits (stop loss / take profit) on completed 5m candles
        for sym in self.symbols:
            fr = frames.get((sym, EXIT_TIMEFRAME))
            if fr is not None:
                res.closed += pf.process_candles(sym, fr, TF_SECONDS[EXIT_TIMEFRAME])

        # 2) signals -> risk gate -> paper entries
        for sym in self.symbols:
            htf = frames.get((sym, HTF))
            for tf in TIMEFRAMES:
                fr = frames.get((sym, tf))
                if fr is None:
                    continue
                if htf is None:
                    res.notes.append(f"{sym} {tf}: no 1H data, trend unconfirmed")
                    continue
                sig = generate_signal(fr, htf, sym, tf, cfg)
                res.signals.append(sig)
                if sig.action == HOLD:
                    continue
                if self._is_duplicate(sig):
                    continue
                if not self.store.record_signal(sig, now):
                    continue  # another process (dashboard/worker) already handled this candle
                age = now - sig.close_time
                if tf not in cfg.trade_timeframes:
                    self.store.update_signal(sym, tf, sig.candle_time, False, "timeframe not traded")
                    continue
                if age > pd.Timedelta(seconds=TF_SECONDS[tf] * cfg.max_signal_age_candles):
                    self.store.update_signal(sym, tf, sig.candle_time, False, "stale signal, not executed")
                    continue
                side = LONG if sig.action == BUY else SHORT
                pos, why = pf.open_position(sym, side, tf, sig.price, sig.atr, sig.close_time, now, marks)
                if pos:
                    res.opened.append(pos)
                    self.store.update_signal(sym, tf, sig.candle_time, True, f"opened {side}: {why}")
                else:
                    self.store.update_signal(sym, tf, sig.candle_time, False, f"blocked: {why}")
                    res.notes.append(f"{sym} {tf} {sig.action} blocked: {why}")

        # 3) equity snapshot
        res.equity = pf.equity(marks)
        self.store.snapshot_equity(now, res.equity, pf.balance, pf.unrealized(marks), len(pf.positions))
        return res
