"""Historical backtest using the SAME signal, risk, fee and slippage logic as paper trading.

Honest-results notes:
  * Indicators are causal, so computing them over the whole series does not leak the future.
  * Entries happen at the CLOSE of the signal candle (+ slippage); exits are checked on later candles.
  * Exits are checked on the base timeframe candles (coarser than live 5m checks), and a candle that
    touches both stop and target is counted as a stop.
  * Open positions at the end are closed at the last close with reason END_OF_TEST.
Run `python -m trader.backtest --days 30` for a CLI report on live exchange history.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .config import HTF, SYMBOLS, TF_SECONDS, Config
from .indicators import ensure_indicators
from .metrics import compute_metrics
from .portfolio import Portfolio
from .risk import LONG, SHORT
from .signals import BUY, HOLD, evaluate, trend_labels

WARMUP = 60


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    signals_seen: int = 0
    notes: list = field(default_factory=list)


def run_backtest(data: Mapping[str, Mapping[str, pd.DataFrame]], cfg: Config | None = None,
                 base_tf: str = "15m") -> BacktestResult:
    cfg = cfg or Config()
    base: dict = {}
    htf_info: dict = {}
    for sym, frames in data.items():
        if base_tf not in frames or HTF not in frames:
            continue
        b = ensure_indicators(frames[base_tf], cfg).reset_index(drop=True)
        h = ensure_indicators(frames[HTF], cfg).reset_index(drop=True)
        base[sym] = b
        htf_info[sym] = (pd.DatetimeIndex(h["timestamp"] + pd.Timedelta(seconds=TF_SECONDS[HTF])),
                         trend_labels(h))
    if not base:
        raise ValueError(f"need both {base_tf} and {HTF} candles for at least one symbol")

    step = pd.Timedelta(seconds=TF_SECONDS[base_tf])
    index = {s: {t: i for i, t in enumerate(df["timestamp"])} for s, df in base.items()}
    timeline = sorted(set().union(*[set(df["timestamp"]) for df in base.values()]))

    pf = Portfolio(cfg)  # in-memory, no database
    last_sig: dict = {}
    eq_times, eq_vals = [], []
    marks: dict = {}
    seen = 0
    cooldown = step * cfg.signal_cooldown_candles

    for ts in timeline:
        now = ts + step  # candle close time: the earliest moment its data is known
        for sym, df in base.items():
            i = index[sym].get(ts)
            if i is not None:
                marks[sym] = float(df["close"].iloc[i])
        pf.roll_day(now, marks)

        # exits first, using the candle that just closed (entries are always strictly earlier)
        for sym, df in base.items():
            i = index[sym].get(ts)
            if i is not None:
                pf.process_candles(sym, df.iloc[[i]], TF_SECONDS[base_tf])

        for sym, df in base.items():
            i = index[sym].get(ts)
            if i is None or i < WARMUP:
                continue
            h_close, labels = htf_info[sym]
            pos = h_close.get_indexer([now], method="pad")[0]
            trend = labels[pos] if pos >= 0 else "NEUTRAL"
            sig = evaluate(df, i, trend, sym, base_tf, cfg)
            if sig.action == HOLD:
                continue
            seen += 1
            prev = last_sig.get(sym)
            if prev and prev[0] == sig.action and (sig.candle_time - prev[1]) < cooldown:
                continue
            last_sig[sym] = (sig.action, sig.candle_time)
            side = LONG if sig.action == BUY else SHORT
            pf.open_position(sym, side, base_tf, sig.price, sig.atr, sig.close_time, now, marks)

        eq_times.append(now)
        eq_vals.append(pf.equity(marks))

    for pos in list(pf.positions):  # close leftovers at last known price
        px = marks.get(pos.symbol, pos.entry_price)
        pf.close_position(pos, px, timeline[-1] + step, "END_OF_TEST")
    if eq_vals:
        eq_vals[-1] = pf.equity(marks)

    cols = ["symbol", "side", "timeframe", "qty", "entry_price", "exit_price", "entry_time", "exit_time",
            "gross_pnl", "fees", "net_pnl", "exit_reason", "risk_amount", "r_multiple"]
    trades = pd.DataFrame([t.__dict__ for t in pf.trades], columns=cols)
    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_times), name="equity")
    notes = []
    if len(trades) < 30:
        notes.append(f"Only {len(trades)} trades: too few to draw statistical conclusions.")
    span_days = (timeline[-1] - timeline[0]).total_seconds() / 86400 if len(timeline) > 1 else 0
    if span_days < 30:
        notes.append(f"Test covers only {span_days:.1f} days of one market regime.")
    return BacktestResult(trades, equity, compute_metrics(trades, cfg.initial_balance, equity), seen, notes)


def main(argv=None) -> int:  # pragma: no cover - needs live network
    from .data import DataError, MarketData

    p = argparse.ArgumentParser(description="Backtest the strategy on real exchange history (paper only).")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--base-tf", default="15m", choices=["5m", "15m", "1h"])
    a = p.parse_args(argv)
    cfg = Config()
    md = MarketData(cfg.exchange_id)
    data: dict = {}
    for sym in SYMBOLS:
        try:
            data[sym] = {a.base_tf: md.fetch_history(sym, a.base_tf, a.days),
                         HTF: md.fetch_history(sym, HTF, a.days + 10)}
        except DataError as e:
            print(f"skip {sym}: {e}")
    if not data:
        print("No market data available; nothing was backtested.")
        return 1
    r = run_backtest(data, cfg, a.base_tf)
    for k, v in r.metrics.items():
        print(f"{k:>18}: {v}")
    for n in r.notes:
        print("NOTE:", n)
    print("Backtest results describe the simulated past. They do not prove future profit.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
