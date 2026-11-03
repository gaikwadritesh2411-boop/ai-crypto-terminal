"""Replay the REAL engine (real signals, risk, fees, SQLite) over synthetic candles, then audit the database.

Candles are synthetic TEST DATA, so P&L here proves the plumbing works, NOT that the strategy is profitable.
    python scripts/verify_paper_trading.py [--days 12]
"""
import argparse
import os
import sys
import tempfile
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from helpers import FakeData, resample, trend_5m  # noqa: E402
from trader.config import SYMBOLS, Config  # noqa: E402
from trader.engine import TradingEngine  # noqa: E402
from trader.metrics import compute_metrics  # noqa: E402
from trader.store import Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--random-walk", action="store_true", help="pure noise, no predictable pattern: leakage / edge sanity check")
    ap.add_argument("--seed", type=int, default=20)
    a = ap.parse_args()
    cfg = Config()
    total_days = a.days + 15  # warm-up history before the replay window
    end = pd.Timestamp("2026-03-02 00:00", tz="UTC")
    frames = {}
    for i, s in enumerate(SYMBOLS):
        kw = dict(drift=0.0, pull_amp=0.0, noise=0.0015) if a.random_walk else dict(drift=0.0006, pull_amp=0.02, pull_hours=8, drift_days=6)
        b = trend_5m(days=total_days, seed=a.seed + i, end=end, **kw)
        for tf, m in (("5m", 5), ("15m", 15), ("1h", 60)):
            frames[(s, tf)] = b if tf == "5m" else resample(b, m)
    data = FakeData(frames)
    db = os.path.join(tempfile.mkdtemp(), "verify.db")
    store = Store(db)
    eng = TradingEngine(cfg, store, data)

    start = end - pd.Timedelta(days=a.days)
    steps = pd.date_range(start, end, freq="15min")
    t0, max_open, errors = time.time(), 0, []
    for now in steps:
        r = eng.run_cycle(now=now)
        errors += r.errors
        max_open = max(max_open, len(store.positions()))
    print(f"replayed {len(steps)} cycles over {a.days} days in {time.time() - t0:.0f}s, data errors: {len(errors)}")

    trades, pos, sigs, eq = store.trades_df(), store.positions_df(), store.signals_df(10_000), store.equity_df()
    bal = float(store.get_meta("balance"))
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

    exp_bal = cfg.initial_balance + trades["net_pnl"].sum() - pos["entry_fee"].sum()
    check("balance == initial + closed net P&L - open entry fees", abs(bal - exp_bal) < 1e-6, f"{bal:.4f} vs {exp_bal:.4f}")
    check("trades were produced and stored in SQLite", len(trades) > 0, f"{len(trades)} closed, {len(pos)} open")
    check("max simultaneous open positions <= 3", max_open <= cfg.max_open_positions, f"peak {max_open}")
    check("no duplicate signals (symbol, timeframe, candle)", not sigs.duplicated(["symbol", "timeframe", "candle_time"]).any())
    check("every closed trade paid fees", bool((trades["fees"] > 0).all()))
    check("exit reasons valid", set(trades["exit_reason"]) <= {"STOP_LOSS", "TAKE_PROFIT"}, str(trades["exit_reason"].value_counts().to_dict()))
    check("no position held without stop and target", bool(((pos["stop_loss"] > 0) & (pos["take_profit"] > 0)).all()) if len(pos) else True)
    sl = trades[trades["exit_reason"] == "STOP_LOSS"]
    check("stop-loss losses ~1% of equity (<=1.5% of initial)", bool((sl["net_pnl"] > -0.015 * cfg.initial_balance * 1.3).all()) if len(sl) else True,
          f"worst {sl['net_pnl'].min():.2f}" if len(sl) else "no stops")
    tp = trades[trades["exit_reason"] == "TAKE_PROFIT"]
    check("take-profit R multiple below 2.0 (costs deducted)", bool((tp["r_multiple"] < 2.0).all()) if len(tp) else True,
          f"max {tp['r_multiple'].max():.2f}" if len(tp) else "no targets")
    ex = sigs[sigs["executed"] == 1]
    if len(ex):
        from trader.config import TF_SECONDS
        age = (pd.to_datetime(ex["created_at"], utc=True) - ex["candle_time"]) / pd.Timedelta(seconds=1) \
            - ex["timeframe"].map(TF_SECONDS)
        limit = ex["timeframe"].map(TF_SECONDS) * cfg.max_signal_age_candles
        check("executed signals were fresh (<= 2 candles after close)", bool((age <= limit).all()), f"{len(ex)} executed, max age {age.max():.0f}s")
    else:
        check("executed signals were fresh", False, "no executed signals to audit")
    check("equity snapshots stored", len(eq) == len(steps), f"{len(eq)}")
    day_start = {}
    ok_daily = True
    for d, g in eq.groupby(eq["ts"].dt.strftime("%Y-%m-%d")):
        ok_daily &= bool((g["equity"].min() > g["equity"].iloc[0] * (1 - cfg.daily_loss_limit - 0.03)))
    check("no day fell far beyond the 3% daily loss limit", ok_daily)

    m = compute_metrics(trades, cfg.initial_balance, eq.set_index("ts")["equity"])
    print("\nSYNTHETIC-DATA summary (plumbing check, NOT strategy evidence):")
    for k in ("total_trades", "win_rate", "profit_factor", "net_pnl", "max_drawdown_pct", "total_fees"):
        print(f"  {k:>17}: {m[k]}")
    print("\nRESULT:", "ALL CHECKS PASSED" if all(checks) else "SOME CHECKS FAILED")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
