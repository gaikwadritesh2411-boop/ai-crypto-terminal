"""Test helpers. Synthetic candles exist ONLY in the test suite; the app never fabricates data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.config import TF_SECONDS
from trader.data import DataError, drop_incomplete
from trader.signals import Signal

NOW = pd.Timestamp("2026-03-02 12:00", tz="UTC")


def make_candles(n: int, tf: str, end=NOW, price: float = 100.0, seed: int | None = None,
                 sigma: float = 0.0, volume: float = 100.0) -> pd.DataFrame:
    """n completed candles; the last one CLOSES exactly at `end`."""
    step = pd.Timedelta(seconds=TF_SECONDS[tf])
    ts = pd.date_range(end=end - step, periods=n, freq=step)
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(sigma * rng.standard_normal(n))) if sigma else np.full(n, price)
    open_ = np.r_[price, close[:-1]]
    return pd.DataFrame({
        "timestamp": ts, "open": open_,
        "high": np.maximum(open_, close) * 1.0005, "low": np.minimum(open_, close) * 0.9995,
        "close": close, "volume": volume * (1 + 0.3 * rng.random(n)),
    })


def wave_5m(days: int = 25, base: float = 100.0, amp: float = 0.06, period_h: float = 30.0,
            noise: float = 0.0006, seed: int = 1, end=NOW) -> pd.DataFrame:
    """Oscillating random-ish 5m path so EMAs cross repeatedly. Test data only."""
    n = days * 288
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    path = base * (1 + amp * np.sin(2 * np.pi * k / (period_h * 12))) * np.exp(np.cumsum(noise * rng.standard_normal(n)))
    ts = pd.date_range(end=end - pd.Timedelta(minutes=5), periods=n, freq="5min")
    open_ = np.r_[path[0], path[:-1]]
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": np.maximum(open_, path) * 1.0004,
        "low": np.minimum(open_, path) * 0.9996, "close": path,
        "volume": 100 * (1 + rng.random(n) * 2),
    })


def trend_5m(days: int = 25, base: float = 100.0, seed: int = 1, end=NOW, drift_days: float = 4.0,
             drift: float = 0.0004, pull_amp: float = 0.012, pull_hours: float = 5.0, noise: float = 0.0008) -> pd.DataFrame:
    """Trending path with pullbacks (regime flips every `drift_days`) so 15m crosses can align with 1H trend. Test data only."""
    n = days * 288
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    sign = np.where((k // int(drift_days * 288)) % 2 == 0, 1.0, -1.0)
    path = base * np.exp(np.cumsum(sign * drift + noise * rng.standard_normal(n))) \
        * (1 + pull_amp * np.sin(2 * np.pi * k / (pull_hours * 12)))
    ts = pd.date_range(end=end - pd.Timedelta(minutes=5), periods=n, freq="5min")
    open_ = np.r_[path[0], path[:-1]]
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": np.maximum(open_, path) * 1.0004,
        "low": np.minimum(open_, path) * 0.9996, "close": path,
        "volume": 100 * (1 + rng.random(n) * 2),
    })


def resample(df5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = df5.set_index("timestamp").resample(f"{minutes}min", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return out.reset_index()


class FakeData:
    """Stands in for MarketData. frames: {(symbol, tf): DataFrame of candles}."""

    def __init__(self, frames=None, prices=None, fail=()):
        self.frames = frames or {}
        self.prices = prices or {}
        self.fail = set(fail)

    def fetch_ohlcv(self, symbol, timeframe, limit=300, now=None):
        if (symbol, timeframe) in self.fail or (symbol, timeframe) not in self.frames:
            raise DataError(f"{symbol} {timeframe}: simulated outage")
        return drop_incomplete(self.frames[(symbol, timeframe)], timeframe, now).tail(limit).reset_index(drop=True)

    def fetch_last_prices(self, symbols):
        return dict(self.prices)


def flat_frames(symbols, end=NOW, price=100.0):
    f = {}
    for s in symbols:
        for tf in ("5m", "15m", "1h"):
            f[(s, tf)] = make_candles(120, tf, end=end, price=price, seed=3)
    return f


def sig(symbol="BTC/USDT", tf="15m", action="BUY", price=100.0, atr=1.0, end=NOW, trend="UP"):
    step = pd.Timedelta(seconds=TF_SECONDS[tf])
    return Signal(symbol, tf, action, price, end - step, atr, trend, "test signal", 55.0, 1.5)
