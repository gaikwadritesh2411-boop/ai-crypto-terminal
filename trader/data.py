"""Market data via CCXT (public endpoints only, no API keys).

Rules enforced here:
  * only COMPLETED candles are ever returned (the in-progress candle is dropped)
  * no data is ever invented: any failure raises DataError
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .config import TF_SECONDS
from .utils import now_utc, utc

log = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DataError(Exception):
    """Market data could not be fetched or failed validation."""


def tf_delta(timeframe: str) -> pd.Timedelta:
    if timeframe not in TF_SECONDS:
        raise DataError(f"unsupported timeframe: {timeframe}")
    return pd.Timedelta(seconds=TF_SECONDS[timeframe])


def to_dataframe(raw) -> pd.DataFrame:
    if not raw:
        raise DataError("exchange returned no candles")
    df = pd.DataFrame(raw, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=COLUMNS[1:5])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise DataError("no valid candles after cleaning")
    bad = (df["high"] < df["low"]) | (df["close"] <= 0)
    if bad.any():
        df = df[~bad].reset_index(drop=True)
    if df.empty:
        raise DataError("no valid candles after validation")
    return df


def drop_incomplete(df: pd.DataFrame, timeframe: str, now=None) -> pd.DataFrame:
    """Keep only candles whose close time is <= now."""
    now = utc(now) if now is not None else now_utc()
    close_time = df["timestamp"] + tf_delta(timeframe)
    return df[close_time <= now].reset_index(drop=True)


class MarketData:
    """Thin CCXT wrapper. `exchange` can be injected for tests."""

    def __init__(self, exchange_id: str = "kraken", exchange=None):
        self.exchange_id = exchange_id
        self._exchange = exchange

    @property
    def exchange(self):
        if self._exchange is None:
            try:
                import ccxt

                klass = getattr(ccxt, self.exchange_id)
            except (ImportError, AttributeError) as e:
                raise DataError(f"cannot load exchange '{self.exchange_id}': {e}") from e
            # No apiKey / secret: public market data only.
            self._exchange = klass({"enableRateLimit": True, "timeout": 10_000})
        return self._exchange

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300, now=None) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except DataError:
            raise
        except Exception as e:  # network, rate limit, geo-block, bad symbol ...
            raise DataError(f"{self.exchange_id} {symbol} {timeframe}: {type(e).__name__}: {e}") from e
        df = drop_incomplete(to_dataframe(raw), timeframe, now)
        if df.empty:
            raise DataError(f"{symbol} {timeframe}: no completed candles")
        return df

    def fetch_history(self, symbol: str, timeframe: str, days: int, now=None) -> pd.DataFrame:
        """Paginated history for backtests. Exchanges may cap how far back they serve."""
        now = utc(now) if now is not None else now_utc()
        since = int((now - pd.Timedelta(days=days)).timestamp() * 1000)
        end = int(now.timestamp() * 1000)
        rows: list = []
        last_seen: Optional[int] = None
        for _ in range(60):
            try:
                batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            except Exception as e:
                raise DataError(f"{self.exchange_id} {symbol} {timeframe}: {type(e).__name__}: {e}") from e
            if not batch:
                break
            rows.extend(batch)
            newest = batch[-1][0]
            if last_seen is not None and newest <= last_seen:
                break
            last_seen = newest
            if newest >= end:
                break
            since = newest + 1
        df = drop_incomplete(to_dataframe(rows), timeframe, now)
        if df.empty:
            raise DataError(f"{symbol} {timeframe}: no completed candles in history")
        return df

    def fetch_last_prices(self, symbols) -> dict:
        """Latest traded prices. Returns {} on failure (never made-up numbers)."""
        out: dict = {}
        try:
            tickers = self.exchange.fetch_tickers(list(symbols))
            for s in symbols:
                last = (tickers.get(s) or {}).get("last")
                if last:
                    out[s] = float(last)
        except Exception as e:
            log.warning("ticker fetch failed: %s", e)
        return out
