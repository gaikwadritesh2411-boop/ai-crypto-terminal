import pandas as pd
import pytest

from trader.data import DataError, MarketData, drop_incomplete, to_dataframe
from helpers import NOW

MS = lambda ts: int(pd.Timestamp(ts).timestamp() * 1000)


class FakeEx:
    def __init__(self, raw=None, boom=False):
        self.raw, self.boom = raw, boom

    def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
        if self.boom:
            raise RuntimeError("451 restricted location")
        return self.raw

    def fetch_tickers(self, symbols):
        if self.boom:
            raise RuntimeError("down")
        return {"BTC/USDT": {"last": 123.4}, "ETH/USDT": {"last": None}}


def raw_rows(start, n, step_min=15):
    return [[MS(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=step_min * i)), 100, 101, 99, 100.5, 10]
            for i in range(n)]


def test_to_dataframe_sorts_dedupes_and_validates():
    rows = raw_rows("2026-03-02 10:00", 3)
    rows = [rows[2], rows[0], rows[1], rows[1]]
    df = to_dataframe(rows)
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    assert len(df) == 3
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_to_dataframe_empty_raises():
    with pytest.raises(DataError):
        to_dataframe([])


def test_incomplete_candle_is_dropped():
    # candles open 11:15, 11:30, 11:45 ; at 11:50 the 11:45 candle is still forming
    df = to_dataframe(raw_rows("2026-03-02 11:15", 3))
    out = drop_incomplete(df, "15m", now=pd.Timestamp("2026-03-02 11:50", tz="UTC"))
    assert out["timestamp"].iloc[-1] == pd.Timestamp("2026-03-02 11:30", tz="UTC")
    out2 = drop_incomplete(df, "15m", now=pd.Timestamp("2026-03-02 12:00", tz="UTC"))
    assert len(out2) == 3  # exactly at close time the candle counts as complete


def test_fetch_ohlcv_returns_completed_only():
    md = MarketData("fake", exchange=FakeEx(raw_rows("2026-03-02 11:15", 3)))
    df = md.fetch_ohlcv("BTC/USDT", "15m", now=pd.Timestamp("2026-03-02 11:50", tz="UTC"))
    assert len(df) == 2


def test_fetch_failure_raises_dataerror_never_fake_data():
    md = MarketData("fake", exchange=FakeEx(boom=True))
    with pytest.raises(DataError):
        md.fetch_ohlcv("BTC/USDT", "15m")
    with pytest.raises(DataError):
        md.fetch_history("BTC/USDT", "15m", 3)


def test_last_prices_skip_missing_and_survive_errors():
    assert MarketData("fake", exchange=FakeEx()).fetch_last_prices(["BTC/USDT", "ETH/USDT"]) == {"BTC/USDT": 123.4}
    assert MarketData("fake", exchange=FakeEx(boom=True)).fetch_last_prices(["BTC/USDT"]) == {}


def test_unknown_exchange_is_dataerror():
    with pytest.raises(DataError):
        MarketData("definitely_not_an_exchange").exchange
