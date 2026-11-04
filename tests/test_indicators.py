import numpy as np
import pandas as pd

from trader.config import Config
from trader.indicators import add_indicators, atr, ema, macd, rsi
from helpers import make_candles


def test_ema_constant_series_is_constant():
    s = pd.Series([5.0] * 60)
    assert np.allclose(ema(s, 9).dropna(), 5.0)


def test_ema_reacts_faster_with_shorter_span():
    s = pd.Series([1.0] * 40 + [2.0] * 5)
    assert ema(s, 9).iloc[-1] > ema(s, 21).iloc[-1]


def test_rsi_extremes_and_flat():
    up = pd.Series(np.arange(1, 60, dtype=float))
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    flat = pd.Series([10.0] * 60)
    assert rsi(up).iloc[-1] == 100.0
    assert rsi(down).iloc[-1] < 1.0
    assert rsi(flat).iloc[-1] == 50.0


def test_rsi_bounded_on_random_walk():
    df = make_candles(400, "15m", seed=7, sigma=0.01)
    r = rsi(df["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_macd_histogram_is_line_minus_signal():
    df = make_candles(200, "15m", seed=2, sigma=0.01)
    line, sig_, hist = macd(df["close"])
    assert np.allclose((line - sig_).dropna(), hist.dropna())


def test_atr_positive_and_warmup_nan():
    df = make_candles(100, "5m", seed=4, sigma=0.01)
    a = atr(df, 14)
    assert a.iloc[:13].isna().all()
    assert (a.dropna() > 0).all()


def test_add_indicators_columns_and_no_mutation():
    df = make_candles(120, "1h", seed=5, sigma=0.01)
    before = df.copy()
    out = add_indicators(df, Config())
    for c in ("ema9", "ema21", "rsi14", "macd", "macd_signal", "macd_hist", "atr", "vol_sma", "vol_ratio"):
        assert c in out.columns
    pd.testing.assert_frame_equal(df, before)


def test_indicators_are_causal():
    """Adding future candles must not change indicator values of earlier candles."""
    df = make_candles(200, "15m", seed=9, sigma=0.01)
    short = add_indicators(df.iloc[:150])
    full = add_indicators(df)
    cols = ["ema9", "ema21", "rsi14", "macd_hist", "atr", "vol_ratio"]
    pd.testing.assert_frame_equal(short[cols].reset_index(drop=True), full[cols].iloc[:150].reset_index(drop=True))
