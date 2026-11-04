import numpy as np
import pandas as pd

from trader.config import Config
from trader.signals import BUY, DOWN, HOLD, NEUTRAL, SELL, UP, evaluate, generate_signal, trend_of_row
from helpers import make_candles

CFG = Config()


def crafted(cross="up", rsi=55.0, hist=None, vol=1.5, n=12, atr=1.0):
    ts = pd.date_range("2026-03-01", periods=n, freq="15min", tz="UTC")
    if cross == "up":
        ema9 = np.r_[np.full(n - 1, 99.0), 101.0]
        hist = 0.5 if hist is None else hist
        close = 101.0
    elif cross == "down":
        ema9 = np.r_[np.full(n - 1, 101.0), 99.0]
        hist = -0.5 if hist is None else hist
        close = 99.0
    else:
        ema9 = np.full(n, 101.0)
        hist = 0.5 if hist is None else hist
        close = 101.0
    return pd.DataFrame({
        "timestamp": ts, "close": close, "ema9": ema9, "ema21": 100.0, "rsi14": rsi,
        "macd_hist": hist, "atr": atr, "vol_ratio": vol,
    })


def test_buy_when_all_conditions_met():
    s = evaluate(crafted("up"), 11, UP, "BTC/USDT", "15m", CFG)
    assert s.action == BUY and s.atr == 1.0 and s.candle_time == pd.Timestamp("2026-03-01 02:45", tz="UTC")


def test_sell_when_all_conditions_met():
    s = evaluate(crafted("down", rsi=45.0), 11, DOWN, "BTC/USDT", "15m", CFG)
    assert s.action == SELL


def test_trend_filter_blocks_trade_against_1h():
    s = evaluate(crafted("up"), 11, DOWN, "BTC/USDT", "15m", CFG)
    assert s.action == HOLD and "1H trend" in s.reason
    s = evaluate(crafted("up"), 11, NEUTRAL, "BTC/USDT", "15m", CFG)
    assert s.action == HOLD


def test_rsi_overbought_blocks_buy():
    assert evaluate(crafted("up", rsi=78), 11, UP, "X", "15m", CFG).action == HOLD


def test_low_volume_blocks_signal():
    assert evaluate(crafted("up", vol=0.4), 11, UP, "X", "15m", CFG).action == HOLD


def test_macd_must_agree():
    assert evaluate(crafted("up", hist=-0.2), 11, UP, "X", "15m", CFG).action == HOLD


def test_no_cross_means_hold():
    s = evaluate(crafted("none"), 11, UP, "X", "15m", CFG)
    assert s.action == HOLD and "no fresh" in s.reason


def test_stale_cross_outside_lookback_is_ignored():
    df = crafted("up")
    df.loc[df.index[-1], "ema9"] = 101.0
    df["ema9"] = np.r_[np.full(4, 99.0), np.full(8, 101.0)]  # cross happened 7 candles ago
    assert evaluate(df, 11, UP, "X", "15m", CFG).action == HOLD


def test_insufficient_history_or_nan_is_hold():
    assert evaluate(crafted("up", n=3), 2, UP, "X", "15m", CFG).action == HOLD
    df = crafted("up")
    df.loc[df.index[-1], "rsi14"] = np.nan
    assert evaluate(df, 11, UP, "X", "15m", CFG).action == HOLD


def test_evaluate_uses_only_rows_up_to_i():
    """Changing FUTURE rows must not change the signal for row i."""
    df = crafted("up", n=12)  # EMA cross happens on the last row (index 11)
    a = evaluate(df, 11, UP, "X", "15m", CFG)
    future = df.iloc[[-1]].copy()
    future["timestamp"] = future["timestamp"] + pd.Timedelta(minutes=15)
    future[["close", "ema9", "rsi14"]] = [50.0, 50.0, 99.0]
    df2 = pd.concat([df, future], ignore_index=True)
    assert a.action == BUY
    assert evaluate(df2, 11, UP, "X", "15m", CFG).action == BUY


def test_trend_of_row():
    assert trend_of_row({"ema9": 2, "ema21": 1, "close": 2}) == UP
    assert trend_of_row({"ema9": 1, "ema21": 2, "close": 1}) == DOWN
    assert trend_of_row({"ema9": 2, "ema21": 1, "close": 0.5}) == NEUTRAL
    assert trend_of_row({"ema9": np.nan, "ema21": 1, "close": 1}) == NEUTRAL


def test_generate_signal_on_real_pipeline_returns_valid_signal():
    f15 = make_candles(200, "15m", seed=11, sigma=0.004)
    f1h = make_candles(200, "1h", seed=12, sigma=0.004)
    s = generate_signal(f15, f1h, "ETH/USDT", "15m", CFG)
    assert s.action in (BUY, SELL, HOLD)
    assert s.candle_time == f15["timestamp"].iloc[-1]
    assert s.close_time == f15["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)


def test_generate_signal_without_htf_cannot_buy():
    f15 = make_candles(200, "15m", seed=11, sigma=0.004)
    s = generate_signal(f15, None, "ETH/USDT", "15m", CFG)
    assert s.trend == NEUTRAL and s.action != BUY and s.action != SELL
