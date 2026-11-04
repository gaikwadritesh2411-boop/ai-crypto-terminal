"""Rule-based BUY / SELL / HOLD signals on COMPLETED candles only.

BUY  : fresh EMA9-over-EMA21 cross (within `cross_lookback` candles), EMA9 still above EMA21,
       MACD histogram > 0, RSI inside [rsi_buy_min, rsi_buy_max), volume >= average,
       and the 1H trend is UP.
SELL : mirror image, with the 1H trend DOWN.
Anything else is HOLD (with a reason explaining what was missing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import HTF, Config, TF_SECONDS
from .indicators import ensure_indicators

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"
UP, DOWN, NEUTRAL = "UP", "DOWN", "NEUTRAL"

_REQUIRED = ("ema9", "ema21", "rsi14", "macd_hist", "atr", "vol_ratio")


@dataclass(frozen=True)
class Signal:
    symbol: str
    timeframe: str
    action: str
    price: float
    candle_time: pd.Timestamp  # OPEN time of the completed candle that produced the signal
    atr: float
    trend: str
    reason: str
    rsi: float = float("nan")
    vol_ratio: float = float("nan")

    @property
    def close_time(self) -> pd.Timestamp:
        return self.candle_time + pd.Timedelta(seconds=TF_SECONDS[self.timeframe])


def trend_of_row(row) -> str:
    vals = (row["ema9"], row["ema21"], row["close"])
    if any(pd.isna(v) for v in vals):
        return NEUTRAL
    if row["ema9"] > row["ema21"] and row["close"] > row["ema21"]:
        return UP
    if row["ema9"] < row["ema21"] and row["close"] < row["ema21"]:
        return DOWN
    return NEUTRAL


def trend_labels(df_htf: pd.DataFrame) -> np.ndarray:
    return np.array([trend_of_row(r) for _, r in df_htf.iterrows()], dtype=object)


def evaluate(df: pd.DataFrame, i: int, trend: str, symbol: str, timeframe: str, cfg: Config) -> Signal:
    """Evaluate row `i` of an indicator-enriched frame (all rows must be completed candles)."""
    row = df.iloc[i]

    def hold(reason: str) -> Signal:
        return Signal(symbol, timeframe, HOLD, float(row["close"]), row["timestamp"],
                      float(row["atr"]) if pd.notna(row["atr"]) else float("nan"),
                      trend, reason, _f(row["rsi14"]), _f(row["vol_ratio"]))

    if i < cfg.cross_lookback + 1 or any(pd.isna(row[c]) for c in _REQUIRED):
        return hold("insufficient history for indicators")

    diff = (df["ema9"] - df["ema21"]).iloc[i - cfg.cross_lookback : i + 1].to_numpy()
    cross_up = bool(np.any((diff[:-1] <= 0) & (diff[1:] > 0))) and diff[-1] > 0
    cross_dn = bool(np.any((diff[:-1] >= 0) & (diff[1:] < 0))) and diff[-1] < 0
    rsi_v, hist, vr = float(row["rsi14"]), float(row["macd_hist"]), float(row["vol_ratio"])

    def build(action: str, why: list) -> Signal:
        return Signal(symbol, timeframe, action, float(row["close"]), row["timestamp"],
                      float(row["atr"]), trend, "; ".join(why), rsi_v, vr)

    if cross_up:
        missing = []
        if trend != UP:
            missing.append(f"1H trend is {trend}, need UP")
        if hist <= 0:
            missing.append("MACD histogram not positive")
        if not (cfg.rsi_buy_min <= rsi_v < cfg.rsi_buy_max):
            missing.append(f"RSI {rsi_v:.1f} outside {cfg.rsi_buy_min:.0f}-{cfg.rsi_buy_max:.0f}")
        if vr < cfg.min_volume_ratio:
            missing.append(f"volume {vr:.2f}x avg below {cfg.min_volume_ratio:.1f}x")
        if not missing:
            return build(BUY, ["EMA9 crossed above EMA21", "1H trend UP", "MACD hist > 0",
                               f"RSI {rsi_v:.1f}", f"volume {vr:.2f}x avg"])
        return hold("bullish cross rejected: " + ", ".join(missing))

    if cross_dn:
        missing = []
        if trend != DOWN:
            missing.append(f"1H trend is {trend}, need DOWN")
        if hist >= 0:
            missing.append("MACD histogram not negative")
        if not (cfg.rsi_sell_min < rsi_v <= cfg.rsi_sell_max):
            missing.append(f"RSI {rsi_v:.1f} outside {cfg.rsi_sell_min:.0f}-{cfg.rsi_sell_max:.0f}")
        if vr < cfg.min_volume_ratio:
            missing.append(f"volume {vr:.2f}x avg below {cfg.min_volume_ratio:.1f}x")
        if not missing:
            return build(SELL, ["EMA9 crossed below EMA21", "1H trend DOWN", "MACD hist < 0",
                                f"RSI {rsi_v:.1f}", f"volume {vr:.2f}x avg"])
        return hold("bearish cross rejected: " + ", ".join(missing))

    return hold("no fresh EMA9/EMA21 cross")


def _f(v) -> float:
    return float(v) if pd.notna(v) else float("nan")


def generate_signal(df_tf: pd.DataFrame, df_htf: Optional[pd.DataFrame], symbol: str,
                    timeframe: str, cfg: Optional[Config] = None) -> Signal:
    """Signal for the latest completed candle. Callers must pass completed candles only."""
    cfg = cfg or Config()
    df = ensure_indicators(df_tf, cfg)
    if timeframe == HTF or df_htf is None:
        htf_df = df if timeframe == HTF else None
    else:
        htf_df = ensure_indicators(df_htf, cfg)
    trend = trend_of_row(htf_df.iloc[-1]) if htf_df is not None and len(htf_df) else NEUTRAL
    return evaluate(df, len(df) - 1, trend, symbol, timeframe, cfg)
