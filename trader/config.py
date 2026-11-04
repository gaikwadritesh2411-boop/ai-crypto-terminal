"""Central configuration. Everything is overridable by environment variables.

No exchange API keys are ever read: the app only uses PUBLIC market data, so it
cannot place orders or withdraw funds even by accident.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
TIMEFRAMES = ("5m", "15m", "1h")
HTF = "1h"  # higher timeframe used to confirm the trend
TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600}
EXIT_TIMEFRAME = "5m"  # finest timeframe used to check stop loss / take profit


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # --- paper account ---
    initial_balance: float = field(default_factory=lambda: _f("INITIAL_BALANCE", 10_000.0))
    # --- risk ---
    risk_per_trade: float = field(default_factory=lambda: _f("RISK_PER_TRADE", 0.01))
    reward_risk: float = field(default_factory=lambda: _f("REWARD_RISK", 2.0))
    daily_loss_limit: float = field(default_factory=lambda: _f("DAILY_LOSS_LIMIT", 0.03))
    max_open_positions: int = field(default_factory=lambda: _i("MAX_OPEN_POSITIONS", 3))
    # Simulated leverage cap on total notional. 1% risk with tight ATR stops needs
    # notional above equity, so the default is 3x. Set MAX_LEVERAGE=1 for a
    # strictly un-leveraged account (actual risk per trade will then often be < 1%).
    max_leverage: float = field(default_factory=lambda: _f("MAX_LEVERAGE", 3.0))
    allow_short: bool = field(default_factory=lambda: _b("ALLOW_SHORT", True))
    min_notional: float = 10.0
    # --- costs ---
    fee_rate: float = field(default_factory=lambda: _f("FEE_RATE", 0.001))  # 0.10% per side
    slippage_rate: float = field(default_factory=lambda: _f("SLIPPAGE_RATE", 0.0005))  # 0.05%
    # --- stops ---
    atr_period: int = 14
    atr_stop_mult: float = field(default_factory=lambda: _f("ATR_STOP_MULT", 1.5))
    # --- signal rules ---
    cross_lookback: int = 3  # EMA cross must have happened within N completed candles
    rsi_buy_min: float = 45.0
    rsi_buy_max: float = 70.0
    rsi_sell_min: float = 30.0
    rsi_sell_max: float = 55.0
    min_volume_ratio: float = 1.0  # volume / 20-candle average volume
    signal_cooldown_candles: int = 6
    max_signal_age_candles: int = 2  # stale signals are recorded but never executed
    trade_timeframes: tuple = ("5m", "15m", "1h")
    candle_limit: int = 300
    # --- infrastructure ---
    exchange_id: str = field(default_factory=lambda: os.getenv("EXCHANGE_ID", "kraken"))
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "data/paper_trading.db"))
    ai_model: str = field(default_factory=lambda: os.getenv("AI_MODEL", "claude-sonnet-5-5"))
