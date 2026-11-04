"""Position sizing and risk rules. Pure functions, no I/O.

AI output never reaches this module: sizing and limits are deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .config import Config

LONG, SHORT = "LONG", "SHORT"


@dataclass(frozen=True)
class TradePlan:
    side: str
    entry_price: float  # fill price after slippage
    stop_loss: float
    take_profit: float
    stop_distance: float


def build_plan(side: str, price: float, atr: float, cfg: Config) -> Optional[TradePlan]:
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (price, atr)):
        return None
    if price <= 0 or atr <= 0:
        return None
    slip = cfg.slippage_rate
    dist = cfg.atr_stop_mult * atr
    if side == LONG:
        entry = price * (1 + slip)
        stop, tp = entry - dist, entry + cfg.reward_risk * dist
    elif side == SHORT:
        entry = price * (1 - slip)
        stop, tp = entry + dist, entry - cfg.reward_risk * dist
    else:
        return None
    if stop <= 0 or tp <= 0:
        return None
    return TradePlan(side, entry, stop, tp, dist)


def size_position(equity: float, open_notional: float, plan: TradePlan, cfg: Config):
    """Return (qty, capped). Risk 1% of equity at the stop, INCLUDING fees and slippage.

    qty is then limited by the leverage cap: each position may use at most
    equity * max_leverage / max_open_positions of notional, and total notional may not
    exceed equity * max_leverage.
    """
    if equity <= 0:
        return 0.0, True
    risk_amount = equity * cfg.risk_per_trade
    per_unit = (
        plan.stop_distance
        + cfg.fee_rate * (plan.entry_price + plan.stop_loss)
        + cfg.slippage_rate * plan.stop_loss
    )
    qty_risk = risk_amount / per_unit
    max_notional = min(
        equity * cfg.max_leverage / cfg.max_open_positions,
        equity * cfg.max_leverage - open_notional,
    )
    qty_cap = max(max_notional, 0.0) / (plan.entry_price * (1 + cfg.fee_rate))
    return max(min(qty_risk, qty_cap), 0.0), qty_cap < qty_risk


def daily_loss_breached(equity: float, day_start_equity: Optional[float], cfg: Config) -> bool:
    if not day_start_equity or day_start_equity <= 0:
        return False
    return (equity - day_start_equity) / day_start_equity <= -cfg.daily_loss_limit
