import math

import pytest

from trader.config import Config
from trader.risk import LONG, SHORT, build_plan, daily_loss_breached, size_position

CFG = Config()


def test_long_plan_geometry_and_rr():
    p = build_plan(LONG, 100.0, 1.0, CFG)
    assert p.entry_price == pytest.approx(100 * (1 + CFG.slippage_rate))
    assert p.stop_loss < p.entry_price < p.take_profit
    assert (p.take_profit - p.entry_price) / (p.entry_price - p.stop_loss) == pytest.approx(2.0)


def test_short_plan_geometry_and_rr():
    p = build_plan(SHORT, 100.0, 1.0, CFG)
    assert p.entry_price == pytest.approx(100 * (1 - CFG.slippage_rate))
    assert p.take_profit < p.entry_price < p.stop_loss
    assert (p.entry_price - p.take_profit) / (p.stop_loss - p.entry_price) == pytest.approx(2.0)


@pytest.mark.parametrize("price,atr", [(0, 1), (100, 0), (100, -1), (float("nan"), 1), (100, float("nan")), (100, math.inf)])
def test_invalid_inputs_give_no_plan(price, atr):
    assert build_plan(LONG, price, atr, CFG) is None


def test_stop_below_zero_is_rejected():
    assert build_plan(LONG, 1.0, 5.0, CFG) is None


def test_unknown_side():
    assert build_plan("FLAT", 100, 1, CFG) is None


def test_size_risks_about_one_percent_including_costs():
    plan = build_plan(LONG, 100.0, 1.0, CFG)
    qty, capped = size_position(10_000, 0.0, plan, CFG)
    assert not capped
    exit_px = plan.stop_loss * (1 - CFG.slippage_rate)
    loss = (plan.entry_price - exit_px) * qty + CFG.fee_rate * (plan.entry_price + exit_px) * qty
    assert loss == pytest.approx(100.0, rel=0.01)  # 1% of 10,000


def test_size_is_capped_by_leverage_limit():
    plan = build_plan(LONG, 100.0, 0.05, CFG)  # tiny stop => huge risk-based size
    qty, capped = size_position(10_000, 0.0, plan, CFG)
    assert capped
    assert qty * plan.entry_price <= 10_000 * CFG.max_leverage / CFG.max_open_positions + 1e-6


def test_total_exposure_cap():
    plan = build_plan(LONG, 100.0, 0.05, CFG)
    qty, _ = size_position(10_000, 29_900.0, plan, CFG)
    assert qty * plan.entry_price <= 100.0 + 1e-6
    qty, _ = size_position(10_000, 31_000.0, plan, CFG)
    assert qty == 0.0


def test_zero_equity_gives_zero_size():
    plan = build_plan(LONG, 100.0, 1.0, CFG)
    assert size_position(0, 0, plan, CFG)[0] == 0.0


def test_daily_loss_limit():
    assert daily_loss_breached(9_700, 10_000, CFG)       # exactly -3%
    assert daily_loss_breached(9_500, 10_000, CFG)
    assert not daily_loss_breached(9_701, 10_000, CFG)
    assert not daily_loss_breached(9_000, None, CFG)
