import pandas as pd
import pytest

from trader.config import Config
from trader.portfolio import Portfolio, Position, check_exit
from trader.risk import LONG, SHORT
from trader.store import Store
from helpers import NOW

CFG = Config()


def candle(ts, o, h, l, c=None):
    return pd.DataFrame([{"timestamp": ts, "open": o, "high": h, "low": l, "close": c or o, "volume": 1}])


def open_long(pf, sym="BTC/USDT", price=100.0, atr=1.0, marks=None):
    return pf.open_position(sym, LONG, "15m", price, atr, NOW, NOW, marks or {})


def test_open_charges_fee_and_slippage():
    pf = Portfolio(CFG)
    pos, why = open_long(pf)
    assert pos and pos.entry_price == pytest.approx(100.05)
    assert pf.balance == pytest.approx(10_000 - pos.entry_fee)
    assert pos.entry_fee == pytest.approx(CFG.fee_rate * pos.entry_price * pos.qty)


def test_stop_loss_loses_about_one_percent_net():
    pf = Portfolio(CFG)
    pos, _ = open_long(pf)
    t = pf.process_candles("BTC/USDT", candle(NOW, 100, 100.2, pos.stop_loss - 0.1), 900)[0]
    assert t.exit_reason == "STOP_LOSS"
    assert t.net_pnl == pytest.approx(-100.0, rel=0.02)
    assert pf.balance == pytest.approx(10_000 + t.net_pnl)
    assert not pf.positions


def test_take_profit_pays_about_two_r_minus_costs():
    pf = Portfolio(CFG)
    pos, _ = open_long(pf)
    t = pf.process_candles("BTC/USDT", candle(NOW, 100, pos.take_profit + 0.5, 99.9), 900)[0]
    assert t.exit_reason == "TAKE_PROFIT"
    assert 1.6 < t.r_multiple < 2.0
    assert t.net_pnl > 0 and t.fees > 0
    assert pf.balance == pytest.approx(10_000 + t.net_pnl)


def test_candle_touching_both_levels_counts_as_stop():
    pf = Portfolio(CFG)
    pos, _ = open_long(pf)
    t = pf.process_candles("BTC/USDT", candle(NOW, 100, pos.take_profit + 1, pos.stop_loss - 1), 900)[0]
    assert t.exit_reason == "STOP_LOSS"


def test_gap_through_stop_fills_at_open_not_stop():
    pf = Portfolio(CFG)
    pos, _ = open_long(pf)
    t = pf.process_candles("BTC/USDT", candle(NOW, pos.stop_loss - 5, pos.stop_loss - 4, pos.stop_loss - 6), 900)[0]
    assert t.exit_price < pos.stop_loss - 5  # worse than the stop, includes slippage


def test_short_position_profits_when_price_falls():
    pf = Portfolio(CFG)
    pos, _ = pf.open_position("ETH/USDT", SHORT, "15m", 100.0, 1.0, NOW, NOW, {})
    assert pos.entry_price == pytest.approx(99.95)
    t = pf.process_candles("ETH/USDT", candle(NOW, 100, 100, pos.take_profit - 0.5), 900)[0]
    assert t.exit_reason == "TAKE_PROFIT" and t.net_pnl > 0


def test_candles_before_entry_are_ignored():
    pf = Portfolio(CFG)
    pos, _ = open_long(pf)
    early = candle(NOW - pd.Timedelta(minutes=15), 100, 200, 1)
    assert pf.process_candles("BTC/USDT", early, 900) == []
    assert pf.positions


def test_unrealized_pnl_and_equity():
    pf = Portfolio(CFG)
    pos, _ = open_long(pf)
    marks = {"BTC/USDT": pos.entry_price + 1.0}
    assert pf.unrealized(marks) == pytest.approx(pos.qty)
    assert pf.equity(marks) == pytest.approx(pf.balance + pos.qty)


def test_one_position_per_symbol_and_max_three():
    pf = Portfolio(CFG)
    assert open_long(pf, "A/USDT")[0]
    pos, why = open_long(pf, "A/USDT")
    assert pos is None and "already open" in why
    assert open_long(pf, "B/USDT")[0] and open_long(pf, "C/USDT")[0]
    pos, why = open_long(pf, "D/USDT")
    assert pos is None and "max open positions" in why
    assert len(pf.positions) == 3


def test_daily_loss_limit_blocks_new_entries():
    pf = Portfolio(CFG)
    pf.roll_day(NOW, {})
    pf.balance = 10_000 * (1 - 0.031)
    pos, why = open_long(pf)
    assert pos is None and "daily loss limit" in why


def test_daily_limit_resets_next_utc_day():
    pf = Portfolio(CFG)
    pf.roll_day(NOW, {})
    pf.balance = 9_600
    assert not pf.can_open("X/USDT", NOW, {})[0]
    tomorrow = NOW + pd.Timedelta(days=1)
    assert pf.can_open("X/USDT", tomorrow, {})[0]


def test_unrealized_losses_count_toward_daily_limit():
    pf = Portfolio(CFG)
    pos, _ = open_long(pf)
    pf.roll_day(NOW, {})
    marks = {"BTC/USDT": pos.entry_price - 10_000 / pos.qty * 0.05}  # -5% of equity unrealized
    assert not pf.can_open("ETH/USDT", NOW, marks)[0]


def test_shorting_can_be_disabled():
    cfg = Config(allow_short=False)
    pf = Portfolio(cfg)
    pos, why = pf.open_position("X/USDT", SHORT, "15m", 100, 1, NOW, NOW, {})
    assert pos is None and "disabled" in why


def test_below_min_notional_or_no_capital_is_rejected():
    pf = Portfolio(Config(initial_balance=1.0))
    pos, why = open_long(pf)
    assert pos is None


def test_persistence_roundtrip_through_sqlite(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    pf = Portfolio.load(CFG, store)
    pos, _ = open_long(pf)
    assert pos.id is not None
    pf2 = Portfolio.load(CFG, store)  # simulates an app restart
    assert len(pf2.positions) == 1 and pf2.balance == pytest.approx(pf.balance)
    assert pf2.positions[0].entry_time == pos.entry_time
    pf2.process_candles("BTC/USDT", candle(NOW, 100, 100.1, pos.stop_loss - 1), 900)
    assert store.positions() == []
    tr = store.trades_df()
    assert len(tr) == 1 and tr["net_pnl"].iloc[0] < 0
    assert Portfolio.load(CFG, store).balance == pytest.approx(10_000 + tr["net_pnl"].iloc[0])


def test_check_exit_none_inside_range():
    pos = Position("X", LONG, "15m", 1, 100, 98, 104, NOW, 0, 2)
    assert check_exit(pos, 100, 101, 99, CFG) is None
