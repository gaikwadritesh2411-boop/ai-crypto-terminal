import pandas as pd
import pytest

from trader.metrics import compute_metrics, daily_pnl, max_drawdown
from trader.store import Store
from helpers import NOW, sig


def test_signal_dedupe_by_candle(tmp_path):
    st = Store(str(tmp_path / "s.db"))
    s = sig()
    assert st.record_signal(s, NOW) is True
    assert st.record_signal(s, NOW) is False
    assert st.signal_exists(s.symbol, s.timeframe, s.candle_time)
    assert st.last_signal(s.symbol, s.timeframe)["action"] == "BUY"
    st.update_signal(s.symbol, s.timeframe, s.candle_time, True, "opened")
    assert st.signals_df().iloc[0]["executed"] == 1


def test_meta_and_reset(tmp_path):
    st = Store(str(tmp_path / "s.db"))
    st.set_meta("balance", 123.5)
    assert float(st.get_meta("balance")) == 123.5
    assert st.get_meta("missing", "d") == "d"
    st.snapshot_equity(NOW, 1, 1, 0, 0)
    st.record_signal(sig(), NOW)
    st.reset()
    assert st.get_meta("balance") is None and st.signals_df().empty and st.equity_df().empty


def test_max_drawdown():
    assert max_drawdown(pd.Series([100, 120, 90, 110])) == pytest.approx(0.25)
    assert max_drawdown(pd.Series([100, 101, 102])) == 0.0
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


def _trades(pnls):
    t = pd.DataFrame({"net_pnl": pnls, "r_multiple": [p / 100 for p in pnls], "fees": [1.0] * len(pnls),
                      "exit_time": [NOW + pd.Timedelta(hours=i * 12) for i in range(len(pnls))]})
    return t


def test_metrics_values():
    m = compute_metrics(_trades([200, -100, 200, -100]), 10_000)
    assert m["total_trades"] == 4 and m["wins"] == 2 and m["losses"] == 2
    assert m["win_rate"] == 50.0
    assert m["profit_factor"] == pytest.approx(2.0)
    assert m["net_pnl"] == 200 and m["return_pct"] == pytest.approx(2.0)
    assert m["max_drawdown_pct"] == pytest.approx(100 / 10_200 * 100, rel=1e-3)


def test_metrics_no_losses_and_empty():
    assert compute_metrics(_trades([50, 60]), 10_000)["profit_factor"] is None
    m = compute_metrics(_trades([]), 10_000)
    assert m["total_trades"] == 0 and m["win_rate"] == 0.0


def test_daily_pnl_groups_by_utc_day():
    d = daily_pnl(_trades([10, 20, -5]))
    assert d["pnl"].sum() == 25 and len(d) == 2
    assert daily_pnl(_trades([])).empty
