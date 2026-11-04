import inspect

import numpy as np
import pandas as pd
import pytest

import trader.ai as ai_mod
from trader.ai import build_snapshot, get_commentary, rule_based_commentary
from trader.backtest import run_backtest
from trader.charts import candlestick_figure, daily_pnl_figure, equity_figure
from trader.config import SYMBOLS, Config
from trader.indicators import add_indicators
from trader.metrics import daily_pnl
from trader.signals import generate_signal
from helpers import make_candles, resample, trend_5m


def synthetic_dataset(days=40):
    """Synthetic trending paths (TEST DATA ONLY). Results say nothing about real-market profitability."""
    data = {}
    for i, s in enumerate(SYMBOLS):
        b = trend_5m(days=days, seed=20 + i, drift=0.0006, pull_amp=0.02, pull_hours=8, drift_days=6)
        data[s] = {"15m": resample(b, 15), "1h": resample(b, 60), "5m": b}
    return data


@pytest.fixture(scope="module")
def result():
    return run_backtest(synthetic_dataset(), Config(), "15m")


def test_backtest_produces_trades_and_consistent_accounting(result):
    t = result.trades
    assert len(t) >= 5, 'synthetic data should trigger several trades'
    assert result.equity.iloc[-1] == pytest.approx(10_000 + t["net_pnl"].sum(), abs=1e-6)
    assert (t["fees"] > 0).all()
    assert set(t["exit_reason"]) <= {"STOP_LOSS", "TAKE_PROFIT", "END_OF_TEST"}
    assert (t["exit_time"] > t["entry_time"]).all()


def test_backtest_is_deterministic(result):
    again = run_backtest(synthetic_dataset(), Config(), "15m")
    pd.testing.assert_frame_equal(again.trades, result.trades)


def test_backtest_respects_risk_limits(result):
    t = result.trades
    # never more than 3 positions open at once
    events = sorted([(r.entry_time, 1) for r in t.itertuples()] + [(r.exit_time, -1) for r in t.itertuples()],
                    key=lambda e: (e[0], e[1]))
    open_n = peak = 0
    for _, d in events:
        open_n += d
        peak = max(peak, open_n)
    assert peak <= 3
    # per-trade stop-loss losses stay near/below 1% of equity (10k start; allow equity drift + gap slippage)
    sl = t[t["exit_reason"] == "STOP_LOSS"]
    if len(sl):
        assert (sl["net_pnl"] > -0.015 * 12_000).all()


def test_backtest_costs_reduce_results():
    free = Config(fee_rate=0.0, slippage_rate=0.0)
    costly = Config(fee_rate=0.002, slippage_rate=0.001)
    d = synthetic_dataset()
    assert run_backtest(d, costly, "15m").metrics["net_pnl"] < run_backtest(d, free, "15m").metrics["net_pnl"]


def test_backtest_flags_small_samples(result):
    assert any("days" in n or "trades" in n for n in result.notes) or len(result.trades) >= 30


def test_backtest_requires_data():
    with pytest.raises(ValueError):
        run_backtest({}, Config(), "15m")


def _frames():
    return {"5m": add_indicators(make_candles(200, "5m", seed=1, sigma=0.003)),
            "15m": add_indicators(make_candles(200, "15m", seed=2, sigma=0.003)),
            "1h": add_indicators(make_candles(200, "1h", seed=3, sigma=0.003))}


def test_rule_based_commentary_uses_only_real_numbers_and_needs_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fr = _frames()
    sigs = [generate_signal(fr[t], fr["1h"], "BTC/USDT", t) for t in fr]
    snap = build_snapshot("BTC/USDT", fr, sigs)
    text, src = get_commentary(snap, use_ai=True, api_key=None)
    assert src == "rules"
    assert f"{fr['15m']['close'].iloc[-1]:.4g}"[:3] in text or "close" in text
    assert "not financial advice" in text.lower()


def test_ai_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(ai_mod, "llm_commentary", lambda *a, **k: None)
    fr = _frames()
    snap = build_snapshot("ETH/USDT", fr, [])
    text, src = get_commentary(snap, True, "sk-test")
    assert src == "rules" and text == rule_based_commentary(snap)


def test_ai_success_path_is_text_only(monkeypatch):
    monkeypatch.setattr(ai_mod, "llm_commentary", lambda *a, **k: "AI says: range-bound.")
    text, src = get_commentary({"symbol": "X", "timeframes": {}}, True, "sk-test")
    assert (text, src) == ("AI says: range-bound.", "ai")


def test_no_data_commentary_is_honest():
    assert "No market data" in rule_based_commentary({"symbol": "SOL/USDT", "timeframes": {}})


def test_ai_module_cannot_touch_trading_or_risk():
    src = inspect.getsource(ai_mod)
    for forbidden in ("portfolio", "risk", "engine", "open_position", "store"):
        assert f"import {forbidden}" not in src and f"from .{forbidden}" not in src


def test_charts_build_with_signals_positions_and_empty_data(tmp_path, result):
    from trader.portfolio import Position
    from helpers import NOW, sig
    df = add_indicators(make_candles(200, "15m", seed=5, sigma=0.003))
    sigs = pd.DataFrame([{"symbol": "BTC/USDT", "timeframe": "15m", "action": "BUY", "price": 100.0,
                          "candle_time": df["timestamp"].iloc[-5]}])
    pos = Position("BTC/USDT", "LONG", "15m", 1, 100, 98, 104, NOW, 0, 2)
    fig = candlestick_figure(df, "BTC/USDT", "15m", sigs, [pos])
    assert len(fig.data) >= 8
    assert equity_figure(pd.DataFrame({"ts": [], "equity": []})) is not None
    assert len(daily_pnl_figure(daily_pnl(result.trades)).data) == 1
