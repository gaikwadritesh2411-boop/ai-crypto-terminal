import pandas as pd
import pytest

import trader.engine as engine_mod
from trader.config import SYMBOLS, Config
from trader.engine import TradingEngine
from trader.signals import HOLD, Signal
from trader.store import Store
from helpers import NOW, FakeData, flat_frames, make_candles, sig

CFG = Config()


def hold_sig(symbol, tf, *a, **k):
    return Signal(symbol, tf, HOLD, 100.0, NOW - pd.Timedelta(minutes=15), 1.0, "UP", "test hold")


def make_engine(tmp_path, frames=None, signal_fn=None, monkeypatch=None, cfg=CFG, fail=()):
    store = Store(str(tmp_path / "e.db"))
    data = FakeData(frames or flat_frames(SYMBOLS), fail=fail)
    if monkeypatch is not None and signal_fn is not None:
        monkeypatch.setattr(engine_mod, "generate_signal", signal_fn)
    return TradingEngine(cfg, store, data), store, data


def only_btc_buy(symbol, tf, *a, **k):
    pass


def buy_btc_15m(df, htf, symbol, tf, cfg):
    return sig(symbol, tf, "BUY", 100.0, 1.0) if (symbol, tf) == ("BTC/USDT", "15m") else hold_sig(symbol, tf)


def test_cycle_opens_paper_position_and_stores_everything(tmp_path, monkeypatch):
    eng, store, _ = make_engine(tmp_path, signal_fn=buy_btc_15m, monkeypatch=monkeypatch)
    r = eng.run_cycle(now=NOW)
    assert len(r.opened) == 1 and r.opened[0].symbol == "BTC/USDT" and r.opened[0].side == "LONG"
    assert len(store.positions()) == 1
    assert store.signals_df().iloc[0]["executed"] == 1
    assert len(store.equity_df()) == 1
    assert r.errors == []


def test_duplicate_signal_not_reexecuted_on_next_cycle(tmp_path, monkeypatch):
    eng, store, _ = make_engine(tmp_path, signal_fn=buy_btc_15m, monkeypatch=monkeypatch)
    eng.run_cycle(now=NOW)
    r2 = eng.run_cycle(now=NOW + pd.Timedelta(seconds=30))
    assert r2.opened == []
    assert len(store.positions()) == 1 and len(store.signals_df()) == 1


def test_same_direction_signal_inside_cooldown_is_suppressed(tmp_path, monkeypatch):
    eng, store, _ = make_engine(tmp_path, signal_fn=buy_btc_15m, monkeypatch=monkeypatch)
    eng.run_cycle(now=NOW)
    later = lambda df, htf, s, tf, cfg: sig(s, tf, "BUY", 100, 1.0, end=NOW + pd.Timedelta(minutes=15)) \
        if (s, tf) == ("BTC/USDT", "15m") else hold_sig(s, tf)
    monkeypatch.setattr(engine_mod, "generate_signal", later)
    frames = flat_frames(SYMBOLS, end=NOW + pd.Timedelta(minutes=15))
    eng.data = FakeData(frames)
    r = eng.run_cycle(now=NOW + pd.Timedelta(minutes=15))
    assert r.opened == [] and len(store.signals_df()) == 1


def test_stale_signal_is_recorded_but_not_executed(tmp_path, monkeypatch):
    eng, store, _ = make_engine(tmp_path, signal_fn=buy_btc_15m, monkeypatch=monkeypatch)
    r = eng.run_cycle(now=NOW + pd.Timedelta(hours=2))  # candles are old relative to "now"
    # frames end at NOW, so with now=NOW+2h the signal candle closed 2h ago -> stale
    assert r.opened == []
    row = store.signals_df().iloc[0]
    assert row["executed"] == 0 and "stale" in row["note"]


def test_stop_loss_closes_trade_and_updates_pnl(tmp_path, monkeypatch):
    eng, store, data = make_engine(tmp_path, signal_fn=buy_btc_15m, monkeypatch=monkeypatch)
    eng.run_cycle(now=NOW)
    pos = store.positions()[0]
    t1 = NOW + pd.Timedelta(minutes=10)
    frames = flat_frames(SYMBOLS, end=t1)
    f5 = frames[("BTC/USDT", "5m")]
    f5.loc[f5["timestamp"] == NOW, "low"] = pos["stop_loss"] - 1.0  # candle after entry hits the stop
    eng.data = FakeData(frames)
    monkeypatch.setattr(engine_mod, "generate_signal", lambda df, htf, s, tf, cfg: hold_sig(s, tf))
    r = eng.run_cycle(now=t1)
    assert len(r.closed) == 1 and r.closed[0].exit_reason == "STOP_LOSS"
    tr = store.trades_df()
    assert len(tr) == 1 and tr["net_pnl"].iloc[0] == pytest.approx(-100, rel=0.03)
    assert store.positions() == []
    assert float(store.get_meta("balance")) == pytest.approx(10_000 + tr["net_pnl"].iloc[0])


def test_max_three_positions_enforced_across_symbols(tmp_path, monkeypatch):
    fn = lambda df, htf, s, tf, cfg: sig(s, tf, "BUY", 100.0, 1.0) if tf == "15m" else hold_sig(s, tf)
    eng, store, _ = make_engine(tmp_path, signal_fn=fn, monkeypatch=monkeypatch, cfg=Config(max_open_positions=2))
    r = eng.run_cycle(now=NOW)
    assert len(r.opened) == 2 and len(store.positions()) == 2
    assert any("max open positions" in n for n in r.notes)


def test_data_failure_is_reported_not_raised_and_never_faked(tmp_path, monkeypatch):
    eng, store, _ = make_engine(tmp_path, signal_fn=buy_btc_15m, monkeypatch=monkeypatch,
                                fail=[("BTC/USDT", "15m"), ("ETH/USDT", "1h")])
    r = eng.run_cycle(now=NOW)
    assert len(r.errors) == 2 and r.opened == []
    assert store.positions() == []


def test_total_outage_does_not_crash(tmp_path):
    store = Store(str(tmp_path / "x.db"))
    r = TradingEngine(CFG, store, FakeData({})).run_cycle(now=NOW)
    assert len(r.errors) == 9 and r.opened == [] and r.equity == pytest.approx(10_000)


def test_real_signal_pipeline_runs_end_to_end_without_patching(tmp_path):
    frames = {}
    for s in SYMBOLS:
        for tf in ("5m", "15m", "1h"):
            frames[(s, tf)] = make_candles(250, tf, seed=hash((s, tf)) % 1000, sigma=0.003)
    eng, store, _ = make_engine(tmp_path, frames=frames)
    r = eng.run_cycle(now=NOW)
    assert r.errors == [] and len(r.signals) == 9
    r2 = eng.run_cycle(now=NOW)
    assert r2.opened == []  # idempotent
