"""Smoke-test the Streamlit UI headlessly. Network is replaced by deterministic stubs."""
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from trader.data import DataError, MarketData
from helpers import NOW, make_candles

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "ui.db"))
    st.cache_data.clear()
    st.cache_resource.clear()


def test_app_renders_when_exchange_is_down(monkeypatch):
    def boom(self, *a, **k):
        raise DataError("simulated exchange outage")

    monkeypatch.setattr(MarketData, "fetch_ohlcv", boom)
    monkeypatch.setattr(MarketData, "fetch_history", boom)
    monkeypatch.setattr(MarketData, "fetch_last_prices", lambda self, s: {})
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("PAPER TRADING ONLY" in m.value for m in at.markdown)
    assert len(at.warning) >= 1  # outage is shown, nothing is invented


def test_app_renders_with_data_and_runs_a_cycle(monkeypatch):
    def fake(self, symbol, tf, limit=300, now=None):
        return make_candles(250, tf, end=pd.Timestamp.now(tz="UTC").floor("1h"), seed=len(symbol), sigma=0.003)

    monkeypatch.setattr(MarketData, "fetch_ohlcv", fake)
    monkeypatch.setattr(MarketData, "fetch_last_prices", lambda self, s: {})
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.tabs) == 5
