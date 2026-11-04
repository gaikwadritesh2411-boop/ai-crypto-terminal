"""SQLite persistence for the paper account: positions, trades, signals, equity, meta."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

import pandas as pd

from .utils import iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, side TEXT NOT NULL, timeframe TEXT NOT NULL,
    qty REAL NOT NULL, entry_price REAL NOT NULL, stop_loss REAL NOT NULL, take_profit REAL NOT NULL,
    entry_time TEXT NOT NULL, entry_fee REAL NOT NULL, risk_amount REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, side TEXT NOT NULL, timeframe TEXT NOT NULL,
    qty REAL NOT NULL, entry_price REAL NOT NULL, exit_price REAL NOT NULL,
    entry_time TEXT NOT NULL, exit_time TEXT NOT NULL,
    gross_pnl REAL NOT NULL, fees REAL NOT NULL, net_pnl REAL NOT NULL,
    exit_reason TEXT NOT NULL, risk_amount REAL NOT NULL, r_multiple REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, action TEXT NOT NULL,
    price REAL NOT NULL, candle_time TEXT NOT NULL, trend TEXT, reason TEXT,
    created_at TEXT NOT NULL, executed INTEGER NOT NULL DEFAULT 0, note TEXT,
    UNIQUE(symbol, timeframe, candle_time)
);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, equity REAL NOT NULL, balance REAL NOT NULL,
    unrealized REAL NOT NULL, open_positions INTEGER NOT NULL
);
"""

_TABLES = ("positions", "trades", "signals", "equity_snapshots", "meta")


class Store:
    def __init__(self, path: str):
        self.path = path
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")  # dashboard and worker may share the file
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- meta ----
    def get_meta(self, key: str, default=None):
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, str(value)))

    # ---- positions ----
    def insert_position(self, p) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO positions(symbol, side, timeframe, qty, entry_price, stop_loss, take_profit,"
                " entry_time, entry_fee, risk_amount) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (p.symbol, p.side, p.timeframe, p.qty, p.entry_price, p.stop_loss, p.take_profit,
                 iso(p.entry_time), p.entry_fee, p.risk_amount),
            )
            return int(cur.lastrowid)

    def delete_position(self, pos_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM positions WHERE id=?", (pos_id,))

    def positions(self) -> list:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM positions ORDER BY id")]

    # ---- trades ----
    def insert_trade(self, t) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO trades(symbol, side, timeframe, qty, entry_price, exit_price, entry_time,"
                " exit_time, gross_pnl, fees, net_pnl, exit_reason, risk_amount, r_multiple)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t.symbol, t.side, t.timeframe, t.qty, t.entry_price, t.exit_price, iso(t.entry_time),
                 iso(t.exit_time), t.gross_pnl, t.fees, t.net_pnl, t.exit_reason, t.risk_amount, t.r_multiple),
            )

    def trades_df(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY id", c)
        for col in ("entry_time", "exit_time"):
            df[col] = pd.to_datetime(df[col], utc=True)
        return df

    def positions_df(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM positions ORDER BY id", c)

    # ---- signals ----
    def signal_exists(self, symbol: str, timeframe: str, candle_time) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM signals WHERE symbol=? AND timeframe=? AND candle_time=?",
                (symbol, timeframe, iso(candle_time)),
            ).fetchone()
        return row is not None

    def last_signal(self, symbol: str, timeframe: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM signals WHERE symbol=? AND timeframe=? ORDER BY candle_time DESC LIMIT 1",
                (symbol, timeframe),
            ).fetchone()
        return dict(row) if row else None

    def record_signal(self, sig, created_at, executed: bool = False, note: str = "") -> bool:
        """Insert a BUY/SELL signal. Returns False if this exact candle was already recorded."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO signals(symbol, timeframe, action, price, candle_time, trend, reason,"
                " created_at, executed, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sig.symbol, sig.timeframe, sig.action, sig.price, iso(sig.candle_time), sig.trend,
                 sig.reason, iso(created_at), int(executed), note),
            )
            return cur.rowcount == 1

    def update_signal(self, symbol: str, timeframe: str, candle_time, executed: bool, note: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE signals SET executed=?, note=? WHERE symbol=? AND timeframe=? AND candle_time=?",
                (int(executed), note, symbol, timeframe, iso(candle_time)),
            )

    def signals_df(self, limit: int = 200) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM signals ORDER BY candle_time DESC LIMIT ?", c, params=(limit,))
        if not df.empty:
            df["candle_time"] = pd.to_datetime(df["candle_time"], utc=True)
        return df

    # ---- equity ----
    def snapshot_equity(self, ts, equity: float, balance: float, unrealized: float, n_open: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO equity_snapshots(ts, equity, balance, unrealized, open_positions) VALUES (?,?,?,?,?)",
                (iso(ts), equity, balance, unrealized, n_open),
            )

    def equity_df(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM equity_snapshots ORDER BY id", c)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def reset(self) -> None:
        with self._conn() as c:
            for t in _TABLES:
                c.execute(f"DELETE FROM {t}")
