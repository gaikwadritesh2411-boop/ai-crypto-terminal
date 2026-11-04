from __future__ import annotations

import pandas as pd


def utc(ts) -> pd.Timestamp:
    """Return a timezone-aware UTC Timestamp for any timestamp-like input."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def iso(ts) -> str:
    return utc(ts).isoformat()


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")
