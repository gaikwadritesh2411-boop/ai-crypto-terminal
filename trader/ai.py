"""Market commentary. Text only: nothing here can open, close or size a trade.

* Rule-based commentary works with no API key and uses only real indicator values.
* Optional LLM commentary receives the same computed numbers and is told not to invent data.
  Any failure falls back to the rule-based text.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import pandas as pd

from .config import TIMEFRAMES

log = logging.getLogger(__name__)


def build_snapshot(symbol: str, frames: dict, signals: list) -> dict:
    """frames: {tf: indicator-enriched DataFrame of completed candles}. Real values only."""
    snap: dict = {"symbol": symbol, "timeframes": {}}
    for tf in TIMEFRAMES:
        df = frames.get(tf)
        if df is None or len(df) == 0:
            continue
        r = df.iloc[-1]

        def g(k):
            v = r.get(k)
            return None if v is None or pd.isna(v) else round(float(v), 4)

        snap["timeframes"][tf] = {
            "candle_open_utc": r["timestamp"].isoformat(), "close": g("close"), "ema9": g("ema9"),
            "ema21": g("ema21"), "rsi14": g("rsi14"), "macd_hist": g("macd_hist"),
            "volume_ratio": g("vol_ratio"), "atr": g("atr"),
        }
    snap["signals"] = {s.timeframe: {"action": s.action, "trend_1h": s.trend, "reason": s.reason}
                       for s in signals if s.symbol == symbol}
    return snap


def rule_based_commentary(snap: dict) -> str:
    sym = snap["symbol"]
    lines = [f"**{sym}: rule-based read of completed candles**"]
    tfs = snap.get("timeframes", {})
    if not tfs:
        return f"No market data available for {sym}, so there is nothing to comment on."
    for tf, d in tfs.items():
        parts = []
        if d["ema9"] is not None and d["ema21"] is not None:
            parts.append("EMA9 above EMA21 (bullish bias)" if d["ema9"] > d["ema21"]
                         else "EMA9 below EMA21 (bearish bias)")
        rsi = d["rsi14"]
        if rsi is not None:
            state = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
            parts.append(f"RSI {rsi:.1f} ({state})")
        if d["macd_hist"] is not None:
            parts.append("MACD momentum positive" if d["macd_hist"] > 0 else "MACD momentum negative")
        if d["volume_ratio"] is not None:
            parts.append(f"volume {d['volume_ratio']:.2f}x average")
        sig = snap.get("signals", {}).get(tf)
        tail = f" Signal: **{sig['action']}** ({sig['reason']})." if sig else ""
        lines.append(f"- **{tf}** (close {d['close']}): " + ", ".join(parts) + "." + tail)
    h = snap.get("signals", {}).get("1h")
    if h:
        lines.append(f"\n1H trend filter: **{h['trend_1h']}**. Entries on 5m/15m only fire in the direction of this filter.")
    lines.append("\n_Indicator readout only, not financial advice. Risk limits are enforced separately by code._")
    return "\n".join(lines)


def llm_commentary(snap: dict, api_key: str, model: str) -> Optional[str]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=20.0)
        msg = client.messages.create(
            model=model, max_tokens=450,
            system=("You write short, plain-language market commentary for a PAPER trading dashboard. "
                    "Use ONLY the numbers in the JSON provided. Never invent prices, news, events or "
                    "statistics. You cannot place trades or change risk settings; do not give orders or "
                    "position sizes. Under 180 words. End with a one-line risk reminder."),
            messages=[{"role": "user", "content": json.dumps(snap)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return text or None
    except Exception as e:
        log.warning("AI commentary failed, using rules: %s", e)
        return None


def get_commentary(snap: dict, use_ai: bool, api_key: Optional[str] = None, model: str = "claude-sonnet-5-5"):
    """Return (text, source) where source is 'ai' or 'rules'."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if use_ai and key:
        text = llm_commentary(snap, key, model)
        if text:
            return text, "ai"
    return rule_based_commentary(snap), "rules"
