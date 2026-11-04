"""Plotly figures (dark theme, touch friendly)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG, GRID = "#0b0f17", "#1c2433"
UP_C, DN_C, BLUE, AMBER, TXT = "#26a69a", "#ef5350", "#4aa3ff", "#ffb74d", "#c9d4e5"


def _base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        margin=dict(l=8, r=8, t=28, b=8), font=dict(color=TXT, size=11), dragmode="pan",
        legend=dict(orientation="h", y=1.04, x=0), xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, showspikes=False)
    fig.update_yaxes(gridcolor=GRID, fixedrange=True)
    return fig


def candlestick_figure(df: pd.DataFrame, symbol: str, timeframe: str, signals: Optional[pd.DataFrame] = None,
                       positions: Optional[list] = None, bars: int = 150, height: int = 680) -> go.Figure:
    d = df.tail(bars)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.52, 0.14, 0.17, 0.17])
    x = d["timestamp"]
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name=f"{symbol} {timeframe}", increasing_line_color=UP_C,
                                 decreasing_line_color=DN_C), row=1, col=1)
    if "ema9" in d:
        fig.add_trace(go.Scatter(x=x, y=d["ema9"], name="EMA 9", line=dict(color=BLUE, width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["ema21"], name="EMA 21", line=dict(color=AMBER, width=1.3)), row=1, col=1)

    if signals is not None and len(signals):
        s = signals[(signals["symbol"] == symbol) & (signals["timeframe"] == timeframe)]
        s = s[s["candle_time"] >= x.iloc[0]] if len(x) else s
        for action, color, sym_, name in (("BUY", UP_C, "triangle-up", "BUY"), ("SELL", DN_C, "triangle-down", "SELL")):
            ss = s[s["action"] == action]
            if len(ss):
                fig.add_trace(go.Scatter(x=ss["candle_time"], y=ss["price"], mode="markers", name=name,
                                         marker=dict(symbol=sym_, size=13, color=color,
                                                     line=dict(width=1, color="white"))), row=1, col=1)

    for p in positions or []:
        if p.symbol != symbol:
            continue
        fig.add_hline(y=p.entry_price, line=dict(color="#9aa7bd", width=1, dash="dot"), row=1, col=1)
        fig.add_hline(y=p.stop_loss, line=dict(color=DN_C, width=1, dash="dash"), row=1, col=1,
                      annotation_text="SL", annotation_position="right")
        fig.add_hline(y=p.take_profit, line=dict(color=UP_C, width=1, dash="dash"), row=1, col=1,
                      annotation_text="TP", annotation_position="right")

    colors = [UP_C if c >= o else DN_C for o, c in zip(d["open"], d["close"])]
    fig.add_trace(go.Bar(x=x, y=d["volume"], marker_color=colors, name="Volume", opacity=0.7), row=2, col=1)
    if "vol_sma" in d:
        fig.add_trace(go.Scatter(x=x, y=d["vol_sma"], name="Vol avg 20", line=dict(color=AMBER, width=1)), row=2, col=1)
    if "rsi14" in d:
        fig.add_trace(go.Scatter(x=x, y=d["rsi14"], name="RSI 14", line=dict(color="#b388ff", width=1.4)), row=3, col=1)
        for lvl in (30, 70):
            fig.add_hline(y=lvl, line=dict(color="#55627a", width=1, dash="dot"), row=3, col=1)
    if "macd" in d:
        hist_c = [UP_C if v >= 0 else DN_C for v in d["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=x, y=d["macd_hist"], marker_color=hist_c, name="MACD hist", opacity=0.6), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["macd"], name="MACD", line=dict(color=BLUE, width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["macd_signal"], name="Signal", line=dict(color=AMBER, width=1.2)), row=4, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    return _base(fig, height)


def equity_figure(equity: pd.DataFrame, x: str = "ts", y: str = "equity", height: int = 300) -> go.Figure:
    fig = go.Figure()
    if len(equity):
        fig.add_trace(go.Scatter(x=equity[x], y=equity[y], mode="lines", name="Equity",
                                 line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(74,163,255,0.08)"))
        lo, hi = float(equity[y].min()), float(equity[y].max())
        pad = max((hi - lo) * 0.2, hi * 0.001)
        fig.update_yaxes(range=[lo - pad, hi + pad])
    return _base(fig, height)


def daily_pnl_figure(daily: pd.DataFrame, height: int = 280) -> go.Figure:
    fig = go.Figure()
    if len(daily):
        fig.add_trace(go.Bar(x=daily["date"], y=daily["pnl"], name="Daily P&L",
                             marker_color=[UP_C if v >= 0 else DN_C for v in daily["pnl"]]))
    return _base(fig, height)
