"""AI Crypto Intraday Trading Terminal - PAPER TRADING ONLY (no real orders, no API keys)."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from trader.ai import build_snapshot, get_commentary
from trader.backtest import run_backtest
from trader.charts import candlestick_figure, daily_pnl_figure, equity_figure
from trader.config import HTF, SYMBOLS, TIMEFRAMES, Config
from trader.data import DataError, MarketData
from trader.engine import TradingEngine
from trader.indicators import ensure_indicators
from trader.metrics import compute_metrics, daily_pnl
from trader.portfolio import Portfolio
from trader.signals import generate_signal
from trader.store import Store
from trader.utils import now_utc

logging.basicConfig(level=logging.INFO)
cfg = Config()

st.set_page_config(page_title="AI Crypto Terminal", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
  .stApp { background:#0b0f17; }
  .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }
  div[data-testid="stMetric"] { background:#121826; border:1px solid #1c2433; border-radius:10px; padding:10px 12px; }
  div[data-testid="stMetricValue"] { font-size:1.35rem; }
  .badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:700;
           background:#1f3a2e; color:#5ee0a0; border:1px solid #2e5a46; letter-spacing:.04em; }
  .sub { color:#8a97ad; font-size:.85rem; margin-top:-6px; }
  .stTabs [data-baseweb="tab-list"] { gap:2px; overflow-x:auto; }
  .stTabs [data-baseweb="tab"] { padding:8px 12px; white-space:nowrap; }
  @media (max-width: 640px) {
    .block-container { padding-left:.6rem; padding-right:.6rem; }
    div[data-testid="stMetricValue"] { font-size:1.05rem; }
    div[data-testid="stMetric"] { padding:6px 8px; }
    h1 { font-size:1.35rem !important; }
    .stDataFrame { font-size:.8rem; }
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_store(path: str) -> Store:
    return Store(path)


@st.cache_resource
def get_data(exchange_id: str) -> MarketData:
    return MarketData(exchange_id)


@st.cache_data(ttl=30, show_spinner=False)
def load_frames(exchange_id: str, symbol: str):
    md = get_data(exchange_id)
    frames, errors = {}, []
    for tf in TIMEFRAMES:
        try:
            frames[tf] = ensure_indicators(md.fetch_ohlcv(symbol, tf, cfg.candle_limit), cfg)
        except DataError as e:
            errors.append(str(e))
    return frames, errors


@st.cache_data(ttl=10, show_spinner=False)
def load_prices(exchange_id: str) -> dict:
    return get_data(exchange_id).fetch_last_prices(SYMBOLS)


def ai_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def run_engine() -> None:
    engine = TradingEngine(cfg, get_store(cfg.db_path), get_data(cfg.exchange_id))
    try:
        st.session_state["last_cycle"] = engine.run_cycle()
        st.session_state.pop("cycle_error", None)
    except Exception as e:  # keep the UI alive no matter what
        logging.exception("engine cycle failed")
        st.session_state["cycle_error"] = f"{type(e).__name__}: {e}"


def money(v: float) -> str:
    return f"{'-' if v < 0 else ''}${abs(v):,.2f}"


def pf_marks(prices: dict, frames_by_symbol: dict) -> dict:
    marks = {}
    for s in SYMBOLS:
        fr = frames_by_symbol.get(s, {}).get("5m")
        if fr is not None and len(fr):
            marks[s] = float(fr["close"].iloc[-1])
    marks.update(prices or {})
    return marks


# ------------------------------------------------------------------ header & controls
h1, h2 = st.columns([3, 2])
h1.markdown("# 📈 AI Crypto Terminal")
h2.markdown('<div style="text-align:right;padding-top:14px"><span class="badge">PAPER TRADING ONLY</span></div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sub">Virtual ${cfg.initial_balance:,.0f} account · data: {cfg.exchange_id} public API · '
            f'no real orders, ever</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([2, 3, 2])
symbol = c1.selectbox("Market", SYMBOLS)
tf = c2.radio("Chart timeframe", TIMEFRAMES, index=1, horizontal=True)
auto = c3.toggle("Auto-run engine (60 s)", value=False,
                 help="Runs the paper-trading cycle every minute while this page is open.")
use_ai = c3.toggle("AI commentary", value=False, help="Needs ANTHROPIC_API_KEY. Falls back to rules.")
if st.button("▶ Run paper-trading cycle now", use_container_width=True):
    with st.spinner("Fetching completed candles and evaluating signals..."):
        run_engine()


def dashboard() -> None:
    store = get_store(cfg.db_path)
    if auto:
        run_engine()

    if "cycle_error" in st.session_state:
        st.error(f"Engine error: {st.session_state['cycle_error']}")

    prices = load_prices(cfg.exchange_id)
    all_frames, all_errors = {}, []
    for s in SYMBOLS:
        fr, er = load_frames(cfg.exchange_id, s)
        all_frames[s] = fr
        all_errors += er
    marks = pf_marks(prices, all_frames)

    # ---- live prices
    pcols = st.columns(len(SYMBOLS))
    for col, s in zip(pcols, SYMBOLS):
        p = prices.get(s) or marks.get(s)
        delta = None
        h = all_frames[s].get("1h")
        if p and h is not None and len(h) > 24:
            delta = f"{(p / float(h['close'].iloc[-25]) - 1) * 100:+.2f}% 24h"
        col.metric(s, f"${p:,.2f}" if p else "n/a", delta)
    if not prices:
        st.warning("Live ticker unavailable; showing last completed 5m close where available.")
    if all_errors:
        st.warning("Market data problems (nothing is simulated or made up):\n\n- " + "\n- ".join(all_errors[:6]))

    t_market, t_port, t_hist, t_bt, t_sys = st.tabs(
        ["📊 Market", "💼 Portfolio", "📜 History", "🧪 Backtest", "⚙️ System"])

    # ================================================================== MARKET
    with t_market:
        frames = all_frames[symbol]
        if tf not in frames:
            st.info(f"No {tf} candles for {symbol} right now.")
        else:
            pf_view = Portfolio.load(cfg, store)
            sig_hist = store.signals_df(300)
            st.plotly_chart(
                candlestick_figure(frames[tf], symbol, tf, sig_hist, pf_view.positions),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False},
            )
        sigs = [generate_signal(frames[t], frames.get(HTF), symbol, t, cfg) for t in TIMEFRAMES if t in frames]
        if sigs:
            st.subheader("Current signals (completed candles)")
            st.dataframe(pd.DataFrame([{
                "TF": s.timeframe, "Signal": s.action, "1H trend": s.trend, "Price": round(s.price, 4),
                "RSI": round(s.rsi, 1) if pd.notna(s.rsi) else None, "Why": s.reason} for s in sigs]),
                hide_index=True, use_container_width=True)
            st.subheader("AI market commentary")
            snap = build_snapshot(symbol, frames, sigs)
            ck = f"comm-{symbol}-{use_ai}-{frames[tf].iloc[-1]['timestamp'] if tf in frames else ''}"
            if ck not in st.session_state:
                st.session_state[ck] = get_commentary(snap, use_ai, ai_key(), cfg.ai_model)
            text, src = st.session_state[ck]
            st.caption("Source: " + ("AI model (numbers supplied by the app)" if src == "ai"
                                     else "rule-based analysis (no AI key needed)"))
            st.markdown(text)
            if use_ai and src == "rules":
                st.caption("AI unavailable (no ANTHROPIC_API_KEY or request failed); showing rule-based analysis.")
        sh = store.signals_df(30)
        if len(sh):
            st.subheader("Recorded BUY/SELL signals")
            st.dataframe(sh[["candle_time", "symbol", "timeframe", "action", "price", "executed", "note"]],
                         hide_index=True, use_container_width=True)

    # ================================================================== PORTFOLIO
    with t_port:
        pf = Portfolio.load(cfg, store)
        pf.roll_day(now_utc(), marks)
        eq = pf.equity(marks)
        dpnl = pf.daily_pnl(marks)
        a, b = st.columns(2)
        a.metric("Equity", money(eq), f"{(eq / cfg.initial_balance - 1) * 100:+.2f}% total")
        b.metric("Cash balance", money(pf.balance))
        c, d = st.columns(2)
        c.metric("Unrealized P&L", money(pf.unrealized(marks)))
        d.metric("Today's P&L", money(dpnl), f"limit -{cfg.daily_loss_limit:.0%}")
        if pf.day_start_equity and dpnl / pf.day_start_equity <= -cfg.daily_loss_limit:
            st.error("Daily loss limit reached: new entries are blocked until the next UTC day.")
        st.caption(f"Open positions: {len(pf.positions)}/{cfg.max_open_positions} · "
                   f"exposure {money(pf.open_notional())} ({pf.open_notional() / max(eq, 1):.2f}x equity, "
                   f"cap {cfg.max_leverage:.1f}x)")
        if pf.positions:
            st.subheader("Open positions")
            st.dataframe(pd.DataFrame([{
                "Symbol": p.symbol, "Side": p.side, "TF": p.timeframe, "Qty": round(p.qty, 6),
                "Entry": round(p.entry_price, 4), "Mark": round(marks.get(p.symbol, p.entry_price), 4),
                "Stop": round(p.stop_loss, 4), "Target": round(p.take_profit, 4),
                "Unrealized": round(p.unrealized(marks.get(p.symbol, p.entry_price)), 2),
                "Opened": p.entry_time} for p in pf.positions]), hide_index=True, use_container_width=True)
        else:
            st.info("No open positions.")
        eqdf = store.equity_df()
        if len(eqdf) > 1:
            st.subheader("Equity curve")
            st.plotly_chart(equity_figure(eqdf), use_container_width=True, config={"displayModeBar": False})

    # ================================================================== HISTORY
    with t_hist:
        trades = store.trades_df()
        m = compute_metrics(trades, cfg.initial_balance,
                            store.equity_df().set_index("ts")["equity"] if len(store.equity_df()) else None)
        k1, k2 = st.columns(2)
        k1.metric("Win rate", f"{m['win_rate']:.1f}%", f"{m['wins']}W / {m['losses']}L")
        k2.metric("Profit factor", "n/a" if m["profit_factor"] is None else f"{m['profit_factor']:.2f}")
        k3, k4 = st.columns(2)
        k3.metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
        k4.metric("Net realized P&L", money(m["net_pnl"]), f"{m['total_trades']} trades")
        if m["total_trades"] < 30:
            st.caption("Fewer than 30 closed trades: these statistics are not meaningful yet.")
        dd = daily_pnl(trades)
        if len(dd):
            st.subheader("Daily P&L (realized, UTC)")
            st.plotly_chart(daily_pnl_figure(dd), use_container_width=True, config={"displayModeBar": False})
        if len(trades):
            st.subheader("Trade history")
            show = trades.drop(columns=["id", "risk_amount"]).sort_values("exit_time", ascending=False)
            st.dataframe(show, hide_index=True, use_container_width=True)
        else:
            st.info("No closed trades yet. Run the engine; trades appear after a stop loss or take profit is hit.")

    # ================================================================== BACKTEST
    with t_bt:
        st.markdown("Backtest uses the same signals, risk rules, fees and slippage as paper trading, "
                    "on real exchange history. **It does not prove future profit.**")
        days = st.slider("History (days)", 7, 90, 30)
        base_tf = st.radio("Base timeframe", ["5m", "15m", "1h"], index=1, horizontal=True, key="bt_tf")
        if st.button("Run backtest", use_container_width=True):
            md = get_data(cfg.exchange_id)
            data, errs = {}, []
            with st.spinner("Downloading history and simulating..."):
                for s in SYMBOLS:
                    try:
                        data[s] = {base_tf: md.fetch_history(s, base_tf, days),
                                   HTF: md.fetch_history(s, HTF, days + 10)}
                    except DataError as e:
                        errs.append(str(e))
                if data:
                    st.session_state["bt"] = run_backtest(data, cfg, base_tf)
                    st.session_state["bt_meta"] = (days, base_tf, sorted(data))
            for e in errs:
                st.warning(e)
            if not data:
                st.error("No history could be downloaded, so no backtest was run.")
        if "bt" in st.session_state:
            r = st.session_state["bt"]
            d_, tf_, syms = st.session_state["bt_meta"]
            st.caption(f"Result for {', '.join(syms)} · {d_} days requested · {tf_} candles")
            m = r.metrics
            b1, b2 = st.columns(2)
            b1.metric("Net P&L", money(m["net_pnl"]), f"{m['return_pct']:+.2f}%")
            b2.metric("Trades", m["total_trades"])
            b3, b4 = st.columns(2)
            b3.metric("Win rate", f"{m['win_rate']:.1f}%")
            b4.metric("Profit factor", "n/a" if m["profit_factor"] is None else f"{m['profit_factor']:.2f}")
            b5, b6 = st.columns(2)
            b5.metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
            b6.metric("Fees paid", money(m["total_fees"]))
            for n in r.notes:
                st.warning(n)
            if len(r.equity):
                st.plotly_chart(equity_figure(r.equity.reset_index().rename(columns={"index": "ts"})),
                                use_container_width=True, config={"displayModeBar": False})
            if len(r.trades):
                st.dataframe(r.trades, hide_index=True, use_container_width=True)

    # ================================================================== SYSTEM
    with t_sys:
        lc = st.session_state.get("last_cycle")
        if lc:
            st.write(f"Last cycle: {lc.ran_at:%Y-%m-%d %H:%M:%S} UTC · equity {money(lc.equity)} · "
                     f"opened {len(lc.opened)} · closed {len(lc.closed)}")
            for e in lc.errors:
                st.warning(e)
            for n in lc.notes:
                st.caption(n)
        else:
            st.info("Engine has not run in this session yet.")
        st.json({"exchange": cfg.exchange_id, "db_path": cfg.db_path, "risk_per_trade": cfg.risk_per_trade,
                 "reward_risk": cfg.reward_risk, "daily_loss_limit": cfg.daily_loss_limit,
                 "max_open_positions": cfg.max_open_positions, "max_leverage": cfg.max_leverage,
                 "fee_rate": cfg.fee_rate, "slippage_rate": cfg.slippage_rate,
                 "allow_short": cfg.allow_short})
        st.caption("The SQLite file is the source of truth. On free cloud hosts it may be wiped on redeploy.")
        with st.expander("Reset paper account"):
            sure = st.checkbox("I understand this deletes all virtual trades and resets to the initial balance")
            if st.button("Reset account", disabled=not sure):
                store.reset()
                st.session_state.pop("last_cycle", None)
                st.rerun(scope="app")


st.fragment(run_every=60 if auto else None)(dashboard)()
