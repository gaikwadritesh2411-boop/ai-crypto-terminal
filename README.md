# AI Crypto Intraday Trading Terminal (PAPER TRADING ONLY)

Streamlit dashboard + rule-based signal engine for **BTC/USDT, ETH/USDT, SOL/USDT** on **5m / 15m / 1h**,
with a virtual **$10,000** account stored in SQLite.

> **This app never places real orders.** It reads *public* market data through CCXT, needs **no exchange
> API key**, and has no code path for orders or withdrawals.
> **No claim of profitability is made.** See "What is and is not verified" below.

## Features
- Dark, mobile-friendly dashboard: live prices, interactive candlestick chart (EMA 9/21, volume, RSI 14, MACD),
  BUY/SELL markers, signal table, AI/rule-based commentary, portfolio, trade history, win rate,
  profit factor, max drawdown, daily P&L, backtest tab.
- Signals use **completed candles only** (the forming candle is dropped), are **de-duplicated**
  (unique per symbol/timeframe/candle + cooldown), and stale signals are never executed.
- 1H EMA trend must agree: BUY needs 1H UP, SELL needs 1H DOWN.
- Paper execution with **fees (0.10%/side)** and **slippage (0.05%)**, realized + unrealized P&L, all in SQLite.
- Optional AI commentary (text only). Without `ANTHROPIC_API_KEY` it uses rule-based analysis built from real indicator values.

## Strategy (all thresholds in `trader/config.py`)
| | BUY | SELL (short) |
|---|---|---|
| Trigger | EMA9 crossed above EMA21 within last 3 candles | EMA9 crossed below EMA21 |
| 1H filter | EMA9>EMA21 and close>EMA21 | mirror |
| MACD hist | > 0 | < 0 |
| RSI 14 | 45 to 70 | 30 to 55 |
| Volume | >= 20-candle average | same |
Otherwise **HOLD** (with the reason shown).

## Risk management (enforced in code; AI cannot change it)
- Risk **1% of virtual equity** per trade, sized from the stop distance **including fees and slippage**.
- Stop = 1.5 x ATR(14); target = **2R** (risk:reward 1:2). A candle touching both counts as a **stop**.
- **Max 3 open positions**, 1 per symbol. **Daily loss limit 3%** (UTC day, includes unrealized P&L) blocks new entries.
- Simulated leverage cap `MAX_LEVERAGE=3` on total notional (tight ATR stops need notional > equity for 1% risk).
  Set `MAX_LEVERAGE=1` for an un-leveraged account; actual risk per trade will then often be below 1%.

## Run locally / in a cloud shell
```bash
pip install -r requirements-dev.txt
streamlit run app.py          # dashboard
python worker.py              # optional headless trading loop (one cycle / 60 s)
python -m pytest -q           # tests
python scripts/verify_paper_trading.py --days 8              # audit paper trading on synthetic candles
python scripts/verify_paper_trading.py --days 14 --random-walk   # leakage / "no edge" sanity check
python -m trader.backtest --days 30                          # backtest on REAL exchange history
```
Press **Run paper-trading cycle now**, or enable **Auto-run engine (60 s)** (works only while the page is open).
For 24/7 paper trading use the worker (Docker option below).

## Deployment (Android-only friendly)
**Option A: Streamlit Community Cloud (free, easiest)**
1. Put this folder in a GitHub repo (from Android: github.com in "Desktop site" mode -> *Add file -> Upload files*, or open the repo in GitHub Codespaces / github.dev in the browser).
2. share.streamlit.io -> *New app* -> pick the repo, branch `main`, main file `app.py`.
3. Optional: *Settings -> Secrets* -> `ANTHROPIC_API_KEY = "..."`.
4. Open the app URL on your phone. Add to home screen from Chrome's menu.
Limits: the filesystem is **ephemeral** (SQLite resets on redeploy/restart) and the app sleeps when idle, so use the
auto-run toggle only while the page is open.

**Option B: Docker host with a persistent disk (Render, Railway, Fly.io...) for 24/7 paper trading**
1. Create a *Web Service* from the repo (it detects the `Dockerfile`).
2. Attach a persistent disk mounted at `/app/data`.
3. `start.sh` runs the worker and dashboard in one container sharing the SQLite file.
4. Set env vars from `.env.example` if needed. Paid plans are usually required for disks / always-on.

**Exchange note:** the default exchange is `kraken`. Binance/Bybit often return HTTP 451 from US cloud regions.
If data fails, the dashboard shows the error and **never fills in made-up numbers**. Try `EXCHANGE_ID=coinbase`
(check the exchange lists the USDT pairs) or another region.

## Safety checklist
- Paper only: no order-placing code, no exchange keys read, no withdrawal capability.
- No hardcoded secrets; the only optional key is read from env/Streamlit secrets and used only for commentary text.
- All network calls wrapped in error handling; failures surface as warnings.
- `trader/ai.py` imports nothing from risk/portfolio/engine (enforced by a test).

## What is and is not verified
Verified here (88 automated tests + replay audit):
- Indicators, signal rules, completed-candle logic, dedupe, risk sizing, stops/targets, fees/slippage, daily limit,
  max positions, SQLite persistence/restart, backtest accounting, AI fallback, dashboard renders (also with the exchange down).
- Replay of the real engine over 14 days of **synthetic random-walk** candles (no real edge): 101 trades, win rate 31.7%,
  profit factor 0.48, net loss. This shows the bookkeeping is right, there is no look-ahead leakage,
  and that fees + slippage hurt a strategy with no edge.

**Not verified:**
- **Live exchange connectivity** was not testable from the build sandbox (network allowlist). First run in your deployment will show whether your chosen exchange is reachable.
- **Profitability on real markets: unknown.** No backtest on real history has been run. Run `python -m trader.backtest`
  (or the Backtest tab) after deployment, then paper trade for weeks before drawing conclusions. Fewer than ~30 trades
  or one market regime means the statistics are not meaningful. The 5m timeframe is especially fee-sensitive.
- Paper fills assume you always get filled at close +/- slippage; real fills, liquidity and latency will differ.

Not financial advice.
