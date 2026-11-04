from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.12 = 12%)."""
    if equity is None or len(equity) == 0:
        return 0.0
    eq = pd.Series(equity).astype(float)
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, np.nan)
    return float(dd.max()) if dd.notna().any() else 0.0


def compute_metrics(trades: pd.DataFrame, initial_balance: float,
                    equity_curve: Optional[pd.Series] = None) -> dict:
    n = len(trades)
    base = {"total_trades": n, "wins": 0, "losses": 0, "win_rate": 0.0, "profit_factor": None,
            "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "avg_r": 0.0, "total_fees": 0.0, "max_drawdown_pct": 0.0,
            "return_pct": 0.0}
    if equity_curve is not None and len(equity_curve):
        base["max_drawdown_pct"] = max_drawdown(equity_curve) * 100
    if n == 0:
        return base
    pnl = trades["net_pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    if equity_curve is None or len(equity_curve) == 0:
        curve = pd.concat([pd.Series([initial_balance]), initial_balance + pnl.cumsum()], ignore_index=True)
        base["max_drawdown_pct"] = max_drawdown(curve) * 100
    base.update({
        "wins": int(len(wins)), "losses": int(len(losses)),
        "win_rate": float(len(wins) / n * 100),
        "profit_factor": (gp / gl) if gl > 0 else None,  # None = no losing trades yet
        "net_pnl": float(pnl.sum()), "gross_profit": gp, "gross_loss": gl,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(pnl.mean()),
        "avg_r": float(trades["r_multiple"].mean()) if "r_multiple" in trades else 0.0,
        "total_fees": float(trades["fees"].sum()) if "fees" in trades else 0.0,
        "return_pct": float(pnl.sum() / initial_balance * 100),
    })
    return base


def daily_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Realized net P&L per UTC day (by exit time)."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame({"date": [], "pnl": []})
    d = trades.copy()
    d["date"] = pd.to_datetime(d["exit_time"], utc=True).dt.strftime("%Y-%m-%d")
    return d.groupby("date", as_index=False)["net_pnl"].sum().rename(columns={"net_pnl": "pnl"})
