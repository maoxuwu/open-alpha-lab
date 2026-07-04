"""Dollar-neutral daily backtest for cross-sectional signals.

Model: at each close, weights ∝ signal (already demeaned by rank()); positions earn
next day's returns (t signal → t+1 return: no lookahead). Costs = flat bps on turnover.
Adequate for hypothesis ranking; NOT a capacity/execution study (see README limitations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily: pd.Series          # net daily returns of the L/S book (unit gross)
    turnover: pd.Series       # daily one-sided turnover
    weights: pd.DataFrame

    @property
    def sharpe(self) -> float:
        x = self.daily.dropna()
        return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 60 and x.std() > 0 else float("nan")

    @property
    def ann_return(self) -> float:
        return float(self.daily.mean() * 252)

    @property
    def avg_turnover(self) -> float:
        return float(self.turnover.mean())

    def summary(self) -> dict:
        x = self.daily.dropna()
        dd = (x.cumsum() - x.cumsum().cummax()).min()
        return {"sharpe": round(self.sharpe, 2), "ann_ret": round(self.ann_return, 4),
                "turnover": round(self.avg_turnover, 3), "max_dd": round(float(dd), 4),
                "T": len(x)}


def run(signal: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 5.0,
        delay: int = 1) -> BacktestResult:
    """signal: wide demeaned scores at date t. returns: wide simple returns at date t.
    Position formed from signal at t is exposed to returns at t+delay."""
    sig = signal.reindex_like(returns)
    # normalize to unit gross exposure per day, keep dollar neutrality
    demeaned = sig.sub(sig.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1)
    w = demeaned.div(gross.replace(0, np.nan), axis=0)
    w_lag = w.shift(delay)
    pnl = (w_lag * returns).sum(axis=1, min_count=1)
    to = (w_lag - w_lag.shift(1)).abs().sum(axis=1, min_count=1) / 2.0
    net = pnl - to * cost_bps * 1e-4 * 2  # both sides trade
    return BacktestResult(daily=net, turnover=to, weights=w_lag)
