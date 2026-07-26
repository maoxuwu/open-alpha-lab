"""Regression tests for the backtest engine: look-ahead guard, dollar
neutrality, and cost accounting. These encode the failure modes that most
often silently corrupt backtests."""
import numpy as np
import pandas as pd

from oal import backtest
from oal import signals as S

RNG = np.random.default_rng(11)


def _panel(T=1200, n=200):
    dates = pd.bdate_range("2019-01-01", periods=T)
    rets = pd.DataFrame(RNG.normal(0, 0.02, (T, n)), index=dates,
                        columns=[f"S{i}" for i in range(n)])
    return rets


def test_lookahead_guard():
    """Signal = today's return (known at close). Trading it TOMORROW must earn
    ~nothing on iid noise; trading it same-day (delay=0) is look-ahead and
    must explode. If this test ever fails, the shift logic broke."""
    rets = _panel()
    legal = backtest.run(S.rank(rets), rets, cost_bps=0, delay=1)
    cheat = backtest.run(S.rank(rets), rets, cost_bps=0, delay=0)
    assert abs(legal.sharpe) < 0.6
    assert cheat.sharpe > 50


def test_dollar_neutral_and_unit_gross():
    rets = _panel()
    sig = pd.DataFrame(RNG.normal(0, 1, rets.shape), index=rets.index, columns=rets.columns)
    r = backtest.run(S.rank(sig), rets, cost_bps=0)
    w = r.weights.dropna(how="all")
    net = w.sum(axis=1).abs().max()
    gross = w.abs().sum(axis=1)
    assert net < 1e-8                       # dollar neutral every day
    assert (gross.dropna() - 1).abs().max() < 1e-8  # unit gross exposure


def test_costs_reduce_returns_by_turnover():
    rets = _panel()
    sig = pd.DataFrame(RNG.normal(0, 1, rets.shape), index=rets.index, columns=rets.columns)
    free = backtest.run(S.rank(sig), rets, cost_bps=0)
    paid = backtest.run(S.rank(sig), rets, cost_bps=10)
    drag = (free.daily - paid.daily).dropna()
    expected = free.turnover.reindex(drag.index) * 10 * 1e-4 * 2
    assert np.allclose(drag.values, expected.values, atol=1e-12)


def test_constant_signal_zero_turnover():
    rets = _panel()
    const = pd.DataFrame(np.tile(RNG.normal(0, 1, rets.shape[1]), (len(rets), 1)),
                         index=rets.index, columns=rets.columns)
    r = backtest.run(S.rank(const), rets, cost_bps=5)
    assert r.turnover.dropna().iloc[5:].max() < 1e-10
