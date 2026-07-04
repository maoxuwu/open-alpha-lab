"""Cross-sectional & time-series operators on wide DataFrames (index=date, columns=ticker).

Semantics deliberately mirror the WorldQuant BRAIN FASTEXPR operators of the same name,
so hypotheses can be translated between the two environments verbatim.
All operators are NaN-tolerant and vectorized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- cross-sectional (row-wise) ----------
def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank mapped to [-0.5, 0.5] (demeaned uniform, dollar-neutral ready)."""
    r = x.rank(axis=1, pct=True)
    return r.sub(r.mean(axis=1), axis=0)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=1), axis=0)


def winsorize(x: pd.DataFrame, std: float = 4.0) -> pd.DataFrame:
    m, s = x.mean(axis=1), x.std(axis=1, ddof=1)
    lo, hi = m - std * s, m + std * s
    return x.clip(lower=lo, upper=hi, axis=0)


def signed_power(x: pd.DataFrame, p: float) -> pd.DataFrame:
    return np.sign(x) * np.abs(x) ** p


def group_neutralize(x: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Demean within groups (groups: ticker -> label). BRAIN's industry neutralization."""
    out = x.copy()
    for _, cols in groups.groupby(groups):
        cs = [c for c in cols.index if c in x.columns]
        out[cs] = x[cs].sub(x[cs].mean(axis=1), axis=0)
    return out


# ---------- time-series (column-wise) ----------
def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(2, d // 2)).mean()


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(2, d // 2)).sum()


def ts_std_dev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=max(2, d // 2)).std(ddof=1)


def ts_delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)


def ts_delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x - x.shift(d)


def ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return (x - ts_mean(x, d)) / ts_std_dev(x, d)


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).rank(pct=True) - 0.5


def ts_backfill(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.ffill(limit=d)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).corr(y)


def ts_decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d).apply(lambda a: np.dot(a, w), raw=True)
