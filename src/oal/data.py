"""Panel download & cache (yfinance) + Ken French factor benchmarks.

Universe default = current S&P 500 constituents (Wikipedia) — SURVIVORSHIP-BIASED,
documented in README. Cache: parquet under ./cache/.
"""
from __future__ import annotations

import os

import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache")
os.makedirs(CACHE, exist_ok=True)


FALLBACK_100 = [  # 大市值流动股备胎池:Wikipedia 抓取失败时管线仍可校准
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","BRK-B","LLY","AVGO","TSLA",
    "JPM","V","XOM","UNH","MA","PG","COST","JNJ","HD","ABBV",
    "WMT","NFLX","BAC","CRM","ORCL","CVX","MRK","KO","AMD","PEP",
    "ADBE","TMO","LIN","WFC","ACN","CSCO","MCD","ABT","IBM","GE",
    "CAT","QCOM","DHR","INTU","AXP","DIS","VZ","AMGN","TXN","PFE",
    "MS","GS","NOW","ISRG","NEE","RTX","UBER","CMCSA","SPGI","HON",
    "UNP","BKNG","LOW","T","AMAT","PGR","BLK","ETN","SYK","ELV",
    "COP","TJX","LMT","BSX","VRTX","C","PLD","MU","ADP","MDT",
    "PANW","REGN","ADI","SBUX","DE","BMY","GILD","MMC","CB","LRCX",
    "MO","SO","BA","NKE","MDLZ","SCHW","KLAC","INTC","CI","UPS",
]


def sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        ticks = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(ticks)
    except Exception as e:
        print(f"[data] Wikipedia 成分表抓取失败({e}),退回内置 100 只大市值池(校准足够)")
        return FALLBACK_100


def load_panel(start: str = "2015-01-01", end: str | None = None,
               tickers: list[str] | None = None, refresh: bool = False):
    """Returns dict of wide DataFrames: close (adj), volume, returns."""
    import yfinance as yf
    path = os.path.join(CACHE, "panel.parquet")
    if os.path.exists(path) and not refresh:
        px = pd.read_parquet(path)
    else:
        tickers = tickers or sp500_tickers()
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=True, group_by="column")
        px = raw["Close"]
        vol = raw["Volume"]
        px.to_parquet(path)
        vol.to_parquet(os.path.join(CACHE, "volume.parquet"))
    vol = pd.read_parquet(os.path.join(CACHE, "volume.parquet")) if \
        os.path.exists(os.path.join(CACHE, "volume.parquet")) else None
    # hygiene: drop tickers with <70% history; forward-fill gaps ≤5 days
    px = px.loc[:, px.notna().mean() > 0.7].ffill(limit=5)
    rets = px.pct_change(fill_method=None)
    # guard: clip absurd single-day moves from bad data (documented, not silent)
    n_clip = int((rets.abs() > 1.0).sum().sum())
    if n_clip:
        print(f"[data] clipped {n_clip} daily |returns|>100% points (data errors)")
        rets = rets.clip(-1.0, 1.0)
    return {"close": px, "volume": (vol.reindex_like(px) if vol is not None else None),
            "returns": rets}


def french_momentum(start: str = "2015-01-01") -> pd.Series:
    """Ken French UMD (monthly) as external benchmark for H001 calibration."""
    from pandas_datareader import data as pdr
    umd = pdr.DataReader("F-F_Momentum_Factor", "famafrench", start=start)[0]
    s = umd.iloc[:, 0] / 100.0
    s.index = s.index.to_timestamp()
    return s.rename("UMD")
