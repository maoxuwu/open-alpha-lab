# open-alpha-lab

A disciplined cross-sectional equity research pipeline on **open data** — hypothesis registry, minimal constructs, multiple-testing-aware screening, and a public death ledger.

This repo is the open-data twin of a research process I developed on WorldQuant BRAIN (13 production alphas, IQC 2026 UK #32). Platform work can't be inspected by outsiders; this can. The point here is not any single signal — most registered hypotheses are *expected to die* — but the **process that decides which ones live**, visible end to end.

## What makes this different from another backtest repo

1. **Hypotheses are registered before simulation** (`hypotheses/`): each entry states the economic mechanism ("who pays, and why they keep paying"), the state variable, the predicted correlation with existing live signals, and a **pre-registered kill condition**. No post-hoc sign flips; deaths are recorded, not deleted (`LEDGER.md`).
2. **Multiple-testing arithmetic is built in** (`src/oal/stats.py`): every batch reports the Deflated Sharpe Ratio (Bailey & López de Prado 2014) against the batch's own noise ceiling — scanning N variants manufactures an expected max Sharpe of roughly √(2·ln N)/√T even from pure noise, and the screener charges for it. The implementation is Monte-Carlo verified (see docstrings).
3. **Honest limitations, stated up front** (see below), because knowing what your backtest cannot claim is the half of research that survives contact with live capital.

## Structure

```
src/oal/
  data.py       # panel download & cache (yfinance), Ken French factor benchmarks
  signals.py    # cross-sectional / time-series operators (rank, ts_zscore, ts_corr, ...)
  backtest.py   # dollar-neutral rank-weighted daily backtest, turnover & cost model
  stats.py      # PSR / Deflated Sharpe / yearly stability (MC-verified)
  ledger.py     # hypothesis registry helpers
hypotheses/     # one file per registered hypothesis (mechanism, kill condition)
scripts/        # run_hXXX.py — one runnable script per hypothesis
LEDGER.md       # the death ledger: every verdict, with cause
```

## Quick start

```bash
uv sync                          # or: pip install -e .
uv run python scripts/run_h001.py   # H001: 12-1 momentum, the calibration hypothesis
```

H001 (12-1 momentum) is deliberately the first entry: it is the most-replicated anomaly in the literature, so it doubles as a **pipeline calibration** — if the pipeline can't reproduce the sign and rough magnitude of momentum on liquid US equities, the bug is in the pipeline, not the market. Results are benchmarked against Ken French's UMD factor.

## Known limitations (read before believing any number)

- **Survivorship bias**: the default universe is current index constituents fetched from yfinance; delisted names are absent, which inflates long-side returns. Treated as an upper bound; conclusions rely on cross-sectional *relative* statements and on effects also verified in the French data (which is survivorship-free).
- **No intraday fills, simplistic costs**: costs are a flat bps parameter on turnover; adequate for ranking hypotheses, not for capacity claims.
- **Point-in-time discipline is approximate**: prices are adjusted-close series; no PIT fundamentals are used at all for this reason (price/volume constructs only, until a PIT source is added).

## Roadmap

- [x] Skeleton: operators, backtest, DSR screening, registry
- [x] H001 momentum calibration vs French UMD — **passed** (monthly corr 0.70; see `LEDGER.md`)
- [ ] H002+ from the registry, one per week
- [ ] Cost sensitivity + capacity notes per surviving signal
- [ ] PIT fundamental source (SEC EDGAR) → accounting-based hypotheses

MIT License. Research/educational use; nothing here is investment advice.
