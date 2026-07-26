# H002 — Short-horizon reversal (net of costs)

**Registered**: 2026-07-25 · **Status**: ☠️ dead (kill condition met; see LEDGER.md)

## Mechanism (who pays, and why they keep paying)
Liquidity provision: short-term price moves partly reflect non-informational
order-flow pressure (fund flows, forced rebalancing); fading them supplies
liquidity and earns its premium (Jegadeesh 1990; Nagel 2012). The payer is the
impatient trader who demands immediacy.

## State variable
Volatility-scaled 10-day return, faded: `-ts_zscore(returns, 10)`.

## The actual question being tested
Gross reversal alpha is well documented; the open question at daily rebalancing
in liquid mega caps is whether it survives **costs**. This hypothesis is
registered as a *cost-viability* test, not an existence test. Prior: turnover
near 100%/day at 5 bps per side is likely fatal in this universe — an expected
death, registered anyway because the death is informative (it quantifies the
cost hurdle that any fast signal must clear).

## Constructs
`-rank(ts_zscore(rets, 10))` raw; 5-day variant; decayed variant
(`ts_decay_linear`, 3d) as the turnover-mitigation arm.

## Pre-registered kill condition
Net Sharpe (5 bps/side) < 0.3 across all variants → dead, cause recorded as
gross-vs-net decomposition (report both). No sign flips; no post-hoc variants.
