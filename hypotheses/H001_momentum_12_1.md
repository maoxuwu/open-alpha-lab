# H001 — 12-1 price momentum (calibration hypothesis)

**Registered**: 2026-07-04 · **Status**: ✅ calibration passed (corr vs UMD = 0.70; see LEDGER.md)

## Mechanism (who pays, and why they keep paying)
Underreaction driven by the disposition effect: investors sell winners too early and hold
losers too long, so prices adjust to good/bad news with a lag. The payer is the
behaviorally-anchored investor; the friction preventing instant correction is the sheer
breadth of the effect and momentum's crash risk, which deters leveraged arbitrage
(Jegadeesh & Titman 1993; Barberis, Shleifer & Vishny 1998; Daniel & Moskowitz 2016).

## State variable
Cumulative return over months t-12..t-1 (skipping the most recent month, which reverses).

## Why it is hypothesis #001
Deliberate calibration choice: the most-replicated cross-sectional anomaly in existence.
If this pipeline cannot recover momentum's sign and rough magnitude on liquid US equities,
the pipeline is broken — not the market. External benchmark: correlation of the strategy's
monthly returns with Ken French's UMD factor should exceed 0.6.

## Construct
`rank(ts_sum(ts_delay(returns, 21), 210))`, dollar-neutral, variants:
raw / industry-demeaned / risk-managed (÷ rolling σ, Barroso & Santa-Clara 2015).

## Pre-registered kill / success conditions
- Calibration FAILS if all variants' Sharpe < 0.3 net of 5bps, or corr(strategy, UMD) < 0.4
  → debug pipeline before touching any other hypothesis.
- Note: survivorship bias inflates the long side; the UMD-correlation check is the
  bias-robust criterion, not the Sharpe level.
