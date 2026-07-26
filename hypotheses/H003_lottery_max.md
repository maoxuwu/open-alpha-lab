# H003 — Lottery demand / MAX effect

**Registered**: 2026-07-25 · **Status**: ☠️ dead (kill condition met; see LEDGER.md)

## Mechanism (who pays, and why they keep paying)
Retail preference for lottery-like payoffs: stocks with recent extreme daily
gains get overbought and subsequently underperform (Bali, Cakici & Whitelaw
2011, *Maxing Out*). Short the recent-jackpot names; the payer is the lottery
buyer, and the friction is the cost/constraint of shorting glamour names.

## State variable
Maximum daily return over the trailing month: `-rank(ts_max(returns, 20))`.

## Prior (stated before running)
The effect is documented mainly in small/illiquid names. This universe is 100
mega caps — the least favourable habitat. Expected weak-to-dead; registered to
map the effect's capitalisation boundary honestly rather than to find alpha.

## Constructs
Windows 10/20/60; vol-scaled variant (`ts_max / ts_std_dev(60)`).

## Pre-registered kill condition
|Net Sharpe| (5 bps) < 0.3 for all variants **in both directions** → dead.
If the *positive* direction is stronger, record it, do not flip signs
(a momentum-flavoured result here would be a different hypothesis).
