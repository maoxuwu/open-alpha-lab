"""H001: 12-1 momentum calibration. See hypotheses/H001_momentum_12_1.md.

Usage:  uv run python scripts/run_h001.py          (first run downloads ~500 tickers, minutes)
Output: batch DSR table + UMD benchmark correlation + verdict vs pre-registered criteria.
"""
import numpy as np
from oal import backtest, data, signals as S
from oal.stats import screen_batch


def main():
    panel = data.load_panel(start="2015-01-01")
    rets, close = panel["returns"], panel["close"]

    mom = S.ts_sum(S.ts_delay(rets, 21), 210)
    variants = {
        "raw":  S.rank(mom),
        "rm":   S.rank(mom / (S.ts_std_dev(rets, 210) + 1e-4)),   # risk-managed
        "w7":   S.rank(S.ts_sum(S.ts_delay(rets, 21), 126)),      # window perturbation
    }
    results, dailies = {}, {}
    for name, sig in variants.items():
        r = backtest.run(sig, rets, cost_bps=5.0)
        results[name] = r
        dailies[name] = r.daily
        print(f"{name:5s} {r.summary()}")

    print("\n== batch DSR screen (multiple-testing-aware) ==")
    print(screen_batch(dailies).to_string(index=False))

    # external benchmark: monthly corr vs Ken French UMD
    try:
        umd = data.french_momentum(start="2015-01-01")
        m = results["raw"].daily.resample("ME").sum()
        m.index = m.index.to_period("M")          # align on monthly periods (month-end vs month-start labels never intersect)
        umd.index = umd.index.to_period("M")
        joint = m.to_frame("strat").join(umd, how="inner").dropna()
        c = joint["strat"].corr(joint["UMD"])
        print(f"(overlap: {len(joint)} months)")
        print(f"\ncorr(monthly strat, French UMD) = {c:.2f}  (pre-registered pass: >0.4, target >0.6)")
    except Exception as e:
        print(f"\n[warn] UMD benchmark unavailable: {e}")

    best = max(r.sharpe for r in results.values())
    print(f"\nVERDICT inputs: best net Sharpe = {best:.2f} (kill line: all <0.3)."
          f" Record outcome in LEDGER.md — deaths included.")

    # persist for the record (and for anyone auditing the ledger)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    screen_batch(dailies).to_csv(os.path.join(out_dir, "h001_screen.csv"), index=False)
    import pandas as pd
    pd.DataFrame({k: v for k, v in dailies.items()}).to_csv(os.path.join(out_dir, "h001_daily.csv"))
    print(f"saved -> results/h001_screen.csv, results/h001_daily.csv")


if __name__ == "__main__":
    main()
