"""H002: short-horizon reversal, net of costs. See hypotheses/H002_short_horizon_reversal.md."""
import pandas as pd
from oal import backtest, data, signals as S
from oal.stats import screen_batch


def main():
    panel = data.load_panel(start="2015-01-01")
    rets = panel["returns"]
    variants = {
        "rev10": -S.rank(S.ts_zscore(rets, 10)),
        "rev5":  -S.rank(S.ts_zscore(rets, 5)),
        "rev10_decay": -S.rank(S.ts_decay_linear(S.ts_zscore(rets, 10), 3)),
    }
    rows, dailies = [], {}
    for name, sig in variants.items():
        gross = backtest.run(sig, rets, cost_bps=0.0)
        net = backtest.run(sig, rets, cost_bps=5.0)
        dailies[name] = net.daily
        rows.append({"variant": name, "gross_sharpe": round(gross.sharpe, 2),
                     "net_sharpe": round(net.sharpe, 2),
                     "turnover": round(net.avg_turnover, 3)})
        print(rows[-1])
    print("\n== batch DSR screen (net) ==")
    scr = screen_batch(dailies)
    print(scr.to_string(index=False))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "h002_summary.csv"), index=False)
    scr.to_csv(os.path.join(out, "h002_screen.csv"), index=False)
    print("\nKill condition: net Sharpe (5bps) < 0.3 across all variants -> dead."
          " Record verdict with gross-vs-net decomposition in LEDGER.md.")


if __name__ == "__main__":
    main()
