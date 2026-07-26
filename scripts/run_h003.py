"""H003: lottery/MAX effect. See hypotheses/H003_lottery_max.md."""
import pandas as pd
from oal import backtest, data, signals as S
from oal.stats import screen_batch


def main():
    panel = data.load_panel(start="2015-01-01")
    rets = panel["returns"]
    variants = {
        "max10": -S.rank(S.ts_max(rets, 10)),
        "max20": -S.rank(S.ts_max(rets, 20)),
        "max60": -S.rank(S.ts_max(rets, 60)),
        "max20_vol": -S.rank(S.ts_max(rets, 20) / (S.ts_std_dev(rets, 60) + 1e-4)),
    }
    rows, dailies = [], {}
    for name, sig in variants.items():
        net = backtest.run(sig, rets, cost_bps=5.0)
        dailies[name] = net.daily
        rows.append({"variant": name, "net_sharpe": round(net.sharpe, 2),
                     "turnover": round(net.avg_turnover, 3)})
        print(rows[-1])
    print("\n== batch DSR screen ==")
    scr = screen_batch(dailies)
    print(scr.to_string(index=False))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "h003_summary.csv"), index=False)
    scr.to_csv(os.path.join(out, "h003_screen.csv"), index=False)
    print("\nKill condition: |net Sharpe| < 0.3 for all variants in both directions -> dead."
          " No sign flips (see hypothesis file).")


if __name__ == "__main__":
    main()
