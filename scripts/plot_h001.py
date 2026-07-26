"""Generate results/h001_vs_umd.png: monthly cumulative returns of the H001
momentum strategy vs Ken French's UMD factor (the corr-0.70 calibration,
visualised). Falls back to a strategy-only plot if the French fetch fails."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def main():
    daily = pd.read_csv(os.path.join(RES, "h001_daily.csv"), index_col=0, parse_dates=True)["raw"]
    m = daily.resample("ME").sum()
    m.index = m.index.to_period("M")

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    ax.plot(m.index.to_timestamp(), m.cumsum().values, lw=1.6,
            label="H001 12-1 momentum (this pipeline, net 5 bps)")
    title = "H001 calibration: pipeline momentum vs French UMD"
    try:
        from oal.data import french_momentum
        umd = french_momentum(start="2015-01-01")
        umd.index = umd.index.to_period("M")
        joint = m.to_frame("strat").join(umd, how="inner").dropna()
        c = joint["strat"].corr(joint["UMD"])
        ax.plot(joint.index.to_timestamp(), joint["UMD"].cumsum().values, lw=1.6, ls="--",
                label="Ken French UMD (survivorship-free benchmark)")
        title += f"  —  monthly corr = {c:.2f}"
    except Exception as e:
        print(f"[warn] UMD unavailable ({e}); plotting strategy only")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("cumulative monthly return")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RES, "h001_vs_umd.png")
    fig.savefig(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
