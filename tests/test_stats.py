"""Monte-Carlo verification of the DSR screening machinery.

These tests ARE the evidence behind the "Monte-Carlo verified" claim in
src/oal/stats.py: a batch of pure-noise strategies must see its champion's
PSR deflated by DSR, while a genuine signal survives; the analytic
expected-max formula must track the simulated maximum; and the inverse
normal must be accurate. Kept small enough to run in seconds in CI.
"""
import math
import numpy as np
import pandas as pd

from oal.stats import (norm_ppf, psr, sr_stats, expected_max_sr,
                       effective_trials, screen_batch)

RNG = np.random.default_rng(7)


def test_norm_ppf_accuracy():
    for p, want in [(0.975, 1.959964), (0.5, 0.0), (0.999, 3.090232), (0.99, 2.326348)]:
        assert abs(norm_ppf(p) - want) < 1e-5


def test_psr_matches_normal_theory():
    # 5y daily, true ann Sharpe 1.0, normal returns: PSR(0) ≈ Φ(sr_d·√(T−1)) ≈ 0.987
    T, sr_d = 1260, 1.0 / math.sqrt(252)
    assert abs(psr(sr_d, 0.0, T, 0.0, 3.0) - 0.9873) < 0.002


def test_effective_trials_boundaries():
    assert effective_trials(8, 0.0) == 8            # independent → full count
    assert effective_trials(8, 1.0) == 1            # clones → one trial
    assert abs(effective_trials(10, 0.5) - 5.5) < 1e-9


def test_expected_max_tracks_simulation():
    # E[max SR] of N noise strategies: formula vs Monte Carlo
    N, T, sims = 50, 750, 40
    maxes = []
    for _ in range(sims):
        X = RNG.normal(0, 0.01, (N, T))
        srs = X.mean(1) / X.std(1, ddof=1)
        maxes.append(srs.max())
    v_sr = 1.0 / (T - 1)
    formula = expected_max_sr(N, v_sr)
    assert abs(np.mean(maxes) - formula) / formula < 0.15  # within 15%


def test_dsr_deflates_noise_champion_but_keeps_signal():
    N, T = 100, 1260
    X = RNG.normal(0, 0.01, (N, T))
    dates = pd.bdate_range("2019-01-01", periods=T)
    dailies = {f"n{i}": pd.Series(X[i], index=dates) for i in range(N)}
    # add one genuine signal, ann Sharpe 2
    dailies["real"] = pd.Series(RNG.normal(2 / math.sqrt(252) * 0.01, 0.01, T), index=dates)
    out = screen_batch(dailies).set_index("name")
    champ = out.drop(index="real").sort_values("psr", ascending=False).iloc[0]
    assert champ["psr"] > 0.95          # noise champion LOOKS significant...
    assert champ["dsr"] < 0.80          # ...and DSR takes it away
    assert out.loc["real", "dsr"] > 0.85  # while the true signal survives
