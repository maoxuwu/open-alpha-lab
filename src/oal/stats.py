"""PSR / Deflated Sharpe / yearly stability.

Ported from a Monte-Carlo-verified implementation (2026-07-02): with 100 pure-noise
strategies over T=1260 days, the batch max annualized Sharpe was 1.22 vs formula 1.17;
the noise champion's PSR read 0.997 but DSR correctly deflated it to 0.539, while a
true ann-Sharpe-2.0 signal retained DSR 0.905. References: Bailey & López de Prado
(2012, 2014); Mertens (2002); Lo (2002); Harvey, Liu & Zhu (2016).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EULER = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam inverse-normal approximation, |err| < 1.15e-9."""
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sr_stats(x: np.ndarray):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 60 or x.std(ddof=1) == 0:
        return T, float("nan"), float("nan"), float("nan")
    z = (x - x.mean()) / x.std(ddof=1)
    return T, float(x.mean() / x.std(ddof=1)), float((z**3).mean()), float((z**4).mean())


def psr(sr_hat: float, sr0: float, T: int, g3: float, g4: float) -> float:
    """P(true SR > sr0 | observed), per-period units, with skew/kurtosis correction."""
    if any(map(math.isnan, (sr_hat, g3, g4))) or T < 60:
        return float("nan")
    var_adj = 1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat**2
    if var_adj <= 0:
        return float("nan")
    return norm_cdf((sr_hat - sr0) * math.sqrt(T - 1) / math.sqrt(var_adj))


def expected_max_sr(n_eff: float, v_sr: float) -> float:
    """Expected max per-period SR of n_eff independent zero-skill trials."""
    if n_eff <= 1 or v_sr <= 0:
        return 0.0
    return math.sqrt(v_sr) * ((1 - EULER) * norm_ppf(1 - 1 / n_eff)
                              + EULER * norm_ppf(1 - 1 / (n_eff * math.e)))


def effective_trials(n: int, rho_bar: float) -> float:
    """N_eff = 1 + (N-1)(1-ρ̄): linear shrinkage for correlated trials."""
    return 1 + (n - 1) * (1.0 - max(0.0, min(1.0, rho_bar)))


def yearly_stability(daily: pd.Series):
    by = daily.groupby(daily.index.year)
    yr = {y: g.mean() / g.std(ddof=1) * math.sqrt(252) for y, g in by
          if len(g) >= 100 and g.std(ddof=1) > 0}
    if not yr:
        return float("nan"), float("nan")
    vals = list(yr.values())
    return sum(v > 0 for v in vals) / len(vals), min(vals)


def screen_batch(dailies: dict[str, pd.Series]) -> pd.DataFrame:
    """Batch screen: DSR against the batch's own noise ceiling. Input: name -> daily returns."""
    df = pd.DataFrame(dailies)
    n = df.shape[1]
    rho = float(np.nanmean(df.corr().values[np.triu_indices(n, 1)])) if n > 1 else 1.0
    n_eff = effective_trials(n, rho)
    stats = {k: sr_stats(v.values) for k, v in dailies.items()}
    srs = [s[1] for s in stats.values() if not math.isnan(s[1])]
    t_med = int(np.median([s[0] for s in stats.values()]))
    v_sr = float(np.var(srs, ddof=1)) if len(srs) >= 3 else 1.0 / (t_med - 1)
    sr0 = expected_max_sr(n_eff, v_sr)
    rows = []
    for k, (T, srh, g3, g4) in stats.items():
        ypos, ymin = yearly_stability(dailies[k])
        rows.append({"name": k, "sharpe_ann": round(srh * math.sqrt(252), 2) if not math.isnan(srh) else np.nan,
                     "T": T, "psr": round(psr(srh, 0, T, g3, g4), 4),
                     "sr0_ann": round(sr0 * math.sqrt(252), 2),
                     "dsr": round(psr(srh, sr0, T, g3, g4), 4),
                     "yr_pos": round(ypos, 2) if not math.isnan(ypos) else np.nan,
                     "yr_min": round(ymin, 2) if not math.isnan(ymin) else np.nan})
    return pd.DataFrame(rows).sort_values("dsr", ascending=False).reset_index(drop=True)
