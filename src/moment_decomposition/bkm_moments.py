"""
Bakshi-Kapadia-Madan (2003) Risk-Neutral Moment Extraction.

Computes the risk-neutral variance, cubic contract, and quartic contract from OTM option prices evaluated on the fitted SSVI

  V_t = integral of [2(1 - ln(K/F)) / K^2] * OTM(K) dK

  W_t = integral of [6 ln(K/F) - 3 ln(K/F)^2] / K^2 * OTM(K) dK

  X_t = integral of [12 ln(K/F)^2 - 4 ln(K/F)^3] / K^2 * OTM(K) dK

From these, the risk-neutral moments are:

  mu_Q    = -(e^{r*tau} * V/2 + e^{r*tau} * W/6 + e^{r*tau} * X/24)

  var_Q   = e^{r*tau} * V - mu_Q^2
  skew_Q  = (e^{r*tau} * W - 3*mu_Q*e^{r*tau}*V + 2*mu_Q^3) / var_Q^{3/2}
  kurt_Q  = (e^{r*tau} * X - 4*mu_Q*e^{r*tau}*W + 6*mu_Q^2*e^{r*tau}*V - 3*mu_Q^4) / var_Q^2

"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm
from typing import NamedTuple

class BKMMoments(NamedTuple):
    V: float            # variance contract
    W: float            # cubic contract
    X: float            # quartic contract
    mu_Q: float         # risk-neutral expected log forward return
    var_Q: float        # risk-neutral variance
    skew_Q: float       # risk-neutral skewness
    kurt_Q: float       # risk-neutral kurtosis (raw, i.e. 3 under lognormality)
    forward: float      # forward price used
    tau: float          # time to maturity (years)
    n_strikes: int      # number of strike points used
    kappa_bound: float  # |log-moneyness| truncation used for the integrals

def bs_call_price(F, K, tau, sigma, r=0.0):
    if tau < 1e-8:
        return np.maximum(F - K, 0.0)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def bs_put_price(F, K, tau, sigma, r=0.0):
    C = bs_call_price(F, K, tau, sigma, r)
    return C - np.exp(-r * tau) * (F - K)

def evaluate_iv_grid(ssvi_model, tau, n_strikes=500, kappa_max=1.5):
    F = float(ssvi_model._forward_interp(tau))
    kappa_grid = np.linspace(-kappa_max, kappa_max, n_strikes)

    iv_grid = np.zeros(n_strikes)
    for i, k in enumerate(kappa_grid):
        try:
            w = ssvi_model.total_variance(tau, k)
            iv_grid[i] = np.sqrt(max(w, 1e-12) / tau)
        except Exception:
            iv_grid[i] = np.nan

    valid = ~np.isnan(iv_grid)
    if valid.sum() < 10:
        raise ValueError(f"Too few valid IV points ({valid.sum()}) for BKM extraction")
    if not valid.all():
        iv_grid = np.interp(kappa_grid, kappa_grid[valid], iv_grid[valid])

    return F, kappa_grid, iv_grid

def bkm_from_iv_grid(F, kappa_grid, iv_grid, tau, r=0.0, kappa_bound=None):
    if kappa_bound is not None:
        mask = np.abs(kappa_grid) <= kappa_bound + 1e-12
        kappa = kappa_grid[mask]
        iv = iv_grid[mask]
    else:
        kappa = kappa_grid
        iv = iv_grid
        kappa_bound = float(np.max(np.abs(kappa_grid)))

    K_grid = F * np.exp(kappa)
    n = len(K_grid)

    # OTM option prices
    otm_prices = np.where(
        K_grid >= F,
        bs_call_price(F, K_grid, tau, iv, r),
        bs_put_price(F, K_grid, tau, iv, r),
    )

    # Log forward moneyness
    log_KF = kappa

    # BKM payoff kernels (forward-based, log(K/F))
    kernel_V = 2.0 * (1.0 - log_KF) / K_grid**2
    kernel_W = (6.0 * log_KF - 3.0 * log_KF**2) / K_grid**2
    kernel_X = (12.0 * log_KF**2 - 4.0 * log_KF**3) / K_grid**2

    V = trapezoid(kernel_V * otm_prices, K_grid)
    W = trapezoid(kernel_W * otm_prices, K_grid)
    X = trapezoid(kernel_X * otm_prices, K_grid)

    ert = np.exp(r * tau)

    # Forward-measure-consistent mean
    mu_Q = -(ert * V / 2.0 + ert * W / 6.0 + ert * X / 24.0)
    var_Q = ert * V - mu_Q**2

    if var_Q > 1e-20:
        skew_Q = (ert * W - 3.0 * mu_Q * ert * V + 2.0 * mu_Q**3) / var_Q**1.5
        kurt_Q = (ert * X - 4.0 * mu_Q * ert * W
                  + 6.0 * mu_Q**2 * ert * V - 3.0 * mu_Q**4) / var_Q**2
    else:
        skew_Q = np.nan
        kurt_Q = np.nan

    return BKMMoments(V=V, W=W, X=X, mu_Q=mu_Q, var_Q=var_Q, skew_Q=skew_Q, kurt_Q=kurt_Q,
        forward=F, tau=tau, n_strikes=n, kappa_bound=float(kappa_bound))

def extract_bkm_moments(ssvi_model, tau, n_strikes=500, r=0.0, kappa_bound=None):
    F, kappa_grid, iv_grid = evaluate_iv_grid(ssvi_model, tau, n_strikes=n_strikes, kappa_max=1.5)
    return bkm_from_iv_grid(F, kappa_grid, iv_grid, tau, r=r, kappa_bound=kappa_bound)
