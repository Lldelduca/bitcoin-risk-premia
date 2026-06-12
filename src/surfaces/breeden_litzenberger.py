"""
Risk-Neutral Density Extraction via Breeden-Litzenberger (1978).

Converts implied volatilities to Black-Scholes call prices, computes the second derivative 
and recovers the risk-neutral density q_t(R) as a function of gross returns.

"""

import numpy as np
from scipy.stats import norm, genpareto
from scipy.integrate import trapezoid

def bs_call_price(F, K, tau, sigma, r=0.0):
    """ C = e^{-rτ} [F N(d1) - K N(d2)]"""
    if tau < 1e-8:
        return np.maximum(F - K, 0.0)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def extract_rnd_from_ssvi(ssvi_model, tau, n_strikes=500, r=0.0):
    """Extracts the risk-neutral density from a fitted SSVI surface at a specific maturity τ using Breeden-Litzenberger."""
    F = float(ssvi_model._forward_interp(tau))
    kappa_min, kappa_max = -1.5, 1.5 
    kappa_grid = np.linspace(kappa_min, kappa_max, n_strikes)
    K_grid = F * np.exp(kappa_grid)

    # Step 1: Evaluate SSVI-smoothed IV at each strike
    iv_grid = np.zeros(n_strikes)
    for i, k in enumerate(kappa_grid):
        try:
            w = ssvi_model.total_variance(tau, k)
            iv_grid[i] = np.sqrt(max(w, 1e-12) / tau)
        except Exception:
            iv_grid[i] = np.nan

    valid = ~np.isnan(iv_grid)
    if valid.sum() < 10:
        raise ValueError(f"Too few valid IV points ({valid.sum()}) for BL extraction")
    if not valid.all():
        iv_grid = np.interp(kappa_grid, kappa_grid[valid], iv_grid[valid])

    # Step 2: Convert IV → Black-Scholes call prices
    C_grid = bs_call_price(F, K_grid, tau, iv_grid, r)

    # Step 3: Compute ∂²C/∂K² via central finite differences
    dK = np.diff(K_grid)
    d2C_dK2 = np.zeros(n_strikes)
    for i in range(1, n_strikes - 1):
        dK_left = K_grid[i] - K_grid[i - 1]
        dK_right = K_grid[i + 1] - K_grid[i]
        d2C_dK2[i] = 2 * (
            C_grid[i + 1] / (dK_right * (dK_left + dK_right))
            - C_grid[i] / (dK_left * dK_right)
            + C_grid[i - 1] / (dK_left * (dK_left + dK_right))
        )

    # Apply BL formula: q(K) = e^{rτ} * ∂²C/∂K²
    q_K = np.exp(r * tau) * d2C_dK2
    q_K[0] = 0.0
    q_K[-1] = 0.0

    # Enforce non-negativity (butterfly condition)
    q_K = np.maximum(q_K, 0.0)

    # Step 4: Change of variable from K-space to return-space
    # R = K/F (gross return), so q(R) = F * q(K*F) by change of variable
    R_grid = K_grid / F 
    q_R = q_K * F 

    # Step 5: Normalize to integrate to 1
    integral = trapezoid(q_R, R_grid)
    if integral > 1e-8:
        q_R = q_R / integral
    else:
        raise ValueError("RND integral is near zero — density extraction failed")

    return {
        'returns': R_grid,
        'density': q_R,
        'strikes': K_grid,
        'call_prices': C_grid,
        'forward': F,
        'tau': tau,
        'kappa': kappa_grid,
        'iv': iv_grid,
    }

def _solve_gpd_shape(f_splice, f_match, mass, x_match):
    """Solves for the GPD (shape xi, scale beta) of an appended tail.

    The tail density beyond the splice point is modeled as
        f_tail(x) = mass * gpd_pdf(x; xi, loc=0, scale=beta),
    where x is the excess beyond the splice. Two identifying conditions, both
    taken from the option-implied density itself (Figlewski 2010; Birru &
    Figlewski 2012):

      1. Mass: the appended tail integrates to the residual probability
         `mass` by construction (the GPD pdf integrates to one).
         Continuity at the splice then pins the scale:
             mass / beta = f_splice  =>  beta = mass / f_splice.
      2. Shape: the tail density must also pass through the BL density at a
         second, deeper quantile point (excess x_match, level f_match);
         this one-dimensional condition pins xi.

    Returns (xi, beta). Falls back to the exponential tail xi = 0 (which
    still satisfies continuity and the mass condition) when no root exists.
    """
    beta = mass / f_splice

    def gap(xi):
        return mass * genpareto.pdf(x_match, xi, loc=0, scale=beta) - f_match

    # Bracket the root: gap is monotone increasing in xi for fixed x_match
    # beyond the scale, but scan a grid to be robust.
    xi_grid = np.linspace(-0.9, 10.0, 120)
    vals = np.array([gap(x) for x in xi_grid])
    finite = np.isfinite(vals)
    xi_grid, vals = xi_grid[finite], vals[finite]

    sign_change = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_change) == 0:
        return 0.0, beta  # exponential fallback (level + mass still matched)

    i = sign_change[0]
    try:
        from scipy.optimize import brentq
        xi = brentq(gap, xi_grid[i], xi_grid[i + 1], xtol=1e-10)
    except Exception:
        return 0.0, beta
    return float(xi), float(beta)


def append_gpd_tails(R, q, tail_quantile=0.10):
    """Replaces the tails of a normalized density on grid R with GPD tails
    matched to the density's own level and mass (Figlewski-style splice).

    Splice points are the `tail_quantile` and `1 - tail_quantile` quantiles
    of the density's CDF (probability quantiles, not grid-index fractions).
    The shape parameter of each tail is identified by additionally matching
    the density level at the `tail_quantile/2` (resp. `1 - tail_quantile/2`)
    quantile. The appended tails carry exactly the residual probability mass,
    so the spliced density integrates to one by construction (a final
    renormalization absorbs only trapezoid error and any mass beyond the
    finite grid).

    Returns (q_spliced, diagnostics_dict).
    """
    q = np.asarray(q, dtype=float).copy()
    R = np.asarray(R, dtype=float)

    # CDF of the input density
    dR = np.diff(R)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * dR)])
    total = cdf[-1]
    if total <= 1e-8:
        raise ValueError("append_gpd_tails: degenerate input density")
    cdf = cdf / total
    q = q / total

    def _quantile(alpha):
        idx = int(np.searchsorted(cdf, alpha))
        idx = min(max(idx, 1), len(R) - 2)
        return idx

    diag = {}

    # ---- Left tail ----
    iL = _quantile(tail_quantile)
    iL_match = _quantile(tail_quantile / 2.0)
    u_L, f_L = R[iL], q[iL]
    mass_L = cdf[iL]
    if f_L > 1e-12 and mass_L > 1e-8 and iL_match < iL:
        x_match = u_L - R[iL_match]
        xi_L, beta_L = _solve_gpd_shape(f_L, q[iL_match], mass_L, x_match)
        left = R < u_L
        q[left] = mass_L * genpareto.pdf(u_L - R[left], xi_L, loc=0, scale=beta_L)
        diag.update({"xi_left": xi_L, "beta_left": beta_L,
                     "splice_left": float(u_L), "mass_left": float(mass_L)})
    else:
        diag.update({"xi_left": np.nan, "beta_left": np.nan,
                     "splice_left": float(u_L), "mass_left": float(mass_L)})

    # ---- Right tail ----
    iR = _quantile(1.0 - tail_quantile)
    iR_match = _quantile(1.0 - tail_quantile / 2.0)
    u_R, f_R = R[iR], q[iR]
    mass_R = 1.0 - cdf[iR]
    if f_R > 1e-12 and mass_R > 1e-8 and iR_match > iR:
        x_match = R[iR_match] - u_R
        xi_R, beta_R = _solve_gpd_shape(f_R, q[iR_match], mass_R, x_match)
        right = R > u_R
        q[right] = mass_R * genpareto.pdf(R[right] - u_R, xi_R, loc=0, scale=beta_R)
        diag.update({"xi_right": xi_R, "beta_right": beta_R,
                     "splice_right": float(u_R), "mass_right": float(mass_R)})
    else:
        diag.update({"xi_right": np.nan, "beta_right": np.nan,
                     "splice_right": float(u_R), "mass_right": float(mass_R)})

    # Renormalize (absorbs trapezoid error and mass beyond the finite grid)
    integral = trapezoid(q, R)
    if integral > 1e-8:
        q = q / integral
    diag["mass_renorm"] = float(integral)

    return q, diag


def extract_rnd_with_gpd_tails(ssvi_model, tau, n_strikes=500, r=0.0, tail_quantile=0.10):
    """
    Extracts the RND with GPD tail extrapolation.

    The core density is extracted via BL (above). Beyond the tail_quantile
    probability quantiles of the extracted density, the tails are replaced
    with GPD tails whose scale is pinned by density continuity at the splice
    and whose shape is pinned by matching the BL density at a second, deeper
    quantile (Figlewski 2010 splice). Both parameters are therefore
    identified from the option-implied density itself.

    (The previous implementation called genpareto.fit on the deterministic
    grid coordinates — an i.i.d. fit to evenly spaced points — so its tail
    parameters reflected grid geometry rather than market information.)
    """

    # Extract the base RND
    rnd = extract_rnd_from_ssvi(ssvi_model, tau, n_strikes, r)
    q_spliced, tail_diag = append_gpd_tails(rnd['returns'], rnd['density'],
                                            tail_quantile=tail_quantile)
    rnd['density'] = q_spliced
    rnd['tail_diagnostics'] = tail_diag
    return rnd

def validate_rnd(rnd, rtol=0.02):
    """
    Checks that the extracted RND satisfies basic validity conditions:
      1. Integrates to ~1 (within rtol)
      2. Non-negative everywhere
      3. Mean return is close to the forward (risk-neutral condition)
      4. Density is not degenerate (std > 0)
    """
    R = rnd['returns']
    q = rnd['density']

    integral = trapezoid(q, R)
    mean_R = trapezoid(R * q, R)
    var_R = trapezoid((R - mean_R)**2 * q, R)
    std_R = np.sqrt(max(var_R, 0))
    non_negative = (q >= -1e-10).all()
    mean_check = abs(mean_R - 1.0) < 0.10 

    return {
        'integral': integral,
        'integral_ok': abs(integral - 1.0) < rtol,
        'mean_return': mean_R,
        'mean_ok': mean_check,
        'std_return': std_R,
        'non_negative': non_negative,
        'valid': abs(integral - 1.0) < rtol and non_negative and mean_check,
    }
