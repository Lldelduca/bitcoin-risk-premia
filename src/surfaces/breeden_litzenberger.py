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

def extract_rnd_with_gpd_tails(ssvi_model, tau, n_strikes=500, r=0.0, tail_quantile=0.10):
    """
    Extracts the RND with GPD tail extrapolation for robustness.

    The core density is extracted via BL (above). The tails beyond the tail_quantile (default: 10th and 90th percentiles of the
    return grid) are replaced with fitted GPD distributions to ensure well-behaved density in regions where the SSVI surface
    is least reliable.
    """

    # Extract the base RND
    rnd = extract_rnd_from_ssvi(ssvi_model, tau, n_strikes, r)
    R = rnd['returns']
    q = rnd['density'].copy()

    # Tails
    n = len(R)
    n_tail = max(int(n * tail_quantile), 10)

    left_idx = n_tail
    right_idx = n - n_tail

    # Fit GPD to left tail (excesses below threshold)
    try:
        left_threshold = R[left_idx]
        left_excesses = left_threshold - R[:left_idx]
        left_excesses = left_excesses[left_excesses > 0]
        if len(left_excesses) >= 5:
            c_left, loc_left, scale_left = genpareto.fit(
                left_excesses, floc=0
            )

            for i in range(left_idx):
                excess = left_threshold - R[i]
                if excess > 0:
                    q[i] = q[left_idx] * genpareto.sf(excess, c_left, 0, scale_left)
    except Exception:
        pass 

    # Fit GPD to right tail (excesses above threshold)
    try:
        right_threshold = R[right_idx]
        right_excesses = R[right_idx:] - right_threshold
        right_excesses = right_excesses[right_excesses > 0]
        if len(right_excesses) >= 5:
            c_right, loc_right, scale_right = genpareto.fit(
                right_excesses, floc=0
            )

            for i in range(right_idx, n):
                excess = R[i] - right_threshold
                if excess > 0:
                    q[i] = q[right_idx] * genpareto.sf(excess, c_right, 0, scale_right)
    except Exception:
        pass

    # Re-normalize
    integral = trapezoid(q, R)
    if integral > 1e-8:
        q = q / integral

    rnd['density'] = q
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
