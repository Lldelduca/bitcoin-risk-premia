"""
Two estimators for the unconditional physical density p(R) of 27-day Bitcoin gross spot returns:

  1. Almeida et al. (2026)
     Full-sample 27-day returns -> histogram body -> 10th-order polynomial smoothing -> GEV tails

  2. Gaussian KDE with Scott's rule 
     Full-sample 27-day returns -> Gaussian KDE with plug-inbandwidth -> GPD tails

"""

import numpy as np
from scipy import stats
from typing import NamedTuple

class PhysicalDensity(NamedTuple):
    R_grid: np.ndarray       # gross return grid
    p_R: np.ndarray          # density values on grid
    method: str              # "almeida" or "kde"
    n_returns: int           # number of overlapping returns used
    bandwidth: float | None  # KDE bandwidth (None for Almeida)

# Shared utilities
def compute_overlapping_returns(spot: np.ndarray, horizon: int = 27) -> np.ndarray:
    R = spot[horizon:] / spot[:-horizon]
    return R[np.isfinite(R) & (R > 0)]

def _fit_gev_tail(excesses: np.ndarray):
    c, loc, scale = stats.genextreme.fit(excesses)
    return c, loc, scale

def _fit_gpd_tail(excesses: np.ndarray):
    shape, _, scale = stats.genpareto.fit(excesses, floc=0)
    return shape, scale

def _splice_and_normalize(R_grid, p_body, R_data, lower_pct=10, upper_pct=90, tail_type="gev"):
    u_L = np.percentile(R_data, lower_pct)
    u_R = np.percentile(R_data, upper_pct)

    left_data = R_data[R_data < u_L]
    right_data = R_data[R_data > u_R]

    p_out = p_body.copy()

    if tail_type == "gev":
        # Left tail: GEV on (u_L - R) for R < u_L
        if len(left_data) > 10:
            c_L, loc_L, scale_L = _fit_gev_tail(u_L - left_data)
            left_mask = R_grid < u_L
            excesses_L = u_L - R_grid[left_mask]
            p_left = stats.genextreme.pdf(excesses_L, c_L, loc=loc_L, scale=scale_L)
            # Scale to match body at splice point
            idx_splice_L = np.searchsorted(R_grid, u_L)
            if idx_splice_L < len(p_body) and p_body[idx_splice_L] > 0 and len(p_left) > 0 and p_left[-1] > 0:
                p_left *= p_body[idx_splice_L] / p_left[-1]
            p_out[left_mask] = p_left

        # Right tail: GEV on (R - u_R) for R > u_R
        if len(right_data) > 10:
            c_R, loc_R, scale_R = _fit_gev_tail(right_data - u_R)
            right_mask = R_grid > u_R
            excesses_R = R_grid[right_mask] - u_R
            p_right = stats.genextreme.pdf(excesses_R, c_R, loc=loc_R, scale=scale_R)
            idx_splice_R = np.searchsorted(R_grid, u_R) - 1
            if idx_splice_R >= 0 and p_body[idx_splice_R] > 0 and len(p_right) > 0 and p_right[0] > 0:
                p_right *= p_body[idx_splice_R] / p_right[0]
            p_out[right_mask] = p_right

    elif tail_type == "gpd":
        # Left tail: GPD on (u_L - R) for R < u_L
        if len(left_data) > 10:
            shape_L, scale_L = _fit_gpd_tail(u_L - left_data)
            left_mask = R_grid < u_L
            excesses_L = u_L - R_grid[left_mask]
            p_left = stats.genpareto.pdf(excesses_L, shape_L, loc=0, scale=scale_L)
            idx_splice_L = np.searchsorted(R_grid, u_L)
            if idx_splice_L < len(p_body) and p_body[idx_splice_L] > 0 and len(p_left) > 0 and p_left[-1] > 0:
                p_left *= p_body[idx_splice_L] / p_left[-1]
            p_out[left_mask] = p_left

        # Right tail: GPD on (R - u_R) for R > u_R
        if len(right_data) > 10:
            shape_R, scale_R = _fit_gpd_tail(right_data - u_R)
            right_mask = R_grid > u_R
            excesses_R = R_grid[right_mask] - u_R
            p_right = stats.genpareto.pdf(excesses_R, shape_R, loc=0, scale=scale_R)
            idx_splice_R = np.searchsorted(R_grid, u_R) - 1
            if idx_splice_R >= 0 and p_body[idx_splice_R] > 0 and len(p_right) > 0 and p_right[0] > 0:
                p_right *= p_body[idx_splice_R] / p_right[0]
            p_out[right_mask] = p_right

    # Floor at zero and renormalize
    p_out = np.maximum(p_out, 0)
    mass = np.trapezoid(p_out, R_grid)
    if mass > 0:
        p_out /= mass

    return p_out

# Estimator 1: Almeida et al. (2026)
def estimate_physical_density_almeida(
    spot: np.ndarray, R_grid: np.ndarray, horizon: int = 27, n_bins: int = 12, poly_order: int = 10, lower_pct: int = 10, 
    upper_pct: int = 90) -> PhysicalDensity:
    R_data = compute_overlapping_returns(spot, horizon)
    return estimate_physical_density_almeida_from_returns(
        R_data, R_grid, n_bins=n_bins, poly_order=poly_order,
        lower_pct=lower_pct, upper_pct=upper_pct)

def estimate_physical_density_almeida_from_returns(
    R_data: np.ndarray, R_grid: np.ndarray, n_bins: int = 12, poly_order: int = 10, lower_pct: int = 10,
    upper_pct: int = 90) -> PhysicalDensity:

    # Step 1: histogram density estimate
    counts, bin_edges = np.histogram(R_data, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Step 2: polynomial smoothing on body region
    u_L = np.percentile(R_data, lower_pct)
    u_R = np.percentile(R_data, upper_pct)
    body_mask = (bin_centers >= u_L) & (bin_centers <= u_R)

    if body_mask.sum() > poly_order:
        coeffs = np.polyfit(bin_centers[body_mask], counts[body_mask], poly_order)
        poly_fn = np.poly1d(coeffs)
    else:
        coeffs = np.polyfit(bin_centers, counts, min(poly_order, len(bin_centers) - 1))
        poly_fn = np.poly1d(coeffs)

    # Evaluate polynomial body on the full return grid within [u_L, u_R]
    p_body = np.zeros_like(R_grid, dtype=float)
    body_grid_mask = (R_grid >= u_L) & (R_grid <= u_R)
    p_body[body_grid_mask] = poly_fn(R_grid[body_grid_mask])
    p_body = np.maximum(p_body, 0)

    # Step 3: splice GEV tails and renormalize
    p_R = _splice_and_normalize(R_grid, p_body, R_data, lower_pct, upper_pct, tail_type="gev")

    return PhysicalDensity(R_grid=R_grid, p_R=p_R, method="almeida", n_returns=len(R_data), bandwidth=None)

# Estimator 2: Gaussian KDE
def estimate_physical_density_kde(
    spot: np.ndarray, R_grid: np.ndarray, horizon: int = 27, lower_pct: int = 10, upper_pct: int = 90) -> PhysicalDensity:
    R_data = compute_overlapping_returns(spot, horizon)
    return estimate_physical_density_kde_from_returns(
        R_data, R_grid, lower_pct=lower_pct, upper_pct=upper_pct)

def estimate_physical_density_kde_from_returns(
    R_data: np.ndarray, R_grid: np.ndarray, lower_pct: int = 10, upper_pct: int = 90) -> PhysicalDensity:

    # Gaussian KDE with Scott's rule (approximates Sheather-Jones for large n)
    kde = stats.gaussian_kde(R_data, bw_method="scott")
    p_body = kde(R_grid)
    bw = kde.factor * R_data.std(ddof=1)

    # Splice GPD tails and renormalize
    p_R = _splice_and_normalize(R_grid, p_body, R_data, lower_pct, upper_pct,
                                tail_type="gpd")

    return PhysicalDensity(R_grid=R_grid, p_R=p_R, method="kde", n_returns=len(R_data), bandwidth=bw)
