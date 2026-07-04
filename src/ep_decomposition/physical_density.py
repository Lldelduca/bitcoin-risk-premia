"""
Two estimators for the unconditional physical density p(R) of 27-day Bitcoin gross spot returns:

  1. Almeida et al. (2026) / Figlewski (2008)
     Full-sample 27-day returns -> histogram body -> 10th-order polynomial smoothing on [q10, q90] -> GEV tails

  2. Gaussian KDE with Scott's rule
     Full-sample 27-day returns -> Gaussian KDE -> GPD tails via a mass-preserving peaks-over-threshold splice.

"""

import warnings
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from typing import NamedTuple


class PhysicalDensity(NamedTuple):
    R_grid: np.ndarray        # gross return grid
    p_R: np.ndarray           # density values on grid
    method: str               # "almeida" or "kde"
    n_returns: int            # number of overlapping returns used
    bandwidth: float | None   # KDE bandwidth (None for Almeida)
    diagnostics: dict | None = None  # tail-fit diagnostics

# Shared utilities
def compute_overlapping_returns(spot: np.ndarray, horizon: int = 27) -> np.ndarray:
    R = spot[horizon:] / spot[:-horizon]
    return R[np.isfinite(R) & (R > 0)]

def _empirical_pdf_at(x: float, R_data: np.ndarray, n_bins: int) -> float:
    counts, edges = np.histogram(R_data, bins=n_bins, density=True)
    idx = int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1))
    val = float(counts[idx])
    if val <= 0.0:
        val = float(stats.gaussian_kde(R_data)(x)[0])
    return val

def _splice_pdf_target(poly_fn, x: float, R_data: np.ndarray, n_bins: int):

    poly_at = float(poly_fn(x))
    hist_at = _empirical_pdf_at(x, R_data, n_bins)
    if poly_at > 0.0 and poly_at >= 0.5 * hist_at:
        return max(poly_at, 1e-10), "poly_body"
    warnings.warn(
        f"Polynomial body degenerate at splice x={x:.4f} "
        f"(poly={poly_at:.4g}, hist={hist_at:.4g}); using histogram fallback."
    )
    return max(hist_at, 1e-10), "histogram_fallback"

def _fit_gpd_tail(excesses: np.ndarray):
    shape, _, scale = stats.genpareto.fit(excesses, floc=0)
    return shape, scale

# GEV point-matching fit (Figlewski 2008)
def _fit_gev_point_matching(x1: float, F1_target: float, f1_target: float, x2: float, f2_target: float,
    init_loc: float, init_scale: float, w_cdf: float = 10.0):

    mono_grid = x1 + np.linspace(0.0, 6.0, 60) * max(x2 - x1, 1e-3)

    def residuals(params):
        c, loc, log_scale = params
        scale = np.exp(log_scale)
        with np.errstate(all="ignore"):
            rF = stats.genextreme.cdf(x1, c, loc=loc, scale=scale) - F1_target
            rf1 = stats.genextreme.pdf(x1, c, loc=loc, scale=scale) - f1_target
            rf2 = stats.genextreme.pdf(x2, c, loc=loc, scale=scale) - f2_target
            pdf_tail = stats.genextreme.pdf(mono_grid, c, loc=loc, scale=scale)
            r_mono = 10.0 * float(np.sum(np.clip(np.diff(pdf_tail), 0.0, None)))
        out = np.array([w_cdf * rF, rf1, rf2, r_mono])
        return np.where(np.isfinite(out), out, 1e3)

    best = None
    for c0 in (-0.3, -0.1, 0.0, 0.1, 0.3):
        try:
            sol = least_squares(
                residuals,
                x0=np.array([c0, init_loc, np.log(init_scale)]),
                bounds=([-0.95, -np.inf, np.log(1e-6)],
                        [0.95, np.inf, np.log(1e3)]),
                method="trf", xtol=1e-14, ftol=1e-14, max_nfev=5000,
            )
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("GEV point-matching fit failed for all starts.")

    c, loc, log_scale = best.x
    info = {
        "cost": float(best.cost),
        "residuals": best.fun.tolist(),
        "xi_figlewski": float(-c),
        "success": bool(best.success),
    }
    return float(c), float(loc), float(np.exp(log_scale)), info

# Estimator 1: Almeida et al. (2026) body + Figlewski GEV tails
def estimate_physical_density_almeida(spot: np.ndarray, R_grid: np.ndarray, horizon: int = 27, n_bins: int = 12,
    poly_order: int = 10, lower_pct: int = 10, upper_pct: int = 90) -> PhysicalDensity:

    R_data = compute_overlapping_returns(spot, horizon)
    return estimate_physical_density_almeida_from_returns(R_data, R_grid, n_bins=n_bins, poly_order=poly_order,
                                                          lower_pct=lower_pct, upper_pct=upper_pct)

def estimate_physical_density_almeida_from_returns(R_data: np.ndarray, R_grid: np.ndarray, n_bins: int = 12,
    poly_order: int = 10, lower_pct: int = 10, upper_pct: int = 90,anchor_offset_pct: float = 5.0) -> PhysicalDensity:

    R_grid = np.asarray(R_grid, dtype=float)

    # Step 1: histogram density estimate
    counts, bin_edges = np.histogram(R_data, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Step 2: polynomial smoothing on body region
    u_L = float(np.percentile(R_data, lower_pct))
    u_R = float(np.percentile(R_data, upper_pct))
    body_mask = (bin_centers >= u_L) & (bin_centers <= u_R)

    if body_mask.sum() > poly_order:
        coeffs = np.polyfit(bin_centers[body_mask], counts[body_mask], poly_order)
    else:
        coeffs = np.polyfit(bin_centers, counts,
                            min(poly_order, len(bin_centers) - 1))
    poly_fn = np.poly1d(coeffs)

    p_R = np.zeros_like(R_grid, dtype=float)
    body_grid_mask = (R_grid >= u_L) & (R_grid <= u_R)
    p_R[body_grid_mask] = np.maximum(poly_fn(R_grid[body_grid_mask]), 0.0)

    diagnostics = {"u_L": u_L, "u_R": u_R}

    # Step 3a: right tail -- GEV point-matched at (q90, q95)
    x2_R = float(np.percentile(R_data, upper_pct + anchor_offset_pct))
    f1_R, src_R = _splice_pdf_target(poly_fn, u_R, R_data, n_bins)
    f2_R = max(_empirical_pdf_at(x2_R, R_data, n_bins), 1e-10)
    c_R, loc_R, scale_R, info_R = _fit_gev_point_matching(
        x1=u_R, F1_target=upper_pct / 100.0, f1_target=f1_R,
        x2=x2_R, f2_target=f2_R,
        init_loc=float(np.median(R_data)), init_scale=float(np.std(R_data)))
    right_mask = R_grid > u_R
    p_R[right_mask] = stats.genextreme.pdf(
        R_grid[right_mask], c_R, loc=loc_R, scale=scale_R)
    diagnostics["gev_right"] = {"c_scipy": c_R, "loc": loc_R,
                                "scale": scale_R,
                                "f1_source": src_R, **info_R}

    # Step 3b: left tail -- GEV on the mirrored variable Y = -R at (q10, q5)
    y1 = -u_L
    y2 = -float(np.percentile(R_data, lower_pct - anchor_offset_pct))
    f1_L, src_L = _splice_pdf_target(poly_fn, u_L, R_data, n_bins)
    f2_L = max(_empirical_pdf_at(-y2, R_data, n_bins), 1e-10)
    c_L, loc_L, scale_L, info_L = _fit_gev_point_matching(
        x1=y1, F1_target=1.0 - lower_pct / 100.0, f1_target=f1_L,
        x2=y2, f2_target=f2_L,
        init_loc=float(np.median(-R_data)), init_scale=float(np.std(R_data)))
    left_mask = R_grid < u_L
    p_R[left_mask] = stats.genextreme.pdf(
        -R_grid[left_mask], c_L, loc=loc_L, scale=scale_L)
    diagnostics["gev_left"] = {"c_scipy": c_L, "loc": loc_L,
                               "scale": scale_L,
                               "f1_source": src_L, **info_L}

    def _interior_rise(tail_vals, edge_idx):
        if tail_vals.size <= 2 or tail_vals[edge_idx] <= 0:
            return 0.0
        return float(tail_vals.max() / tail_vals[edge_idx] - 1.0)

    rise_R = _interior_rise(p_R[right_mask], 0)
    if rise_R > 0.01:
        c_R, loc_R, scale_R, info_R = _fit_gev_point_matching(
            x1=u_R, F1_target=upper_pct / 100.0, f1_target=f1_R,
            x2=x2_R, f2_target=f2_R,
            init_loc=float(np.median(R_data)), init_scale=float(np.std(R_data)),
            w_cdf=50.0)
        p_R[right_mask] = stats.genextreme.pdf(
            R_grid[right_mask], c_R, loc=loc_R, scale=scale_R)
        diagnostics["gev_right"] = {"c_scipy": c_R, "loc": loc_R,
                                    "scale": scale_R, "f1_source": src_R,
                                    "escalated": True, **info_R}
        rise_R = _interior_rise(p_R[right_mask], 0)
        if rise_R > 0.01:
            warnings.warn(f"Right GEV tail interior rise {rise_R:.1%} "
                          f"persists after escalation; inspect diagnostics.")
    diagnostics["interior_rise_right"] = rise_R

    tail_L = p_R[left_mask]
    rise_L = _interior_rise(tail_L, tail_L.size - 1)
    if rise_L > 0.01:
        c_L, loc_L, scale_L, info_L = _fit_gev_point_matching(
            x1=y1, F1_target=1.0 - lower_pct / 100.0, f1_target=f1_L,
            x2=y2, f2_target=f2_L,
            init_loc=float(np.median(-R_data)), init_scale=float(np.std(R_data)),
            w_cdf=50.0)
        p_R[left_mask] = stats.genextreme.pdf(
            -R_grid[left_mask], c_L, loc=loc_L, scale=scale_L)
        diagnostics["gev_left"] = {"c_scipy": c_L, "loc": loc_L,
                                   "scale": scale_L, "f1_source": src_L,
                                   "escalated": True, **info_L}
        tail_L = p_R[left_mask]
        rise_L = _interior_rise(tail_L, tail_L.size - 1)
        if rise_L > 0.01:
            warnings.warn(f"Left GEV tail interior rise {rise_L:.1%} "
                          f"persists after escalation; inspect diagnostics.")
    diagnostics["interior_rise_left"] = rise_L

    # Continuity gaps at the splices (v3: should be ~ LSQ residual only)
    for name, u in (("left", u_L), ("right", u_R)):
        i = int(np.searchsorted(R_grid, u))
        if 1 <= i < len(R_grid) - 1:
            diagnostics[f"continuity_gap_{name}"] = float(abs(p_R[i] - p_R[i - 1]))

    # Step 4: single global renormalization (mass ~ 1 by construction)
    mass = float(np.trapezoid(p_R, R_grid))
    diagnostics["mass_pre_norm"] = mass
    if mass <= 0:
        raise RuntimeError("Non-positive density mass; tail fit failed.")
    p_R = p_R / mass
    diagnostics["tail_mass_left"] = float(
        np.trapezoid(p_R[left_mask], R_grid[left_mask]))
    diagnostics["tail_mass_right"] = float(
        np.trapezoid(p_R[right_mask], R_grid[right_mask]))

    return PhysicalDensity(R_grid=R_grid, p_R=p_R, method="almeida",
                           n_returns=len(R_data), bandwidth=None,
                           diagnostics=diagnostics)

# Estimator 2: Gaussian KDE body + mass-preserving POT GPD tails
def estimate_physical_density_kde(spot: np.ndarray, R_grid: np.ndarray, horizon: int = 27,
    lower_pct: int = 10, upper_pct: int = 90) -> PhysicalDensity:

    R_data = compute_overlapping_returns(spot, horizon)
    return estimate_physical_density_kde_from_returns(
        R_data, R_grid, lower_pct=lower_pct, upper_pct=upper_pct)

def estimate_physical_density_kde_from_returns(R_data: np.ndarray, R_grid: np.ndarray,
    lower_pct: int = 10, upper_pct: int = 90) -> PhysicalDensity:

    R_grid = np.asarray(R_grid, dtype=float)

    kde = stats.gaussian_kde(R_data, bw_method="scott")
    p_R = np.asarray(kde(R_grid), dtype=float)
    bw = kde.factor * R_data.std(ddof=1)

    u_L = float(np.percentile(R_data, lower_pct))
    u_R = float(np.percentile(R_data, upper_pct))
    tail_prob_L = lower_pct / 100.0
    tail_prob_R = 1.0 - upper_pct / 100.0
    diagnostics = {"u_L": u_L, "u_R": u_R}

    # Left tail: POT splice, unconditional density = tail_prob * f_GPD
    left_data = R_data[R_data < u_L]
    if len(left_data) > 10:
        shape_L, scale_L = _fit_gpd_tail(u_L - left_data)
        left_mask = R_grid < u_L
        p_R[left_mask] = tail_prob_L * stats.genpareto.pdf(
            u_L - R_grid[left_mask], shape_L, loc=0, scale=scale_L)
        diagnostics["gpd_left"] = {"shape": float(shape_L), "scale": float(scale_L)}

    # Right tail: POT splice
    right_data = R_data[R_data > u_R]
    if len(right_data) > 10:
        shape_R, scale_R = _fit_gpd_tail(right_data - u_R)
        right_mask = R_grid > u_R
        p_R[right_mask] = tail_prob_R * stats.genpareto.pdf(
            R_grid[right_mask] - u_R, shape_R, loc=0, scale=scale_R)
        diagnostics["gpd_right"] = {"shape": float(shape_R), "scale": float(scale_R)}

    # Floor at zero and single global renormalization
    p_R = np.maximum(p_R, 0.0)
    mass = float(np.trapezoid(p_R, R_grid))
    diagnostics["mass_pre_norm"] = mass
    if mass > 0:
        p_R = p_R / mass
    diagnostics["tail_mass_left"] = float(
        np.trapezoid(p_R[R_grid < u_L], R_grid[R_grid < u_L]))
    diagnostics["tail_mass_right"] = float(
        np.trapezoid(p_R[R_grid > u_R], R_grid[R_grid > u_R]))

    return PhysicalDensity(R_grid=R_grid, p_R=p_R, method="kde",
                           n_returns=len(R_data), bandwidth=bw,
                           diagnostics=diagnostics)
