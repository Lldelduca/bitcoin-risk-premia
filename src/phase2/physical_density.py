"""
Two estimators for the unconditional physical density p(R) of 27-day Bitcoin gross spot returns:

  1. Almeida et al. (2026) / Figlewski (2008)
     Full-sample 27-day returns -> histogram body -> 10th-order polynomial
     smoothing on [q10, q90] -> GEV tails calibrated by POINT MATCHING.

  2. Gaussian KDE with Scott's rule
     Full-sample 27-day returns -> Gaussian KDE -> GPD tails via a
     mass-preserving peaks-over-threshold splice.
     
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


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
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
    """MLE GPD fit to threshold excesses (correct for the POT splice)."""
    shape, _, scale = stats.genpareto.fit(excesses, floc=0)
    return shape, scale


# ---------------------------------------------------------------------------
# GEV point-matching fit (Figlewski 2008)
# ---------------------------------------------------------------------------
def _fit_gev_point_matching(
    x1: float, F1_target: float, f1_target: float,
    x2: float, f2_target: float,
    init_loc: float, init_scale: float,
    x_support: float,
    w_cdf: float = 10.0,
    w_f2: float = 1.0 / np.sqrt(10.0),
    w_sup: float = 10.0,
    x3: float | None = None,
    f3_target: float | None = None,
    w_f3: float = 0.5,
):

    mono_grid = x1 + np.linspace(0.0, 8.0, 80) * max(init_scale, 0.05)

    def residuals(params):
        c, loc, log_scale = params
        scale = np.exp(log_scale)
        with np.errstate(all="ignore"):
            rF = stats.genextreme.cdf(x1, c, loc=loc, scale=scale) - F1_target
            rf1 = stats.genextreme.pdf(x1, c, loc=loc, scale=scale) - f1_target
            rf2 = stats.genextreme.pdf(x2, c, loc=loc, scale=scale) - f2_target
            if x3 is not None:
                rf3 = stats.genextreme.pdf(x3, c, loc=loc, scale=scale) - f3_target
            else:
                rf3 = 0.0
            pdf_tail = stats.genextreme.pdf(mono_grid, c, loc=loc, scale=scale)
            r_mono = 10.0 * float(np.sum(np.clip(np.diff(pdf_tail), 0.0, None)))
            if c > 1e-8:                      # bounded upper support
                endpoint = loc + scale / c
                r_sup = w_sup * max(0.0, x_support - endpoint)
            else:
                r_sup = 0.0
        out = np.array([w_cdf * rF, rf1, w_f2 * rf2, w_f3 * rf3, r_mono, r_sup])
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


# ---------------------------------------------------------------------------
# Estimator 1: Almeida et al. (2026) body + Figlewski GEV tails
# ---------------------------------------------------------------------------
def estimate_physical_density_almeida(
    spot: np.ndarray, R_grid: np.ndarray, horizon: int = 27, n_bins: int = 12,
    poly_order: int = 10, lower_pct: int = 10, upper_pct: int = 90,
) -> PhysicalDensity:
    R_data = compute_overlapping_returns(spot, horizon)
    return estimate_physical_density_almeida_from_returns(
        R_data, R_grid, n_bins=n_bins, poly_order=poly_order,
        lower_pct=lower_pct, upper_pct=upper_pct)


def estimate_physical_density_almeida_from_returns(
    R_data: np.ndarray, R_grid: np.ndarray, n_bins: int = 12,
    poly_order: int = 10, lower_pct: int = 10, upper_pct: int = 90,
    inner_delta: float = 0.01,
) -> PhysicalDensity:

    R_grid = np.asarray(R_grid, dtype=float)

    # Step 1: histogram density estimate
    counts, bin_edges = np.histogram(R_data, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Step 2: polynomial smoothing
    u_L = float(np.percentile(R_data, lower_pct))
    u_R = float(np.percentile(R_data, upper_pct))
    body_mask = (bin_centers >= u_L) & (bin_centers <= u_R)  # diagnostics

    hist_uL = _empirical_pdf_at(u_L, R_data, n_bins)
    hist_uR = _empirical_pdf_at(u_R, R_data, n_bins)
    ANCHOR_W = 5.0

    x_fit = np.concatenate([[u_L], bin_centers, [u_R]])
    y_fit = np.concatenate([[hist_uL], counts, [hist_uR]])
    w_fit = np.concatenate([[ANCHOR_W], np.ones(len(bin_centers)), [ANCHOR_W]])
    order = np.argsort(x_fit)
    deg = min(poly_order, len(x_fit) - 1)
    coeffs = np.polyfit(x_fit[order], y_fit[order], deg, w=w_fit[order])
    poly_fn = np.poly1d(coeffs)

    p_R = np.zeros_like(R_grid, dtype=float)
    body_grid_mask = (R_grid >= u_L) & (R_grid <= u_R)
    p_R[body_grid_mask] = np.maximum(poly_fn(R_grid[body_grid_mask]), 0.0)

    body_target_mass = (upper_pct - lower_pct) / 100.0
    body_int = float(np.trapezoid(p_R[body_grid_mask], R_grid[body_grid_mask]))
    body_scale = body_target_mass / body_int if body_int > 0 else 1.0
    p_R[body_grid_mask] *= body_scale

    diagnostics = {"u_L": u_L, "u_R": u_R, "body_scale": body_scale}

    # Step 3a: right tail -- GEV point-matched per the AGMW replication

    x2_R = u_R - inner_delta
    x3_R = float(np.percentile(R_data, upper_pct + 5.0))   # outer anchor q95
    f1_R_raw, src_R = _splice_pdf_target(poly_fn, u_R, R_data, n_bins)
    f1_R = body_scale * f1_R_raw
    f2_R = body_scale * max(float(poly_fn(x2_R)), 1e-10)
    f3_R = max(_empirical_pdf_at(x3_R, R_data, n_bins), 1e-10)
    c_R, loc_R, scale_R, info_R = _fit_gev_point_matching(
        x1=u_R, F1_target=upper_pct / 100.0, f1_target=f1_R,
        x2=x2_R, f2_target=f2_R,
        init_loc=float(np.median(R_data)), init_scale=float(np.std(R_data)),
        x_support=float(np.max(R_data)),
        x3=x3_R, f3_target=f3_R)
    right_mask = R_grid > u_R
    p_R[right_mask] = stats.genextreme.pdf(
        R_grid[right_mask], c_R, loc=loc_R, scale=scale_R)
    diagnostics["gev_right"] = {"c_scipy": c_R, "loc": loc_R,
                                "scale": scale_R,
                                "f1_source": src_R, **info_R}

    # Step 3b: left tail -- GEV on the mirrored variable Y = -R, with the
    # inner anchor 0.01 inside the body (i.e., at R = u_L + 0.01).
    y1 = -u_L
    y2 = y1 - inner_delta                     # mirrors R = u_L + inner_delta
    x3_L = float(np.percentile(R_data, lower_pct - 5.0))   # outer anchor q05
    y3 = -x3_L
    f1_L_raw, src_L = _splice_pdf_target(poly_fn, u_L, R_data, n_bins)
    f1_L = body_scale * f1_L_raw
    f2_L = body_scale * max(float(poly_fn(u_L + inner_delta)), 1e-10)
    f3_L = max(_empirical_pdf_at(x3_L, R_data, n_bins), 1e-10)
    c_L, loc_L, scale_L, info_L = _fit_gev_point_matching(
        x1=y1, F1_target=1.0 - lower_pct / 100.0, f1_target=f1_L,
        x2=y2, f2_target=f2_L,
        init_loc=float(np.median(-R_data)), init_scale=float(np.std(R_data)),
        x_support=float(-np.min(R_data)),
        x3=y3, f3_target=f3_L)
    left_mask = R_grid < u_L
    p_R[left_mask] = stats.genextreme.pdf(
        -R_grid[left_mask], c_L, loc=loc_L, scale=scale_L)
    diagnostics["gev_left"] = {"c_scipy": c_L, "loc": loc_L,
                               "scale": scale_L,
                               "f1_source": src_L, **info_L}

    # Guards: tails must be monotone away from the splice.
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
            x_support=float(np.max(R_data)), w_cdf=50.0,
            x3=x3_R, f3_target=f3_R)
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
            x_support=float(-np.min(R_data)), w_cdf=50.0,
            x3=y3, f3_target=f3_L)
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


# ---------------------------------------------------------------------------
# Estimator 2: Gaussian KDE body + mass-preserving POT GPD tails
# ---------------------------------------------------------------------------
def estimate_physical_density_kde(
    spot: np.ndarray, R_grid: np.ndarray, horizon: int = 27,
    lower_pct: int = 10, upper_pct: int = 90,
) -> PhysicalDensity:
    R_data = compute_overlapping_returns(spot, horizon)
    return estimate_physical_density_kde_from_returns(
        R_data, R_grid, lower_pct=lower_pct, upper_pct=upper_pct)


def estimate_physical_density_kde_from_returns(
    R_data: np.ndarray, R_grid: np.ndarray,
    lower_pct: int = 10, upper_pct: int = 90,
) -> PhysicalDensity:

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

# ---------------------------------------------------------------------------
# Estimator 3: Grith et al. vanilla replication (AGMW OA conventions)
# ---------------------------------------------------------------------------
def _matlab_hist_density(R_data, n_bins):
    """MATLAB hist(data, centers) semantics: n_bins CENTERS spanning
    [min, max]; edges at midpoints with open-ended outer bins; density
    normalized by trapz over the centers (their convention)."""
    centers = np.linspace(np.min(R_data), np.max(R_data), n_bins)
    mid = 0.5 * (centers[:-1] + centers[1:])
    edges = np.concatenate([[-np.inf], mid, [np.inf]])
    counts, _ = np.histogram(R_data, bins=edges)
    counts = counts.astype(float)
    Z = np.trapezoid(counts, centers)
    return counts / Z if Z > 0 else counts, centers


def estimate_physical_density_grith_vanilla(
    spot: np.ndarray, R_grid: np.ndarray, horizon: int = 27, n_bins: int = 12,
    poly_order: int = 10, lower_pct: int = 10, upper_pct: int = 90,
) -> PhysicalDensity:
    R_data = compute_overlapping_returns(spot, horizon)
    return estimate_physical_density_grith_vanilla_from_returns(
        R_data, R_grid, n_bins=n_bins, poly_order=poly_order,
        lower_pct=lower_pct, upper_pct=upper_pct)


def estimate_physical_density_grith_vanilla_from_returns(
    R_data: np.ndarray, R_grid: np.ndarray, n_bins: int = 12,
    poly_order: int = 10, lower_pct: int = 10, upper_pct: int = 90,
) -> PhysicalDensity:

    r = np.asarray(R_data, dtype=float) - 1.0          # net returns
    dens, centers = _matlab_hist_density(r, n_bins)

    p10 = float(np.percentile(r, lower_pct))
    p90 = float(np.percentile(r, upper_pct))
    interior = centers[(centers > p10) & (centers < p90)]
    X_cut = np.concatenate([[p10], interior, [p90]])
    f_cut = np.interp(X_cut, centers, dens)

    rank_deficient = False
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        coeffs = np.polyfit(X_cut, f_cut, poly_order)
        rank_deficient = any("conditioned" in str(w.message).lower()
                             or "rank" in str(w.message).lower() for w in wlist)
    poly_fn = np.poly1d(coeffs)

    # body on a fine net grid, rescaled to 0.8 mass (her *0.8 step)
    x_fit = np.linspace(p10, p90, 1000)
    y_fit = np.maximum(poly_fn(x_fit), 0.0)
    Zb = np.trapezoid(y_fit, x_fit)
    body_scale = 0.8 / Zb if Zb > 0 else 1.0
    y_fit *= body_scale

    body_at = lambda x: body_scale * max(float(poly_fn(x)), 0.0)
    rnd_l = [body_at(p10), body_at(p10 + 0.01)]        # her target_l values
    rnd_r = [body_at(p90 - 0.01), body_at(p90)]        # her target_r values

    def gpdf(x, c, s, m):
        with np.errstate(all="ignore"):
            return stats.genextreme.pdf(x, c, loc=m, scale=s)

    def gcdf(x, c, s, m):
        with np.errstate(all="ignore"):
            return stats.genextreme.cdf(x, c, loc=m, scale=s)

    def loss(x):
        cL, sL, mL, cR, sR, mR = x
        t = [
            (gpdf(-p10, cL, sL, mL) - rnd_l[0]) ** 2,
            (gpdf(-(p10 + 0.01), cL, sL, mL) - rnd_l[1]) ** 2 / 10.0,
            ((1.0 - gcdf(-p10, cL, sL, mL)) - lower_pct / 100.0) ** 2,
            (gpdf(p90 - 0.01, cR, sR, mR) - rnd_r[0]) ** 2 / 10.0,
            (gpdf(p90, cR, sR, mR) - rnd_r[1]) ** 2,
            (gcdf(p90, cR, sR, mR) - upper_pct / 100.0) ** 2,
        ]
        v = float(np.sum(t))
        return v if np.isfinite(v) else 1e6

    cons = [
        {"type": "eq",
         "fun": lambda x: gpdf(-p10, x[0], x[1], x[2]) - rnd_l[0]},
        {"type": "eq",
         "fun": lambda x: gpdf(p90, x[3], x[4], x[5]) - rnd_r[1]},
        {"type": "ineq",
         "fun": lambda x: 0.01 - abs((1.0 - gcdf(-p10, x[0], x[1], x[2]))
                                     - lower_pct / 100.0)},
        {"type": "ineq",
         "fun": lambda x: 0.01 - abs(gcdf(p90, x[3], x[4], x[5])
                                     - upper_pct / 100.0)},
    ]
    # her OA bounds/starts, MATLAB k -> scipy c = -k
    k0, s0, m0 = 0.2, 0.13, 0.03
    x0 = np.array([-k0, s0, m0, -k0, s0, m0])
    bounds = [(-0.5, 0.05), (0.01, 0.20), (-0.5, 0.5),     # left  (c, s, mu)
              (-0.5, 0.05), (0.01, 0.25), (-0.5, 0.5)]     # right (c, s, mu)

    from scipy.optimize import minimize as _minimize
    sol = _minimize(loss, x0, method="SLSQP", bounds=bounds,
                    constraints=cons,
                    options={"maxiter": 2000, "ftol": 1e-14})
    cL, sL, mL, cR, sR, mR = sol.x

    # assemble on the GROSS grid (density invariant under the unit shift)
    rg = np.asarray(R_grid, dtype=float) - 1.0
    p_R = np.zeros_like(rg)
    body_m = (rg >= p10) & (rg <= p90)
    p_R[body_m] = np.maximum(body_scale * poly_fn(rg[body_m]), 0.0)
    left_m = rg < p10
    p_R[left_m] = gpdf(-rg[left_m], cL, sL, mL)
    right_m = rg > p90
    p_R[right_m] = gpdf(rg[right_m], cR, sR, mR)

    mass = float(np.trapezoid(p_R, R_grid))
    diagnostics = {
        "u_L": p10 + 1.0, "u_R": p90 + 1.0,
        "body_scale": body_scale,
        "rank_deficient_polyfit": bool(rank_deficient),
        "opt_success": bool(sol.success), "opt_message": str(sol.message),
        "mass_pre_norm": mass,
        "sol_left": {"k": -cL, "sigma": sL, "mu_net": mL},
        "sol_right": {"k": -cR, "sigma": sR, "mu_net": mR},
        "bounds_binding": {
            "sigma_L_lower": abs(sL - 0.01) < 1e-4,
            "sigma_L_upper": abs(sL - 0.20) < 1e-4,
            "sigma_R_lower": abs(sR - 0.01) < 1e-4,
            "sigma_R_upper": abs(sR - 0.25) < 1e-4,
            "k_L_lower": abs(-cL - (-0.05)) < 1e-4,
            "k_R_lower": abs(-cR - (-0.05)) < 1e-4,
            "k_L_upper": abs(-cL - 0.5) < 1e-4,
            "k_R_upper": abs(-cR - 0.5) < 1e-4,
        },
        "cdf_resid_left": float((1.0 - gcdf(-p10, cL, sL, mL)) - lower_pct / 100.0),
        "cdf_resid_right": float(gcdf(p90, cR, sR, mR) - upper_pct / 100.0),
    }
    if mass > 0:
        p_R = p_R / mass
    diagnostics["tail_mass_left"] = float(
        np.trapezoid(p_R[left_m], R_grid[left_m]))
    diagnostics["tail_mass_right"] = float(
        np.trapezoid(p_R[right_m], R_grid[right_m]))

    return PhysicalDensity(R_grid=np.asarray(R_grid, dtype=float), p_R=p_R,
                           method="grith_vanilla", n_returns=len(R_data),
                           bandwidth=None, diagnostics=diagnostics)


# Diagnostic: do the published optimizer bounds bind?
def sigma_binding_check(R_data: np.ndarray, R_grid: np.ndarray, n_bins_list=range(8, 14)):

    import pandas as pd
    rows = []
    for nb in n_bins_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            van = estimate_physical_density_grith_vanilla_from_returns(
                R_data, R_grid, n_bins=nb)
            enh = estimate_physical_density_almeida_from_returns(
                R_data, R_grid, n_bins=nb)
        dv, de = van.diagnostics, enh.diagnostics
        rows.append({
            "n_bins": nb,
            "van_sigma_L": dv["sol_left"]["sigma"],
            "van_sigma_R": dv["sol_right"]["sigma"],
            "van_k_L": dv["sol_left"]["k"],
            "van_k_R": dv["sol_right"]["k"],
            "van_any_bound_binds": any(dv["bounds_binding"].values()),
            "van_binding_detail": ",".join(
                k for k, v in dv["bounds_binding"].items() if v) or "none",
            "van_in_cluster_box_L": 0.05 <= dv["sol_left"]["sigma"] <= 0.11,
            "van_in_cluster_box_R": 0.05 <= dv["sol_right"]["sigma"] <= 0.11,
            "van_rank_deficient": dv["rank_deficient_polyfit"],
            "enh_scale_L": de["gev_left"]["scale"],
            "enh_scale_R": de["gev_right"]["scale"],
            "enh_xi_L": de["gev_left"]["xi_figlewski"],
            "enh_xi_R": de["gev_right"]["xi_figlewski"],
        })
    return pd.DataFrame(rows)