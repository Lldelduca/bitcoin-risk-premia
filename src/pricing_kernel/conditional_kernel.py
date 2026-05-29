"""
Phase 3: Conditional Pricing Kernel (Schreindorfer & Sichert 2025).

Estimates the conditional pricing kernel as a cubic polynomial in returns with state-dependent coefficients:

    ln m_t^j(R) = a_t + b_t R + c_t R^2 + d_t R^3

where each coefficient is linear in the conditioning vector Z_t:

    a_t = a_0 + a_1' Z_t,   b_t = b_0 + b_1' Z_t,   etc.

The estimation minimizes the KL divergence between the model-implied RND and the SSVI-extracted RND across all days

"""

import numpy as np
from scipy.optimize import minimize
from typing import NamedTuple
 
class ConditionalKernelResult(NamedTuple):
    """Container for conditional kernel estimation results."""
    theta: np.ndarray
    n_params: int
    n_days: int
    n_Z: int
    venue: str
    spec_name: str
    kl_total: float
    kl_mean: float
    converged: bool
    hessian_inv: np.ndarray | None

def _unpack_theta(theta, n_Z):
    block = 1 + n_Z
    a_const, a_Z = theta[0], theta[1:block]
    b_const, b_Z = theta[block], theta[block+1:2*block]
    c_const, c_Z = theta[2*block], theta[2*block+1:3*block]
    d_const, d_Z = theta[3*block], theta[3*block+1:4*block]
    return a_const, a_Z, b_const, b_Z, c_const, c_Z, d_const, d_Z
 
def _compute_coefficients(theta, Z_t, n_Z):
    a0, a1, b0, b1, c0, c1, d0, d1 = _unpack_theta(theta, n_Z)
    a_t = a0 + a1 @ Z_t
    b_t = b0 + b1 @ Z_t
    c_t = c0 + c1 @ Z_t
    d_t = d0 + d1 @ Z_t
    return a_t, b_t, c_t, d_t
 
def _log_kernel(R_grid, a_t, b_t, c_t, d_t):
    return a_t + b_t * R_grid + c_t * R_grid**2 + d_t * R_grid**3
 
def _kl_single_day(theta, R_grid, q_obs, p_phys, Z_t, n_Z):
    a_t, b_t, c_t, d_t = _compute_coefficients(theta, Z_t, n_Z)
    log_m = _log_kernel(R_grid, a_t, b_t, c_t, d_t)
 
    # Numerical stability: shift before exp
    log_m_max = np.max(log_m)
    log_m_shifted = log_m - log_m_max
 
    # Normalizing constant (in log space)
    unnorm = p_phys * np.exp(log_m_shifted)
    Z_val = np.trapezoid(unnorm, R_grid)
    if Z_val <= 0:
        return 1e10
    log_Z = np.log(Z_val) + log_m_max
 
    # KL objective
    ln_p = np.log(np.maximum(p_phys, 1e-300))
    integrand = q_obs * (ln_p + log_m)
    kl = -np.trapezoid(integrand, R_grid) + log_Z
 
    return kl if np.isfinite(kl) else 1e10
 
def _objective(theta, R_grid, q_obs_list, p_phys, Z_matrix, n_Z):
    total = 0.0
    for t in range(len(q_obs_list)):
        total += _kl_single_day(theta, R_grid, q_obs_list[t],
                                p_phys, Z_matrix[t], n_Z)
    return total
 
def estimate_conditional_kernel(
    R_grid: np.ndarray, q_obs_list: list, p_phys: np.ndarray, Z_matrix: np.ndarray, venue: str = "unknown", 
    spec_name: str = "unknown", max_iter: int = 20000, method: str = "L-BFGS-B", 
    theta0: np.ndarray | None = None) -> ConditionalKernelResult:

    T, n_Z = Z_matrix.shape
    n_params = 4 * (1 + n_Z)
    assert len(q_obs_list) == T
 
    print(f"    Estimating: venue={venue}, spec={spec_name}, "
          f"T={T}, n_Z={n_Z}, n_params={n_params}")
 
    # Initial guess: use warm-start if provided, otherwise default
    if theta0 is not None:
        if len(theta0) != n_params:
            print(f"    [WARN] theta0 length {len(theta0)} != n_params {n_params}, "
                  f"falling back to default init")
            theta0 = None
 
    if theta0 is None:
        theta0 = np.zeros(n_params)
        block = 1 + n_Z
        theta0[block] = -2.0  # b_0: negative slope (risk aversion)
        print(f"    Using default initialization")
    else:
        print(f"    Using warm-start initialization")
 
    result = minimize(_objective, theta0, args=(R_grid, q_obs_list, p_phys, Z_matrix, n_Z), method=method,
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-10})
 
    hess_inv = None
    if hasattr(result, "hess_inv"):
        if hasattr(result.hess_inv, "todense"):
            hess_inv = np.array(result.hess_inv.todense())
        else:
            hess_inv = np.array(result.hess_inv)
 
    kl_total = result.fun
    kl_mean = kl_total / T
 
    print(f"    Converged: {result.success}, "
          f"KL total={kl_total:.4f}, KL mean={kl_mean:.6f}")
    print(f"    Final |grad|_inf = {np.max(np.abs(result.jac)):.2e}")
 
    return ConditionalKernelResult(
        theta=result.x, n_params=n_params, n_days=T, n_Z=n_Z, venue=venue, spec_name=spec_name, kl_total=kl_total,
        kl_mean=kl_mean, converged=result.success, hessian_inv=hess_inv)
 
def evaluate_kernel(result, R_grid, Z_t):
    a_t, b_t, c_t, d_t = _compute_coefficients(result.theta, Z_t, result.n_Z)
    log_m = _log_kernel(R_grid, a_t, b_t, c_t, d_t)
    log_m -= np.max(log_m)
    return np.exp(log_m)
 
def evaluate_kernel_at_terciles(result, R_grid, Z_matrix):
    vol_proxy = Z_matrix[:, 0]
    tercile_edges = np.percentile(vol_proxy, [33.33, 66.67])
    tercile_labels = np.digitize(vol_proxy, bins=tercile_edges)
 
    out = {}
    for label, name in zip([0, 1, 2], ["low", "mid", "high"]):
        mask = tercile_labels == label
        if mask.sum() == 0:
            continue
        Z_mean = Z_matrix[mask].mean(axis=0)
        a_t, b_t, c_t, d_t = _compute_coefficients(
            result.theta, Z_mean, result.n_Z
        )
        log_m = _log_kernel(R_grid, a_t, b_t, c_t, d_t)
        log_m -= np.max(log_m)
        out[name] = {
            "kernel": np.exp(log_m),
            "Z_mean": Z_mean,
            "a": a_t, "b": b_t, "c": c_t, "d": d_t,
            "n_days": int(mask.sum()),
        }
    return out
 
def get_coefficient_timeseries(result, Z_matrix):
    T = Z_matrix.shape[0]
    coeffs = {"a": np.zeros(T), "b": np.zeros(T),
              "c": np.zeros(T), "d": np.zeros(T)}
    for t in range(T):
        a_t, b_t, c_t, d_t = _compute_coefficients(
            result.theta, Z_matrix[t], result.n_Z
        )
        coeffs["a"][t] = a_t
        coeffs["b"][t] = b_t
        coeffs["c"][t] = c_t
        coeffs["d"][t] = d_t
    return coeffs
 