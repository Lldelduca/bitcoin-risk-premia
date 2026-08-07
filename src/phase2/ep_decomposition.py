"""
Beason & Schreindorfer (2022) Equity Premium Decomposition.

Computes the equity premium curve, cumulative equity premium, and unconditional pricing kernel for each venue, 
given the risk-neutral and the physical density.

All densities in this pipeline live on a GROSS forward-standardized return grid, R = S_{t+tau} / F_t(tau), with R in [0.40, 2.00].

Beason & Schreindorfer state their decomposition in NET return space over [-1, inf):

    E[R_net] - Rf = integral of R_net * (f(R_net) - f*(R_net)) d R_net

Under the change of variable R_net = R_gross - 1 (unit Jacobian), the image of their integrand on our gross grid is:

    ep(R) = (R - 1) * (p(R) - q(R))

Multiplying by R rather than (R - 1) treats the gross return as if it were the net return. Because

    integral of (R - c) * (p - q) dR

gives the SAME total for any constant c (both densities integrate to one)

"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import NamedTuple

REGION_EDGES = (0.90, 1.10)

class EPDecomposition(NamedTuple):
    R_grid: np.ndarray       # gross return grid
    ep: np.ndarray           # EP curve ep^j(R)
    cep: np.ndarray          # cumulative EP CEP^j(R)
    kernel: np.ndarray       # pricing kernel m^j(R) = q/p
    total_ep: float          # integral of ep = E^P[R] - E^Q[R]
    p_R: np.ndarray          # physical density on grid
    q_R: np.ndarray          # risk-neutral density on grid
    venue: str               # venue identifier

def compute_ep_decomposition(R_grid: np.ndarray, q_R: np.ndarray, p_R: np.ndarray, venue: str = "unknown") -> EPDecomposition:

    # EP curve: (R - 1) * (p(R) - q(R)) Beason-Schreindorfer net-return integrand expressed on the gross grid
    ep = (p_R - q_R) * (R_grid - 1.0)

    # Cumulative EP via trapezoidal quadrature
    cep = np.concatenate([[0.0], cumulative_trapezoid(ep, R_grid)])

    # Total EP = integral over full domain
    total_ep = cep[-1]

    # Unconditional pricing kernel m(R) = q(R) / p(R)
    p_safe = np.maximum(p_R, 1e-20)
    kernel = q_R / p_safe
    kernel[p_R < 1e-15] = np.nan

    return EPDecomposition(R_grid=R_grid, ep=ep, cep=cep, kernel=kernel, total_ep=total_ep, p_R=p_R, q_R=q_R, venue=venue)

def _edge_interp_weights(R_grid: np.ndarray, edge: float):
    j = int(np.searchsorted(R_grid, edge) - 1)
    j = int(np.clip(j, 0, len(R_grid) - 2))
    w = (edge - R_grid[j]) / (R_grid[j + 1] - R_grid[j])
    return j, float(np.clip(w, 0.0, 1.0))

def _cum_at(ct: np.ndarray, R_grid: np.ndarray, edge: float) -> float:
    j, w = _edge_interp_weights(R_grid, edge)
    return float((1.0 - w) * ct[j] + w * ct[j + 1])

def compute_ep_contributions(decomp: EPDecomposition, boundaries: tuple = REGION_EDGES, return_residual: bool = False):
    R = decomp.R_grid
    ep = decomp.ep
    total = decomp.total_ep

    lower, upper = boundaries

    ct = cumulative_trapezoid(ep, R, initial=0.0)
    c_lo = _cum_at(ct, R, lower)
    c_hi = _cum_at(ct, R, upper)

    values = {
        "downside": c_lo - float(ct[0]),
        "mid": c_hi - c_lo,
        "upside": float(ct[-1]) - c_hi,
    }
    regions = {
        "downside": (float(R[0]), float(lower)),
        "mid": (float(lower), float(upper)),
        "upside": (float(upper), float(R[-1])),
    }

    contributions = {}
    for name, contrib in values.items():
        contributions[name] = {
            "contribution": float(contrib),
            "share": float(contrib / total) if abs(total) > 1e-10 else np.nan,
            "region": regions[name],
        }

    if return_residual:
        return contributions, float(total - sum(values.values()))
    return contributions