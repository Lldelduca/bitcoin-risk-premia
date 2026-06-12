"""
Beason & Schreindorfer (2022) Equity Premium Decomposition.

Computes the equity premium curve, cumulative equity premium, and unconditional pricing kernel for each venue, 
given the risk-neutral density q^j(R) and the physical density p(R)

"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import NamedTuple

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

    # EP curve: (p(R) - q(R)) * R
    ep = (p_R - q_R) * R_grid

    # Cumulative EP via trapezoidal quadrature
    cep = np.concatenate([[0.0], cumulative_trapezoid(ep, R_grid)])

    # Total EP = integral over full domain
    total_ep = cep[-1]

    # Unconditional pricing kernel m(R) = q(R) / p(R)
    p_safe = np.maximum(p_R, 1e-20)
    kernel = q_R / p_safe
    kernel[p_R < 1e-15] = np.nan

    return EPDecomposition(R_grid=R_grid, ep=ep, cep=cep, kernel=kernel, total_ep=total_ep, p_R=p_R, q_R=q_R, venue=venue)

def compute_ep_contributions(decomp: EPDecomposition, boundaries: tuple = (0.90, 1.10)):
    R = decomp.R_grid
    ep = decomp.ep
    total = decomp.total_ep

    lower, upper = boundaries
    down_mask = R < lower
    mid_mask = (R >= lower) & (R <= upper)
    up_mask = R > upper

    contributions = {}
    for name, mask in [("downside", down_mask), ("mid", mid_mask), ("upside", up_mask)]:
        if mask.sum() > 1:
            contrib = np.trapezoid(ep[mask], R[mask])
        else:
            contrib = 0.0
        contributions[name] = {
            "contribution": contrib,
            "share": contrib / total if abs(total) > 1e-10 else np.nan,
            "region": (R[mask].min() if mask.any() else np.nan,
                       R[mask].max() if mask.any() else np.nan),
        }

    return contributions