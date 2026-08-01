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