"""
Phase 5 explanatory extension.

Deribit BTC options are coin-margined (inverse): payoff and margin are denominated in BTC, not USD. 
The Deribit risk-neutral measure is therefore the BTC-numeraire measure Q^B, while CME (USD cash-settled) reveals the
USD-numeraire measure Q^$. The two are linked by the change-of-numeraire Radon-Nikodym derivative:

    dQ^B/dQ^$ |_T  =  S_T / F_{0,T}  =  R * e^{-r tau}       (R = S_T/S_0)

i.e. the inverse contract reweights every terminal state by its own gross return R. In log-return space x = ln R this 
is an exponential tilt with parameter exactly 1:

    q^B(x) = e^{x} q^$(x) / E^$[e^{x}]  =  e^{x - mu_c} q^$(x).

This module takes a measured CME risk-neutral density, applies the tilt to predict the Deribit density 
WITH NO FREE PARAMETERS, pushes the predicted density through the same BKM log-return integrals (V, W, X) 
and the same CL20 contribution weights used everywhere else in the pipeline, and returns
the predicted cumulant-premium wedge Pi_k^{pred,DER} - Pi_k^{CME} for comparison with the measured wedge.

If the prediction matches the measured wedge, the cross-venue premium is mechanically explained by contract design. 
If a residual survives, it isolates the part attributable to genuine frictions (funding, segmentation, margining costs).

"""

import numpy as np

# Density-space BKM log-return integrals
def bkm_integrals_from_density(R, q, kappa_bound=None):
    x = np.log(R)
    if kappa_bound is not None:
        m = np.abs(x) <= kappa_bound + 1e-12
        R, q, x = R[m], q[m], x[m]

    # Renormalize the (possibly truncated) density
    Z = np.trapezoid(q, R)
    q = q / Z

    V = np.trapezoid(q * x**2, R)
    W = np.trapezoid(q * x**3, R)
    X = np.trapezoid(q * x**4, R)
    return V, W, X, Z

def cyl_weights(theta=2.0):
    l1 = 1.0
    l2 = -(theta + 1.0) / 3.0
    l3 = (theta + 1.0) * (theta + 2.0) / 12.0
    return l1, l2, l3

def contributions_from_density(R, q, theta=2.0, kappa_bound=None):
    V, W, X, _ = bkm_integrals_from_density(R, q, kappa_bound=kappa_bound)
    l1, l2, l3 = cyl_weights(theta)
    Pi_2, Pi_3, Pi_4 = l1 * V, l2 * W, l3 * X
    return {
        "Pi_2": Pi_2, "Pi_3": Pi_3, "Pi_4": Pi_4,
        "lb_total": Pi_2 + Pi_3 + Pi_4,
        "V": V, "W": W, "X": X,
    }

# The inverse-contract tilt
def tilt_to_inverse_measure(R, q):
    num = R * q
    denom = np.trapezoid(num, R)
    return num / denom

def predict_inverse_wedge(R_cme, q_cme, theta=2.0, kappa_bound=None):
    cme = contributions_from_density(R_cme, q_cme, theta, kappa_bound)
    q_der_pred = tilt_to_inverse_measure(R_cme, q_cme)
    der_pred = contributions_from_density(R_cme, q_der_pred, theta, kappa_bound)
    wedge = {f"wedge_Pi_{k}": der_pred[f"Pi_{k}"] - cme[f"Pi_{k}"]
             for k in (2, 3, 4)}
    return {"cme": cme, "der_pred": der_pred, **wedge,
            "q_der_pred": q_der_pred}
