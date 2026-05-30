"""
Chabi-Yo & Loudis (2020, 2024) Cumulant Premium Decomposition.

The k-th cumulant premium contribution is the risk-neutral lower-bound contribution from the Chabi-Yo & Loudis (2020) bound:

    Pi_{k,t}^j = lambda_{k-1} * mu_{k,t}^{Q,j},   k in {2,3,4}

where mu_2 = V (variance contract), mu_3 = W (cubic contract), mu_4 = X (quartic contract) are the raw BKM RN Moments

The sum gives the CL20 lower bound on the conditional expected excess return:

    E_t[R^e | Z_t] >= Pi_{2,t} + Pi_{3,t} + Pi_{4,t}.

Following Martin (2017), the variance term is anchored at lambda_1 = 1. The higher-order weights are pinned
by a single preference parameter theta (relative prudence / higher-order risk aversion) via the CL20 power-utility expansion:

    lambda_1 = 1
    lambda_2 = -(theta + 1) / 3             (skewness weight, < 0)
    lambda_3 =  (theta + 1)(theta + 2) / 12 (kurtosis weight, > 0)

theta = 2 (log/power-utility baseline used by CL20) gives
    lambda_1 = 1, lambda_2 = -1, lambda_3 = 1/2.

We also compute a secondary diagnostic: The classic Bollerslev-type variance risk premium

    VRP_t^j = V_t^{Q,j} - V_t^{P}

is reported as a single well-understood cross-check on the variance contribution to the risk premium
where V_t^P is the physical variance scaled to the tau-day horizon.

"""

import numpy as np
import pandas as pd
from typing import NamedTuple

class CumulantPremia(NamedTuple):
    date: pd.Timestamp
    venue: str
    V: float
    W: float
    X: float
    var_Q: float
    skew_Q: float
    kurt_Q: float
    Pi_2: float     # lambda_1 * V   (variance contribution)
    Pi_3: float     # lambda_2 * W   (skewness contribution)
    Pi_4: float     # lambda_3 * X   (kurtosis contribution)
    lb_total: float # Pi_2 + Pi_3 + Pi_4  (CL20 lower bound)
    var_P: float    # physical variance, scaled to tau-day horizon
    vrp: float      # V_t^Q - V_t^P
    theta: float    # preference parameter used

def cyl_weights(theta=2.0):
    lambda_1 = 1.0
    lambda_2 = -(theta + 1.0) / 3.0
    lambda_3 = (theta + 1.0) * (theta + 2.0) / 12.0
    return lambda_1, lambda_2, lambda_3

def compute_physical_variance(spot_returns, tau_days, window=252):
    var_daily = spot_returns.rolling(window, min_periods=60).var()
    var_P = var_daily * tau_days
    var_P.name = "var_P"
    return var_P

def compute_cumulant_premia(date, venue, V, W, X, var_Q, skew_Q, kurt_Q, var_P=np.nan, theta=2.0):
    l1, l2, l3 = cyl_weights(theta)

    Pi_2 = l1 * V
    Pi_3 = l2 * W
    Pi_4 = l3 * X
    lb_total = Pi_2 + Pi_3 + Pi_4

    vrp = (var_Q - var_P) if np.isfinite(var_P) else np.nan

    return CumulantPremia(
        date=date, venue=venue,
        V=V, W=W, X=X,
        var_Q=var_Q, skew_Q=skew_Q, kurt_Q=kurt_Q,
        Pi_2=Pi_2, Pi_3=Pi_3, Pi_4=Pi_4, lb_total=lb_total,
        var_P=var_P, vrp=vrp, theta=theta,
    )

def compute_cyl_decomposition_table(premia_df, tercile_col="tercile"):
    rows = []

    def _agg(v, venue, regime):
        n = len(v)
        p2, p3, p4 = v["Pi_2"].mean(), v["Pi_3"].mean(), v["Pi_4"].mean()
        tot = p2 + p3 + p4
        return {
            "venue": venue, "regime": regime, "n_days": n,
            "Pi_2": p2, "Pi_3": p3, "Pi_4": p4, "lb_total": tot,
            "share_var": p2 / tot if tot != 0 else np.nan,
            "share_skew": p3 / tot if tot != 0 else np.nan,
            "share_kurt": p4 / tot if tot != 0 else np.nan,
            "mean_vrp": v["vrp"].mean(),
        }

    for venue in premia_df["venue"].unique():
        v = premia_df[premia_df["venue"] == venue]
        rows.append(_agg(v, venue, "unconditional"))

    if tercile_col in premia_df.columns:
        for venue in premia_df["venue"].unique():
            for tercile in ["low", "mid", "high"]:
                mask = (premia_df["venue"] == venue) & (premia_df[tercile_col] == tercile)
                v = premia_df[mask]
                if len(v) > 0:
                    rows.append(_agg(v, venue, tercile))

    return pd.DataFrame(rows)

def robustness_over_theta(premia_inputs_df, theta_grid=(1.0, 2.0, 3.0)):
    rows = []
    for theta in theta_grid:
        l1, l2, l3 = cyl_weights(theta)
        df = premia_inputs_df.copy()
        df["Pi_2"] = l1 * df["V"]
        df["Pi_3"] = l2 * df["W"]
        df["Pi_4"] = l3 * df["X"]
        df["lb_total"] = df["Pi_2"] + df["Pi_3"] + df["Pi_4"]
        for venue in df["venue"].unique():
            v = df[df["venue"] == venue]
            tot = v["lb_total"].mean()
            rows.append({
                "venue": venue, "theta": theta,
                "lambda_1": l1, "lambda_2": l2, "lambda_3": l3,
                "mean_Pi_2": v["Pi_2"].mean(),
                "mean_Pi_3": v["Pi_3"].mean(),
                "mean_Pi_4": v["Pi_4"].mean(),
                "mean_lb_total": tot,
                "lb_annualized_pct": 100.0 * tot * (365.0 / 27.0),
            })
    return pd.DataFrame(rows)
