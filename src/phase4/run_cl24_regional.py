"""
CL24 Regional Decomposition of the CL20 Lower Bound 

Chabi-Yo and Loudis (2024) decompose risk premia across RETURN REGIONS. That is:
down (-inf, -0.1], center (-0.1, 0.1], up (0.1, inf) in net returns via truncated risk-neutral moments. 

This script computes the exact regional analog for the CL20 lower bound used in Phase 4, per venue and
cross-venue:

    Pi_k^A(t) = lambda_{k-1} * INT_A (ln R)^k q_t(R) dR ,  k = 2, 3, 4
    LB^A(t)   = sum_k Pi_k^A(t),      sum_A LB^A(t) = LB(t)   (exact)

with regions in gross returns: down R < 0.90, center 0.90 <= R <= 1.10, up R > 1.10 (the CL24 +-10% partition; 
identical to the regional MFK cuts). Entirely risk-neutral: no physical density enters, so results are
invariant to any p-hat revision.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import get_path, get_return_grid
from src.phase4.cumulant_premia import cyl_weights

THETA = 2.0
NW_LAGS = 27

def _stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

REGION_EDGES = (0.90, 1.10)
REGION_NAMES = ["down", "center", "up"]

def make_regions(R_grid):
    return [("down", R_grid < REGION_EDGES[0]),
            ("center", (R_grid >= REGION_EDGES[0]) & (R_grid <= REGION_EDGES[1])),
            ("up", R_grid > REGION_EDGES[1])]


def _edge_interp_weights(R_grid, edge):
    j = int(np.searchsorted(R_grid, edge) - 1)
    j = min(max(j, 0), len(R_grid) - 2)
    w = (edge - R_grid[j]) / (R_grid[j + 1] - R_grid[j])
    return j, float(w)

def compute_regional_premia(Q, R_grid, lam):
    from scipy.integrate import cumulative_trapezoid
    x = np.log(R_grid)
    weights = {2: lam[0], 3: lam[1], 4: lam[2]}
    (j1, w1) = _edge_interp_weights(R_grid, REGION_EDGES[0])
    (j2, w2) = _edge_interp_weights(R_grid, REGION_EDGES[1])
    out = {}
    for k in (2, 3, 4):
        integrand = Q * (x ** k)[None, :]
        ct = cumulative_trapezoid(integrand, R_grid, axis=1, initial=0.0)
        F1 = ct[:, j1] * (1 - w1) + ct[:, j1 + 1] * w1
        F2 = ct[:, j2] * (1 - w2) + ct[:, j2 + 1] * w2
        Ftot = ct[:, -1]
        out[("down", k)] = weights[k] * F1
        out[("center", k)] = weights[k] * (F2 - F1)
        out[("up", k)] = weights[k] * (Ftot - F2)
        out[("total", k)] = weights[k] * Ftot
    return out

def regional_table(per_venue, terc_map, R_grid):
    """Table 1: venue x regime x region means with shares of the bound."""
    rows = []
    region_names = REGION_NAMES + ["total"]
    for venue, (dates, prem) in per_venue.items():
        lb_total = sum(prem[("total", k)] for k in (2, 3, 4))
        labels = terc_map.reindex(dates)
        regimes = [("unconditional", np.ones(len(dates), dtype=bool))]
        regimes += [(t, (labels == t).values) for t in ["low", "mid", "high"]]
        for regime, rmask in regimes:
            if rmask.sum() == 0:
                continue
            lb_tot_mean = float(np.mean(lb_total[rmask]))
            for name in region_names:
                pk = {k: float(np.mean(prem[(name, k)][rmask]))
                      for k in (2, 3, 4)}
                lb_a = sum(pk.values())
                rows.append({
                    "venue": venue, "regime": regime, "region": name,
                    "n_days": int(rmask.sum()),
                    "Pi_2": pk[2], "Pi_3": pk[3], "Pi_4": pk[4],
                    "LB_region": lb_a,
                    "share_of_LB": (lb_a / lb_tot_mean
                                    if abs(lb_tot_mean) > 1e-12 else np.nan),
                })
    return pd.DataFrame(rows)

def wedge_table(per_venue, R_grid, nw_lags=NW_LAGS):
    import statsmodels.api as sm
    d_c, p_c = per_venue["CME"]
    d_d, p_d = per_venue["DER"]
    common = d_c.intersection(d_d)
    ic, idx = d_c.get_indexer(common), d_d.get_indexer(common)
    region_names = REGION_NAMES + ["total"]
    rows = []
    for name in region_names:
        for k in (2, 3, 4, "LB"):
            if k == "LB":
                delta = (sum(p_d[(name, kk)][idx] for kk in (2, 3, 4))
                         - sum(p_c[(name, kk)][ic] for kk in (2, 3, 4)))
                dep = f"dLB_{name}"
            else:
                delta = p_d[(name, k)][idx] - p_c[(name, k)][ic]
                dep = f"dPi_{k}_{name}"
            res = sm.OLS(delta, np.ones((len(delta), 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": nw_lags})
            rows.append({
                "dep_var": dep, "region": name, "order": str(k),
                "wedge": float(res.params[0]),
                "t_stat": float(res.tvalues[0]),
                "p_value": float(res.pvalues[0]),
                "stars": _stars(float(res.pvalues[0])),
                "n_days": int(len(delta)), "nw_lags": nw_lags,
            })
    return pd.DataFrame(rows), len(common)

def run_cl24_regional():
    R_grid = get_return_grid()
    DATA_P1 = get_path("data_phase1")
    DATA_P4 = get_path("data_phase4")
    RES_P4 = get_path("results_phase4")
    TAB_DIR = RES_P4 / "tables"
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CL24 Regional Decomposition of the CL20 Bound (theta = 2)")
    print("=" * 60)

    lam = cyl_weights(theta=THETA)
    print(f"  CL20 weights (lambda_1, lambda_2, lambda_3) = "
          f"({lam[0]:+.3f}, {lam[1]:+.3f}, {lam[2]:+.3f})")

    Z = pd.read_parquet(DATA_P1 / "Z_crypto.parquet")
    Z["date"] = pd.to_datetime(Z["date"])
    Z["tercile"] = pd.qcut(Z["Z_IVS_1"], q=3, labels=["low", "mid", "high"])
    terc_map = Z.set_index("date")["tercile"]

    per_venue = {}
    for venue in ["CME", "DER"]:
        df = pd.read_parquet(DATA_P1 / f"rnd_{venue}_densities.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["tau_days"] == 27].sort_values("date")
        dates, rnds = [], []
        for _, row in df.iterrows():
            q = np.interp(R_grid, np.array(row["returns"]), np.array(row["density"]), left=0, right=0)
            m = np.trapezoid(q, R_grid)
            if m > 0:
                dates.append(row["date"])
                rnds.append(q / m)
        dates = pd.DatetimeIndex(dates)
        prem = compute_regional_premia(np.stack(rnds), R_grid, lam)
        # additivity: regions must sum to the untruncated total, exactly
        for k in (2, 3, 4):
            err = np.max(np.abs(prem[("total", k)]
                                - sum(prem[(nm, k)] for nm in REGION_NAMES)))
            assert err < 1e-10, f"{venue} k={k} additivity violated ({err:.2e})"
        per_venue[venue] = (dates, prem)
        print(f"  [{venue}] {len(dates)} days; regional additivity exact")

    table = regional_table(per_venue, terc_map, R_grid)
    out1 = TAB_DIR / "cl24_regional.csv"
    table.to_csv(out1, index=False)
    print(f"\n  Saved: {out1}")
    print("\n  Unconditional regional composition of the bound:")
    print(table[table.regime == "unconditional"]
          [["venue", "region", "Pi_2", "Pi_3", "Pi_4", "LB_region",
            "share_of_LB"]].round(4).to_string(index=False))

    wedge, n_common = wedge_table(per_venue, R_grid)
    out2 = TAB_DIR / "cl24_regional_wedge.csv"
    wedge.to_csv(out2, index=False)
    print(f"\n  Saved: {out2}  (matched days: {n_common})")
    print("\n  Regional LB wedge (DER - CME), NW(27):")
    print(wedge[wedge.order == "LB"]
          [["region", "wedge", "t_stat", "stars", "n_days"]]
          .round(5).to_string(index=False))
    return table, wedge

if __name__ == "__main__":
    run_cl24_regional()