"""
Wedge Term Structure

Phases 4-5 measure the cross-venue cumulant wedge at tau = 27 only. This script recomputes the daily premia 
Pi_k = lambda_{k-1} INT (ln R)^k q_t dR (theta = 2 weights, identical construction per maturity) from the SAVED
daily RNDs at tau in {14, 27, 60}, and estimates the matched-day wedge DER - CME per (tau, k) with NW(27) errors. 

Note: Pi levels here use the RND-integral route (vs Phase 4's BKM surface route with kappa bounds), so levels can differ 
slightly from the headline at tau = 27; the tau = 27 WEDGE should nonetheless reproduce the headline sign and significance
printed as a consistency check.

"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import get_path, get_return_grid
from src.phase4.cumulant_premia import cyl_weights

R_GRID = get_return_grid()
TAUS = [14, 27, 60]
NW_LAGS = 27


def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _daily_premia(venue, tau_days, lam):
    df = pd.read_parquet(get_path("data_phase1") / f"rnd_{venue}_densities.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["tau_days"] == tau_days].sort_values("date")
    x = np.log(R_GRID)
    dates, rows = [], []
    for _, r in df.iterrows():
        q = np.interp(R_GRID, np.array(r["returns"]), np.array(r["density"]),
                      left=0, right=0)
        m = np.trapezoid(q, R_GRID)
        if m <= 0:
            continue
        q /= m
        dates.append(r["date"])
        rows.append([lam[k - 2] * np.trapezoid(q * x ** k, R_GRID)
                     for k in (2, 3, 4)])
    return pd.DataFrame(rows, columns=["Pi_2", "Pi_3", "Pi_4"],
                        index=pd.DatetimeIndex(dates))


def run_wedge_term_structure():
    # Corrected: Standardized hardcoded output layout strings
    RES_P5 = get_path("results_phase5")
    TAB = RES_P5 / "tables"
    FIG = RES_P5 / "figures"
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    lam = cyl_weights(theta=2.0)

    print("\n" + "=" * 60)
    print("  Wedge Term Structure (tau in {14, 27, 60})")
    print("=" * 60)

    rows = []
    for tau in TAUS:
        pc = _daily_premia("CME", tau, lam)
        pdd = _daily_premia("DER", tau, lam)
        common = pc.index.intersection(pdd.index)
        if len(common) < 50:
            print(f"  [tau={tau}] only {len(common)} matched days — skipped")
            continue
        for k in (2, 3, 4):
            delta = (pdd.loc[common, f"Pi_{k}"]
                     - pc.loc[common, f"Pi_{k}"]).values
            res = sm.OLS(delta, np.ones((len(delta), 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
            rows.append({"tau_days": tau, "order": k,
                         "wedge": float(res.params[0]),
                         "se": float(res.bse[0]),
                         "t_stat": float(res.tvalues[0]),
                         "p_value": float(res.pvalues[0]),
                         "stars": _stars(float(res.pvalues[0])),
                         "n_days": len(delta)})
        print(f"  [tau={tau}] {len(common)} matched days")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TAB / "wedge_term_structure.csv", index=False)
    print("\n" + tbl.round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, c in [(2, "C0"), (3, "C1"), (4, "C2")]:
        sub = tbl[tbl["order"] == k].sort_values("tau_days")
        ax.errorbar(sub["tau_days"], sub["wedge"], yerr=1.96 * sub["se"],
                    fmt="o-", color=c, capsize=3, label=rf"$\Delta\Pi_{k}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel(r"Maturity $\tau$ (days)")
    ax.set_ylabel("Cross-venue wedge (DER $-$ CME)")
    ax.set_title("Term Structure of the Cumulant Wedge (95% NW CIs)")
    ax.set_xticks(TAUS)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_wedge_term_structure.png", dpi=150)
    plt.close()
    print(f"  Saved: {TAB / 'wedge_term_structure.csv'}")
    return tbl

if __name__ == "__main__":
    run_wedge_term_structure()