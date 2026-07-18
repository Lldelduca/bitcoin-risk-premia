"""
KDE-Tilted Kernel Robustness

Re-estimates the crypto specification on both venues with the cross-family KDE+GPD density as the tilting measure 
(point estimates only, warm-started from the saved enhanced-tilt theta) and tabulates the tercile-mean coefficients 
and curvature at the money, 2c + 6d at R = 1, against the headline results. Stability across tilting measures retires
the single-density-dependence question; instability would itself be a reportable sensitivity.

"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import get_path, get_return_grid
from src.phase3.conditional_kernel import (
    estimate_conditional_kernel, evaluate_kernel_at_terciles)
from src.phase3.run_phase3 import (
    load_daily_rnds_from_parquet, load_conditioning_spec,
    load_volatility_tercile_labels, align_rnds_and_Z)

R_GRID = get_return_grid()

def run_kde_tilt_robustness():
    TAB = get_path("results_phase3") / "tables"
    TAB.mkdir(parents=True, exist_ok=True)

    DATA_P2 = get_path("data_phase2")
    DATA_P3 = get_path("data_phase3")

    print("\n" + "=" * 60)
    print("  Phase 3 robustness: KDE+GPD tilting density (crypto spec)")
    print("=" * 60)

    dens = np.load(DATA_P2 / "phase2_densities.npz")
    p_by_name = {"enhanced": dens["p_almeida"], "kde": dens["p_kde"]}

    z_dates, Z_matrix, z_cols = load_conditioning_spec("crypto")
    tercile_df = load_volatility_tercile_labels()

    cells = {}
    for venue in ["CME", "DER"]:
        rnd_dates, rnds = load_daily_rnds_from_parquet(venue, tau_days=27)
        dates, arnds, aZ, labels = align_rnds_and_Z(
            rnd_dates, rnds, z_dates, Z_matrix, tercile_df=tercile_df)

        theta0 = None
        warm = DATA_P3 / f"phase3_{venue}_crypto.npz"
        if warm.exists():
            theta0 = np.load(warm)["theta"]

        for pname, p_phys in p_by_name.items():
            print(f"\n  [{venue} | tilt = {pname}]")
            res = estimate_conditional_kernel(
                R_GRID, arnds, p_phys, aZ, venue=venue,
                spec_name=f"crypto_{pname}",
                theta0=theta0 if pname == "kde" else theta0,
                verbose=True)
            terc = evaluate_kernel_at_terciles(
                res, R_GRID, aZ, tercile_labels=labels, p_phys=p_phys)
            for tn, t in terc.items():
                cells[(venue, tn, pname)] = {
                    "b": t["b"], "c": t["c"], "d": t["d"],
                    "curv": 2 * t["c"] + 6 * t["d"],
                    "n_days": t["n_days"],
                    "converged": res.converged, "kl_mean": res.kl_mean,
                }

    rows = []
    for venue in ["CME", "DER"]:
        for tn in ["low", "mid", "high"]:
            e = cells.get((venue, tn, "enhanced"))
            k = cells.get((venue, tn, "kde"))
            if e is None or k is None:
                continue
            rows.append({
                "venue": venue, "tercile": tn, "n_days": e["n_days"],
                "b_enh": e["b"], "c_enh": e["c"], "d_enh": e["d"],
                "curv_enh": e["curv"],
                "b_kde": k["b"], "c_kde": k["c"], "d_kde": k["d"],
                "curv_kde": k["curv"],
                "curv_delta": k["curv"] - e["curv"],
                "curv_sign_agrees": bool(np.sign(e["curv"])
                                         == np.sign(k["curv"])),
                "converged_enh": e["converged"],
                "converged_kde": k["converged"],
                "kl_mean_enh": e["kl_mean"], "kl_mean_kde": k["kl_mean"],
            })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TAB / "kde_tilt_robustness.csv", index=False)
    print("\n  Curvature at the money by tilting density:")
    print(tbl[["venue", "tercile", "curv_enh", "curv_kde", "curv_delta",
               "curv_sign_agrees"]].round(3).to_string(index=False))
    print(f"\n  Saved: {TAB / 'kde_tilt_robustness.csv'}")
    return tbl

if __name__ == "__main__":
    run_kde_tilt_robustness()