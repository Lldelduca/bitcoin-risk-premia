"""
Grid-Sensitivity Diagnostic: integration-window dependence of the EP total.

The raw-moment anchor is 4.4% but the density-based totals are 3.6-4.2%. Why? This is mechanical:
The analysis grid [0.40, 2.00] clips probability mass (2020-21 upside returns exceed R = 2) and renormalization 
shifts the grid-restricted mean. 

This script quantifies that channel by recomputing the total EP for all three physical-density estimators on three 
integration windows:

    headline : [0.40, 2.00]   (the pipeline analysis grid)
    wide-1   : [0.35, 2.20]
    wide-2   : [0.30, 2.60]

The number of raw returns outside it, and the average-RND probability mass clipped per venue (the clipped-mass
diagnostic). This summarizes the wide-grid reconciliation check and the q-clipped-mass diagnostic.

"""

import warnings
import numpy as np
import pandas as pd


GRIDS = [
    ("headline", 0.40, 2.00, 1000),
    ("wide_1",   0.35, 2.20, 1200),
    ("wide_2",   0.30, 2.60, 1500),
]

MASTER_GRID = np.linspace(0.25, 4.50, 2500)

# Pure computational core (unit-testable without the repo)
def build_qbar_master(native_rows, master_grid=MASTER_GRID):
    """Average RND on the master grid from native (R, q) per-day arrays."""
    acc = np.zeros_like(master_grid)
    n = 0
    for R, q in native_rows:
        qi = np.interp(master_grid, R, q, left=0.0, right=0.0)
        m = np.trapezoid(qi, master_grid)
        if m > 0:
            acc += qi / m
            n += 1
    qbar = acc / max(n, 1)
    mass = np.trapezoid(qbar, master_grid)
    return qbar / mass if mass > 0 else qbar, n


def compute_grid_sensitivity(R_data, qbar_by_venue, est_fns, grids=GRIDS, master_grid=MASTER_GRID):

    raw_anchor = float(np.mean(R_data) - 1.0)
    rows = []
    for label, lo, hi, pts in grids:
        G = np.linspace(lo, hi, pts)
        n_below = int((R_data < lo).sum())
        n_above = int((R_data > hi).sum())

        # physical densities refit on this window
        p_by_est = {}
        for name, fn in est_fns.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_by_est[name] = fn(R_data, G).p_R

        for venue, qbar in qbar_by_venue.items():
            # clipped q mass: average-RND probability outside the window,
            # measured on the master grid BEFORE renormalization
            inside = (master_grid >= lo) & (master_grid <= hi)
            q_inside_mass = float(np.trapezoid(qbar[inside],
                                               master_grid[inside]))
            q_clipped = 1.0 - q_inside_mass

            # window-restricted, renormalized average RND
            q_w = np.interp(G, master_grid, qbar)
            mw = np.trapezoid(q_w, G)
            q_w = q_w / mw if mw > 0 else q_w

            for name, p_R in p_by_est.items():
                total = float(np.trapezoid((p_R - q_w) * G, G))
                rows.append({
                    "grid": label, "lo": lo, "hi": hi,
                    "estimator": name, "venue": venue,
                    "total_ep": total,
                    "raw_anchor": raw_anchor,
                    "gap_to_anchor": raw_anchor - total,
                    "q_clipped_mass": q_clipped,
                    "n_returns_below": n_below,
                    "n_returns_above": n_above,
                })
    df = pd.DataFrame(rows)

    # gap closed relative to the headline window, per estimator x venue
    head = df[df["grid"] == "headline"].set_index(["estimator", "venue"])
    def _closed(r):
        h = head.loc[(r["estimator"], r["venue"])]
        denom = h["gap_to_anchor"]
        if abs(denom) < 1e-12:
            return np.nan
        return 1.0 - r["gap_to_anchor"] / denom
    df["gap_closed_vs_headline"] = df.apply(_closed, axis=1)
    return df


# Pipeline entry point
def run_grid_sensitivity():
    from src.config import get_path
    from src.phase2.physical_density import (
        estimate_physical_density_almeida_from_returns,
        estimate_physical_density_grith_vanilla_from_returns,
        estimate_physical_density_kde_from_returns,
        compute_overlapping_returns)
    from src.phase2.run_phase2 import load_spot_prices, TAB_DIR

    print("\n" + "=" * 60)
    print("  Grid Sensitivity: integration-window dependence of total EP")
    print("=" * 60)

    spot = load_spot_prices()
    R_data = compute_overlapping_returns(spot, horizon=27)
    print(f"  {len(R_data)} overlapping returns; "
          f"support [{R_data.min():.3f}, {R_data.max():.3f}]")

    DATA_P1 = get_path("data_phase1")
    qbar_by_venue = {}
    for venue in ["CME", "DER"]:
        df = pd.read_parquet(DATA_P1 / f"rnd_{venue}_densities.parquet")
        df = df[df["tau_days"] == 27]
        native_rows = [(np.array(r["returns"]), np.array(r["density"]))
                       for _, r in df.iterrows()]
        qbar, n = build_qbar_master(native_rows)
        qbar_by_venue[venue] = qbar
        print(f"  [{venue}] average RND rebuilt on master grid "
              f"[{MASTER_GRID[0]:.2f}, {MASTER_GRID[-1]:.2f}] from {n} days")

    est_fns = {
        "almeida": estimate_physical_density_almeida_from_returns,
        "vanilla": estimate_physical_density_grith_vanilla_from_returns,
        "kde": estimate_physical_density_kde_from_returns,
    }

    df = compute_grid_sensitivity(R_data, qbar_by_venue, est_fns)
    out = TAB_DIR / "grid_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")

    # console summary + the appendix reconciliation sentence
    piv = df.pivot_table(index=["estimator", "venue"], columns="grid",
                         values="total_ep")
    print("\n  Total EP by integration window:")
    print(piv[["headline", "wide_1", "wide_2"]].round(4).to_string())

    print(f"\n  Raw-moment anchor (full support): "
          f"{df['raw_anchor'].iloc[0]:+.4f}")
    for venue in qbar_by_venue:
        r_h = df[(df.grid == "headline") & (df.estimator == "almeida")
                 & (df.venue == venue)].iloc[0]
        r_w = df[(df.grid == "wide_2") & (df.estimator == "almeida")
                 & (df.venue == venue)].iloc[0]
        print(f"  [{venue}] enhanced: {r_h['total_ep']:+.4f} -> "
              f"{r_w['total_ep']:+.4f} on [0.30, 2.60]; "
              f"gap to anchor closed: {r_w['gap_closed_vs_headline']:+.1%}; "
              f"q mass clipped on headline grid: "
              f"{r_h['q_clipped_mass']:.4f}")
    n_above = df[df.grid == "headline"]["n_returns_above"].iloc[0]
    print(f"  Returns above R = 2.00 (headline window): {n_above}")
    return df


if __name__ == "__main__":
    run_grid_sensitivity()