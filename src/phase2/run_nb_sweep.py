"""
NB-Sweep Diagnostic: bin-count stability of the physical density estimators.

Appendix companion to the AGMW Figure A4 exercise. For each bin count
NB in {8, ..., 13}, both the ENHANCED estimator and the AGMW VANILLA
benchmark are refitted, the EP curve is recomputed per venue, and two
stability metrics are reported per (estimator, venue):

  density_range_pct : max_R [ range_NB p_hat(R) ] / max_R p_hat(R; NB=12)
  ep_range_pct      : max_R [ range_NB ep(R) ]   / max_R |ep(R; NB=12)|

Lower is more stable. The published-bound binding diagnostic
(sigma_binding_check) is run over the same NB grid and saved alongside.

Outputs (results/phase2/):
  tables/nb_sweep_summary.csv
  tables/nb_sweep_totals.csv          (total EP per NB x estimator x venue)
  tables/sigma_binding_check.csv
  figures/fig_nb_sweep_appendix.png   (EP-curve spaghetti, 2x2)

Run AFTER run_phase2 (reuses its loaders); does not modify pipeline state.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import get_return_grid
from src.phase2.physical_density import (
    estimate_physical_density_almeida_from_returns,
    estimate_physical_density_grith_vanilla_from_returns,
    compute_overlapping_returns, sigma_binding_check)
from src.phase2.ep_decomposition import compute_ep_decomposition
from src.phase2.run_phase2 import (
    load_spot_prices, load_daily_rnds_from_parquet, compute_average_rnd,
    FIG_DIR, TAB_DIR, ESTIMATOR_LABELS)

R_GRID = get_return_grid()
NB_LIST = list(range(8, 14))
NB_REF = 12

SWEEP_FNS = {
    "almeida": estimate_physical_density_almeida_from_returns,
    "vanilla": estimate_physical_density_grith_vanilla_from_returns,
}


def run_nb_sweep():
    print("\n" + "=" * 60)
    print("  NB Sweep: bin-count stability (enhanced vs AGMW vanilla)")
    print("=" * 60)

    spot = load_spot_prices()
    R_data = compute_overlapping_returns(spot, horizon=27)
    print(f"  {len(R_data)} overlapping 27d returns; NB grid = {NB_LIST}")

    _, cme_rnds = load_daily_rnds_from_parquet("CME", tau_days=27)
    _, der_rnds = load_daily_rnds_from_parquet("DER", tau_days=27)
    q_by_venue = {"CME": compute_average_rnd(cme_rnds),
                  "DER": compute_average_rnd(der_rnds)}

    # ---- refit both estimators at every NB ----
    dens = {}                                # (est, nb) -> p_R
    ep = {}                                  # (est, venue, nb) -> ep curve
    total_rows = []
    for est, fn in SWEEP_FNS.items():
        for nb in NB_LIST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = fn(R_data, R_GRID, n_bins=nb)
            dens[(est, nb)] = p.p_R
            for venue, q_R in q_by_venue.items():
                d = compute_ep_decomposition(R_GRID, q_R, p.p_R, venue=venue)
                ep[(est, venue, nb)] = d.ep
                total_rows.append({"estimator": est, "venue": venue,
                                   "n_bins": nb, "total_ep": d.total_ep})
            print(f"    [{est:>7s}] NB={nb} done")

    totals = pd.DataFrame(total_rows)
    totals.to_csv(TAB_DIR / "nb_sweep_totals.csv", index=False)

    # ---- stability metrics ----
    rows = []
    for est in SWEEP_FNS:
        d_stack = np.stack([dens[(est, nb)] for nb in NB_LIST])
        d_range = d_stack.max(axis=0) - d_stack.min(axis=0)
        d_ref_peak = float(np.max(dens[(est, NB_REF)]))
        for venue in q_by_venue:
            e_stack = np.stack([ep[(est, venue, nb)] for nb in NB_LIST])
            e_range = e_stack.max(axis=0) - e_stack.min(axis=0)
            e_ref_peak = float(np.max(np.abs(ep[(est, venue, NB_REF)])))
            t = totals[(totals["estimator"] == est) & (totals["venue"] == venue)]
            rows.append({
                "estimator": est, "venue": venue,
                "density_range_pct": float(d_range.max()) / d_ref_peak,
                "ep_range_pct": float(e_range.max()) / e_ref_peak,
                "total_ep_min": t["total_ep"].min(),
                "total_ep_max": t["total_ep"].max(),
                "total_ep_spread": t["total_ep"].max() - t["total_ep"].min(),
                "nb_ref": NB_REF, "nb_grid": str(NB_LIST),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB_DIR / "nb_sweep_summary.csv", index=False)
    print("\n  Stability (range as fraction of NB=12 peak; lower = stabler):")
    print(summary[["estimator", "venue", "density_range_pct",
                   "ep_range_pct", "total_ep_spread"]]
          .round(4).to_string(index=False))

    # ---- published-bound binding diagnostic over the same NB grid ----
    print("\n  Published-bound binding check (vanilla) across NB...")
    bind = sigma_binding_check(R_data, R_GRID, n_bins_list=NB_LIST)
    bind.to_csv(TAB_DIR / "sigma_binding_check.csv", index=False)
    n_bind = int(bind["van_any_bound_binds"].sum())
    print(f"    published bounds bind in {n_bind}/{len(bind)} NB settings; "
          f"detail column saved to sigma_binding_check.csv")

    # ---- spaghetti figure (appendix) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey="row")
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(NB_LIST)))
    for col, venue in enumerate(["CME", "DER"]):
        for row, est in enumerate(["almeida", "vanilla"]):
            ax = axes[row, col]
            for ci, nb in enumerate(NB_LIST):
                ax.plot(R_GRID, ep[(est, venue, nb)], color=cmap[ci],
                        lw=1.0, label=f"NB={nb}")
            ax.axhline(0, color="black", lw=0.5)
            ax.axvline(1.0, color="gray", lw=0.5, ls=":")
            ax.set_xlim(0.50, 1.60)
            vlabel = "Deribit" if venue == "DER" else venue
            ax.set_title(f"{vlabel} — {ESTIMATOR_LABELS[est]}")
            if row == 1:
                ax.set_xlabel("Gross return $R$")
            if col == 0:
                ax.set_ylabel(r"$\mathrm{ep}^j(R)$")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("EP-Curve Stability Across Histogram Bin Counts", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_nb_sweep_appendix.png", dpi=150)
    plt.close()
    print(f"\n  Saved figure: {FIG_DIR / 'fig_nb_sweep_appendix.png'}")
    return summary


if __name__ == "__main__":
    run_nb_sweep()