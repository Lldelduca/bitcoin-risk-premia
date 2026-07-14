"""
Runs the coin-margined (inverse) contract numeraire prediction

Compares the predicted Deribit-minus-CME cumulant-premium wedge against the measured wedge from Phase 4.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import get_path, get_return_grid, get_sample_window
from src.phase5.inverse_contract import (contributions_from_density, tilt_to_inverse_measure, predict_inverse_wedge)
from src.phase3.bootstrap_inference import block_bootstrap_statistic

# Explicit path updates
DATA_P1 = get_path("data_phase1")
DATA_P5 = get_path("data_phase5")
RES_P5 = get_path("results_phase5")

FIG_DIR = RES_P5 / "figures"
TAB_DIR = RES_P5 / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

R_GRID = get_return_grid()
KAPPA_MAX = 1.5
R_GRID_WIDE = np.linspace(np.exp(-(KAPPA_MAX + 0.1)), np.exp(KAPPA_MAX + 0.1), 2000)

TAU_DAYS = 27
THETA = 2.0
KAPPA_BOUND = 1.5 
BOOT_B = 1000
BOOT_BLOCK = 27

SAMPLE_START_STR, SAMPLE_END_STR = get_sample_window()

def _load_matched_densities():
    cme = pd.read_parquet(DATA_P1 / "rnd_CME_densities.parquet")
    der = pd.read_parquet(DATA_P1 / "rnd_DER_densities.parquet")
    for d in (cme, der):
        d["date"] = pd.to_datetime(d["date"])
    cme = cme[cme["tau_days"] == TAU_DAYS].set_index("date").sort_index()
    der = der[der["tau_days"] == TAU_DAYS].set_index("date").sort_index()

    matched = cme.index.intersection(der.index)
    q_cme_rows, q_der_rows = [], []
    for date in matched:
        R_c = np.asarray(cme.loc[date, "returns"])
        q_c = np.asarray(cme.loc[date, "density"])
        R_d = np.asarray(der.loc[date, "returns"])
        q_d = np.asarray(der.loc[date, "density"])
        q_cme_rows.append(np.interp(R_GRID_WIDE, R_c, q_c, left=1e-20, right=1e-20))
        q_der_rows.append(np.interp(R_GRID_WIDE, R_d, q_d, left=1e-20, right=1e-20))
    return (np.array(matched), np.vstack(q_cme_rows), np.vstack(q_der_rows))

def run_inverse_contract():
    print("\n" + "=" * 60)
    print("  Phase 5 extension: Inverse-Contract Numeraire Prediction")
    print(f"  Window: {SAMPLE_START_STR} -> {SAMPLE_END_STR}")
    print(f"  theta = {THETA}; kappa bound = {KAPPA_BOUND}; tau = {TAU_DAYS}d")
    print(f"  BKM grid: R in [{R_GRID_WIDE[0]:.3f}, {R_GRID_WIDE[-1]:.3f}] "
          f"({len(R_GRID_WIDE)} pts, |ln R| up to "
          f"{np.abs(np.log(R_GRID_WIDE[[0,-1]])).max():.2f})")
    print("=" * 60)

    dates, Q_CME, Q_DER = _load_matched_densities()
    n = len(dates)
    print(f"\n  Matched CME-Deribit density days: {n}")

    # Per-day predicted vs measured wedge
    pred_wedge = np.empty((n, 3))
    meas_wedge = np.empty((n, 3))
    daily = []
    for i in range(n):
        cme_i = contributions_from_density(R_GRID_WIDE, Q_CME[i], THETA, KAPPA_BOUND)
        der_i = contributions_from_density(R_GRID_WIDE, Q_DER[i], THETA, KAPPA_BOUND)

        q_pred = tilt_to_inverse_measure(R_GRID_WIDE, Q_CME[i])
        der_pred_i = contributions_from_density(R_GRID_WIDE, q_pred, THETA, KAPPA_BOUND)

        for j, k in enumerate((2, 3, 4)):
            pred_wedge[i, j] = der_pred_i[f"Pi_{k}"] - cme_i[f"Pi_{k}"]
            meas_wedge[i, j] = der_i[f"Pi_{k}"] - cme_i[f"Pi_{k}"]
        daily.append({
            "date": dates[i],
            **{f"pred_wedge_Pi_{k}": pred_wedge[i, j]
               for j, k in enumerate((2, 3, 4))},
            **{f"meas_wedge_Pi_{k}": meas_wedge[i, j]
               for j, k in enumerate((2, 3, 4))},
        })
    daily_df = pd.DataFrame(daily)
    daily_df.to_csv(TAB_DIR / "inverse_contract_daily.csv", index=False)

    # Block-bootstrap CIs on the mean predicted and measured wedges
    def stat_fn(idx):
        return np.concatenate([pred_wedge[idx].mean(0), meas_wedge[idx].mean(0)])
    res = block_bootstrap_statistic(n, stat_fn, block_length=BOOT_BLOCK,
                                    B=BOOT_B, seed=42)

    summary = []
    for j, k in enumerate((2, 3, 4)):
        summary.append({
            "cumulant": f"Pi_{k}",
            "predicted_wedge": res["point"][j],
            "pred_ci_lo": res["lo"][j], "pred_ci_hi": res["hi"][j],
            "measured_wedge": res["point"][3 + j],
            "meas_ci_lo": res["lo"][3 + j], "meas_ci_hi": res["hi"][3 + j],
            "residual": res["point"][3 + j] - res["point"][j],
            "n_days": n, "block_length": BOOT_BLOCK, "B": res["n_success"],
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(TAB_DIR / "inverse_contract_wedge.csv", index=False)

    print("\n  Predicted vs measured cumulant-premium wedge (DER - CME):")
    print(f"    {'':6s} {'predicted':>22s} {'measured':>22s} {'residual':>10s}")
    for r in summary:
        print(f"    {r['cumulant']:6s} "
              f"{r['predicted_wedge']:+.5f} "
              f"[{r['pred_ci_lo']:+.4f},{r['pred_ci_hi']:+.4f}] "
              f"{r['measured_wedge']:+.5f} "
              f"[{r['meas_ci_lo']:+.4f},{r['meas_ci_hi']:+.4f}] "
              f"{r['residual']:+.5f}")

    # Sign-agreement diagnostic
    sign_pred = np.sign([r["predicted_wedge"] for r in summary])
    sign_meas = np.sign([r["measured_wedge"] for r in summary])
    agree = int((sign_pred == sign_meas).sum())
    print(f"\n  Sign agreement: {agree}/3 cumulants. "
          f"{'Contract design explains the wedge direction.'if agree == 3 else 'Contract design predicts the WRONG sign at ' + str(3 - agree) + ' of 3 orders — the friction is real and amplified by a mechanical effect of opposite sign.'}")

    # Psi overlay figure 
    eps = 1e-12

    # Measured Psi_t = ln(q_CME_t / q_DER_t), averaged over matched days
    psi_meas_rows = np.array([
        np.log(np.maximum(Q_CME[i], eps) / np.maximum(Q_DER[i], eps))
        for i in range(n)])
    psi_measured = psi_meas_rows.mean(0)

    # Predicted Psi_t = ln(q_CME_t / q_DER_pred_t), same direction as above
    psi_pred_rows = np.array([
        np.log(np.maximum(Q_CME[i], eps)
               / np.maximum(tilt_to_inverse_measure(R_GRID_WIDE, Q_CME[i]), eps))
        for i in range(n)])
    psi_predicted = psi_pred_rows.mean(0)
    psi_residual = psi_measured - psi_predicted

    m = (R_GRID_WIDE >= 0.5) & (R_GRID_WIDE <= 1.8)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.plot(R_GRID_WIDE[m], psi_measured[m], color="C0", lw=1.8,
            label=r"Measured $\Psi=\ln(q^{\mathrm{CME}}/q^{\mathrm{DER}})$", zorder=4)
    ax.plot(R_GRID_WIDE[m], psi_predicted[m], color="C3", lw=1.6, ls="--",
            label=r"Contract-design prediction (inverse-measure tilt)", zorder=3)
    ax.plot(R_GRID_WIDE[m], psi_residual[m], color="C2", lw=1.4, ls=":",
            label=r"Residual ($=$ measured $-$ predicted)", zorder=2)
    ax.set_xlabel(r"Gross return $R$")
    ax.set_ylabel(r"$\Psi(R)$")
    ax.set_title("Cross-Venue Pricing-Kernel Wedge: Mechanical vs Residual",
                 fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_inverse_contract_psi.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {TAB_DIR / 'inverse_contract_wedge.csv'}")
    print(f"  Saved: {TAB_DIR / 'inverse_contract_daily.csv'}")
    print(f"  Saved: {FIG_DIR / 'fig_inverse_contract_psi.png'}")
    print("\n  Inverse-contract extension complete.")
    return summary_df

if __name__ == "__main__":
    run_inverse_contract()
