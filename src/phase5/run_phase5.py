"""
Phase 5 Orchestrator: Cross-Venue Analysis.

  1. Conditional MFK by volatility tercile
  2. Panel regressions of Π_{k,t}^j on venue dummies + Z_crypto
  3. Regional MFK integration (downside/mid/upside)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import get_path, SAMPLE, get_return_grid
from src.phase5.cross_venue import (compute_conditional_mfk, run_cumulant_panel_regressions,
                                         run_matched_difference_regressions)
from src.phase5.cross_venue import (format_regression_table, compute_regional_mfk)

SAMPLE_START = pd.to_datetime(SAMPLE["start_date"])
SAMPLE_END = pd.to_datetime(SAMPLE["end_date"])

CLEAN_DIR = Path(get_path("cleaned_cme")).parent
SURFACES_DIR = CLEAN_DIR.parent / "surfaces"
COND_DIR = CLEAN_DIR.parent / "conditioning"
PHASE4_DIR = CLEAN_DIR.parent / "phase4"
PHASE5_DIR = CLEAN_DIR.parent / "phase5"
FIG_DIR = Path("results") / "phase5" / "figures"
TAB_DIR = Path("results") / "phase5" / "tables"
for d in [PHASE5_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (13, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

R_GRID = get_return_grid()

def run_phase5():
    print("\n" + "=" * 60)
    print("  Phase 5: Cross-Venue Analysis")
    print(f"  Window: {SAMPLE_START.date()} -> {SAMPLE_END.date()}")
    print("=" * 60)

    # Load tercile labels
    Z_crypto = pd.read_parquet(COND_DIR / "Z_crypto.parquet")
    Z_crypto["date"] = pd.to_datetime(Z_crypto["date"])
    Z_crypto["tercile"] = pd.qcut(Z_crypto["Z_IVS_1"], q=3, labels=["low", "mid", "high"])

    # Component 1: Conditional MFK
    print("\n  Component 1: Conditional MFK by volatility tercile...")

    mfk_results, psi_df = compute_conditional_mfk(
        rnd_cme_path=SURFACES_DIR / "rnd_CME_densities.parquet",
        rnd_der_path=SURFACES_DIR / "rnd_DER_densities.parquet",
        tercile_labels=Z_crypto[["date", "tercile"]],
        tau_days=27,
        R_grid=R_GRID,
    )

    for regime, data in mfk_results.items():
        print(f"    {regime}: {data['n_days']} days, "
              f"peak Ψ = {data['mean_psi'].max():.4f}, "
              f"min Ψ = {data['mean_psi'].min():.4f}")

    psi_df.to_parquet(PHASE5_DIR / "conditional_mfk.parquet")

    _plot_conditional_mfk(mfk_results)

    # Component 2: Venue-wedge regressions
    print("\n  Component 2: Cumulant premium regressions...")

    premia = pd.read_parquet(PHASE4_DIR / "cumulant_premia.parquet")
    premia["date"] = pd.to_datetime(premia["date"])

    # Z_crypto as regressors (standardized)
    z_cols = ["Z_IVS_1", "rv", "fng"]
    Z_reg = Z_crypto[["date"] + z_cols].copy()

    print("\n  (2a) Matched-difference regressions (HEADLINE, NW lags=27):")
    diff_results, diff_panels = run_matched_difference_regressions(
        premia, Z_reg, z_cols=z_cols, nw_lags=27,
    )
    diff_table = format_regression_table(diff_results)
    diff_table.to_csv(TAB_DIR / "matched_difference_regressions.csv", index=False)
    for dep in ["Pi_2", "Pi_3", "Pi_4"]:
        res = diff_results[dep]
        print(f"\n    Δ{dep} (n={int(res.nobs)}):")
        for i, name in enumerate(res.col_names):
            stars = _stars_fn(res.pvalues[i])
            print(f"      {name:>22s}: {res.params[i]:+.5f} "
                  f"({res.bse[i]:.5f}) [{res.tvalues[i]:+.3f}]{stars}")
        print(f"      {'R²':>22s}: {res.rsquared:.4f}")

    # (2b) Secondary: pooled matched panel with Driscoll-Kraay errors
    print("\n  (2b) Pooled panel, Driscoll-Kraay (secondary):")
    reg_results, panel = run_cumulant_panel_regressions(
        premia, Z_reg, z_cols=z_cols, nw_lags=27,
        driscoll_kraay=True, matched_only=True,
    )
    reg_table = format_regression_table(reg_results)
    reg_table.to_csv(TAB_DIR / "panel_regressions_dk.csv", index=False)

    # (2c) Legacy stacked-panel HAC — kept ONLY for comparison with the old
    print("\n  (2c) Legacy stacked-panel HAC (comparison only):")
    legacy_results, _ = run_cumulant_panel_regressions(
        premia, Z_reg, z_cols=z_cols, nw_lags=10,
        driscoll_kraay=False, matched_only=False,
    )
    legacy_table = format_regression_table(legacy_results)
    legacy_table.to_csv(TAB_DIR / "panel_regressions.csv", index=False)

    print("\n  Pooled panel (Driscoll-Kraay) results:")
    for dep in ["Pi_2", "Pi_3", "Pi_4"]:
        res = reg_results[dep]
        print(f"\n    {dep}:")
        for i, name in enumerate(res.col_names):
            stars = _stars_fn(res.pvalues[i])
            print(f"      {name:>15s}: {res.params[i]:+.5f} "
                  f"({res.bse[i]:.5f}) [{res.tvalues[i]:+.3f}]{stars}")
        print(f"      {'R²':>15s}: {res.rsquared:.4f}")
        print(f"      {'N':>15s}: {int(res.nobs)}")
    print("\n  [NOTE] LaTeX: the venue-wedge claims should cite "
          "matched_difference_regressions.csv (headline) and "
          "panel_regressions_dk.csv (secondary). panel_regressions.csv is the "
          "legacy estimator, retained for comparison only.")

    _plot_venue_coefficients(diff_results, wedge_name="const (venue wedge)")

    # Component 3: Regional MFK 
    print("\n  Component 3: Regional MFK integration...")

    regional_df, regional_summary = compute_regional_mfk(
        psi_df, R_GRID, tercile_col="tercile",
    )

    regional_df.to_parquet(PHASE5_DIR / "regional_mfk.parquet")
    regional_summary.to_csv(TAB_DIR / "regional_mfk_summary.csv", index=False)

    print(regional_summary.round(4).to_string(index=False))

    _plot_regional_mfk(regional_summary)
    _plot_regional_mfk_timeseries(regional_df)

    print(f"\n  Phase 5 complete. Figures in {FIG_DIR}/")
    return mfk_results, reg_results, regional_summary

def _stars_fn(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""


# Figures
def _plot_conditional_mfk(mfk_results):
    """Conditional MFK by tercile with unconditional as grey reference."""
    fig, ax = plt.subplots(figsize=(13, 6))

    R = mfk_results["unconditional"]["R_grid"]

    # Unconditional as dashed grey reference
    d = mfk_results["unconditional"]
    ax.plot(R, d["mean_psi"], color="grey", ls="--", lw=1.0, alpha=0.6,
            label=f"Unconditional (n={d['n_days']})")

    # Terciles as primary content (95% block-bootstrap bands)
    colors = {"low": "C0", "mid": "C1", "high": "C2"}
    for tercile in ["low", "mid", "high"]:
        if tercile in mfk_results:
            d = mfk_results[tercile]
            ax.plot(R, d["mean_psi"], color=colors[tercile], lw=1.5,
                    label=f"{tercile}-vol (n={d['n_days']})")
            lo = d.get("lo_psi", d["mean_psi"] - 1.96 * d["se_psi"])
            hi = d.get("hi_psi", d["mean_psi"] + 1.96 * d["se_psi"])
            ax.fill_between(R, lo, hi, alpha=0.1, color=colors[tercile])

    ax.axhline(0, color="black", lw=0.5)
    ax.axvspan(R[0], 0.90, alpha=0.05, color="red")
    ax.axvspan(1.10, R[-1], alpha=0.05, color="green")
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"$\Psi(R \mid \mathrm{tercile})$")
    ax.set_title("Conditional Microstructure Friction Kernel by Volatility Tercile")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_mfk_conditional.png", dpi=150)
    plt.close()

def _plot_venue_coefficients(reg_results, wedge_name="DER"):
    """Bar chart of the venue wedge across the three regressions.

    Works with both the legacy pooled panel (wedge_name='DER') and the
    matched-difference headline (wedge_name='const (venue wedge)')."""
    fig, ax = plt.subplots(figsize=(10, 5))

    dep_vars = ["Pi_2", "Pi_3", "Pi_4"]
    labels = [r"$\Pi_2$ (var)", r"$\Pi_3$ (skew)", r"$\Pi_4$ (kurt)"]
    x = np.arange(len(dep_vars))

    betas = []
    ses = []
    for dep in dep_vars:
        res = reg_results[dep]
        idx = res.col_names.index(wedge_name)
        betas.append(res.params[idx])
        ses.append(1.96 * res.bse[idx])

    bars = ax.bar(x, betas, yerr=ses, capsize=5, color=["C0", "C1", "C2"], alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Deribit venue coefficient β")
    ax.set_title("Cross-Venue Wedge: Deribit Premium over CME (with 95% CI)")

    # Add significance stars
    for i, dep in enumerate(dep_vars):
        res = reg_results[dep]
        idx = res.col_names.index(wedge_name)
        stars = _stars_fn(res.pvalues[idx])
        if stars:
            ax.text(i, betas[i] + ses[i] + 0.001, stars, ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_venue_coefficients.png", dpi=150)
    plt.close()

def _plot_regional_mfk(regional_summary):
    """Bar chart of regional MFK by tercile."""
    fig, ax = plt.subplots(figsize=(12, 5))

    regimes = regional_summary["regime"].values
    x = np.arange(len(regimes))
    width = 0.25

    ax.bar(x - width, regional_summary["mean_down"], width,
           label="Downside (R<0.90)", color="C3", alpha=0.8)
    ax.bar(x, regional_summary["mean_mid"], width,
           label="Mid (0.90≤R≤1.10)", color="C0", alpha=0.8)
    ax.bar(x + width, regional_summary["mean_up"], width,
           label="Upside (R>1.10)", color="C2", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Mean integrated Ψ")
    ax.set_title("Regional MFK by Volatility Regime")
    ax.legend(loc="upper center", ncol=3, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_regional_mfk.png", dpi=150)
    plt.close()

def _plot_regional_mfk_timeseries(regional_df):
    """Time series of the three regional MFK scalars."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    regional_df = regional_df.set_index("date").sort_index()

    for ax, (col, label, color) in zip(axes, [
        ("psi_down", "Downside (R<0.90)", "C3"),
        ("psi_mid", "Mid (0.90≤R≤1.10)", "C0"),
        ("psi_up", "Upside (R>1.10)", "C2"),
    ]):
        ax.plot(regional_df.index, regional_df[col], color=color, lw=0.6, alpha=0.7)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel(label)

    axes[0].set_title("Regional MFK Scalars Over Time")
    axes[2].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_regional_mfk_ts.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_phase5()
