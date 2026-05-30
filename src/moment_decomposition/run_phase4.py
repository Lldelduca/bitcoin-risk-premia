"""
Phase 4 Orchestrator: BKM Moment Extraction & CL20/CL24 Cumulant Decomposition.

Extracts daily risk-neutral moments from the SSVI surface via BKM (2003), forms the Chabi-Yo & Loudis (2020) lower-bound 
contributions, aggregates them unconditionally and by volatility tercile (CL24 conditional decomposition), and reports the 
variance risk premium as a secondary diagnostic. A theta robustness sweep is produced for the preference parameter.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import get_path, SAMPLE
from src.surfaces.ssvi import SSVI
from src.moment_decomposition.bkm_moments import extract_bkm_moments
from src.moment_decomposition.cumulant_premia import (compute_physical_variance, compute_cumulant_premia)
from src.moment_decomposition.cumulant_premia import (compute_cyl_decomposition_table, robustness_over_theta, cyl_weights)

SAMPLE_START = pd.to_datetime(SAMPLE["start_date"])
SAMPLE_END = pd.to_datetime(SAMPLE["end_date"])

CLEAN_DIR = Path(get_path("cleaned_cme")).parent
SURFACES_DIR = CLEAN_DIR.parent / "surfaces"
COND_DIR = CLEAN_DIR.parent / "conditioning"
PHASE4_DIR = CLEAN_DIR.parent / "phase4"
FIG_DIR = Path("results") / "phase4" / "figures"
TAB_DIR = Path("results") / "phase4" / "tables"
for d in [PHASE4_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TAU_DAYS = 27
TAU_YEARS = TAU_DAYS / 365.25
THETA_BASELINE = 2.0
THETA_GRID = (1.0, 2.0, 3.0)

plt.rcParams["figure.figsize"] = (13, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

def load_option_data(venue):
    key = "cleaned_cme" if venue == "CME" else "cleaned_deribit"
    df = pd.read_parquet(get_path(key))
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)
    return df[mask].copy()

def extract_bkm_for_venue(venue, df):
    dates = sorted(df["date"].unique())
    results = []
    for i, date in enumerate(dates):
        df_day = df[df["date"] == date]
        try:
            ssvi = SSVI(df_day, venue=venue, date=date)
            ssvi.fit()
            fitted_days = np.array(ssvi.res["maturities"]) * 365.25
            if TAU_DAYS < fitted_days.min() * 0.8 or TAU_DAYS > fitted_days.max() * 1.2:
                continue
            bkm = extract_bkm_moments(ssvi, TAU_YEARS, n_strikes=500, r=0.0)
            results.append({
                "date": date, "venue": venue,
                "V": bkm.V, "W": bkm.W, "X": bkm.X,
                "mu_Q": bkm.mu_Q, "var_Q": bkm.var_Q,
                "skew_Q": bkm.skew_Q, "kurt_Q": bkm.kurt_Q,
                "forward": bkm.forward,
            })
        except Exception as e:
            if i % 100 == 0:
                print(f"    [{venue}] day {i}/{len(dates)}: {e}")
            continue
        if (i + 1) % 100 == 0:
            print(f"    [{venue}] {i+1}/{len(dates)} ({len(results)} ok)")
    print(f"  [{venue}] BKM extraction complete: {len(results)}/{len(dates)} days")
    return pd.DataFrame(results)

def load_physical_variance():
    panel = pd.read_parquet(CLEAN_DIR / "auxiliary_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date")
    log_ret = np.log(panel["btc_spot"] / panel["btc_spot"].shift(1))
    log_ret.index = panel["date"]
    var_P = compute_physical_variance(log_ret, tau_days=TAU_DAYS, window=252)
    return var_P

def run_phase4():
    print("\n" + "=" * 60)
    print("  Phase 4: BKM Extraction & CL20/CL24 Decomposition")
    print(f"  Window: {SAMPLE_START.date()} -> {SAMPLE_END.date()}")
    print(f"  theta baseline = {THETA_BASELINE}; weights = {cyl_weights(THETA_BASELINE)}")
    print("=" * 60)

    # Step 1: BKM extraction
    all_bkm = []
    for venue in ["CME", "DER"]:
        print(f"\n  Loading {venue} options...")
        df = load_option_data(venue)
        print(f"  {venue}: {len(df):,} options, {df['date'].nunique()} days")
        bkm_df = extract_bkm_for_venue(venue, df)
        all_bkm.append(bkm_df)
    bkm_all = pd.concat(all_bkm, ignore_index=True)
    bkm_all["date"] = pd.to_datetime(bkm_all["date"])
    bkm_all.to_parquet(PHASE4_DIR / "bkm_moments.parquet", index=False)
    print(f"\n  Saved BKM moments: {len(bkm_all)} day-venue pairs")

    # Step 2: physical variance (for VRP diagnostic only)
    print("\n  Computing rolling physical variance (252-day window)...")
    var_P = load_physical_variance()
    print(f"  Physical variance available: {var_P.dropna().shape[0]} days")

    # Step 3: cumulant premium contributions (Option A, theta baseline)
    print(f"\n  Computing CL20 contributions at theta={THETA_BASELINE}...")
    premia_rows = []
    for _, row in bkm_all.iterrows():
        date = row["date"]
        vP = var_P.loc[date] if date in var_P.index else np.nan
        cp = compute_cumulant_premia(
            date=date, venue=row["venue"],
            V=row["V"], W=row["W"], X=row["X"],
            var_Q=row["var_Q"], skew_Q=row["skew_Q"], kurt_Q=row["kurt_Q"],
            var_P=vP, theta=THETA_BASELINE,
        )
        premia_rows.append(cp._asdict())
    premia_df = pd.DataFrame(premia_rows)
    premia_df["date"] = pd.to_datetime(premia_df["date"])

    # Tercile labels from Z_crypto
    Z_crypto = pd.read_parquet(COND_DIR / "Z_crypto.parquet")
    Z_crypto["date"] = pd.to_datetime(Z_crypto["date"])
    Z_crypto["tercile"] = pd.qcut(Z_crypto["Z_IVS_1"], q=3, labels=["low", "mid", "high"])
    premia_df = premia_df.merge(Z_crypto[["date", "tercile"]], on="date", how="left")
    premia_df.to_parquet(PHASE4_DIR / "cumulant_premia.parquet", index=False)
    print(f"  Saved cumulant premia: {len(premia_df)} rows")

    # Step 4: CL24 decomposition table
    print("\n  CL24 conditional decomposition (unconditional + terciles)...")
    decomp = compute_cyl_decomposition_table(premia_df, tercile_col="tercile")
    decomp.to_csv(TAB_DIR / "cyl_decomposition.csv", index=False)
    print(decomp.round(4).to_string(index=False))

    # Step 5: theta robustness
    print("\n  Theta robustness sweep...")
    rob = robustness_over_theta(bkm_all[["date", "venue", "V", "W", "X"]], THETA_GRID)
    rob.to_csv(PHASE4_DIR / "theta_robustness.csv", index=False)
    print(rob.round(4).to_string(index=False))

    # Step 6: moment summary
    summary = premia_df.groupby("venue").agg({
        "var_Q": ["mean", "std"], "skew_Q": ["mean", "std"], "kurt_Q": ["mean", "std"],
        "Pi_2": ["mean", "std"], "Pi_3": ["mean", "std"], "Pi_4": ["mean", "std"],
        "vrp": ["mean", "std"],
    }).round(4)
    summary.to_csv(TAB_DIR / "moment_summary.csv")
    print(f"\n  Moment summary:\n{summary}")

    # Step 7: figures
    _plot_rn_moments_timeseries(bkm_all)
    _plot_cumulant_contributions_ts(premia_df)
    _plot_contributions_boxplot(premia_df)
    _plot_lb_decomposition_bars(decomp)
    _plot_moment_cross_venue_scatter(bkm_all)
    _plot_vrp_timeseries(premia_df)
    _plot_theta_robustness(rob)

    print(f"\n  Phase 4 complete. Figures in {FIG_DIR}/")
    return bkm_all, premia_df, decomp, rob

# Figure functions
def _plot_rn_moments_timeseries(bkm_all):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for venue, color in [("CME", "C0"), ("DER", "C1")]:
        v = bkm_all[bkm_all["venue"] == venue].set_index("date").sort_index()
        axes[0].plot(v.index, v["var_Q"], color=color, lw=0.8, alpha=0.8, label=venue)
        axes[1].plot(v.index, v["skew_Q"], color=color, lw=0.8, alpha=0.8, label=venue)
        axes[2].plot(v.index, v["kurt_Q"], color=color, lw=0.8, alpha=0.8, label=venue)
    axes[0].set_ylabel(r"$V_t^{\mathbb{Q},j}$"); axes[0].set_title("Risk-Neutral Moments (27-day)"); axes[0].legend()
    axes[1].set_ylabel(r"Skew$_t^{\mathbb{Q},j}$"); axes[1].axhline(0, color="black", lw=0.4)
    axes[2].set_ylabel(r"Kurt$_t^{\mathbb{Q},j}$"); axes[2].axhline(3, color="black", lw=0.4, ls="--")
    axes[2].set_xlabel("Date")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_rnd_moments_ts.png", dpi=150); plt.close()

def _plot_cumulant_contributions_ts(premia_df):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for venue, color in [("CME", "C0"), ("DER", "C1")]:
        v = premia_df[premia_df["venue"] == venue].set_index("date").sort_index()
        axes[0].plot(v.index, v["Pi_2"], color=color, lw=0.8, alpha=0.8, label=venue)
        axes[1].plot(v.index, v["Pi_3"], color=color, lw=0.8, alpha=0.8, label=venue)
        axes[2].plot(v.index, v["Pi_4"], color=color, lw=0.8, alpha=0.8, label=venue)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.4)
    axes[0].set_ylabel(r"$\Pi_{2,t}^j$ (variance)"); axes[0].set_title("CL20 Cumulant Premium Contributions"); axes[0].legend()
    axes[1].set_ylabel(r"$\Pi_{3,t}^j$ (skewness)")
    axes[2].set_ylabel(r"$\Pi_{4,t}^j$ (kurtosis)"); axes[2].set_xlabel("Date")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_cumulant_premia_ts.png", dpi=150); plt.close()

def _plot_contributions_boxplot(premia_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, label) in zip(axes, [
        ("Pi_2", r"$\Pi_2$ (variance)"), ("Pi_3", r"$\Pi_3$ (skewness)"), ("Pi_4", r"$\Pi_4$ (kurtosis)")
    ]):
        data, labels_x = [], []
        for venue in ["CME", "DER"]:
            for tercile in ["low", "mid", "high"]:
                mask = (premia_df["venue"] == venue) & (premia_df["tercile"] == tercile)
                data.append(premia_df.loc[mask, col].dropna().values)
                labels_x.append(f"{venue}\n{tercile}")
        ax.boxplot(data, tick_labels=labels_x, showfliers=False)
        ax.axhline(0, color="black", lw=0.4); ax.set_ylabel(label); ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Cumulant Premium Contributions by Venue and Volatility Tercile", fontsize=13)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_cumulant_premia_boxplot.png", dpi=150); plt.close()

def _plot_lb_decomposition_bars(decomp):
    uncond = decomp[decomp["regime"] == "unconditional"]
    if len(uncond) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(uncond)); width = 0.25
    ax.bar(x - width, uncond["Pi_2"], width, label=r"$\Pi_2$ variance")
    ax.bar(x, uncond["Pi_3"], width, label=r"$\Pi_3$ skewness")
    ax.bar(x + width, uncond["Pi_4"], width, label=r"$\Pi_4$ kurtosis")
    ax.set_xticks(x); ax.set_xticklabels(uncond["venue"].values)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("27-day contribution"); ax.set_title("CL20 Lower-Bound Decomposition (Unconditional)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_cyl_contributions.png", dpi=150); plt.close()

def _plot_moment_cross_venue_scatter(bkm_all):
    cme = bkm_all[bkm_all["venue"] == "CME"].set_index("date")
    der = bkm_all[bkm_all["venue"] == "DER"].set_index("date")
    matched = cme.join(der, lsuffix="_cme", rsuffix="_der", how="inner")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, label) in zip(axes, [("var_Q", "Variance"), ("skew_Q", "Skewness"), ("kurt_Q", "Kurtosis")]):
        ax.scatter(matched[f"{col}_cme"], matched[f"{col}_der"], s=3, alpha=0.4, color="C0")
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"CME {label}"); ax.set_ylabel(f"Deribit {label}")
        rho = matched[[f"{col}_cme", f"{col}_der"]].corr().iloc[0, 1]
        ax.set_title(f"{label} (rho = {rho:.3f})")
    fig.suptitle("Cross-Venue RN Moment Agreement (matched days)", fontsize=13)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_moment_cross_venue.png", dpi=150); plt.close()

def _plot_vrp_timeseries(premia_df):
    fig, ax = plt.subplots(figsize=(13, 5))
    for venue, color in [("CME", "C0"), ("DER", "C1")]:
        v = premia_df[premia_df["venue"] == venue].set_index("date").sort_index()
        ax.plot(v.index, v["vrp"], color=color, lw=0.8, alpha=0.8, label=venue)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Date"); ax.set_ylabel(r"$\mathrm{VRP}_t^j = V_t^{\mathbb{Q},j} - V_t^{\mathbb{P}}$")
    ax.set_title("Variance Risk Premium: CME vs Deribit (27-day)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_vrp_timeseries.png", dpi=150); plt.close()

def _plot_theta_robustness(rob):
    fig, ax = plt.subplots(figsize=(10, 5))
    for venue, color in [("CME", "C0"), ("DER", "C1")]:
        v = rob[rob["venue"] == venue].sort_values("theta")
        ax.plot(v["theta"], v["lb_annualized_pct"], "o-", color=color, label=venue)
    ax.set_xlabel(r"Preference parameter $\theta$")
    ax.set_ylabel("Annualized lower bound (%)")
    ax.set_title("CL20 Lower Bound: Sensitivity to Preference Parameter")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_theta_robustness.png", dpi=150); plt.close()

if __name__ == "__main__":
    run_phase4()
