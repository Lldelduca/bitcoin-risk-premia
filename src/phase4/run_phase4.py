"""
Phase 4 Orchestrator: BKM Moment Extraction & CL20 Cumulant Decomposition.

Extracts daily risk-neutral moments from the SSVI surface via BKM (2003), forms the Chabi-Yo & Loudis (2020) lower-bound 
contributions, aggregates them unconditionally and by volatility tercile (regime-conditional decomposition), and reports the 
variance risk premium as a secondary diagnostic. A theta robustness sweep is produced for the preference parameter.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import get_path, SAMPLE
from src.phase1.ssvi import SSVI
from src.phase4.bkm_moments import (extract_bkm_moments, evaluate_iv_grid, bkm_from_iv_grid)
from src.phase4.cumulant_premia import (compute_physical_variance, compute_cumulant_premia)
from src.phase4.cumulant_premia import (compute_cyl_decomposition_table, robustness_over_theta, cyl_weights)

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

KAPPA_BOUNDS = (1.0, 1.25, 1.5)
KAPPA_HEADLINE = 1.5

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

def load_ssvi_params():
    params_path = SURFACES_DIR / "ssvi_params.parquet"
    if not params_path.exists():
        raise FileNotFoundError(
            f"{params_path} not found. Run fit_surfaces first; Phase 4 "
            f"evaluates the saved Phase 1 surfaces, never a refit."
        )
    params = pd.read_parquet(params_path)
    params["date"] = pd.to_datetime(params["date"])
    return params

def extract_bkm_for_venue(venue, df, params):
    params_v = params[params["venue"] == venue]
    dates = sorted(params_v["date"].unique())
    has_forward_col = "forward" in params_v.columns and params_v["forward"].notna().all()
    results = []
    sweep_rows = []
    for i, date in enumerate(dates):
        params_day = params_v[params_v["date"] == date]
        try:
            forward_map = None
            if not has_forward_col:
                df_day = df[df["date"] == date]
                if df_day.empty:
                    continue
                forward_map = df_day.groupby("tau")["forward_price"].mean()
            ssvi = SSVI.from_params(params_day, forward_map=forward_map,
                                    venue=venue, date=date)
            fitted_days = np.array(ssvi.res["maturities"]) * 365.25
            if TAU_DAYS < fitted_days.min() * 0.8 or TAU_DAYS > fitted_days.max() * 1.2:
                continue

            # Evaluate the IV grid ONCE at the widest bound
            F, kgrid, ivgrid = evaluate_iv_grid(ssvi, TAU_YEARS, n_strikes=500,
                                                kappa_max=KAPPA_HEADLINE)
            bkm = None
            for kb in KAPPA_BOUNDS:
                b = bkm_from_iv_grid(F, kgrid, ivgrid, TAU_YEARS, r=0.0, kappa_bound=kb)
                sweep_rows.append({
                    "date": date, "venue": venue, "kappa_bound": kb,
                    "V": b.V, "W": b.W, "X": b.X,
                })
                if kb == KAPPA_HEADLINE:
                    bkm = b
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
    return pd.DataFrame(results), pd.DataFrame(sweep_rows)

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
    print("  Phase 4: BKM Extraction & CL20 Decomposition")
    print(f"  Window: {SAMPLE_START.date()} -> {SAMPLE_END.date()}")
    print(f"  theta baseline = {THETA_BASELINE}; weights = {cyl_weights(THETA_BASELINE)}")
    print("=" * 60)

    # Step 1: BKM extraction from the saved Phase 1 surfaces
    print("\n  Loading fitted SSVI parameters...")
    params = load_ssvi_params()
    print(f"  Loaded {len(params):,} parameter rows")

    all_bkm = []
    all_sweep = []
    for venue in ["CME", "DER"]:
        print(f"\n  Loading {venue} options...")
        df = load_option_data(venue)
        print(f"  {venue}: {len(df):,} options, {df['date'].nunique()} days")
        bkm_df, sweep_df = extract_bkm_for_venue(venue, df, params)
        all_bkm.append(bkm_df)
        all_sweep.append(sweep_df)
    bkm_all = pd.concat(all_bkm, ignore_index=True)
    bkm_all["date"] = pd.to_datetime(bkm_all["date"])
    bkm_all.to_parquet(PHASE4_DIR / "bkm_moments.parquet", index=False)
    print(f"\n  Saved BKM moments: {len(bkm_all)} day-venue pairs")

    sweep_all = pd.concat(all_sweep, ignore_index=True)
    sweep_all["date"] = pd.to_datetime(sweep_all["date"])
    sweep_all.to_parquet(PHASE4_DIR / "bkm_kappa_sweep.parquet", index=False)

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

    # Step 4: Regime-conditional decomposition tables
    cme_days = set(premia_df.loc[premia_df["venue"] == "CME", "date"])
    der_days = set(premia_df.loc[premia_df["venue"] == "DER", "date"])
    matched_days = cme_days & der_days
    premia_matched = premia_df[premia_df["date"].isin(matched_days)].copy()
    print(f"\n  Matched CME-Deribit days: {len(matched_days)} "
          f"({len(premia_matched)} venue-day rows)")

    print("\n  Regime-conditional decomposition — MATCHED DAYS (headline)...")
    decomp_matched = compute_cyl_decomposition_table(premia_matched, tercile_col="tercile")
    decomp_matched.to_csv(TAB_DIR / "cyl_decomposition_matched.csv", index=False)
    print(decomp_matched.round(4).to_string(index=False))

    # Consistency guard
    for venue in ["CME", "DER"]:
        v = decomp_matched[decomp_matched["venue"] == venue]
        n_uncond = int(v.loc[v["regime"] == "unconditional", "n_days"].iloc[0])
        n_terc = int(v.loc[v["regime"] != "unconditional", "n_days"].sum())
        if n_uncond != n_terc:
            print(f"  [WARN] {venue}: unconditional n={n_uncond} != "
                  f"sum of terciles {n_terc} — tercile labels missing on "
                  f"{n_uncond - n_terc} matched days; check Z_crypto coverage.")

    print("\n  Regime-conditional decomposition — ALL DAYS (supplementary)...")
    decomp = compute_cyl_decomposition_table(premia_df, tercile_col="tercile")
    decomp.to_csv(TAB_DIR / "cyl_decomposition.csv", index=False)
    print(decomp.round(4).to_string(index=False))
    print("\n  [NOTE] LaTeX: source the headline cross-venue table from "
          "cyl_decomposition_matched.csv; cyl_decomposition.csv (all days) "
          "is supplementary.")

    # Step 5: theta robustness (matched days, consistent with the headline)
    print("\n  Theta robustness sweep (matched days)...")
    bkm_matched = bkm_all[bkm_all["date"].isin(matched_days)]
    rob = robustness_over_theta(bkm_matched[["date", "venue", "V", "W", "X"]], THETA_GRID)
    rob.to_csv(PHASE4_DIR / "theta_robustness.csv", index=False)
    print(rob.round(4).to_string(index=False))

    # Step 5b: kappa-bound truncation sensitivity (matched days)
    print("\n  Kappa-bound sensitivity (matched days)...")
    kappa_sens = kappa_sensitivity_table(sweep_all, matched_days, theta=THETA_BASELINE)
    kappa_sens.to_csv(TAB_DIR / "kappa_sensitivity.csv", index=False)
    print(kappa_sens.round(4).to_string(index=False))
    _plot_kappa_sensitivity(kappa_sens)

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
    _plot_lb_decomposition_bars(decomp_matched) 
    _plot_moment_cross_venue_scatter(bkm_all)
    _plot_vrp_timeseries(premia_df)
    _plot_theta_robustness(rob)

    print(f"\n  Phase 4 complete. Figures in {FIG_DIR}/")
    return bkm_all, premia_df, decomp_matched, decomp, rob

# Figure functions
def kappa_sensitivity_table(sweep_all, matched_days, theta=2.0):
    """Mean CL20 contributions and shares per venue and kappa truncation,
    on matched days only (consistent with the headline table)."""
    l1, l2, l3 = cyl_weights(theta)
    s = sweep_all[sweep_all["date"].isin(matched_days)].copy()
    s["Pi_2"] = l1 * s["V"]
    s["Pi_3"] = l2 * s["W"]
    s["Pi_4"] = l3 * s["X"]

    rows = []
    for venue in ["CME", "DER"]:
        for kb in sorted(s["kappa_bound"].unique()):
            v = s[(s["venue"] == venue) & (s["kappa_bound"] == kb)]
            if len(v) == 0:
                continue
            p2, p3, p4 = v["Pi_2"].mean(), v["Pi_3"].mean(), v["Pi_4"].mean()
            tot = p2 + p3 + p4
            rows.append({
                "venue": venue, "kappa_bound": kb, "n_days": len(v),
                "Pi_2": p2, "Pi_3": p3, "Pi_4": p4, "lb_total": tot,
                "share_var": p2 / tot if tot != 0 else np.nan,
                "share_skew": p3 / tot if tot != 0 else np.nan,
                "share_kurt": p4 / tot if tot != 0 else np.nan,
                "lb_annualized_pct": 100.0 * tot * (365.0 / 27.0),
            })
    return pd.DataFrame(rows)

def _plot_kappa_sensitivity(kappa_sens):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, label) in zip(axes, [
        ("share_kurt", r"Kurtosis share $\Pi_4 / \Sigma_k \Pi_k$"),
        ("lb_total", r"CL20 bound $\Sigma_k \Pi_k$ (27-day)"),
    ]):
        for venue, color in [("CME", "C0"), ("DER", "C1")]:
            v = kappa_sens[kappa_sens["venue"] == venue].sort_values("kappa_bound")
            ax.plot(v["kappa_bound"], v[col], "o-", color=color, label=venue)
        ax.set_xlabel(r"Strike-domain truncation $|\kappa| \leq$ bound")
        ax.set_ylabel(label)
        ax.legend()
    fig.suptitle("CL20 Decomposition: Sensitivity to Strike-Domain Truncation (matched days)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_kappa_sensitivity.png", dpi=150)
    plt.close()

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