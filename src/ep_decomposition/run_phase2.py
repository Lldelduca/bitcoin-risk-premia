"""
Phase 2 Orchestrator: Beason-Schreindorfer EP Decomposition.

Loads Phase 1 outputs (daily RNDs from parquet + BTC spot prices), estimates the unconditional physical density 
(Almeida headline + KDE robustness), computes the EP decomposition for each venue x estimator combination.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import get_path, SAMPLE, get_return_grid

from src.ep_decomposition.physical_density import (estimate_physical_density_almeida, estimate_physical_density_kde,
estimate_physical_density_almeida_from_returns, estimate_physical_density_kde_from_returns,compute_overlapping_returns)
from src.ep_decomposition.ep_decomposition import (compute_ep_decomposition, compute_ep_contributions)
from src.inference.bootstrap_inference import block_bootstrap_statistic

CLEAN_DIR = Path(get_path("cleaned_cme")).parent
SURFACES_DIR = CLEAN_DIR.parent / "surfaces"
PHASE2_DIR = CLEAN_DIR.parent / "phase2"
FIG_DIR = Path("results") / "phase2" / "figures"
TAB_DIR = Path("results") / "phase2" / "tables"
for d in [PHASE2_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

R_GRID = get_return_grid()  
EP_BOOT_B = 500
EP_BOOT_BLOCK = 54
EP_BOOT_SEED = 42

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

def load_spot_prices():
    panel = pd.read_parquet(CLEAN_DIR / "auxiliary_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").dropna(subset=["btc_spot"])
    return panel["btc_spot"].values

def load_daily_rnds_from_parquet(venue, tau_days=27):
    density_path = SURFACES_DIR / f"rnd_{venue}_densities.parquet"
    df = pd.read_parquet(density_path)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to target maturity
    df = df[df["tau_days"] == tau_days].sort_values("date").reset_index(drop=True)

    dates = []
    rnds = []
    for _, row in df.iterrows():
        R = np.array(row["returns"])
        q = np.array(row["density"])

        # Interpolate onto common grid
        q_interp = np.interp(R_GRID, R, q, left=0, right=0)
        mass = np.trapezoid(q_interp, R_GRID)
        if mass > 0:
            q_interp /= mass
        dates.append(row["date"])
        rnds.append(q_interp)

    print(f"  [{venue}] Loaded {len(rnds)} daily RNDs at tau={tau_days}d from parquet")
    return dates, rnds

def compute_average_rnd(rnds):
    mean_q = np.mean(rnds, axis=0)
    mass = np.trapezoid(mean_q, R_GRID)
    if mass > 0:
        mean_q /= mass
    return mean_q

def bootstrap_ep_inference(spot, q_by_venue, estimator="almeida", B=EP_BOOT_B, block_length=EP_BOOT_BLOCK, seed=EP_BOOT_SEED):
    R_data = compute_overlapping_returns(spot, horizon=27)
    n = len(R_data)
    est_fn = (estimate_physical_density_almeida_from_returns if estimator == "almeida"
              else estimate_physical_density_kde_from_returns)
    venues = list(q_by_venue.keys())

    def stat_fn(idx):
        R_b = np.asarray(R_data)[idx]
        p_b = est_fn(R_b, R_GRID).p_R
        out = [float(np.mean(R_b) - 1.0)]
        for v in venues:
            d = compute_ep_decomposition(R_GRID, q_by_venue[v], p_b, venue=v)
            c = compute_ep_contributions(d)
            out += [d.total_ep,
                    c["downside"]["contribution"],
                    c["mid"]["contribution"],
                    c["upside"]["contribution"]]
        return np.array(out)

    res = block_bootstrap_statistic(n, stat_fn, block_length=block_length,
                                    B=B, seed=seed)
    draws = res["draws"]

    rows = [{
        "estimator": "raw_moment", "venue": "ALL", "statistic": "total_ep",
        "point": res["point"][0],
        "ci_lo": res["lo"][0], "ci_hi": res["hi"][0],
        "se_boot": res["se_boot"][0],
        "B_effective": res["n_success"], "n_excluded": res["n_fail"],
        "block_length": block_length, "n_returns": n,
    }]
    region_names = ["downside", "mid", "upside"]
    for vi, v in enumerate(venues):
        base = 1 + vi * 4
        # Levels: total EP + regional contributions
        for j, stat in enumerate(["total_ep"] + [f"{r}_contrib" for r in region_names]):
            rows.append({
                "estimator": estimator, "venue": v, "statistic": stat,
                "point": res["point"][base + j],
                "ci_lo": res["lo"][base + j], "ci_hi": res["hi"][base + j],
                "se_boot": res["se_boot"][base + j],
                "B_effective": res["n_success"], "n_excluded": res["n_fail"],
                "block_length": block_length, "n_returns": n,
            })
        # Shares: derived from the level draws, excluding near-zero totals
        tot = draws[:, base]
        ok = np.abs(tot) > 1e-8
        for j, r in enumerate(region_names):
            sh = draws[ok, base + 1 + j] / tot[ok]
            pt_tot = res["point"][base]
            pt = (res["point"][base + 1 + j] / pt_tot) if abs(pt_tot) > 1e-10 else np.nan
            rows.append({
                "estimator": estimator, "venue": v, "statistic": f"{r}_share",
                "point": pt,
                "ci_lo": np.quantile(sh, 0.025), "ci_hi": np.quantile(sh, 0.975),
                "se_boot": sh.std(ddof=1),
                "B_effective": int(ok.sum()),
                "n_excluded": res["n_fail"] + int((~ok).sum()),
                "block_length": block_length, "n_returns": n,
            })
    return pd.DataFrame(rows)

def run_phase2():
    print("\n" + "=" * 60)
    print("  Phase 2: Beason-Schreindorfer EP Decomposition")
    print("=" * 60)

    # Load inputs
    print("\n  Loading BTC spot prices...")
    spot = load_spot_prices()
    print(f"  Spot series: {len(spot)} days")

    print("\n  Loading daily RNDs from parquet...")
    cme_dates, cme_rnds = load_daily_rnds_from_parquet("CME", tau_days=27)
    der_dates, der_rnds = load_daily_rnds_from_parquet("DER", tau_days=27)

    q_cme = compute_average_rnd(cme_rnds)
    q_der = compute_average_rnd(der_rnds)

    # Estimate physical density (both methods)
    print("\n  Estimating physical density (Almeida et al.)...")
    p_almeida = estimate_physical_density_almeida(spot, R_GRID, horizon=27)
    print(f"    n_returns = {p_almeida.n_returns}, "
          f"integral = {np.trapezoid(p_almeida.p_R, R_GRID):.4f}")

    print("  Estimating physical density (KDE)...")
    p_kde = estimate_physical_density_kde(spot, R_GRID, horizon=27)
    print(f"    n_returns = {p_kde.n_returns}, bandwidth = {p_kde.bandwidth:.4f}, "
          f"integral = {np.trapezoid(p_kde.p_R, R_GRID):.4f}")

    # EP decomposition: 2 venues x 2 estimators
    results = {}
    for est_name, p_est in [("almeida", p_almeida), ("kde", p_kde)]:
        for venue, q_R in [("CME", q_cme), ("DER", q_der)]:
            key = f"{venue}_{est_name}"
            print(f"\n  Computing EP decomposition: {key}")
            decomp = compute_ep_decomposition(R_GRID, q_R, p_est.p_R, venue=venue)
            contribs = compute_ep_contributions(decomp)
            results[key] = {"decomp": decomp, "contribs": contribs}
            print(f"    Total EP = {decomp.total_ep:.4f}")
            for region, c in contribs.items():
                print(f"    {region:>10s}: {c['contribution']:+.4f} "
                      f"({c['share']:+.1%})")

    # Save
    summary_rows = []
    for key, res in results.items():
        venue, est = key.split("_")
        d = res["decomp"]
        c = res["contribs"]
        summary_rows.append({
            "venue": venue, "estimator": est,
            "total_ep": d.total_ep,
            "downside_contrib": c["downside"]["contribution"],
            "downside_share": c["downside"]["share"],
            "mid_contrib": c["mid"]["contribution"],
            "mid_share": c["mid"]["share"],
            "upside_contrib": c["upside"]["contribution"],
            "upside_share": c["upside"]["share"],
        })
    pd.DataFrame(summary_rows).to_csv(TAB_DIR / "ep_decomposition_summary.csv", index=False)

    # Block-bootstrap inference for total EP and regional contributions. The raw-moment total EP is also
    print(f"\n  Block-bootstrap EP inference "
          f"(B={EP_BOOT_B}, block={EP_BOOT_BLOCK} days, "
          f"~{p_almeida.n_returns // 27} effectively independent return obs)...")
    boot_tables = []
    q_by_venue = {"CME": q_cme, "DER": q_der}
    for est in ["almeida", "kde"]:
        bt = bootstrap_ep_inference(spot, q_by_venue, estimator=est)
        boot_tables.append(bt)
        for v in q_by_venue:
            r = bt[(bt["venue"] == v) & (bt["statistic"] == "total_ep")].iloc[0]
            print(f"    [{est:>7s}] {v}: total EP = {r['point']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                  f"(se = {r['se_boot']:.4f}, B_eff = {int(r['B_effective'])})")
    boot_df = pd.concat(boot_tables, ignore_index=True)

    boot_df = boot_df.drop_duplicates(subset=["estimator", "venue", "statistic"])
    boot_df.to_csv(TAB_DIR / "ep_bootstrap_ci.csv", index=False)
    raw = boot_df[boot_df["estimator"] == "raw_moment"].iloc[0]
    print(f"    [raw moment] ALL: total EP = {raw['point']:+.4f} "
          f"[{raw['ci_lo']:+.4f}, {raw['ci_hi']:+.4f}] "
          f"(no density estimation; anchors the estimator-distortion gap)")
    print(f"  Saved: {TAB_DIR / 'ep_bootstrap_ci.csv'}")

    np.savez(
        PHASE2_DIR / "phase2_densities.npz",
        R_grid=R_GRID,
        p_almeida=p_almeida.p_R,
        p_kde=p_kde.p_R,
        q_cme=q_cme,
        q_der=q_der,
    )

    # Figures
    _plot_densities(p_almeida, p_kde, q_cme, q_der)
    _plot_ep_curves(results)
    _plot_cep_curves(results)
    _plot_kernels(results)

    print(f"\n  Phase 2 complete. Figures in {FIG_DIR}/")
    return results

def _plot_densities(p_almeida, p_kde, q_cme, q_der):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (venue, q_R, color) in zip(axes,
            [("CME", q_cme, "C0"), ("Deribit", q_der, "C1")]):
        ax.plot(R_GRID, p_almeida.p_R, "k-", lw=1.5, label=r"$\hat{p}(R)$ (Almeida)")
        ax.plot(R_GRID, p_kde.p_R, "k--", lw=1.0, alpha=0.5, label=r"$\hat{p}(R)$ (KDE)")
        ax.plot(R_GRID, q_R, color=color, lw=1.5,
                label=rf"$\hat{{q}}^{{\mathrm{{{venue}}}}}(R)$")
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("Gross return $R$")
        ax.set_xlim(0.50, 1.60)
        ax.set_title(venue)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Density")
    fig.suptitle("Physical vs Risk-Neutral Densities (27-day horizon)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_densities_overlay.png", dpi=150)
    plt.close()

def _plot_ep_curves(results):
    fig, ax = plt.subplots(figsize=(12, 5))
    d_cme = results["CME_almeida"]["decomp"]
    d_der = results["DER_almeida"]["decomp"]
    ax.plot(R_GRID, d_cme.ep, "C0-", lw=1.5, label="CME")
    ax.plot(R_GRID, d_der.ep, "C1-", lw=1.5, label="Deribit")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.axvspan(R_GRID[0], 0.90, alpha=0.05, color="red")
    ax.axvspan(1.10, R_GRID[-1], alpha=0.05, color="green")
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"$\mathrm{ep}^j(R)$")
    ax.set_xlim(0.50, 1.60)
    ax.set_title("Equity Premium Curve: CME vs Deribit (27-day, Almeida estimator)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_ep_curve.png", dpi=150)
    plt.close()

def _plot_cep_curves(results):
    fig, ax = plt.subplots(figsize=(12, 5))
    d_cme = results["CME_almeida"]["decomp"]
    d_der = results["DER_almeida"]["decomp"]
    ax.plot(R_GRID, d_cme.cep, "C0-", lw=1.5, label="CME")
    ax.plot(R_GRID, d_der.cep, "C1-", lw=1.5, label="Deribit")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"$\mathrm{CEP}^j(R)$")
    ax.set_xlim(0.50, 1.60)
    ax.set_title("Cumulative Equity Premium: CME vs Deribit (27-day, Almeida estimator)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_cep_curve.png", dpi=150)
    plt.close()

def _plot_kernels(results):
    fig, ax = plt.subplots(figsize=(12, 5))
    d_cme = results["CME_almeida"]["decomp"]
    d_der = results["DER_almeida"]["decomp"]
    ax.plot(R_GRID, d_cme.kernel, "C0-", lw=1.5, label="CME")
    ax.plot(R_GRID, d_der.kernel, "C1-", lw=1.5, label="Deribit")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"$\hat{m}^j(R)$")
    ax.set_xlim(0.50, 1.60)
    ax.set_ylim(0, 5)
    ax.set_title("Unconditional Pricing Kernel: CME vs Deribit (27-day, Almeida estimator)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_kernel_unconditional.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_phase2()
    