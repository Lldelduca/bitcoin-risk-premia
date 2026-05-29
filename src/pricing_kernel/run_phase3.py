"""
Phase 3 Orchestrator: Conditional Pricing Kernel Estimation.

Runs Schreindorfer-Sichert estimation for 2 venues x 3 specs = 6 runs. Also computes unconditional MFK.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.config import get_path, SAMPLE

from src.pricing_kernel.conditional_kernel import (
    estimate_conditional_kernel,
    evaluate_kernel_at_terciles,
    get_coefficient_timeseries,
)

CLEAN_DIR = Path(get_path("cleaned_cme")).parent
SURFACES_DIR = CLEAN_DIR.parent / "surfaces"
PHASE2_DIR = CLEAN_DIR.parent / "phase2"
PHASE3_DIR = CLEAN_DIR.parent / "phase3"
COND_DIR = CLEAN_DIR.parent / "conditioning"
FIG_DIR = Path("results") / "phase3" / "figures"
TAB_DIR = Path("results") / "phase3" / "tables"
for d in [PHASE3_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

R_GRID = np.linspace(0.40, 2.00, 1000)

SPECS = {
    "macro": "Z_macro.parquet",
    "crypto": "Z_crypto.parquet",
    "full": "Z_full.parquet",
}
VENUES = ["CME", "DER"]

MAX_ITER = 20000
N_WORKERS = 6

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

def load_daily_rnds_from_parquet(venue, tau_days=27):
    density_path = SURFACES_DIR / f"rnd_{venue}_densities.parquet"
    df = pd.read_parquet(density_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["tau_days"] == tau_days].sort_values("date").reset_index(drop=True)

    dates, rnds = [], []
    for _, row in df.iterrows():
        R = np.array(row["returns"])
        q = np.array(row["density"])
        q_interp = np.interp(R_GRID, R, q, left=0, right=0)
        mass = np.trapezoid(q_interp, R_GRID)
        if mass > 0:
            q_interp /= mass
        dates.append(row["date"])
        rnds.append(q_interp)

    print(f"  [{venue}] Loaded {len(rnds)} daily RNDs at tau={tau_days}d")
    return dates, rnds

def load_conditioning_spec(spec_name):
    df = pd.read_parquet(COND_DIR / SPECS[spec_name])
    df["date"] = pd.to_datetime(df["date"])
    z_cols = [c for c in df.columns if c != "date" and not c.endswith("_raw")]
    return df["date"].values, df[z_cols].values, z_cols

def align_rnds_and_Z(rnd_dates, rnds, z_dates, Z_matrix):
    rnd_df = pd.DataFrame({"date": rnd_dates, "idx": range(len(rnd_dates))})
    z_df = pd.DataFrame({"date": z_dates, "z_idx": range(len(z_dates))})
    merged = rnd_df.merge(z_df, on="date", how="inner").sort_values("date")
    dates = merged["date"].values
    aligned_rnds = [rnds[i] for i in merged["idx"].values]
    aligned_Z = Z_matrix[merged["z_idx"].values]
    return dates, aligned_rnds, aligned_Z

def _run_single_estimation(job):
    venue = job["venue"]
    spec_name = job["spec_name"]
    R_grid = job["R_grid"]
    q_obs_list = job["q_obs_list"]
    p_phys = job["p_phys"]
    Z_matrix = job["Z_matrix"]
    dates = job["dates"]
    z_cols = job["z_cols"]

    result = estimate_conditional_kernel(
        R_grid, q_obs_list, p_phys, Z_matrix,
        venue=venue, spec_name=spec_name,
        max_iter=MAX_ITER, theta0=None,
    )

    terciles = evaluate_kernel_at_terciles(result, R_grid, Z_matrix)
    coeffs = get_coefficient_timeseries(result, Z_matrix)

    return {
        "key": f"{venue}_{spec_name}",
        "venue": venue,
        "spec_name": spec_name,
        "result": result,
        "terciles": terciles,
        "coeffs": coeffs,
        "dates": dates,
        "Z": Z_matrix,
        "z_cols": z_cols,
    }

def run_phase3():
    print("\n" + "=" * 60)
    print("  Phase 3: Conditional Pricing Kernel Estimation")
    print("=" * 60)
    print(f"  max_iter = {MAX_ITER}, workers = {N_WORKERS}")

    # Load physical density from Phase 2
    print("\n  Loading Phase 2 physical density...")
    p_data = np.load(PHASE2_DIR / "phase2_densities.npz")
    p_phys = p_data["p_almeida"]
    print(f"  Physical density integral = {np.trapezoid(p_phys, R_GRID):.4f}")

    # Load daily RNDs per venue
    print("\n  Loading daily RNDs...")
    venue_rnds = {}
    for venue in VENUES:
        venue_rnds[venue] = load_daily_rnds_from_parquet(venue, tau_days=27)

    # Pre-load conditioning specs
    spec_data = {}
    for spec_name in SPECS:
        z_dates, Z_matrix, z_cols = load_conditioning_spec(spec_name)
        spec_data[spec_name] = {"dates": z_dates, "Z": Z_matrix, "cols": z_cols}
        print(f"  Loaded {spec_name}: Z shape {Z_matrix.shape}, cols {z_cols}")

    # Build all 6 jobs
    print("\n  Building estimation jobs...")
    jobs = []
    for spec_name in SPECS:
        sd = spec_data[spec_name]
        for venue in VENUES:
            rnd_dates, rnds = venue_rnds[venue]
            dates, aligned_rnds, aligned_Z = align_rnds_and_Z(
                rnd_dates, rnds, sd["dates"], sd["Z"]
            )
            print(f"    [{venue}|{spec_name}] Aligned: {len(dates)} days, "
                  f"n_Z={aligned_Z.shape[1]}")

            if len(dates) < 50:
                print(f"    [WARN] Too few days, skipping.")
                continue

            jobs.append({
                "venue": venue,
                "spec_name": spec_name,
                "R_grid": R_GRID,
                "q_obs_list": aligned_rnds,
                "p_phys": p_phys,
                "Z_matrix": aligned_Z,
                "dates": dates,
                "z_cols": sd["cols"],
            })

    # Run all jobs in parallel
    print(f"\n  Launching {len(jobs)} estimations across {N_WORKERS} workers...")
    all_results = {}
    summary_rows = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        future_to_key = {
            pool.submit(_run_single_estimation, job): f"{job['venue']}_{job['spec_name']}"
            for job in jobs
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                out = future.result()
                all_results[key] = out

                result = out["result"]
                terciles = out["terciles"]
                coeffs = out["coeffs"]
                venue = out["venue"]
                spec_name = out["spec_name"]
                dates = out["dates"]

                print(f"\n  [{key}] Converged: {result.converged}, "
                      f"KL mean={result.kl_mean:.6f}")

                # Summary row
                row = {
                    "venue": venue, "spec": spec_name,
                    "n_days": result.n_days, "n_params": result.n_params,
                    "kl_total": result.kl_total, "kl_mean": result.kl_mean,
                    "converged": result.converged,
                }
                for cn in ["a", "b", "c", "d"]:
                    row[f"mean_{cn}"] = coeffs[cn].mean()
                    row[f"std_{cn}"] = coeffs[cn].std()
                for tn in ["low", "mid", "high"]:
                    if tn in terciles:
                        row[f"c_{tn}"] = terciles[tn]["c"]
                summary_rows.append(row)

                # Save per-estimation outputs
                np.savez(
                    PHASE3_DIR / f"phase3_{key}.npz",
                    theta=result.theta, dates=dates,
                    coeffs_a=coeffs["a"], coeffs_b=coeffs["b"],
                    coeffs_c=coeffs["c"], coeffs_d=coeffs["d"],
                )

            except Exception as e:
                print(f"\n  [{key}] FAILED: {e}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    spec_order = {"crypto": 0, "macro": 1, "full": 2}
    summary_df["_order"] = summary_df["spec"].map(spec_order)
    summary_df = summary_df.sort_values(["venue", "_order"]).drop(columns="_order")
    summary_df.to_csv(TAB_DIR / "phase3_summary.csv", index=False)
    print(f"\n  Saved summary: {TAB_DIR / 'phase3_summary.csv'}")

    # Generate figures sequentially (matplotlib is not thread-safe)
    print("\n  Generating figures...")
    for key, out in all_results.items():
        _plot_kernel_terciles(out["result"], out["terciles"],
                              out["venue"], out["spec_name"])
        _plot_coefficient_timeseries(out["coeffs"], out["dates"],
                                     out["venue"], out["spec_name"])

    # Unconditional MFK
    _compute_and_plot_mfk(venue_rnds)

    print(f"\n  Phase 3 complete. Figures in {FIG_DIR}/")
    return all_results

def _plot_kernel_terciles(result, terciles, venue, spec_name):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"low": "C2", "mid": "C7", "high": "C3"}
    for name in ["low", "mid", "high"]:
        if name not in terciles:
            continue
        t = terciles[name]
        ax.plot(R_GRID, t["kernel"], color=colors[name], lw=1.5,
                label=f"{name}-vol (c={t['c']:.3f}, n={t['n_days']})")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"$\hat{m}^j(R \mid Z_t)$")
    ax.set_xlim(0.50, 1.60)
    ax.set_ylim(0, 5)
    ax.set_title(f"Conditional Kernel by Vol Tercile: {venue}, $Z^{{({spec_name})}}$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"fig_kernel_terciles_{venue}_{spec_name}.png", dpi=150)
    plt.close()

def _plot_coefficient_timeseries(coeffs, dates, venue, spec_name):
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    labels = [("a", "$a_t$ (level)"), ("b", "$b_t$ (slope)"),
              ("c", "$c_t$ (curvature)"), ("d", "$d_t$ (cubic)")]
    for ax, (name, label) in zip(axes, labels):
        ax.plot(dates, coeffs[name], lw=0.8, color="C0")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel(label)
    axes[0].set_title(f"Kernel Coefficients: {venue}, $Z^{{({spec_name})}}$")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"fig_coeffs_ts_{venue}_{spec_name}.png", dpi=150)
    plt.close()

def _compute_and_plot_mfk(venue_rnds):
    """Unconditional MFK: Psi(R) = log(q^CME / q^DER) averaged over matched days."""
    print("\n  Computing unconditional MFK...")
    cme_dates, cme_rnds = venue_rnds["CME"]
    der_dates, der_rnds = venue_rnds["DER"]

    cme_df = pd.DataFrame({"date": cme_dates, "idx": range(len(cme_dates))})
    der_df = pd.DataFrame({"date": der_dates, "idx": range(len(der_dates))})
    merged = cme_df.merge(der_df, on="date", how="inner", suffixes=("_c", "_d"))

    mfk_daily = []
    for _, row in merged.iterrows():
        q_c = np.maximum(cme_rnds[row["idx_c"]], 1e-20)
        q_d = np.maximum(der_rnds[row["idx_d"]], 1e-20)
        mfk_daily.append(np.log(q_c / q_d))

    mfk_mean = np.mean(mfk_daily, axis=0)
    mfk_std = np.std(mfk_daily, axis=0)
    n = len(mfk_daily)
    print(f"  MFK over {n} matched days.")

    np.savez(PHASE3_DIR / "mfk_unconditional.npz",
             R_grid=R_GRID, mfk_mean=mfk_mean, mfk_std=mfk_std, n_days=n)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(R_GRID, mfk_mean, "k-", lw=1.5, label=r"$\bar{\Psi}(R)$")
    ax.fill_between(R_GRID,
                     mfk_mean - 1.96 * mfk_std / np.sqrt(n),
                     mfk_mean + 1.96 * mfk_std / np.sqrt(n),
                     alpha=0.2, color="gray", label="95% CI")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.axvspan(R_GRID[0], 0.90, alpha=0.05, color="red")
    ax.axvspan(1.10, R_GRID[-1], alpha=0.05, color="green")
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"$\Psi(R)=\log(\hat{q}^{\mathrm{CME}}/\hat{q}^{\mathrm{DER}})$")
    ax.set_xlim(0.50, 1.60)
    ax.set_title("Unconditional Microstructure Friction Kernel")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_mfk_unconditional.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_phase3()
