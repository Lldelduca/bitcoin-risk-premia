"""
Phase 3 Bootstrap: Block-Bootstrap Inference for the Tercile Kernel Coefficients.

Percentile confidence intervals and sign frequencies on (b, c, d) evaluated at the FIXED full-sample tercile-mean states.

Design:
  - Days are resampled with a circular moving-block bootstrap
  - Each replicate is a FULL re-estimation of the conditional kernel, warm-started at the full-sample theta
  - Coefficients are evaluated at the SAME full-sample tercile-mean Z vectors in every replicate,
    so the intervals are intervals for "c in the low-vol state" — parameter uncertainty only, not state-
    definition randomness.
  - The physical density and the tercile-mean states are held fixed at their full-sample values; the dominant
    uncertainty here is theta.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.config import get_path, get_return_grid

from src.phase3.conditional_kernel import (estimate_conditional_kernel, coefficients_at)
from src.phase3.bootstrap_inference import circular_block_indices
from src.phase3.run_phase3 import (load_daily_rnds_from_parquet, load_conditioning_spec, align_rnds_and_Z,
    load_volatility_tercile_labels, PHASE2_DIR, PHASE3_DIR, TAB_DIR, SPECS)

R_GRID = get_return_grid()

BLOCK_LENGTH = 27
DEFAULT_B = 200
MAX_ITER_REPLICATE = 3000

def _tercile_mean_states(Z_matrix, tercile_labels):
    labels = np.asarray(tercile_labels, dtype=object)
    if len(labels) != Z_matrix.shape[0]:
        raise ValueError(
            f"_tercile_mean_states: labels length {len(labels)} != "
            f"T {Z_matrix.shape[0]}")
    out = {}
    for name in ["low", "mid", "high"]:
        mask = labels == name
        if mask.sum() > 0:
            out[name] = Z_matrix[mask].mean(axis=0)
    return out

def _one_replicate(job):
    seed = job["seed"]
    rng = np.random.default_rng(seed)
    Q = job["Q"]                 # (T, G) stacked daily RNDs
    Z = job["Z"]                 # (T, n_Z)
    p_phys = job["p_phys"]
    theta_full = job["theta_full"]
    tercile_states = job["tercile_states"]
    n_Z = Z.shape[1]

    idx = circular_block_indices(Q.shape[0], BLOCK_LENGTH, rng)
    Q_b = Q[idx]
    Z_b = Z[idx]

    res = estimate_conditional_kernel(
        R_GRID, Q_b, p_phys, Z_b,
        venue=job["venue"], spec_name=f"boot{seed}",
        max_iter=MAX_ITER_REPLICATE, theta0=theta_full.copy(), verbose=False,
    )
    row = {"seed": seed, "converged": bool(res.converged),
           "kl_mean": float(res.kl_mean)}
    for name, Z_vec in tercile_states.items():
        b, c, d = coefficients_at(res.theta, Z_vec, n_Z)
        row[f"b_{name}"] = float(b)
        row[f"c_{name}"] = float(c)
        row[f"d_{name}"] = float(d)
    return row

def run_bootstrap(venues, spec_name, B, workers, ci=0.95):
    print("\n" + "=" * 60)
    print("  Phase 3 Bootstrap: Tercile Kernel Coefficient Inference")
    print(f"  spec = {spec_name}, B = {B}, block = {BLOCK_LENGTH}, "
          f"workers = {workers}")
    print("=" * 60)

    # Physical density (fixed at full-sample value)
    p_data = np.load(PHASE2_DIR / "phase2_densities.npz")
    p_phys = p_data["p_almeida"]
    if len(p_phys) != len(R_GRID):
        raise ValueError(
            "phase2_densities.npz grid does not match the shared return grid; "
            "re-run run_phase2 after the grid harmonization."
        )

    z_dates, Z_matrix_full, z_cols = load_conditioning_spec(spec_name)
    tercile_df = load_volatility_tercile_labels()

    summary_rows = []
    for venue in venues:
        print(f"\n  [{venue}] Preparing data...")
        rnd_dates, rnds = load_daily_rnds_from_parquet(venue, tau_days=27)
        dates, aligned_rnds, Z, labels = align_rnds_and_Z(
            rnd_dates, rnds, z_dates, Z_matrix_full, tercile_df=tercile_df
        )
        Q = np.stack(aligned_rnds)
        print(f"  [{venue}] {Q.shape[0]} aligned days, n_Z = {Z.shape[1]}")

        theta_path = PHASE3_DIR / f"phase3_{venue}_{spec_name}.npz"
        if not theta_path.exists():
            print(f"  [{venue}] SKIP: {theta_path.name} not found — "
                  f"run run_phase3 first.")
            continue
        theta_full = np.load(theta_path)["theta"]
        expected = 3 * (1 + Z.shape[1])
        if len(theta_full) != expected:
            print(f"  [{venue}] SKIP: saved theta has length "
                  f"{len(theta_full)}, expected {expected} — the saved run "
                  f"predates the (b, c, d) parameterization; re-run "
                  f"run_phase3 first.")
            continue

        tercile_states = _tercile_mean_states(Z, labels)
        for name, Z_vec in tercile_states.items():
            n_t = int((np.asarray(labels, dtype=object) == name).sum())
            print(f"    tercile '{name}': {n_t} days "
                  f"(full-sample Z_IVS_1 labels)")

        jobs = [{
            "seed": 10_000 + b, "Q": Q, "Z": Z, "p_phys": p_phys,
            "theta_full": theta_full, "tercile_states": tercile_states,
            "venue": venue,
        } for b in range(B)]

        print(f"  [{venue}] Running {B} replicates...")
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one_replicate, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures)):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    print(f"    replicate failed: {e}")
                if (i + 1) % 25 == 0:
                    print(f"    {i+1}/{B} done")

        draws = pd.DataFrame(rows)
        draws.to_parquet(PHASE3_DIR / f"phase3_bootstrap_draws_{venue}_{spec_name}.parquet",
                         index=False)
        n_ok = len(draws)
        n_conv = int(draws["converged"].sum()) if n_ok else 0
        print(f"  [{venue}] {n_ok}/{B} replicates returned, "
              f"{n_conv} flagged converged")

        # Point estimates from the full-sample theta at the same fixed states
        alpha = (1.0 - ci) / 2.0
        for name, Z_vec in tercile_states.items():
            b_pt, c_pt, d_pt = coefficients_at(theta_full, Z_vec, Z.shape[1])
            for coef, pt in [("b", b_pt), ("c", c_pt), ("d", d_pt)]:
                col = f"{coef}_{name}"
                v = draws[col].dropna().values
                summary_rows.append({
                    "venue": venue, "spec": spec_name, "tercile": name,
                    "coef": coef, "point": float(pt),
                    "ci_lo": np.quantile(v, alpha),
                    "ci_hi": np.quantile(v, 1.0 - alpha),
                    "se_boot": v.std(ddof=1),
                    "frac_negative": float((v < 0).mean()),
                    "B_effective": len(v),
                    "block_length": BLOCK_LENGTH,
                })

    summary = pd.DataFrame(summary_rows)
    out_path = TAB_DIR / f"phase3_bootstrap_ci_{spec_name}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    if len(summary):
        c_rows = summary[summary["coef"] == "c"]
        print("\n  Curvature coefficient c by tercile "
              "(point [95% CI], P(c<0)):")
        for _, r in c_rows.iterrows():
            print(f"    {r['venue']:>4s} {r['tercile']:>4s}: "
                  f"{r['point']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"P(c<0) = {r['frac_negative']:.3f}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block-bootstrap CIs for the tercile kernel coefficients."
    )
    parser.add_argument("--B", type=int, default=DEFAULT_B,
                        help=f"Number of bootstrap replicates (default {DEFAULT_B}).")
    parser.add_argument("--venues", nargs="+", default=["CME", "DER"])
    parser.add_argument("--spec", choices=list(SPECS.keys()), default="crypto",
                        help="Conditioning spec (default: crypto, the headline).")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    run_bootstrap(args.venues, args.spec, args.B, args.workers)
    