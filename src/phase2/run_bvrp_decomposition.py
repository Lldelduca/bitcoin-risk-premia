"""
BVRP State-Space Decomposition

Decomposes the Bitcoin variance risk premium across return states, mirroring the BVRP(r) extension in Grith's paper 

    BVRP        = Var_Q(R) - Var_P(R) = sigma^2_Q - sigma^2_P
    bvrp(x)     = (x - mu_Q)^2 q(x) - (x - mu_P)^2 p(x)      [integrand]
    BVRP(r)     = INT_{x_lo}^{r} bvrp(x) dx                   [cum. level]

computed in net-return units x = R - 1 on the analysis grid (variance is shift-invariant, so gross- and net-return variances 
coincide; means are converted). Regional contributions use the thesis convention: downside R < 0.90, mid 0.90-1.10, upside 
R > 1.10.

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import get_path, get_return_grid
from src.phase2.physical_density import (
    estimate_physical_density_almeida_from_returns,
    compute_overlapping_returns)
from src.phase2.run_phase2 import (load_spot_prices,
                                   load_daily_rnds_from_parquet,
                                   compute_average_rnd)

R_GRID = get_return_grid()
R_PLOT = np.arange(R_GRID[0], R_GRID[-1] + 0.001, 0.01)
B = 500
BLOCK_DAYS = 27
BLOCK_RET = 54
SEED = 42
REGIONS = [("downside", R_GRID < 0.90),
           ("mid", (R_GRID >= 0.90) & (R_GRID <= 1.10)),
           ("upside", R_GRID > 1.10)]


def compute_bvrp(q, p, R=R_GRID):
    from scipy.integrate import cumulative_trapezoid
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    mu_Q = np.trapezoid(q * R, R)
    mu_P = np.trapezoid(p * R, R)
    s2_Q = np.trapezoid(q * (R - mu_Q) ** 2, R)
    s2_P = np.trapezoid(p * (R - mu_P) ** 2, R)
    integrand = (R - mu_Q) ** 2 * q - (R - mu_P) ** 2 * p
    total = s2_Q - s2_P
    cum_level = np.concatenate([[0.0], cumulative_trapezoid(integrand, R)])
    cum_share = cum_level / total if abs(total) > 1e-14 else np.full_like(cum_level, np.nan)
    out = {"mu_Q": mu_Q, "mu_P": mu_P, "sigma2_Q": s2_Q, "sigma2_P": s2_P,
           "total": total, "integrand": integrand,
           "cum_level": cum_level, "cum_share": cum_share}

    cum_at = lambda x: float(np.interp(x, R, cum_level))
    lo, hi = 0.90, 1.10
    out["downside_contrib"] = cum_at(lo) - cum_level[0]
    out["mid_contrib"] = cum_at(hi) - cum_at(lo)
    out["upside_contrib"] = float(cum_level[-1]) - cum_at(hi)
    return out

def _circular_blocks(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]

def run_bvrp_decomposition(B=B):
    RES_P2 = get_path("results_phase2")
    TAB = RES_P2 / "tables"
    FIG = RES_P2 / "figures"
    DATA = get_path("data_phase2")
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  BVRP State-Space Decomposition (Grith slide extension)")
    print("=" * 60)

    spot = load_spot_prices()
    R_data = compute_overlapping_returns(spot, horizon=27)
    dens = np.load(DATA / "phase2_densities.npz")
    p_by_est = {"almeida": dens["p_almeida"],
                "vanilla": dens["p_vanilla"],
                "kde": dens["p_kde"]}
    rnds, qbar = {}, {}
    for venue in ["CME", "DER"]:
        _, r = load_daily_rnds_from_parquet(venue, tau_days=27)
        rnds[venue] = np.stack(r)
        qbar[venue] = compute_average_rnd(list(rnds[venue]))

    rows, curves = [], {}
    for venue in ["CME", "DER"]:
        for est, p in p_by_est.items():
            b = compute_bvrp(qbar[venue], p)
            share = {f"{n}_share": (b[f"{n}_contrib"] / b["total"]
                                    if abs(b["total"]) > 1e-12 else np.nan)
                     for n, _ in REGIONS}
            rows.append({"venue": venue, "estimator": est,
                         "sigma2_Q": b["sigma2_Q"],
                         "sigma2_P": b["sigma2_P"],
                         "total": b["total"],
                         **{f"{n}_contrib": b[f"{n}_contrib"]
                            for n, _ in REGIONS},
                         **share})
            if est == "almeida":
                curves[venue] = b

    print(f"\n  Joint block bootstrap for the BVRP total (B={B})...")
    rng = np.random.default_rng(SEED)
    draws = {v: [] for v in rnds}
    n_fail = 0
    for i in range(B):
        idx_ret = _circular_blocks(len(R_data), BLOCK_RET, rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_b = estimate_physical_density_almeida_from_returns(
                    np.asarray(R_data)[idx_ret], R_GRID).p_R
        except Exception:
            n_fail += 1
            continue
        for v, Q in rnds.items():
            idx_d = _circular_blocks(len(Q), BLOCK_DAYS, rng)
            qb = Q[idx_d].mean(axis=0)
            qb /= np.trapezoid(qb, R_GRID)
            draws[v].append(compute_bvrp(qb, p_b)["total"])
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{B} replicates")

    ci = {}
    for v, d in draws.items():
        d = np.asarray(d)
        ci[v] = {"ci_lo": float(np.quantile(d, 0.025)),
                 "ci_hi": float(np.quantile(d, 0.975)),
                 "se_boot": float(d.std(ddof=1)),
                 "B_effective": int(len(d)), "n_fail": n_fail}
    df = pd.DataFrame(rows)
    for v, c in ci.items():
        m = (df["venue"] == v) & (df["estimator"] == "almeida")
        for k, val in c.items():
            df.loc[m, k] = val
    df.to_csv(TAB / "bvrp_decomposition_summary.csv", index=False)

    print("\n  BVRP totals (sigma2_Q - sigma2_P):")
    for _, r in df[df["estimator"] == "almeida"].iterrows():
        print(f"    {r['venue']}: {r['total']:+.5f} "
              f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]  "
              f"(s2_Q={r['sigma2_Q']:.5f}, s2_P={r['sigma2_P']:.5f})")
    print("\n  Regional contributions (enhanced, x100 pp):")
    for _, r in df[df["estimator"] == "almeida"].iterrows():
        print(f"    {r['venue']}: down {100*r['downside_contrib']:+.3f}, "
              f"mid {100*r['mid_contrib']:+.3f}, "
              f"up {100*r['upside_contrib']:+.3f}")

    try:
        panel = pd.read_parquet(get_path("data_phase4") / "cumulant_premia.parquet")
        vp_cols = [c for c in panel.columns if "var" in c.lower()
                   and "premium" in c.lower()]
        if vp_cols:
            vp = panel.groupby("venue")[vp_cols[0]].mean()
            print(f"\n  Reconciliation: Phase 4 mean variance premium "
                  f"({vp_cols[0]}, quadratic-variation-of-log-returns "
                  f"convention): "
                  + ", ".join(f"{v}={vp[v]:+.5f}" for v in vp.index))
            print("  (Differs from BVRP above by construction: central "
                  "moments of simple returns vs log-return quadratic "
                  "variation.)")
    except Exception:
        pass

    # Save both cum_level and cum_share to npz
    np.savez(DATA / "bvrp_curves.npz", R_grid=R_GRID,
             **{f"{k}_{v}": curves[v][k] for v in curves
                for k in ("integrand", "cum_share", "cum_level")})

    # Figure 1: integrand (main text exhibit — shows WHERE the cancellation happens)
    fig, ax = plt.subplots(figsize=(12, 5))
    for v, c in [("CME", "C0"), ("DER", "C1")]:
        ax.plot(R_PLOT, np.interp(R_PLOT, R_GRID, curves[v]["integrand"]),
                color=c, lw=1.5, label="Deribit" if v == "DER" else v)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.axvspan(R_PLOT[0], 0.90, alpha=0.05, color="red")
    ax.axvspan(1.10, R_PLOT[-1], alpha=0.05, color="green")
    ax.set_xlim(0.5, 1.6)
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"$(x-\mu_Q)^2 q - (x-\mu_P)^2 p$")
    ax.set_title("BVRP State Decomposition: Integrand by Return State")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_bvrp_curve.png", dpi=150)
    plt.close()

    # Figure 2: cumulative LEVEL (not share — near-zero total makes
    # the share swing by hundreds of percent, same pathology as EP shares)
    fig, ax = plt.subplots(figsize=(12, 5))
    for v, c in [("CME", "C0"), ("DER", "C1")]:
        cum_pp = 100 * np.interp(R_PLOT, R_GRID, curves[v]["cum_level"])
        ax.plot(R_PLOT, cum_pp,
                color=c, lw=1.5, label="Deribit" if v == "DER" else v)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.axvspan(R_PLOT[0], 0.90, alpha=0.05, color="red")
    ax.axvspan(1.10, R_PLOT[-1], alpha=0.05, color="green")
    ax.set_xlim(0.5, 1.6)
    ax.set_xlabel("Gross return $R$")
    ax.set_ylabel(r"Cumulative BVRP$(r)$  (pp)")
    ax.set_title("Cumulative BVRP by Return State (level, not share)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_bvrp_cumulative.png", dpi=150)
    plt.close()

    print(f"\n  Saved: {TAB / 'bvrp_decomposition_summary.csv'}")
    return df

if __name__ == "__main__":
    run_bvrp_decomposition()