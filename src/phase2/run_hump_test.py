"""
Formal Significance Test for the Pricing-Kernel Hump 

The central cross-venue claim -- the hump-shaped kernel appears on BOTH venues -- currently rests on point curves. 
This script upgrades it to a formal finding via a joint block bootstrap: per replicate, trading days
are circular-block resampled (rebuilding q-bar per venue) and the overlapping 27d returns are independently circular-block 
resampled (refitting the v8 enhanced physical density), giving B draws of m(R) = q-bar(R) / p-hat(R). 
The hump statistic per venue is

    H = max_{R in PEAK} m(R) - min_{R in TROUGH} m(R),  TROUGH left of PEAK,

with PEAK = [0.90, 1.10] and TROUGH = [0.70, 0.95] (the trough is searched strictly left of the realized peak). 
Monotonicity implies H <= 0; P_boot(H <= 0) is the one-sided p-value. Pointwise 95% bands for
m(R) are saved for the figure.

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import get_path, get_return_grid
from src.phase2.physical_density import (
    estimate_physical_density_almeida_from_returns, compute_overlapping_returns)
from src.phase2.run_phase2 import (load_spot_prices,
                                   load_daily_rnds_from_parquet)

R_GRID = get_return_grid()
B = 500
BLOCK_DAYS = 27
BLOCK_RET = 54
SEED = 42
PEAK_WIN = (R_GRID >= 0.90) & (R_GRID <= 1.10)
TROUGH_LO = 0.70

def _circular_blocks(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def hump_statistic(m):
    """H = m(peak in PEAK_WIN) - m(trough left of the peak, R >= 0.70)."""
    m = np.asarray(m, dtype=float)
    i_peak_local = int(np.nanargmax(np.where(PEAK_WIN, m, -np.inf)))
    trough_mask = (R_GRID >= TROUGH_LO) & (R_GRID < R_GRID[i_peak_local])
    if trough_mask.sum() == 0:
        return np.nan, np.nan, np.nan
    i_trough_local = int(np.nanargmin(np.where(trough_mask, m, np.inf)))
    return (m[i_peak_local] - m[i_trough_local],
            float(R_GRID[i_peak_local]), float(R_GRID[i_trough_local]))


def run_hump_test(B=B):
    RES_P2 = get_path("results_phase2")
    TAB = RES_P2 / "tables"
    FIG = RES_P2 / "figures"
    DATA = Path(get_path("data_phase2"))
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  Kernel Hump Significance Test (joint block bootstrap, B={B})")
    print("=" * 60)

    spot = load_spot_prices()
    R_data = compute_overlapping_returns(spot, horizon=27)
    rnds = {}
    for venue in ["CME", "DER"]:
        _, r = load_daily_rnds_from_parquet(venue, tau_days=27)
        rnds[venue] = np.stack(r)

    rng = np.random.default_rng(SEED)
    draws_m = {v: np.empty((B, len(R_GRID))) for v in rnds}
    draws_H = {v: np.empty(B) for v in rnds}
    n_fail = 0
    for b in range(B):
        idx_ret = _circular_blocks(len(R_data), BLOCK_RET, rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_b = estimate_physical_density_almeida_from_returns(
                    R_data[idx_ret], R_GRID).p_R
        except Exception:
            n_fail += 1
            for v in rnds:
                draws_m[v][b] = np.nan
                draws_H[v][b] = np.nan
            continue
        p_safe = np.maximum(p_b, 1e-15)
        for v, Q in rnds.items():
            idx_d = _circular_blocks(len(Q), BLOCK_DAYS, rng)
            qbar = Q[idx_d].mean(axis=0)
            qbar /= np.trapezoid(qbar, R_GRID)
            m = qbar / p_safe
            draws_m[v][b] = m
            draws_H[v][b], _, _ = hump_statistic(m)
        if (b + 1) % 100 == 0:
            print(f"    {b + 1}/{B} replicates")

    rows, bands = [], {}
    for v, Q in rnds.items():
        qbar0 = Q.mean(axis=0)
        qbar0 /= np.trapezoid(qbar0, R_GRID)
        p0 = estimate_physical_density_almeida_from_returns(
            R_data, R_GRID).p_R
        m0 = qbar0 / np.maximum(p0, 1e-15)
        H0, Rp, Rt = hump_statistic(m0)
        Hd = draws_H[v][np.isfinite(draws_H[v])]
        rows.append({
            "venue": v, "H_point": H0, "peak_R": Rp, "trough_R": Rt,
            "H_boot_mean": float(Hd.mean()),
            "ci_lo": float(np.quantile(Hd, 0.025)),
            "ci_hi": float(np.quantile(Hd, 0.975)),
            "p_monotone": float((Hd <= 0).mean()),
            "B_effective": int(len(Hd)), "n_fail": n_fail,
            "block_days": BLOCK_DAYS, "block_returns": BLOCK_RET,
        })
        M = draws_m[v][np.isfinite(draws_m[v]).all(axis=1)]
        bands[f"m_{v}"] = m0
        bands[f"lo_{v}"] = np.quantile(M, 0.025, axis=0)
        bands[f"hi_{v}"] = np.quantile(M, 0.975, axis=0)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TAB / "hump_test.csv", index=False)
    np.savez(DATA / "hump_bands.npz", R_grid=R_GRID, **bands)
    print("\n" + tbl.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, v in zip(axes, ["CME", "DER"]):
        ax.plot(R_GRID, bands[f"m_{v}"], "k-", lw=1.5)
        ax.fill_between(R_GRID, bands[f"lo_{v}"], bands[f"hi_{v}"],
                        alpha=0.2, color="gray")
        ax.axhline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_xlim(0.5, 1.6)
        ax.set_ylim(0, 5)
        ax.set_xlabel("Gross return $R$")
        r = tbl[tbl.venue == v].iloc[0]
        ax.set_title(f"{'Deribit' if v == 'DER' else v}: "
                     f"H = {r['H_point']:.3f}, "
                     f"p(monotone) = {r['p_monotone']:.3f}")
    axes[0].set_ylabel(r"$\hat{m}(R)$")
    fig.suptitle("Unconditional Kernel with 95% Joint-Bootstrap Bands")
    plt.tight_layout()
    plt.savefig(FIG / "fig_kernel_hump_bands.png", dpi=150)
    plt.close()
    print(f"  Saved: {TAB / 'hump_test.csv'}")
    return tbl


if __name__ == "__main__":
    run_hump_test()