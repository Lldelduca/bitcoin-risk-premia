"""
Paired Bootstrap CI for the Enhanced - Vanilla EP Difference

Completes the benchmark story with formal inference: per circular-block bootstrap replicate (same resampled return sample), 
BOTH the enhanced and the AGMW vanilla estimators are refit and the difference in total EP is recorded per venue. 

The claim it supports: "statistically indistinguishable in level, distinguishable in NB-stability."

"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import get_path, get_return_grid
from src.phase2.physical_density import (
    estimate_physical_density_almeida_from_returns,
    estimate_physical_density_grith_vanilla_from_returns,
    compute_overlapping_returns)
from src.phase2.run_phase2 import (load_spot_prices,
                                   load_daily_rnds_from_parquet,
                                   compute_average_rnd)

R_GRID = get_return_grid()
B = 500
BLOCK = 54
SEED = 42

def _circular_blocks(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]

def run_ep_diff_ci(B=B):
    TAB = get_path("results_phase2") / "tables"
    TAB.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  Paired bootstrap: enhanced - vanilla total EP (B={B})")
    print("=" * 60)

    spot = load_spot_prices()
    R_data = compute_overlapping_returns(spot, horizon=27)
    q = {}
    for venue in ["CME", "DER"]:
        _, r = load_daily_rnds_from_parquet(venue, tau_days=27)
        q[venue] = compute_average_rnd(r)

    rng = np.random.default_rng(SEED)
    diffs = {v: [] for v in q}
    n_fail = 0
    for b in range(B):
        idx = _circular_blocks(len(R_data), BLOCK, rng)
        Rb = np.asarray(R_data)[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_e = estimate_physical_density_almeida_from_returns(
                    Rb, R_GRID).p_R
                p_v = estimate_physical_density_grith_vanilla_from_returns(
                    Rb, R_GRID).p_R
        except Exception:
            n_fail += 1
            continue
        for v, qv in q.items():
            te = float(np.trapezoid((p_e - qv) * R_GRID, R_GRID))
            tv = float(np.trapezoid((p_v - qv) * R_GRID, R_GRID))
            diffs[v].append(te - tv)
        if (b + 1) % 100 == 0:
            print(f"    {b + 1}/{B} replicates")

    rows = []
    for v, d in diffs.items():
        d = np.asarray(d)
        p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
        rows.append({
            "venue": v, "mean_diff": float(d.mean()),
            "ci_lo": float(np.quantile(d, 0.025)),
            "ci_hi": float(np.quantile(d, 0.975)),
            "se_boot": float(d.std(ddof=1)),
            "p_two_sided": float(p_two),
            "B_effective": int(len(d)), "n_fail": n_fail,
            "block_length": BLOCK,
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TAB / "ep_diff_ci.csv", index=False)
    print("\n" + tbl.round(5).to_string(index=False))
    print(f"  Saved: {TAB / 'ep_diff_ci.csv'}")
    return tbl


if __name__ == "__main__":
    run_ep_diff_ci()