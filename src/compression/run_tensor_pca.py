"""
CP Tensor Decomposition Orchestrator.

Constructs the 4th-order IVS tensor, selects the rank via CORCONDIA, and runs CP-ALS

"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import get_path, SAMPLE, TENSOR_GRID
from src.compression.tensor_pca import (build_ivs_tensor, standardize_tensor, cp_als, select_rank)

SAMPLE_START = SAMPLE["start_date"]
SAMPLE_END = SAMPLE["end_date"]

MATURITY_GRIDS = {
    "almeida": [9, 27, 45],
    "broad": [14, 21, 27, 35, 45, 60, 90, 120, 180],
}

SURFACES_DIR = Path(get_path("cleaned_cme")).parent.parent / "surfaces"
SURFACES_DIR.mkdir(parents=True, exist_ok=True)

def run_tensor_decomposition(grid_name: str = "almeida"):
    print("\n" + "=" * 60)
    print("  CP Tensor Decomposition Pipeline")
    print("=" * 60)
    print(f"  Sample: {SAMPLE_START} → {SAMPLE_END}")

    # Load fitted SSVI parameters
    params_path = SURFACES_DIR / "ssvi_params.parquet"
    params = pd.read_parquet(params_path)
    params["date"] = pd.to_datetime(params["date"])
    print(f"\n  Loaded {len(params):,} fitted SSVI parameter rows")

    # Load cleaned option data per venue
    cleaned_dfs = {}
    for venue in ["CME", "DER"]:
        path = get_path("cleaned_cme" if venue == "CME" else "cleaned_deribit")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)
        cleaned_dfs[venue] = df[mask].copy()
        print(f"  {venue}: {len(cleaned_dfs[venue]):,} cleaned options")

    # Select maturity grid
    if grid_name not in MATURITY_GRIDS:
        raise ValueError(
            f"Unknown grid '{grid_name}'. Available: {list(MATURITY_GRIDS.keys())}"
        )
    t_grid_days = MATURITY_GRIDS[grid_name]
    print(f"\n  Maturity grid: {grid_name} = {t_grid_days}")

    # Build tensor
    k_grid = np.linspace(TENSOR_GRID["k_grid_min"], TENSOR_GRID["k_grid_max"], TENSOR_GRID["k_grid_points"])

    print(f"\n  Building 4th-order tensor...")
    X, meta = build_ivs_tensor(
        params_df=params,
        cleaned_dfs=cleaned_dfs,
        t_grid_days=t_grid_days,
        k_grid=k_grid,
        venues=["CME", "DER"],
    )

    print(f"  Tensor shape: {X.shape}")
    print(f"  Non-NaN entries: {(~np.isnan(X)).sum():,} / {X.size:,} "
          f"({(~np.isnan(X)).mean():.1%})")

    # Standardize
    print(f"\n  Standardizing tensor along time axis...")
    X_std, mu, sigma = standardize_tensor(X)

    # Rank selection via CORCONDIA
    print(f"\n  Selecting CP rank via CORCONDIA (Bro & Kiers, 2003):")
    print(f"  Checking seed robustness across [42, 7, 100, 2024]...")
    seed_results = {}
    seed_chosen = {}
    for seed in [42, 7, 100, 2024]:
        print(f"\n  --- Seed {seed} ---")
        np.random.seed(seed)
        res = select_rank(X_std, max_rank=6, corcondia_threshold=70.0)
        seed_results[seed] = res["diagnostics"]
        seed_chosen[seed] = res["chosen_rank"]
        print(f"  Seed {seed}: chosen rank = {res['chosen_rank']}")

    # Confirm rank stability
    unique_ranks = set(seed_chosen.values())
    if len(unique_ranks) == 1:
        chosen_rank = unique_ranks.pop()
        print(f"\n  ✓ Rank R = {chosen_rank} is stable across all four seeds")
    else:
        # Fall back to majority rank if seeds disagree
        from collections import Counter
        chosen_rank = Counter(seed_chosen.values()).most_common(1)[0][0]
        print(f"\n  ⚠ Seeds disagree: {seed_chosen}")
        print(f"   Using majority rank R = {chosen_rank}")

    # Combine diagnostics across seeds for robustness reporting
    diag_long = pd.concat(
        [df.assign(seed=s) for s, df in seed_results.items()],
        ignore_index=True,
    )
    rank_result = {"diagnostics": seed_results[42], "diagnostics_all_seeds": diag_long}
    print(f"\n  Chosen rank: R = {chosen_rank}")

    # Final CP fit at chosen rank
    print(f"\n  Running final CP-ALS at rank {chosen_rank}...")
    np.random.seed(42)
    factors, weights, final_err = cp_als(X_std, rank=chosen_rank, max_iter=1000)
    print(f"  Final relative reconstruction error: {final_err:.4f}")
    print(f"  Variance explained: {1 - final_err**2:.3f}")

    # Extract temporal factor (mode 0) → Z^{IVS}_t
    U_time = factors[0]
    Z_state = pd.DataFrame(
        U_time,
        index=meta["dates"],
        columns=[f"Z_IVS_{r+1}" for r in range(chosen_rank)],
    )
    Z_state.index.name = "date"

    # Save outputs
    suffix = f"_{grid_name}"
    state_path = SURFACES_DIR / f"tensor_pca_state{suffix}.parquet"
    Z_state.to_parquet(state_path)
    print(f"\n  Saved state vector: {state_path}")

    factors_path = SURFACES_DIR / f"tensor_pca_factors{suffix}.npz"
    np.savez(
        factors_path,
        U_time=factors[0],
        V_moneyness=factors[1],
        W_maturity=factors[2],
        S_venue=factors[3],
        weights=weights,
        k_grid=meta["k_grid"],
        tau_grid=meta["tau_grid"],
        t_grid_days=np.asarray(meta["t_grid_days"], dtype=int),
        venues=np.array(meta["venues"]),
        mu_tensor=mu,
        sigma_tensor=sigma,
    )
    print(f"  Saved factors: {factors_path}")

    diag_path = SURFACES_DIR / f"tensor_pca_diagnostics{suffix}.csv"
    rank_result["diagnostics"].to_csv(diag_path, index=False)
    print(f"  Saved diagnostics: {diag_path}")

    seed_diag_path = SURFACES_DIR / f"tensor_pca_diagnostics_seeds{suffix}.csv"
    rank_result["diagnostics_all_seeds"].to_csv(seed_diag_path, index=False)
    print(f"  Saved seed-robustness diagnostics: {seed_diag_path}")

    if grid_name == "almeida":
        Z_state.to_parquet(SURFACES_DIR / "tensor_pca_state.parquet")
        np.savez(
            SURFACES_DIR / "tensor_pca_factors.npz",
            U_time=factors[0],
            V_moneyness=factors[1],
            W_maturity=factors[2],
            S_venue=factors[3],
            weights=weights,
            k_grid=meta["k_grid"],
            tau_grid=meta["tau_grid"],
            t_grid_days=np.asarray(meta["t_grid_days"], dtype=int),
            venues=np.array(meta["venues"]),
            mu_tensor=mu,
            sigma_tensor=sigma,
        )
        rank_result["diagnostics"].to_csv(SURFACES_DIR / "tensor_pca_diagnostics.csv", index=False)
        rank_result["diagnostics_all_seeds"].to_csv(
            SURFACES_DIR / "tensor_pca_diagnostics_seeds.csv", index=False
        )
        print(f"  Saved headline (unsuffixed) copies for downstream consumption.")

    print(f"\n  {'='*60}")
    print(f"  COMPONENT INTERPRETATION (venue loadings S_venue)")
    print(f"  {'='*60}")
    for r in range(chosen_rank):
        cme_load = factors[3][0, r]
        der_load = factors[3][1, r]
        print(f"  Component {r+1}: weight={weights[r]:.3f}, "
              f"CME loading={cme_load:+.3f}, Deribit loading={der_load:+.3f}")
        if np.sign(cme_load) == np.sign(der_load):
            print(f"               → COMMON factor (same sign on both venues)")
        else:
            print(f"               → VENUE WEDGE factor (opposite signs)")

    return Z_state, factors, weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CP tensor decomposition on the IVS panel."
    )
    parser.add_argument(
        "--grid",
        choices=list(MATURITY_GRIDS.keys()),
        default="almeida",
        help="Maturity grid: 'almeida' (3 maturities, headline) or 'broad' (9 maturities, robustness).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both grids sequentially (headline first, then robustness).",
    )
    args = parser.parse_args()

    np.random.seed(42)
    if args.both:
        for g in ["almeida", "broad"]:
            print(f"\n{'#' * 60}\n# Running grid: {g}\n{'#' * 60}")
            run_tensor_decomposition(grid_name=g)
    else:
        run_tensor_decomposition(grid_name=args.grid)