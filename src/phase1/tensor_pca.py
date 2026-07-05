"""
CP Tensor Decomposition for the IVS Panel.

Constructs a 4th-order tensor X ∈ R^{T × M × Δ × J} from the fitted SSVI surfaces. Applies (CP) decomposition via 
alternating least squares to recover a low-dimensional state vector capturing systemic features of the surface panel.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from src.config import get_path, SAMPLE, TENSOR_GRID
from src.phase1.ssvi import SSVI

def build_ivs_tensor(params_df: pd.DataFrame, cleaned_dfs: dict = None, t_grid_days: list = None, k_grid: np.ndarray = None,
                     venues: list = ["CME", "DER"],) -> Tuple[np.ndarray, dict]:
   
    t_grid_days = np.asarray(t_grid_days, dtype=int)
    tau_grid = t_grid_days.astype(float) / 365.25

    # Find common dates across all venues
    date_sets = []
    for v in venues:
        v_dates = set(params_df[params_df["venue"] == v]["date"].unique())
        date_sets.append(v_dates)
    common_dates = sorted(set.intersection(*date_sets))
    common_dates = pd.to_datetime(common_dates)

    print(f"  Common dates across venues: {len(common_dates)}")
    print(f"  Tensor shape will be: ({len(common_dates)}, {len(k_grid)}, "
          f"{len(tau_grid)}, {len(venues)})")

    T, M, D, J = len(common_dates), len(k_grid), len(tau_grid), len(venues)
    X = np.full((T, M, D, J), np.nan)

    for j, venue in enumerate(venues):
        params_v = params_df[params_df["venue"] == venue].copy()
        params_v["date"] = pd.to_datetime(params_v["date"])
        has_forward_col = ("forward" in params_v.columns
                           and params_v["forward"].notna().all())
        df_v = cleaned_dfs.get(venue) if cleaned_dfs is not None else None
        if df_v is not None:
            df_v["date"] = pd.to_datetime(df_v["date"])
        n_filled = 0
        n_skipped = 0

        for t_idx, d in enumerate(common_dates):
            params_day = params_v[params_v["date"] == d]
            if params_day.empty:
                n_skipped += 1
                continue

            # Reconstruct the saved Phase 1a surface
            try:
                forward_map = None
                if not has_forward_col:
                    if df_v is None:
                        raise ValueError(
                            "ssvi_params.parquet has no 'forward' column and no "
                            "cleaned data was provided to recover forwards"
                        )
                    df_day = df_v[df_v["date"] == d]
                    if df_day.empty:
                        n_skipped += 1
                        continue
                    forward_map = df_day.groupby("tau")["forward_price"].mean()
                ssvi = SSVI.from_params(params_day, forward_map=forward_map,
                                        venue=venue, date=d)
            except Exception:
                n_skipped += 1
                continue

            # Evaluate SSVI on the (κ × τ) grid
            for d_idx, tau in enumerate(tau_grid):
                fitted_taus = ssvi.res["maturities"]

                if tau < fitted_taus.min() * 0.8 or tau > fitted_taus.max() * 1.2:
                    continue
                for m_idx, k in enumerate(k_grid):
                    try:
                        w = ssvi.total_variance(tau, k)
                        X[t_idx, m_idx, d_idx, j] = w
                    except Exception:
                        continue

            n_filled += 1

            if (t_idx + 1) % 100 == 0:
                print(f"    [{venue}] {t_idx+1}/{T} days processed")

        print(f"  [{venue}] Filled {n_filled}, skipped {n_skipped}")

    meta = {"dates": common_dates, "k_grid": k_grid, "tau_grid": tau_grid, "t_grid_days": t_grid_days, "venues": venues}
    return X, meta

def standardize_tensor(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    X_std = (X - mu[np.newaxis, ...]) / sigma[np.newaxis, ...]
    X_std = np.nan_to_num(X_std, nan=0.0)
    return X_std, mu, sigma

def cp_als(X: np.ndarray, rank: int, max_iter: int = 500, tol: float = 1e-7, random_state: int = 42) -> Tuple[list, np.ndarray, float]:
    """ CP decomposition via Alternating Least Squares. Approximates the 4-mode tensor X ≈ Σ_r λ_r · u_r ∘ v_r ∘ w_r ∘ s_r"""
    rng = np.random.default_rng(random_state)
    shape = X.shape
    N = X.ndim

    # Initialize factor matrices with random orthonormal columns
    factors = []
    for n in range(N):
        A = rng.standard_normal((shape[n], rank))
        Q, _ = np.linalg.qr(A)
        factors.append(Q if Q.shape[1] == rank else A / np.linalg.norm(A, axis=0))

    weights = np.ones(rank)
    X_norm = np.linalg.norm(X)
    prev_error = np.inf

    for iteration in range(max_iter):
        for n in range(N):
            X_n = unfold(X, n)
            # Compute Khatri-Rao product of all other factors
            kr = khatri_rao([factors[m] for m in range(N) if m != n])
            # Update factor n via least squares
            V = np.ones((rank, rank))
            for m in range(N):
                if m != n:
                    V = V * (factors[m].T @ factors[m])

            U_new = X_n @ kr @ np.linalg.pinv(V)

            # Normalize columns and absorb norms into weights
            norms = np.linalg.norm(U_new, axis=0)
            norms = np.where(norms < 1e-12, 1.0, norms)
            U_new = U_new / norms
            weights = norms if n == N - 1 else weights * norms / np.maximum(
                np.linalg.norm(factors[n], axis=0), 1e-12
            )
            factors[n] = U_new

        # Compute reconstruction error
        X_hat = cp_to_tensor(factors, weights)
        error = np.linalg.norm(X - X_hat) / X_norm

        if abs(prev_error - error) < tol:
            print(f"    CP-ALS converged in {iteration+1} iterations (rel. err = {error:.6f})")
            break
        prev_error = error

    return factors, weights, error

def unfold(X: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)

def khatri_rao(matrices: list) -> np.ndarray:
    n_cols = matrices[0].shape[1]
    n_rows = int(np.prod([M.shape[0] for M in matrices]))
    result = np.ones((n_rows, n_cols))
    for r in range(n_cols):
        kron = matrices[0][:, r]
        for M in matrices[1:]:
            kron = np.outer(kron, M[:, r]).flatten()
        result[:, r] = kron
    return result

def cp_to_tensor(factors: list, weights: np.ndarray) -> np.ndarray:
    shape = tuple(f.shape[0] for f in factors)
    rank = factors[0].shape[1]
    X = np.zeros(shape)
    for r in range(rank):
        component = weights[r]
        outer = factors[0][:, r]
        for f in factors[1:]:
            outer = np.multiply.outer(outer, f[:, r])
        X += component * outer
    return X

def corcondia(X: np.ndarray, factors: list, weights: np.ndarray) -> float:
    """
    Core Consistency Diagnostic (Bro & Kiers, 2003).

    A value close to 100 indicates a valid trilinear (or higher-order) structure at the chosen rank. 
    Values dropping below ~70 suggest the rank is too high..
    """
    N = X.ndim
    R = factors[0].shape[1]

    # Scale factors by cube-root of weights for symmetric distribution
    scaled_factors = [
        f * np.power(np.maximum(weights, 1e-12), 1.0 / N)[np.newaxis, :]
        for f in factors
    ]

    # Compute Tucker core: G = X ×_1 U_1^+ ×_2 U_2^+ ... ×_N U_N^+
    G = X.copy()
    for n in range(N):
        U_pinv = np.linalg.pinv(scaled_factors[n])
        G = mode_n_product(G, U_pinv, n)

    # G should be approximately superdiagonal (1 on superdiag, 0 elsewhere)
    superdiag_mask = np.zeros([R] * N)
    for r in range(R):
        idx = tuple([r] * N)
        superdiag_mask[idx] = 1.0

    numerator = np.sum((G - superdiag_mask) ** 2)
    denominator = R  # sum of squared superdiag entries (each = 1)
    consistency = 100 * (1 - numerator / denominator)
    return float(consistency)

def mode_n_product(X: np.ndarray, M: np.ndarray, mode: int) -> np.ndarray:
    X_unf = unfold(X, mode)
    result = M @ X_unf
    new_shape = list(X.shape)
    new_shape[mode] = M.shape[0]
    return np.moveaxis(result.reshape([new_shape[mode]] + [X.shape[m] for m in range(X.ndim) if m != mode]), 0, mode)

def select_rank(X: np.ndarray, max_rank: int = 6, corcondia_threshold: float = 70.0, verbose: bool = True,
                random_state: int = 42) -> dict:

    diagnostics = []
    for r in range(1, max_rank + 1):
        factors, weights, err = cp_als(X, rank=r, random_state=random_state)
        cc = corcondia(X, factors, weights)
        diagnostics.append({"rank": r, "reconstruction_error": err, "explained": 1 - err ** 2, "corcondia": cc})
        if verbose:
            print(f"  Rank {r}: rec.err={err:.4f}, explained={1-err**2:.3f}, "
                  f"CORCONDIA={cc:.1f}")

    diag_df = pd.DataFrame(diagnostics)
    valid = diag_df[diag_df["corcondia"] >= corcondia_threshold]

    if len(valid) == 0:
        chosen = 1
    else:
        chosen = int(valid["rank"].max())
    return {"diagnostics": diag_df, "chosen_rank": chosen}
