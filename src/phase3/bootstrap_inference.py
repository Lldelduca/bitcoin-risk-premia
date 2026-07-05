"""
Block-Bootstrap Inference Utilities.

The circular moving-block bootstrap resamples contiguous blocks of days, preserving the within-block dependence
with block lengths at or above the 27-day overlap horizon, the resulting percentile intervals are
valid under weak dependence.

"""

import numpy as np

def circular_block_indices(n: int, block_length: int, rng) -> np.ndarray:
    """Returns n resampled indices formed from circular contiguous blocks."""
    block_length = max(int(block_length), 1)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_length)[None, :]) % n
    return idx.reshape(-1)[:n]


def block_bootstrap_mean_bands(X, block_length=27, B=1000, ci=0.95, seed=42):
    """Pointwise bootstrap bands for the time-mean of daily curves."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    means = np.empty((B, X.shape[1]))
    for b in range(B):
        idx = circular_block_indices(n, block_length, rng)
        means[b] = X[idx].mean(axis=0)
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": X.mean(axis=0),
        "lo": np.quantile(means, alpha, axis=0),
        "hi": np.quantile(means, 1.0 - alpha, axis=0),
        "se_boot": means.std(axis=0, ddof=1),
        "B": B,
        "block_length": block_length,
        "ci": ci,
        "n_days": n,
    }

def block_bootstrap_group_mean_bands(X, labels, groups, block_length=27, B=1000, ci=0.95, seed=42):
    """Pointwise bootstrap bands for group means of daily curves."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    groups = list(groups)
    means = {g: np.full((B, X.shape[1]), np.nan) for g in groups}
    for b in range(B):
        idx = circular_block_indices(n, block_length, rng)
        lab_b = labels[idx]
        X_b = X[idx]
        for g in groups:
            mask = lab_b == g
            if mask.sum() > 0:
                means[g][b] = X_b[mask].mean(axis=0)
    alpha = (1.0 - ci) / 2.0
    out = {}
    for g in groups:
        mask0 = labels == g
        if mask0.sum() == 0:
            continue
        out[g] = {
            "mean": X[mask0].mean(axis=0),
            "lo": np.nanquantile(means[g], alpha, axis=0),
            "hi": np.nanquantile(means[g], 1.0 - alpha, axis=0),
            "se_boot": np.nanstd(means[g], axis=0, ddof=1),
            "B": B,
            "block_length": block_length,
            "ci": ci,
            "n_days": int(mask0.sum()),
        }
    return out

def block_bootstrap_statistic(n, stat_fn, block_length=27, B=1000, ci=0.95, seed=42, allow_failures=True):
    """Generic block bootstrap for an arbitrary statistic of day indices."""
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(np.asarray(stat_fn(np.arange(n)), dtype=float))
    draws = []
    n_fail = 0
    for b in range(B):
        idx = circular_block_indices(n, block_length, rng)
        try:
            val = np.atleast_1d(np.asarray(stat_fn(idx), dtype=float))
            if val.shape != point.shape or not np.all(np.isfinite(val)):
                raise ValueError("non-finite or misshaped replicate")
            draws.append(val)
        except Exception:
            if not allow_failures:
                raise
            n_fail += 1
    draws = np.array(draws)
    alpha = (1.0 - ci) / 2.0
    return {
        "point": point,
        "lo": np.quantile(draws, alpha, axis=0),
        "hi": np.quantile(draws, 1.0 - alpha, axis=0),
        "se_boot": draws.std(axis=0, ddof=1),
        "draws": draws,
        "B": B,
        "n_success": len(draws),
        "n_fail": n_fail,
        "block_length": block_length,
        "ci": ci,
    }
