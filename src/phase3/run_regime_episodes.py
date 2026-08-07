"""
Phase 3 robustness: episode-based regime test (read-only post-process).

MOTIVATION
The tercile classification on Z_IVS_1 is clustered in calendar time: high-volatility days fall
almost entirely in 2020-2021 and low-volatility days almost entirely in 2023. A regime effect
estimated on that partition is therefore partly confounded with a monotone time trend and with
anything else that moved over the sample.

The sample contains two SEPARATED high-volatility episodes, however, divided by roughly a year of
calm. A monotone trend cannot produce the same signed deviation at two dates separated by a return
to baseline. If the at-the-money curvature contrast carries the same sign in each high-volatility
episode taken on its own, the contrast tracks the volatility state rather than the passage of time.

WHAT THIS DOES
Episodes are identified from the data rather than by eye: contiguous runs of high-tercile days,
merged across gaps shorter than MIN_GAP_DAYS and required to last at least MIN_RUN_DAYS. The two
longest are the high-volatility episodes; the longest run of low-tercile days is the calm
comparison episode.

The fitted coefficient functions are then evaluated at each episode's mean conditioning state. This
is a function evaluation of the already-estimated theta, NOT a re-estimation, so no headline result
is touched and nothing is refitted.

INFERENCE
The stored bootstrap draws record coefficients only at the three tercile-mean states, not the full
theta per replicate. Because the coefficient functions are AFFINE in Z, however, any state lying in
the affine hull of the three tercile-mean states can be written as

    Z_ep = sum_g w_g Z_g    with    sum_g w_g = 1,

and then b(Z_ep) = sum_g w_g b(Z_g) exactly, replicate by replicate. Bootstrap intervals at the
episode states therefore come free from the existing draws.

Three tercile-mean states span only a two-dimensional affine plane in R^3, so a general episode
state need not lie exactly within it. The weights are obtained by constrained least squares and the
projection residual is reported alongside every interval. Where the residual is small relative to
the spread of the basis states the interval is trustworthy; where it is not, the point estimate
(which is exact regardless) stands alone and the interval should not be quoted.

To make the intervals exact on any future rerun, add

    row["theta"] = res.theta.tolist()

to _one_replicate in run_phase3_bootstrap.py. That is a one-line change and costs nothing.

Usage:  python -m src.phase3.run_regime_episodes
"""

import numpy as np
import pandas as pd

from src.config import get_path, get_sample_window
from src.phase3.conditional_kernel import coefficients_at
from src.phase3.run_phase3 import (load_conditioning_spec,
                                   load_volatility_tercile_labels)

SPEC = "crypto"
VENUES = ("CME", "DER")
TERCILES = ("low", "mid", "high")
MIN_RUN_DAYS = 15          # a run shorter than this is not an episode
MIN_GAP_DAYS = 60          # episodes must be separated by this much non-episode time
CI = 0.95


# ----------------------------------------------------------------------
# Episode identification
# ----------------------------------------------------------------------

def _contiguous_runs(dates: pd.DatetimeIndex, flag: np.ndarray):
    """Maximal runs of consecutive True in `flag`, as (start, end, n_days)."""
    runs, i, n = [], 0, len(flag)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1]:
            j += 1
        runs.append((dates[i], dates[j], j - i + 1))
        i = j + 1
    return runs


def _merge_close(runs, min_gap_days: int):
    """Merge runs separated by fewer than `min_gap_days` calendar days."""
    if not runs:
        return runs
    out = [list(runs[0])]
    for s, e, k in runs[1:]:
        if (s - out[-1][1]).days < min_gap_days:
            out[-1][1] = e
            out[-1][2] += k
        else:
            out.append([s, e, k])
    return [tuple(r) for r in out]


def identify_episodes(labels: pd.DataFrame) -> dict:
    """Two longest high-volatility episodes plus the longest calm episode."""
    lab = labels.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(lab["date"])
    terc = lab["tercile"].astype(str).to_numpy()

    episodes = {}
    for tercile, tag, keep in (("high", "high", 2), ("low", "calm", 1)):
        runs = _merge_close(_contiguous_runs(dates, terc == tercile), MIN_GAP_DAYS)
        runs = [r for r in runs if r[2] >= MIN_RUN_DAYS]
        runs.sort(key=lambda r: r[2], reverse=True)
        for i, (s, e, k) in enumerate(runs[:keep], start=1):
            name = f"{tag}_{i}" if keep > 1 else tag
            episodes[name] = {"start": s, "end": e, "n_days": int(k)}
    # Order high episodes chronologically for readable output.
    return dict(sorted(episodes.items(), key=lambda kv: kv[1]["start"]))


# ----------------------------------------------------------------------
# Affine projection onto the tercile-state hull
# ----------------------------------------------------------------------

def _affine_weights(Z_target: np.ndarray, Z_basis: np.ndarray):
    """Least-squares w with sum(w) = 1 minimising ||Z_basis.T @ w - Z_target||.

    Z_basis is (3, dim_Z), one tercile-mean state per row. Returns
    (w, residual_norm, residual_relative_to_basis_spread).
    """
    A = np.vstack([Z_basis.T, np.ones((1, Z_basis.shape[0]))])   # (dim_Z + 1, 3)
    b = np.concatenate([Z_target, [1.0]])
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.linalg.norm(Z_basis.T @ w - Z_target))
    scale = float(np.linalg.norm(Z_basis.max(axis=0) - Z_basis.min(axis=0)))
    return w, resid, (resid / scale if scale > 1e-12 else np.nan)


# ----------------------------------------------------------------------

def run_regime_episodes(spec_name: str = SPEC, ci: float = CI):
    data_p3 = get_path("data_phase3")
    tab_dir = get_path("results_phase3") / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    start, end = get_sample_window()

    print("=" * 70)
    print("  Phase 3 robustness: episode-based regime test")
    print("  Two separated high-volatility episodes against one calm episode")
    print("=" * 70)

    labels = load_volatility_tercile_labels()
    labels["date"] = pd.to_datetime(labels["date"])
    labels = labels[(labels["date"] >= start) & (labels["date"] <= end)]

    episodes = identify_episodes(labels)
    print(f"\n  Episodes identified (contiguous tercile runs, min {MIN_RUN_DAYS} days, "
          f"merged across gaps under {MIN_GAP_DAYS} days):")
    for name, ep in episodes.items():
        print(f"    {name:8s}: {ep['start'].date()} -> {ep['end'].date()} "
              f"({ep['n_days']} days)")
    n_high = sum(1 for k in episodes if k.startswith("high"))
    if n_high < 2:
        print("\n  [WARN] Fewer than two separable high-volatility episodes were found; "
              "the separation argument cannot be made on this sample.")

    z_dates_raw, Z_full, z_cols = load_conditioning_spec(spec_name)
    z_dates = pd.DatetimeIndex(pd.to_datetime(z_dates_raw))
    n_Z = Z_full.shape[1]

    rows, rep_store = [], {}

    for venue in VENUES:
        theta_path = data_p3 / f"phase3_{venue}_{spec_name}.npz"
        draws_path = data_p3 / f"phase3_bootstrap_draws_{venue}_{spec_name}.parquet"
        if not theta_path.exists():
            print(f"\n  [{venue}] SKIP: {theta_path.name} not found.")
            continue
        theta = np.load(theta_path)["theta"]

        draws = None
        if draws_path.exists():
            draws = pd.read_parquet(draws_path)
            if "converged" in draws.columns:
                draws = draws[draws["converged"]].copy()
            missing = [f"{c}_{g}" for c in ("b", "c", "d") for g in TERCILES
                       if f"{c}_{g}" not in draws.columns]
            if missing:
                print(f"  [{venue}] draws lack {missing}; intervals skipped.")
                draws = None

        # Tercile-mean states form the affine basis used for interval construction.
        Z_basis, basis_ok = [], True
        for g in TERCILES:
            d = pd.DatetimeIndex(labels.loc[labels["tercile"].astype(str) == g, "date"])
            m = z_dates.isin(d)
            if m.sum() == 0:
                basis_ok = False
                break
            Z_basis.append(Z_full[m].mean(axis=0))
        Z_basis = np.array(Z_basis) if basis_ok else None

        print(f"\n  [{venue}]")
        for name, ep in episodes.items():
            m = (z_dates >= ep["start"]) & (z_dates <= ep["end"])
            if m.sum() == 0:
                continue
            Z_ep = Z_full[m].mean(axis=0)

            # Exact point estimate: a function evaluation of the fitted theta.
            b, c, d = coefficients_at(theta, Z_ep, n_Z)
            curv = float(2.0 * c + 6.0 * d)

            row = {"venue": venue, "episode": name,
                   "start": ep["start"].date(), "end": ep["end"].date(),
                   "n_days_state": int(m.sum()),
                   "b": float(b), "c": float(c), "d": float(d),
                   "curv_at_money": curv}
            for k, col in enumerate(z_cols):
                row[f"Zbar_{col}"] = float(Z_ep[k])

            # Bootstrap interval by affine recombination of the tercile draws.
            if draws is not None and Z_basis is not None:
                w, resid, rel = _affine_weights(Z_ep, Z_basis)
                row["affine_resid"] = resid
                row["affine_resid_rel"] = rel
                for coef in ("b", "c", "d"):
                    V = np.column_stack([draws[f"{coef}_{g}"].to_numpy() for g in TERCILES])
                    rep = V @ w
                    lo, hi = np.percentile(rep, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
                    row[f"{coef}_lo"], row[f"{coef}_hi"] = float(lo), float(hi)
                Vc = np.column_stack([
                    2.0 * draws[f"c_{g}"].to_numpy() + 6.0 * draws[f"d_{g}"].to_numpy()
                    for g in TERCILES])
                rep_c = Vc @ w
                lo, hi = np.percentile(rep_c, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
                row["curv_lo"], row["curv_hi"] = float(lo), float(hi)
                row["B_effective"] = int(len(draws))
                rep_store[(venue, name)] = rep_c

            rows.append(row)
            if "curv_lo" in row:
                print(f"    {name:8s}: curvature at money = {curv:+.3f} "
                      f"[{row['curv_lo']:+.3f}, {row['curv_hi']:+.3f}]   "
                      f"(affine resid = {row['affine_resid']:.3f}, "
                      f"rel = {row['affine_resid_rel']:.3f})")
            else:
                print(f"    {name:8s}: curvature at money = {curv:+.3f}   "
                      f"(no draws available; point estimate only)")

    out = pd.DataFrame(rows)
    path1 = tab_dir / f"regime_episodes_{spec_name}.csv"
    out.to_csv(path1, index=False)

    # ------------------------------------------------------------------
    # Separation test: same signed contrast in BOTH high-vol episodes?
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Separation test: curvature contrast against the calm episode")
    print("-" * 70)

    contrast_rows = []
    for venue in out["venue"].unique():
        sub = out[out["venue"] == venue].set_index("episode")
        if "calm" not in sub.index:
            continue
        calm = float(sub.loc["calm", "curv_at_money"])
        highs = [e for e in sub.index if str(e).startswith("high")]
        signs = []
        for e in highs:
            diff = calm - float(sub.loc[e, "curv_at_money"])
            rec = {"venue": venue, "high_episode": e,
                   "curv_high": float(sub.loc[e, "curv_at_money"]),
                   "curv_calm": calm,
                   "curv_diff_calm_minus_high": float(diff)}
            kh, kc = (venue, e), (venue, "calm")
            if kh in rep_store and kc in rep_store:
                rep_d = rep_store[kc] - rep_store[kh]
                lo, hi = np.percentile(rep_d, [100 * (1 - CI) / 2, 100 * (1 + CI) / 2])
                rec["diff_lo"], rec["diff_hi"] = float(lo), float(hi)
                rec["P_diff_lt_0"] = float((rep_d < 0).mean())
            signs.append(np.sign(diff))
            contrast_rows.append(rec)

        if len(signs) == 2:
            agree = bool(signs[0] == signs[1] and signs[0] != 0)
            print(f"    {venue}: sign agreement across the two high-volatility "
                  f"episodes = {agree}")
            if agree:
                print("       A monotone time trend cannot generate the same signed "
                      "deviation at two dates separated by a return to baseline, so the "
                      "contrast tracks the volatility state.")
            else:
                print("       Signs differ across episodes; the separation evidence is "
                      "inconclusive on this sample and should be reported as such.")

    contrasts = pd.DataFrame(contrast_rows)
    path2 = tab_dir / f"regime_episode_contrasts_{spec_name}.csv"
    contrasts.to_csv(path2, index=False)
    if not contrasts.empty:
        cols = [c for c in ["venue", "high_episode", "curv_high", "curv_calm",
                            "curv_diff_calm_minus_high", "diff_lo", "diff_hi",
                            "P_diff_lt_0"] if c in contrasts.columns]
        print("\n" + contrasts[cols].round(3).to_string(index=False))

    print(f"\n  [checkpoint] Saved: {path1}")
    print(f"  [checkpoint] Saved: {path2}")
    print("\n  NOTE: point estimates are exact evaluations of the fitted theta. Intervals "
          "rely on the affine recombination described in the module docstring; inspect "
          "affine_resid_rel before quoting them.")

    return out, contrasts


if __name__ == "__main__":
    run_regime_episodes()
