"""
Joint bootstrap test of kernel regime-dependence

MARGINAL curvature coefficient c is poorly identified at each tercile because the cubic terms (R, R^2, R^3) 
are collinear over R in [0.40, 2.00], so b, c, d trade off and no single one is pinned down in isolation. 

However, that does NOT mean the kernel is regime-independent, but it means the marginal test is the wrong question. 
The right question is JOINT: Does the coefficient VECTOR (b, c, d) differ across volatility regimes?

This module answers it by computing per venue:

  (A) Pairwise joint test  H0: (b,c,d)_low = (b,c,d)_high
      via a bootstrap Wald / Mahalanobis statistic on the 3-vector difference delta = (b,c,d)_low - (b,c,d)_high, 
      using the bootstrap covariance of delta. Reports the bootstrap p-value (fraction of recentred replicate
      statistics exceeding the observed) and the share of replicates in which the whole-vector ordering is preserved.

  (B) Per-coefficient low-vs-high difference with bootstrap CI and P(diff<0)

  (C) A curvature-at-a-point contrast. Because the individual cubic coefficients are collinear, a more interpretable scalar 
      is the second derivative of the log-kernel at the money, kappa(R) = 2c + 6dR evaluated at R = 1

"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import get_path

DATA_P3 = get_path("data_phase3")
TAB_DIR = get_path("results_phase3") / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

COEFS = ("b", "c", "d")

def _vec(draws, tercile):
    return draws[[f"{c}_{tercile}" for c in COEFS]].to_numpy(dtype=float)

def _curv_at_money(draws, tercile):
    return 2.0 * draws[f"c_{tercile}"].to_numpy(float) + 6.0 * draws[f"d_{tercile}"].to_numpy(float)

def joint_regime_test(venue, spec="crypto", lo="low", hi="high", point_theta=None):
    path = DATA_P3 / f"phase3_bootstrap_draws_{venue}_{spec}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run run_phase3_bootstrap.py first "
            f"(this test is a read-only post-process of its draws).")
    draws = pd.read_parquet(path)
    if "converged" in draws.columns:
        draws = draws[draws["converged"]].copy()
    B = len(draws)

    V_lo, V_hi = _vec(draws, lo), _vec(draws, hi)

    D = V_lo - V_hi                      # (B, 3) per-replicate vector difference
    d_mean = D.mean(axis=0)              # observed mean difference
    Sigma = np.cov(D, rowvar=False)      # (3,3) bootstrap covariance of the difference
    Sigma_inv = np.linalg.pinv(Sigma)

    # (A) Bootstrap Wald / Mahalanobis test of H0: delta = 0
    W_obs = float(d_mean @ Sigma_inv @ d_mean)

    # Null distribution: recentre the replicate differences to mean zero
    Dc = D - d_mean
    W_rep = np.einsum("ij,jk,ik->i", Dc, Sigma_inv, Dc)
    p_joint = float((W_rep >= W_obs).mean())

    # Whole-vector ordering
    sign_point = np.sign(d_mean)
    same_all = np.all(np.sign(D) == sign_point, axis=1)
    frac_vector_consistent = float(same_all.mean())

    # (B) Per-coefficient differences
    per_coef = []
    for j, c in enumerate(COEFS):
        dj = D[:, j]
        per_coef.append({
            "venue": venue, "test": "per_coef_diff", "coef": c,
            "diff_point": float(d_mean[j]),
            "ci_lo": float(np.quantile(dj, 0.025)),
            "ci_hi": float(np.quantile(dj, 0.975)),
            "frac_negative": float((dj < 0).mean()),
            "se_boot": float(dj.std(ddof=1)),
            "B_effective": B,
        })

    # (C) Curvature-at-money contrast (2c + 6d at R=1)
    m2_lo = _curv_at_money(draws, lo)
    m2_hi = _curv_at_money(draws, hi)
    dm2 = m2_lo - m2_hi
    curv = {
        "venue": venue, "test": "curv_at_money_diff", "coef": "2c+6d@R=1",
        "diff_point": float(dm2.mean()),
        "ci_lo": float(np.quantile(dm2, 0.025)),
        "ci_hi": float(np.quantile(dm2, 0.975)),
        "frac_negative": float((dm2 < 0).mean()),
        "se_boot": float(dm2.std(ddof=1)),
        "B_effective": B,
    }

    joint = {
        "venue": venue, "test": "joint_wald", "coef": "(b,c,d)",
        "wald_stat": W_obs, "p_value": p_joint,
        "frac_vector_consistent": frac_vector_consistent,
        "B_effective": B,
        "delta_b": float(d_mean[0]), "delta_c": float(d_mean[1]),
        "delta_d": float(d_mean[2]),
    }
    return joint, per_coef, curv

def run_all(venues=("CME", "DER"), spec="crypto"):
    print("\n" + "=" * 64)
    print("  Joint bootstrap test of kernel regime-dependence (low vs high)")
    print(f"  spec = {spec}; reads existing 4b draws (no re-estimation)")
    print("=" * 64)

    joint_rows, detail_rows = [], []
    for v in venues:
        joint, per_coef, curv = joint_regime_test(v, spec)
        joint_rows.append(joint)
        detail_rows.extend(per_coef)
        detail_rows.append(curv)

        print(f"\n  [{v}]  (B = {joint['B_effective']} converged replicates)")
        print(f"    (A) JOINT H0: (b,c,d)_low = (b,c,d)_high")
        print(f"        Wald = {joint['wald_stat']:.2f},  "
              f"bootstrap p = {joint['p_value']:.4f}  "
              f"{'***' if joint['p_value']<0.001 else '**' if joint['p_value']<0.01 else '*' if joint['p_value']<0.05 else '(n.s.)'}")
        print(f"        mean vector diff: db={joint['delta_b']:+.3f}, "
              f"dc={joint['delta_c']:+.3f}, dd={joint['delta_d']:+.3f}")
        print(f"        replicates preserving full-vector ordering: "
              f"{joint['frac_vector_consistent']:.1%}")
        print(f"    (B) per-coefficient low - high [95% CI], P(diff<0):")
        for r in per_coef:
            print(f"        {r['coef']}: {r['diff_point']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"P<0 = {r['frac_negative']:.3f}")
        print(f"    (C) curvature at money (2c+6d|R=1), low - high: "
              f"{curv['diff_point']:+.3f} "
              f"[{curv['ci_lo']:+.3f}, {curv['ci_hi']:+.3f}]  "
              f"P<0 = {curv['frac_negative']:.3f}")

    pd.DataFrame(joint_rows).to_csv(TAB_DIR / f"joint_regime_test_{spec}.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(TAB_DIR / f"joint_regime_detail_{spec}.csv", index=False)
    print(f"\n  Saved: {TAB_DIR / f'joint_regime_test_{spec}.csv'}")
    print(f"  Saved: {TAB_DIR / f'joint_regime_detail_{spec}.csv'}")
    return joint_rows, detail_rows

if __name__ == "__main__":
    run_all()
