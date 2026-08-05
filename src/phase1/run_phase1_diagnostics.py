"""
Phase 1 supplementary diagnostics (read-only post-process).

Produces two tables that the results text requires but the pipeline does not currently emit:

  1. tail_shape_summary.csv  -- cross-sectional distribution of the per-day option-implied GPD tail
     shapes xi_L, xi_R at the headline maturity, the exponential-fallback frequency, and the share
     of days violating the finite-fourth-moment condition xi < 0.25 that the CL20 bound of Phase 4
     requires. These are already persisted by extract_densities.py in rnd_{venue}_summary.parquet;
     nothing is recomputed here.

  2. ssvi_skew_summary.csv   -- distribution of the calibrated SSVI skew parameter rho on the slice
     nearest the headline maturity, by venue and by year, with the frequency of positive skew, plus
     bound-binding frequencies for the curvature parameters eta and gamma. This supports the
     inverse-leverage claim of the introduction, which is currently asserted without a statistic,
     and quantifies the parameter dispersion that the appendix discusses from the figures.

Reads only existing Phase 1 outputs and writes only into results/phase1/tables. Nothing here alters
any headline estimate.

Usage:  python -m src.phase1.run_phase1_diagnostics
"""

import numpy as np
import pandas as pd

from src.config import get_path, get_sample_window

TAU_DAYS = 27
XI_CRITICAL = 0.25          # finite fourth moment requires xi < 1/4
VENUES = ("CME", "DER")


# ----------------------------------------------------------------------
# 1. Tail shape distribution
# ----------------------------------------------------------------------

def tail_shape_summary(tau_days: int = TAU_DAYS) -> pd.DataFrame:
    data_p1 = get_path("data_phase1")
    start, end = get_sample_window()
    rows = []

    for venue in VENUES:
        path = data_p1 / f"rnd_{venue}_summary.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run extract_densities.py first.")

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["tau_days"] == tau_days)
                & (df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            raise ValueError(f"No {venue} rows at tau={tau_days}d in the sample window.")

        row = {"venue": venue, "n_days": int(len(df))}

        for side, col in (("L", "xi_left"), ("R", "xi_right")):
            xi = df[col]
            skipped = xi.isna()                    # splice not attempted
            exp_fb = (xi == 0.0)                   # exponential tail used
            fitted = xi[~skipped & ~exp_fb]        # genuine GPD root

            row[f"xi_{side}_mean"] = fitted.mean() if len(fitted) else np.nan
            row[f"xi_{side}_sd"] = fitted.std(ddof=1) if len(fitted) > 1 else np.nan
            row[f"xi_{side}_p25"] = fitted.quantile(0.25) if len(fitted) else np.nan
            row[f"xi_{side}_median"] = fitted.median() if len(fitted) else np.nan
            row[f"xi_{side}_p75"] = fitted.quantile(0.75) if len(fitted) else np.nan
            row[f"xi_{side}_min"] = fitted.min() if len(fitted) else np.nan
            row[f"xi_{side}_max"] = fitted.max() if len(fitted) else np.nan
            row[f"n_fitted_{side}"] = int(len(fitted))
            row[f"exp_fallback_{side}_pct"] = 100.0 * float(exp_fb.mean())
            row[f"splice_skipped_{side}_pct"] = 100.0 * float(skipped.mean())

            # Moment conditions. An exponential tail (xi = 0) has all moments
            # finite. A skipped splice leaves the BL body in place, which is
            # compactly supported on the grid, so it also has all moments finite.
            xi_eff = xi.fillna(0.0)
            row[f"share_xi_{side}_ge_025_pct"] = 100.0 * float((xi_eff >= 0.25).mean())
            row[f"share_xi_{side}_ge_050_pct"] = 100.0 * float((xi_eff >= 0.50).mean())
            row[f"share_xi_{side}_ge_100_pct"] = 100.0 * float((xi_eff >= 1.00).mean())

        # The bound needs a finite fourth moment in BOTH tails simultaneously.
        xi_l = df["xi_left"].fillna(0.0)
        xi_r = df["xi_right"].fillna(0.0)
        row["share_both_tails_ok_pct"] = 100.0 * float(
            ((xi_l < XI_CRITICAL) & (xi_r < XI_CRITICAL)).mean())
        row["share_either_tail_violates_pct"] = 100.0 * float(
            ((xi_l >= XI_CRITICAL) | (xi_r >= XI_CRITICAL)).mean())

        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. SSVI parameter distribution near the headline maturity
# ----------------------------------------------------------------------

def ssvi_skew_summary(tau_days: int = TAU_DAYS):
    data_p1 = get_path("data_phase1")
    start, end = get_sample_window()

    path = data_p1 / "ssvi_params.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run fit_surfaces.py first.")

    params = pd.read_parquet(path)
    params["date"] = pd.to_datetime(params["date"])
    params = params[(params["date"] >= start) & (params["date"] <= end)].copy()

    # ssvi_params carries days_to_expiry directly; pick, per (venue, date), the slice
    # whose maturity is closest to the headline horizon.
    params["gap"] = (params["days_to_expiry"] - tau_days).abs()
    near = (params.sort_values(["venue", "date", "gap"])
                  .groupby(["venue", "date"], as_index=False)
                  .first())

    def _block(g):
        return pd.Series({
            "n_days": int(len(g)),
            "rho_mean": g["rho"].mean(),
            "rho_sd": g["rho"].std(ddof=1),
            "rho_p05": g["rho"].quantile(0.05),
            "rho_p25": g["rho"].quantile(0.25),
            "rho_median": g["rho"].median(),
            "rho_p75": g["rho"].quantile(0.75),
            "rho_p95": g["rho"].quantile(0.95),
            "share_rho_pos_pct": 100.0 * float((g["rho"] > 0).mean()),
            # Bound-binding diagnostics for the weakly identified curvature parameters.
            # Boxes are eta in (1e-6, 4) and gamma in (1e-6, 1).
            "eta_median": g["eta"].median(),
            "share_eta_at_upper_pct": 100.0 * float((g["eta"] > 3.99).mean()),
            "gamma_median": g["gamma"].median(),
            "share_gamma_at_bound_pct": 100.0 * float(
                ((g["gamma"] < 0.01) | (g["gamma"] > 0.99)).mean()),
            "median_days_to_expiry": g["days_to_expiry"].median(),
        })

    overall = near.groupby("venue")[["rho", "eta", "gamma", "days_to_expiry"]] \
                  .apply(_block).reset_index()
    overall.insert(1, "year", "all")

    near["year"] = near["date"].dt.year
    by_year = near.groupby(["venue", "year"])[["rho", "eta", "gamma", "days_to_expiry"]] \
                  .apply(_block).reset_index()
    by_year["year"] = by_year["year"].astype(str)

    out = pd.concat([overall, by_year], ignore_index=True)

    # Cross-venue agreement on the sign of the skew, day by day: does the smile
    # tilt the same way on both venues on the same date?
    wide = near.pivot_table(index="date", columns="venue", values="rho").dropna()
    agree = pd.DataFrame([{
        "n_matched_days": int(len(wide)),
        "corr_rho_cme_der": float(wide["CME"].corr(wide["DER"])),
        "share_same_sign_pct": 100.0 * float(
            (np.sign(wide["CME"]) == np.sign(wide["DER"])).mean()),
        "share_both_positive_pct": 100.0 * float(
            ((wide["CME"] > 0) & (wide["DER"] > 0)).mean()),
        "share_both_negative_pct": 100.0 * float(
            ((wide["CME"] < 0) & (wide["DER"] < 0)).mean()),
    }])

    return out, agree


# ----------------------------------------------------------------------

def run_phase1_diagnostics():
    tab_dir = get_path("results_phase1") / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Phase 1 supplementary diagnostics")
    print("=" * 70)

    tails = tail_shape_summary()
    out1 = tab_dir / "tail_shape_summary.csv"
    tails.to_csv(out1, index=False)

    print(f"\n  Tail shapes at tau={TAU_DAYS}d "
          f"(exponential-fallback days excluded from the moments, counted as xi = 0 "
          f"in the conditions below):")
    show = ["venue", "n_days",
            "xi_L_median", "xi_L_p25", "xi_L_p75", "exp_fallback_L_pct",
            "xi_R_median", "xi_R_p25", "xi_R_p75", "exp_fallback_R_pct"]
    print(tails[show].round(3).to_string(index=False))

    print(f"\n  Finite-fourth-moment condition (xi < {XI_CRITICAL}):")
    cond = ["venue", "share_xi_L_ge_025_pct", "share_xi_R_ge_025_pct",
            "share_both_tails_ok_pct", "share_either_tail_violates_pct"]
    print(tails[cond].round(2).to_string(index=False))

    print("\n  Heavier violations (xi >= 0.5 kills the second moment of the "
          "parametric tail; xi >= 1.0 kills the first):")
    heavy = ["venue", "share_xi_L_ge_050_pct", "share_xi_R_ge_050_pct",
             "share_xi_L_ge_100_pct", "share_xi_R_ge_100_pct"]
    print(tails[heavy].round(2).to_string(index=False))
    print(f"\n  [checkpoint] Saved: {out1}")

    skew, agree = ssvi_skew_summary()
    out2 = tab_dir / "ssvi_skew_summary.csv"
    skew.to_csv(out2, index=False)
    out3 = tab_dir / "ssvi_skew_agreement.csv"
    agree.to_csv(out3, index=False)

    print(f"\n  SSVI skew parameter rho on the slice nearest tau={TAU_DAYS}d:")
    cols = ["venue", "year", "n_days", "rho_median", "rho_p05", "rho_p95",
            "share_rho_pos_pct"]
    print(skew[cols].round(3).to_string(index=False))

    print("\n  Curvature-parameter bound binding:")
    cols2 = ["venue", "year", "eta_median", "share_eta_at_upper_pct",
             "gamma_median", "share_gamma_at_bound_pct"]
    print(skew[cols2].round(3).to_string(index=False))

    print("\n  Cross-venue skew agreement (same date, both venues):")
    print(agree.round(3).to_string(index=False))
    print(f"\n  [checkpoint] Saved: {out2}")
    print(f"  [checkpoint] Saved: {out3}")

    return tails, skew, agree


if __name__ == "__main__":
    run_phase1_diagnostics()
