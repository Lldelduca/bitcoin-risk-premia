"""
Wedge Term Structure

Phases 4-5 measure the cross-venue cumulant wedge at tau = 27 only. This script recomputes the daily premia 
Pi_k = lambda_{k-1} INT (ln R)^k q_t dR (theta = 2 weights, identical construction per maturity) from the SAVED
daily RNDs at tau in {14, 27, 60}, and estimates the matched-day wedge DER - CME per (tau, k) with NW(27) errors. 

"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import get_path, get_return_grid
from src.phase4.cumulant_premia import cyl_weights
from src.phase1.ssvi import SSVI
from src.phase4.bkm_moments import extract_bkm_moments

R_GRID = get_return_grid()
TAUS = [14, 27, 60]
NW_LAGS = 27
KAPPA_BOUND = 1.5

def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

def _load_ssvi_params():
    params = pd.read_parquet(get_path("data_phase1") / "ssvi_params.parquet")
    params["date"] = pd.to_datetime(params["date"])
    return params

def _load_options(venue):
    key = "cleaned_cme" if venue == "CME" else "cleaned_deribit"
    df = pd.read_parquet(get_path(key))
    df["date"] = pd.to_datetime(df["date"])
    return df

def _daily_premia(venue, tau_days, lam, params=None, opts=None):
    """CL20 contributions by BKM spanning, matching the Phase 4 route."""
    if params is None:
        params = _load_ssvi_params()
    if opts is None:
        opts = _load_options(venue)
    pv = params[params["venue"] == venue]
    has_fwd = "forward" in pv.columns and pv["forward"].notna().all()
    tau_years = tau_days / 365.25

    dates, rows, n_guard = [], [], 0
    for date in sorted(pv["date"].unique()):
        params_day = pv[pv["date"] == date]
        try:
            forward_map = None
            if not has_fwd:
                df_day = opts[opts["date"] == date]
                if df_day.empty:
                    continue
                forward_map = df_day.groupby("tau")["forward_price"].mean()
            ssvi = SSVI.from_params(params_day, forward_map=forward_map,
                                    venue=venue, date=date)
            fitted = np.array(ssvi.res["maturities"]) * 365.25
            if tau_days < fitted.min() * 0.8 or tau_days > fitted.max() * 1.2:
                n_guard += 1
                continue
            b = extract_bkm_moments(ssvi, tau_years, n_strikes=500,
                                    r=0.0, kappa_bound=KAPPA_BOUND)
        except Exception:
            continue
        dates.append(date)
        rows.append([lam[0] * b.V, lam[1] * b.W, lam[2] * b.X])

    print(f"    [{venue} tau={tau_days}] {len(rows)} days extracted, "
          f"{n_guard} dropped by maturity guard")
    return pd.DataFrame(rows, columns=["Pi_2", "Pi_3", "Pi_4"],
                        index=pd.DatetimeIndex(dates))

def run_wedge_term_structure():
    # Corrected: Standardized hardcoded output layout strings
    RES_P5 = get_path("results_phase5")
    TAB = RES_P5 / "tables"
    FIG = RES_P5 / "figures"
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    lam = cyl_weights(theta=2.0)

    print("\n" + "=" * 60)
    print("  Wedge Term Structure (tau in {14, 27, 60})")
    print("=" * 60)

    params = _load_ssvi_params()
    opts = {v: _load_options(v) for v in ("CME", "DER")}

    premia, common_per_tau = {}, {}
    for tau in TAUS:
        pc = _daily_premia("CME", tau, lam, params, opts["CME"])
        pdd = _daily_premia("DER", tau, lam, params, opts["DER"])
        common = pc.index.intersection(pdd.index)
        premia[tau] = (pc, pdd)
        common_per_tau[tau] = common
        print(f"  [tau={tau}] {len(common)} matched days")

    # Days matched at ALL THREE maturities simultaneously
    common_all = None
    for tau in TAUS:
        common_all = (common_per_tau[tau] if common_all is None else common_all.intersection(common_per_tau[tau]))
    print(f"  Common to all {len(TAUS)} maturities: {len(common_all)} days")

    rows = []
    for tau in TAUS:
        pc, pdd = premia[tau]
        common = common_per_tau[tau]
        if len(common) < 50:
            print(f"  [tau={tau}] only {len(common)} matched days — skipped")
            continue
        samples = [("own", common)]
        if len(common_all) >= 50:
            samples.append(("common_all_tau", common_all))
        for k in (2, 3, 4):
            for sample_name, sample_idx in samples:
                delta = (pdd.loc[sample_idx, f"Pi_{k}"]
                         - pc.loc[sample_idx, f"Pi_{k}"]).values
                finite = np.isfinite(delta)
                n_dropped = int((~finite).sum())
                if n_dropped:
                    print(f"    [tau={tau}, k={k}, {sample_name}] dropped "
                          f"{n_dropped} non-finite day(s) before OLS")
                delta = delta[finite]
                res = sm.OLS(delta, np.ones((len(delta), 1))).fit(
                    cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
                rows.append({"tau_days": tau, "order": k, "sample": sample_name,
                             "wedge": float(res.params[0]),
                             "se": float(res.bse[0]),
                             "t_stat": float(res.tvalues[0]),
                             "p_value": float(res.pvalues[0]),
                             "stars": _stars(float(res.pvalues[0])),
                             "n_days": len(delta)})

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TAB / "wedge_term_structure.csv", index=False)
    print("\n" + tbl.round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_tbl = tbl[tbl["sample"] == "own"]
    for k, c in [(2, "C0"), (3, "C1"), (4, "C2")]:
        sub = plot_tbl[plot_tbl["order"] == k].sort_values("tau_days")
        ax.errorbar(sub["tau_days"], sub["wedge"], yerr=1.96 * sub["se"],
                    fmt="o-", color=c, capsize=3, label=rf"$\Delta\Pi_{k}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel(r"Maturity $\tau$ (days)")
    ax.set_ylabel("Cross-venue wedge (DER $-$ CME)")
    ax.set_title("Term Structure of the Cumulant Wedge (95% NW CIs)")
    ax.set_xticks(TAUS)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig_wedge_term_structure.png", dpi=150)
    plt.close()
    print(f"  Saved: {TAB / 'wedge_term_structure.csv'}")
    return tbl

if __name__ == "__main__":
    run_wedge_term_structure()