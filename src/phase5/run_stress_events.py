"""
Stress-Event Dynamics of the Wedge

The Z-interaction tests use smooth volatility terciles; event dummies are the sharp instrument. Regresses the daily 
cross-venue wedge dPi_k on indicator windows for the three major crypto stress episodes in the sample, with NW(27) errors:

    COVID crash   2020-03-01 .. 2020-04-15
    Terra/LUNA    2022-05-01 .. 2022-06-15
    FTX collapse  2022-11-01 .. 2022-12-15

If the wedge widens exactly when arbitrage capital evaporates, that is limits-to-arbitrage evidence complementing the no-trade 
band; if it does not move, its stability is the finding (structural clientele segmentation rather than funding-constrained 
arbitrage).

"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import get_path

NW_LAGS = 27
EVENTS = {
    "COVID": ("2020-03-01", "2020-04-15"),
    "Terra": ("2022-05-01", "2022-06-15"),
    "FTX":   ("2022-11-01", "2022-12-15"),
}

def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def stress_regression_table(wedges: pd.DataFrame, events=EVENTS,
                            nw_lags=NW_LAGS) -> pd.DataFrame:
    """wedges: DataFrame indexed by date, columns dPi_2..4."""
    D = pd.DataFrame(index=wedges.index)
    for name, (a, b) in events.items():
        D[name] = ((wedges.index >= pd.Timestamp(a))
                   & (wedges.index <= pd.Timestamp(b))).astype(float)
    rows = []
    for dep in wedges.columns:
        X = sm.add_constant(D.values)
        res = sm.OLS(wedges[dep].values, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        names = ["const (non-event wedge)"] + list(events.keys())
        for j, nm in enumerate(names):
            rows.append({
                "dep_var": dep, "regressor": nm,
                "coef": float(res.params[j]),
                "t_stat": float(res.tvalues[j]),
                "p_value": float(res.pvalues[j]),
                "stars": _stars(float(res.pvalues[j])),
                "n_days": int(res.nobs),
                "n_event_days": int(D[nm].sum()) if nm in D.columns else np.nan,
                "nw_lags": nw_lags,
            })
    return pd.DataFrame(rows)


def run_stress_events():
    # Corrected: Swapped hardcoded output strings for dynamic config variables
    RES_P5 = get_path("results_phase5")
    TAB = RES_P5 / "tables"
    FIG = RES_P5 / "figures"
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Stress-Event Dynamics of the Cumulant Wedge")
    print("=" * 60)

    # Corrected: Pointed directly to Phase 4 input directory registry key
    panel = pd.read_parquet(get_path("data_phase4") / "cumulant_premia.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    wedges = pd.DataFrame({
        f"dPi_{k}": (panel.pivot_table(index="date", columns="venue",
                                       values=f"Pi_{k}")
                     .pipe(lambda p: p["DER"] - p["CME"]))
        for k in (2, 3, 4)}).dropna()
    print(f"  Daily wedges: {len(wedges)} matched days")

    tbl = stress_regression_table(wedges)
    tbl.to_csv(TAB / "stress_event_regressions.csv", index=False)
    for dep in ["dPi_2", "dPi_3", "dPi_4"]:
        sub = tbl[tbl.dep_var == dep]
        line = "  ".join(f"{r['regressor'].split()[0]}={r['coef']:+.5f}"
                         f"(t={r['t_stat']:+.2f}){r['stars']}"
                         for _, r in sub.iterrows())
        print(f"  {dep}: {line}")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(wedges.index, wedges["dPi_2"], lw=0.6, color="C0",
            label=r"$\Delta\Pi_2$ (DER $-$ CME)")
    ax.plot(wedges.index,
            wedges["dPi_2"].rolling(27, min_periods=10).mean(),
            lw=1.5, color="C3", label="27-day rolling mean")
    for name, (a, b) in EVENTS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.15,
                   color="red")
        ax.text(pd.Timestamp(a), ax.get_ylim()[1], f" {name}", va="top",
                fontsize=8, color="darkred")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(r"$\Delta\Pi_2$")
    ax.set_title("Daily Variance Wedge Around Stress Events")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig_wedge_stress_events.png", dpi=150)
    plt.close()
    print(f"  Saved: {TAB / 'stress_event_regressions.csv'}")
    return tbl


if __name__ == "__main__":
    run_stress_events()