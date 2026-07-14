"""
Explanatory extension 2 (driver) - What explains the cross-venue cumulant-premium wedge?

Question: what frictions drive the wedge? Are they observable, or are they latent and option-implied? 

Solution: Run a regression of the matched-day wedge on two observable frictions that are independent of 
the option-implied densities that produce Π_k. The two frictions are:

  1. CME annualized futures basis:  (F_CME / S - 1) / tau
     from the CME futures settlement price and BTC spot. A regulated-venue cost-of-carry measure

  2. Deribit annualized perpetual funding:
     the financing cost of leveraged exposure on the offshore venue

These enter the DER-CME wedge with opposite expected signs: 
A Deribit-specific friction (funding) should widen the wedge, a CME-specific carry measure should narrow it. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from src.config import get_path, get_return_grid, get_sample_window

DATA_P4 = get_path("data_phase4")
DATA_P5 = get_path("data_phase5")
RES_P5 = get_path("results_phase5")

FIG_DIR = RES_P5 / "figures"
TAB_DIR = RES_P5 / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

NW_LAGS = 27
SAMPLE_START_STR, SAMPLE_END_STR = get_sample_window()
SAMPLE_START = pd.to_datetime(SAMPLE_START_STR)
SAMPLE_END = pd.to_datetime(SAMPLE_END_STR)

def _annualize(level_basis, tau_years):
    return level_basis / tau_years

def build_cme_basis():
    cme = pd.read_parquet(get_path("cleaned_cme"))
    cme["date"] = pd.to_datetime(cme["date"])
    if "tau" not in cme.columns:
        cme["tau"] = (pd.to_datetime(cme["expiration"]) - cme["date"]).dt.days / 365.25

    spot = pd.read_parquet(get_path("cleaned_auxiliary"))[["date", "btc_spot"]].dropna()
    spot["date"] = pd.to_datetime(spot["date"])

    fwd = (cme[["date", "expiration", "tau", "futuresettlementprice"]]
           .dropna(subset=["futuresettlementprice"])
           .drop_duplicates(["date", "expiration"]))
    fwd = fwd.merge(spot, on="date", how="inner")
    fwd = fwd[fwd["btc_spot"] > 0]
    fwd["basis_period"] = fwd["futuresettlementprice"] / fwd["btc_spot"] - 1.0
    fwd["tau"] = fwd["tau"].clip(lower=1e-6)
    fwd["basis_annual"] = _annualize(fwd["basis_period"], fwd["tau"])

    target = 27.0 / 365.25
    fwd["dist"] = (fwd["tau"] - target).abs()
    idx = fwd.groupby("date")["dist"].idxmin()
    basis = fwd.loc[idx, ["date", "basis_annual"]].rename(
        columns={"basis_annual": "cme_basis"}).sort_values("date")
    return basis

def load_deribit_funding():
    f = pd.read_parquet(get_path("cleaned_dir") / "funding_deribit.parquet")
    f["date"] = pd.to_datetime(f["date"])
    return f[["date", "funding_der_annual"]].rename(columns={"funding_der_annual": "der_funding"}).dropna().sort_values("date")

def build_wedge():
    premia = pd.read_parquet(DATA_P4 / "cumulant_premia.parquet")
    premia["date"] = pd.to_datetime(premia["date"])
    wide = premia.pivot_table(index="date", columns="venue",
                              values=["Pi_2", "Pi_3", "Pi_4"])
    out = {}
    for k in ("Pi_2", "Pi_3", "Pi_4"):
        try:
            d = (wide[(k, "DER")] - wide[(k, "CME")]).dropna()
        except KeyError:
            continue
        out[k] = d.rename(f"d_{k}")
    return pd.concat(out.values(), axis=1).reset_index()

def _fit(y, Xcols_df, names):
    X = sm.add_constant(Xcols_df.values)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = res.rsquared if tss > 1e-15 else np.nan
    return res, ["const"] + names, r2

def run_friction_regressions():
    print("\n" + "=" * 60)
    print("  Phase 5 extension 2: Friction-proxy regressions")
    print(f"  Window: {SAMPLE_START_STR} -> {SAMPLE_END_STR} "
          f"(common, CME-bounded)")
    print("=" * 60)

    wedge = build_wedge()
    basis = build_cme_basis()
    funding = load_deribit_funding()

    df = (wedge.merge(basis, on="date", how="inner").merge(funding, on="date", how="inner"))
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)]
    df = df.dropna(subset=["d_Pi_2", "d_Pi_3", "d_Pi_4", "cme_basis", "der_funding"])
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    print(f"\n  Matched days with both frictions on the common window: {n}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    for c in ("cme_basis", "der_funding"):
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

    rho = df[["cme_basis", "der_funding"]].corr().iloc[0, 1]
    print(f"  corr(CME basis, Deribit funding) = {rho:+.3f}  "
          f"{'(watch collinearity)' if abs(rho) > 0.6 else '(low — both safe in one regression)'}")

    df.to_csv(TAB_DIR / "friction_proxies_daily.csv", index=False)

    rows = []
    specs = {
        "basis_only": ["cme_basis_z"],
        "funding_only": ["der_funding_z"],
        "joint": ["cme_basis_z", "der_funding_z"],
    }
    print(f"\n  Newey-West({NW_LAGS}) regressions of ΔΠ_k on standardized frictions:")
    for dep in ("d_Pi_2", "d_Pi_3", "d_Pi_4"):
        y = df[dep].values
        print(f"\n    {dep}  (n={n}):")
        for spec_name, cols in specs.items():
            res, names, r2 = _fit(y, df[cols], [c[:-2] for c in cols])
            for j, nm in enumerate(names):
                rows.append({
                    "dep_var": dep, "spec": spec_name, "regressor": nm,
                    "coef": res.params[j], "se": res.bse[j],
                    "t_stat": res.tvalues[j], "p_value": res.pvalues[j],
                    "stars": ("***" if res.pvalues[j] < 0.01 else
                              "**" if res.pvalues[j] < 0.01 else
                              "*" if res.pvalues[j] < 0.05 else ""),
                    "r2": r2, "n": n,
                })
            if spec_name == "joint":
                coefs = ", ".join(
                    f"{nm}={res.params[j]:+.5f}{'***' if res.pvalues[j]<0.01 else '**' if res.pvalues[j]<0.05 else '*' if res.pvalues[j]<0.10 else ''}"
                    for j, nm in enumerate(names))
                r2s = f"{r2:.3f}" if np.isfinite(r2) else "n/a"
                print(f"      [joint]  {coefs}   R²={r2s}")

    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "friction_regressions.csv", index=False)
    print(f"\n  Saved: {TAB_DIR / 'friction_regressions.csv'}")
    print(f"  Saved: {TAB_DIR / 'friction_proxies_daily.csv'}")

    # Binned-scatter figure: wedge (Pi_2) vs each proxy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (proxy, lab) in zip(axes, [("cme_basis", "CME futures basis (ann.)"),
                                        ("der_funding", "Deribit perp funding (ann.)")]):
        x = df[proxy].values
        y = df["d_Pi_2"].values
        bins = np.quantile(x, np.linspace(0, 1, 11))
        bins = np.unique(bins)
        idx = np.clip(np.digitize(x, bins[1:-1]), 0, len(bins) - 2)
        bx = np.array([x[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).any()])
        by = np.array([y[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).any()])
        ax.scatter(x, y, s=6, alpha=0.15, color="0.6")
        ax.scatter(bx, by, s=45, color="C0", zorder=5, label="decile means")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel(r"$\Delta\Pi_2$ (DER $-$ CME)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Cross-venue variance-premium wedge vs market frictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_friction_wedge.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_friction_wedge.png'}")
    print("\n  Friction-proxy extension complete.")
    return out

if __name__ == "__main__":
    run_friction_regressions()