"""
Phase 5: Cross-Venue Analysis with Conditional MFK, Panel Regressions, Regional MFK

Component 1: Conditional Microstructure Friction Kernel

Component 2: Cumulant Premium Panel Regressions

Component 3: Regional MFK Integration

"""

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from typing import Dict, List
import statsmodels.api as sm

# Component 1: Conditional MFK by volatility tercile
def compute_conditional_mfk(rnd_cme_path, rnd_der_path, tercile_labels, tau_days=27, R_grid=None):
    if R_grid is None:
        R_grid = np.linspace(0.40, 2.00, 500)

    rnd_cme = pd.read_parquet(rnd_cme_path)
    rnd_der = pd.read_parquet(rnd_der_path)
    rnd_cme["date"] = pd.to_datetime(rnd_cme["date"])
    rnd_der["date"] = pd.to_datetime(rnd_der["date"])

    # Filter to target maturity
    cme = rnd_cme[rnd_cme["tau_days"] == tau_days].set_index("date")
    der = rnd_der[rnd_der["tau_days"] == tau_days].set_index("date")

    # Matched days
    matched_dates = cme.index.intersection(der.index)
    tercile_labels = tercile_labels.set_index("date")

    # Compute daily Ψ_t(R) on the common grid
    daily_psi = {}
    for date in matched_dates:
        R_cme = np.array(cme.loc[date, "returns"])
        q_cme = np.array(cme.loc[date, "density"])
        R_der = np.array(der.loc[date, "returns"])
        q_der = np.array(der.loc[date, "density"])

        # Interpolate both densities onto common grid
        q_c = np.interp(R_grid, R_cme, q_cme, left=1e-20, right=1e-20)
        q_d = np.interp(R_grid, R_der, q_der, left=1e-20, right=1e-20)

        # Floor to avoid log(0)
        q_c = np.maximum(q_c, 1e-20)
        q_d = np.maximum(q_d, 1e-20)

        psi = np.log(q_c / q_d)
        daily_psi[date] = psi

    psi_df = pd.DataFrame(daily_psi, index=R_grid).T
    psi_df.index.name = "date"

    # Merge with tercile labels
    psi_df = psi_df.join(tercile_labels[["tercile"]], how="left")

    results = {}

    # Unconditional
    psi_vals = psi_df.drop(columns=["tercile"]).values
    results["unconditional"] = {
        "R_grid": R_grid,
        "mean_psi": np.mean(psi_vals, axis=0),
        "se_psi": np.std(psi_vals, axis=0) / np.sqrt(len(psi_vals)),
        "n_days": len(psi_vals),
    }

    # By tercile
    for tercile in ["low", "mid", "high"]:
        mask = psi_df["tercile"] == tercile
        vals = psi_df.loc[mask].drop(columns=["tercile"]).values
        if len(vals) > 0:
            results[tercile] = {
                "R_grid": R_grid,
                "mean_psi": np.mean(vals, axis=0),
                "se_psi": np.std(vals, axis=0) / np.sqrt(len(vals)),
                "n_days": len(vals),
            }

    return results, psi_df

# Component 2: Panel Regressions: Π_{k,t}^j on Venue dummy + state variables + interactions
def run_cumulant_panel_regressions(premia_df, Z_df, z_cols=None, nw_lags=10):
    premia = premia_df.copy()
    premia["date"] = pd.to_datetime(premia["date"])
    Z = Z_df.copy()
    Z["date"] = pd.to_datetime(Z["date"])

    if z_cols is None:
        z_cols = [c for c in Z.columns if c != "date"]

    # Merge
    panel = premia.merge(Z, on="date", how="inner")

    # Venue dummy
    panel["DER"] = (panel["venue"] == "DER").astype(float)

    # Interaction terms
    for col in z_cols:
        panel[f"DER_x_{col}"] = panel["DER"] * panel[col]

    # Regressors: constant + DER + Z_cols + DER×Z_cols
    x_cols = ["DER"] + z_cols + [f"DER_x_{col}" for col in z_cols]

    results = {}
    for dep_var in ["Pi_2", "Pi_3", "Pi_4"]:
        y = panel[dep_var].values
        X = sm.add_constant(panel[x_cols].values)
        col_names = ["const", "DER"] + z_cols + [f"DER×{c}" for c in z_cols]

        valid = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        y_clean = y[valid]
        X_clean = X[valid]

        model = sm.OLS(y_clean, X_clean)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        res.col_names = col_names
        results[dep_var] = res

    return results, panel

def format_regression_table(results, dep_vars=None):
    if dep_vars is None:
        dep_vars = ["Pi_2", "Pi_3", "Pi_4"]

    rows = []
    for dep in dep_vars:
        res = results[dep]
        for i, name in enumerate(res.col_names):
            rows.append({
                "dep_var": dep,
                "regressor": name,
                "coef": res.params[i],
                "se": res.bse[i],
                "t_stat": res.tvalues[i],
                "p_value": res.pvalues[i],
                "stars": _stars(res.pvalues[i]),
            })
    df = pd.DataFrame(rows)

    # Add R-squared and N
    for dep in dep_vars:
        res = results[dep]
        df = pd.concat([df, pd.DataFrame([{
            "dep_var": dep, "regressor": "R²",
            "coef": res.rsquared, "se": np.nan, "t_stat": np.nan,
            "p_value": np.nan, "stars": "",
        }, {
            "dep_var": dep, "regressor": "N",
            "coef": int(res.nobs), "se": np.nan, "t_stat": np.nan,
            "p_value": np.nan, "stars": "",
        }])], ignore_index=True)

    return df

def _stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

# Component 3: Regional MFK Integration over three return regions and by tercile
def compute_regional_mfk(psi_df, R_grid, tercile_col="tercile"):
    R = R_grid
    psi_vals = psi_df.drop(columns=[tercile_col], errors="ignore").values

    # Region masks
    down_mask = R < 0.90
    mid_mask = (R >= 0.90) & (R <= 1.10)
    up_mask = R > 1.10

    def _integrate_region(psi_row, mask):
        if mask.sum() < 2:
            return np.nan
        return trapezoid(psi_row[mask], R[mask])

    records = []
    for i, (date, row) in enumerate(psi_df.iterrows()):
        psi = row.drop(labels=[tercile_col], errors="ignore").values.astype(float)
        tercile = row.get(tercile_col, np.nan)
        records.append({
            "date": date,
            "tercile": tercile,
            "psi_down": _integrate_region(psi, down_mask),
            "psi_mid": _integrate_region(psi, mid_mask),
            "psi_up": _integrate_region(psi, up_mask),
        })

    regional_df = pd.DataFrame(records)

    # Summary table
    summary_rows = []

    def _agg(v, regime):
        return {
            "regime": regime,
            "n_days": len(v),
            "mean_down": v["psi_down"].mean(),
            "mean_mid": v["psi_mid"].mean(),
            "mean_up": v["psi_up"].mean(),
            "se_down": v["psi_down"].std() / np.sqrt(len(v)),
            "se_mid": v["psi_mid"].std() / np.sqrt(len(v)),
            "se_up": v["psi_up"].std() / np.sqrt(len(v)),
        }

    summary_rows.append(_agg(regional_df, "unconditional"))
    for tercile in ["low", "mid", "high"]:
        mask = regional_df["tercile"] == tercile
        if mask.sum() > 0:
            summary_rows.append(_agg(regional_df[mask], tercile))

    return regional_df, pd.DataFrame(summary_rows)
