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
from src.phase3.bootstrap_inference import (
    block_bootstrap_mean_bands, block_bootstrap_group_mean_bands,
    circular_block_indices,
)

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
    psi_df = psi_df.sort_index()  # chronological order required by the block bootstrap

    # Merge with tercile labels
    psi_df = psi_df.join(tercile_labels[["tercile"]], how="left")

    results = {}

    # Inference
    psi_vals = psi_df.drop(columns=["tercile"]).values.astype(float)
    bands = block_bootstrap_mean_bands(psi_vals, block_length=27, B=1000, seed=42)
    results["unconditional"] = {
        "R_grid": R_grid,
        "mean_psi": bands["mean"],
        "se_psi": bands["se_boot"],
        "lo_psi": bands["lo"],
        "hi_psi": bands["hi"],
        "n_days": bands["n_days"],
    }

    # By tercile
    labels = psi_df["tercile"].astype(object).values
    g_bands = block_bootstrap_group_mean_bands(
        psi_vals, labels, ["low", "mid", "high"],
        block_length=27, B=1000, seed=42,
    )
    for tercile, b in g_bands.items():
        results[tercile] = {
            "R_grid": R_grid,
            "mean_psi": b["mean"],
            "se_psi": b["se_boot"],
            "lo_psi": b["lo"],
            "hi_psi": b["hi"],
            "n_days": b["n_days"],
        }

    return results, psi_df

# Component 2: Cumulant Premium Panel Regressions: Π_{k,t}^j on Venue dummy + state variables + interactions
def driscoll_kraay_ols(y, X, dates, maxlags=27):
    from scipy import stats as _stats

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    dates = pd.to_datetime(pd.Series(dates)).values

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Date-level summed scores
    score = X * resid[:, None]
    score_df = pd.DataFrame(score)
    score_df["date"] = dates
    h = score_df.groupby("date").sum().sort_index().values  # T_dates × k

    T = h.shape[0]
    L = min(maxlags, T - 1)
    S = h.T @ h
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        gamma = h[lag:].T @ h[:-lag]
        S += w * (gamma + gamma.T)

    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t_stat = beta / se
    df_resid = T - X.shape[1]
    p_val = 2.0 * _stats.t.sf(np.abs(t_stat), df=max(df_resid, 1))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"params": beta, "bse": se, "tvalues": t_stat, "pvalues": p_val,
            "rsquared": r2, "nobs": len(y), "n_dates": T, "maxlags": L}

def run_matched_difference_regressions(premia_df, Z_df, z_cols=None, nw_lags=27):
    premia = premia_df.copy()
    premia["date"] = pd.to_datetime(premia["date"])
    Z = Z_df.copy()
    Z["date"] = pd.to_datetime(Z["date"])
    if z_cols is None:
        z_cols = [c for c in Z.columns if c != "date"]

    wide = premia.pivot_table(index="date", columns="venue",
                              values=["Pi_2", "Pi_3", "Pi_4"])
    results = {}
    panels = {}
    for dep_var in ["Pi_2", "Pi_3", "Pi_4"]:
        try:
            delta = (wide[(dep_var, "DER")] - wide[(dep_var, "CME")]).dropna()
        except KeyError:
            continue
        df_k = delta.rename("delta").reset_index().merge(
            Z[["date"] + z_cols], on="date", how="inner"
        ).dropna(subset=["delta"] + z_cols).sort_values("date")

        y = df_k["delta"].values
        X = sm.add_constant(df_k[z_cols].values)
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        res.col_names = ["const (venue wedge)"] + z_cols
        results[dep_var] = res
        panels[dep_var] = df_k
    return results, panels

def run_cumulant_panel_regressions(premia_df, Z_df, z_cols=None, nw_lags=27,
                                   driscoll_kraay=True, matched_only=True):
    premia = premia_df.copy()
    premia["date"] = pd.to_datetime(premia["date"])
    Z = Z_df.copy()
    Z["date"] = pd.to_datetime(Z["date"])

    if z_cols is None:
        z_cols = [c for c in Z.columns if c != "date"]

    # Merge
    panel = premia.merge(Z, on="date", how="inner")

    # Balanced matched panel
    if matched_only:
        counts = panel.groupby("date")["venue"].nunique()
        both = counts[counts == 2].index
        panel = panel[panel["date"].isin(both)].copy()

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
        dates_clean = panel.loc[valid, "date"].values

        if driscoll_kraay:
            dk = driscoll_kraay_ols(y_clean, X_clean, dates_clean, maxlags=nw_lags)
            res = _DKResult(dk, col_names)
        else:
            # Legacy stacked-panel HAC — comparison only; SEs unreliable
            # (ignores contemporaneous cross-venue correlation).
            model = sm.OLS(y_clean, X_clean)
            res = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
            res.col_names = col_names
        results[dep_var] = res

    return results, panel

class _DKResult:
    def __init__(self, dk, col_names):
        self.params = dk["params"]
        self.bse = dk["bse"]
        self.tvalues = dk["tvalues"]
        self.pvalues = dk["pvalues"]
        self.rsquared = dk["rsquared"]
        self.nobs = dk["nobs"]
        self.n_dates = dk["n_dates"]
        self.maxlags = dk["maxlags"]
        self.col_names = col_names

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

    # Summary table with block-bootstrap inference (block = 27 days)
    regional_df = regional_df.sort_values("date").reset_index(drop=True)
    vals = regional_df[["psi_down", "psi_mid", "psi_up"]].values.astype(float)
    labels = regional_df["tercile"].astype(object).values

    def _row(regime, band, n_days):
        return {
            "regime": regime, "n_days": n_days,
            "mean_down": band["mean"][0], "mean_mid": band["mean"][1], "mean_up": band["mean"][2],
            "se_down": band["se_boot"][0], "se_mid": band["se_boot"][1], "se_up": band["se_boot"][2],
            "lo_down": band["lo"][0], "hi_down": band["hi"][0],
            "lo_mid": band["lo"][1], "hi_mid": band["hi"][1],
            "lo_up": band["lo"][2], "hi_up": band["hi"][2],
        }

    summary_rows = []
    uncond = block_bootstrap_mean_bands(vals, block_length=27, B=1000, seed=42)
    summary_rows.append(_row("unconditional", uncond, uncond["n_days"]))

    g_bands = block_bootstrap_group_mean_bands(
        vals, labels, ["low", "mid", "high"], block_length=27, B=1000, seed=42,
    )
    for tercile in ["low", "mid", "high"]:
        if tercile in g_bands:
            b = g_bands[tercile]
            summary_rows.append(_row(tercile, b, b["n_days"]))

    return regional_df, pd.DataFrame(summary_rows)
