"""
Builds the three core conditioning specifications used by Phase 3 and Phase 5:

  Z_macro  = (VIX_t, BAA_SPREAD_t, DGS2_t, DXY_t)
  Z_crypto = (Z_IVS_1_t, RV_t, FNG_t)
  Z_full   = Z_macro ∪ Z_crypto

Plus one deferred specification for the extended-sample analysis:

  Z_crypto_hl = Z_crypto ∪ {Δf_t^{HL-DER}}  (post-May 2023 only)

"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import get_path, get_sample_window

start_str, end_str = get_sample_window()
SAMPLE_START, SAMPLE_END = pd.Timestamp(start_str), pd.Timestamp(end_str)

DATA_DIR = get_path("data_phase1")
RESULTS_DIR = get_path("results_phase1")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = get_path("cleaned_auxiliary")
STATE_PATH = DATA_DIR / "tensor_pca_state_almeida.parquet"
FUNDING_PATH = get_path("funding_diff")

MACRO_VARS = ["vix", "baa_spread", "dgs2", "dxy"]
CRYPTO_VARS = ["Z_IVS_1", "rv", "fng"]
HL_VAR = "delta_f_hl_der"

def zscore(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if len(valid) < 2 or valid.std(ddof=0) < 1e-12:
        return s * np.nan
    return (s - valid.mean()) / valid.std(ddof=0)

def load_inputs():
    print(f"  Loading auxiliary panel from {PANEL_PATH}...")
    aux = pd.read_parquet(PANEL_PATH)
    aux["date"] = pd.to_datetime(aux["date"])

    print(f"  Loading tensor state vector from {STATE_PATH}...")
    state = pd.read_parquet(STATE_PATH).reset_index()
    state["date"] = pd.to_datetime(state["date"])

    if FUNDING_PATH.exists():
        print(f"  Loading funding differential from {FUNDING_PATH}...")
        funding = pd.read_parquet(FUNDING_PATH)
        funding["date"] = pd.to_datetime(funding["date"])
        funding = funding[["date", HL_VAR]]
    else:
        print(f"  [INFO] Funding differential not available at {FUNDING_PATH}")
        funding = pd.DataFrame(columns=["date", HL_VAR])

    return aux, state, funding

def merge_all(aux: pd.DataFrame, state: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    df = aux.merge(state, on="date", how="left").merge(funding, on="date", how="left")
    mask = (df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)
    df = df[mask].sort_values("date").reset_index(drop=True)
    return df

def make_spec(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[["date"] + cols].copy()
    for col in cols:
        raw = out[col].copy()
        out[col] = zscore(out[col])
        out[f"{col}_raw"] = raw
    return out

def build_specifications(df: pd.DataFrame):
    Z_macro = make_spec(df, MACRO_VARS)
    Z_crypto = make_spec(df, CRYPTO_VARS)
    Z_full = make_spec(df, MACRO_VARS + CRYPTO_VARS)

    # Deferred HL extension: only build if the column exists and has data
    Z_crypto_hl = None
    if HL_VAR in df.columns and df[HL_VAR].notna().any():
        Z_crypto_hl = make_spec(df, CRYPTO_VARS + [HL_VAR])

    return Z_macro, Z_crypto, Z_full, Z_crypto_hl

def coverage_report(specs: dict[str, tuple[pd.DataFrame, list[str]]]) -> pd.DataFrame:
    rows = []
    for name, (df_z, cols) in specs.items():
        for col in cols:
            non_null = int(df_z[col].notna().sum())
            valid_dates = df_z.loc[df_z[col].notna(), "date"]
            rows.append({
                "specification": name,
                "variable": col,
                "n_obs": len(df_z),
                "n_non_null": non_null,
                "coverage_pct": round(100 * non_null / len(df_z), 2),
                "first_valid": valid_dates.min() if non_null > 0 else None,
                "last_valid": valid_dates.max() if non_null > 0 else None,
            })

        # Complete-case row
        complete = df_z[cols].notna().all(axis=1)
        rows.append({
            "specification": name,
            "variable": "ALL (complete case)",
            "n_obs": len(df_z),
            "n_non_null": int(complete.sum()),
            "coverage_pct": round(100 * complete.mean(), 2),
            "first_valid": df_z.loc[complete, "date"].min() if complete.any() else None,
            "last_valid": df_z.loc[complete, "date"].max() if complete.any() else None,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "conditioning_coverage.csv", index=False)
    print(f"\n  Coverage report:")
    print(summary.to_string(index=False))
    return summary

def correlation_report(Z_full: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    corr = Z_full[cols].corr()
    corr.to_csv(RESULTS_DIR / "conditioning_correlations.csv")
    print(f"\n  Pearson correlation matrix (Z_full, standardized):")
    print(corr.round(3).to_string())
    return corr

def build_conditioning_vectors():
    print("\n" + "=" * 60)
    print("  Conditioning Vector Construction")
    print("=" * 60)
    print(f"  Sample: {SAMPLE_START.date()} -> {SAMPLE_END.date()}")

    aux, state, funding = load_inputs()
    df = merge_all(aux, state, funding)
    print(f"\n  Merged panel: {len(df):,} rows over {df['date'].nunique()} unique dates")

    # Restrict to the common CME-Deribit window (days where Z_IVS_1 exists)
    common_mask = df["Z_IVS_1"].notna()
    df_common = df[common_mask].reset_index(drop=True)
    print(f"  Common CME-Deribit window: {len(df_common):,} days "
          f"({df_common['date'].min().date()} -> {df_common['date'].max().date()})")

    # Guard the CP sign convention
    if df_common["rv"].notna().sum() >= 30:
        rho_check = df_common[["Z_IVS_1", "rv"]].dropna().corr().iloc[0, 1]
        print(f"  Sign-convention check: corr(Z_IVS_1, RV) = {rho_check:+.3f}")
        if rho_check < 0:
            raise ValueError(
                f"corr(Z_IVS_1, RV) = {rho_check:.3f} < 0: the CP time factor "
                f"sign convention was not enforced. Re-run run_tensor_pca "
                f"(which anchors Z_IVS_1 to RV) before building conditioning "
                f"vectors; proceeding would invert all volatility terciles."
            )

    Z_macro, Z_crypto, Z_full, Z_crypto_hl = build_specifications(df_common)

    # Save core specifications
    Z_macro.to_parquet(DATA_DIR / "Z_macro.parquet", index=False)
    Z_crypto.to_parquet(DATA_DIR / "Z_crypto.parquet", index=False)
    Z_full.to_parquet(DATA_DIR / "Z_full.parquet", index=False)
    print(f"\n  Saved Z_macro:  {DATA_DIR / 'Z_macro.parquet'} ({Z_macro.shape})")
    print(f"  Saved Z_crypto: {DATA_DIR / 'Z_crypto.parquet'} ({Z_crypto.shape})")
    print(f"  Saved Z_full:   {DATA_DIR / 'Z_full.parquet'} ({Z_full.shape})")

    if Z_crypto_hl is not None:
        Z_crypto_hl.to_parquet(DATA_DIR / "Z_crypto_hl.parquet", index=False)
        print(f"  Saved Z_crypto_hl: {DATA_DIR / 'Z_crypto_hl.parquet'} ({Z_crypto_hl.shape})")

    # Coverage (on the common window only)
    spec_dict = {
        "Z_macro": (Z_macro, MACRO_VARS),
        "Z_crypto": (Z_crypto, CRYPTO_VARS),
        "Z_full": (Z_full, MACRO_VARS + CRYPTO_VARS),
    }
    if Z_crypto_hl is not None:
        spec_dict["Z_crypto_hl"] = (Z_crypto_hl, CRYPTO_VARS + [HL_VAR])

    coverage = coverage_report(spec_dict)
    corr = correlation_report(Z_full, MACRO_VARS + CRYPTO_VARS)

    return Z_macro, Z_crypto, Z_full, Z_crypto_hl, coverage, corr

if __name__ == "__main__":
    build_conditioning_vectors()
