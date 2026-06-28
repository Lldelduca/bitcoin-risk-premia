"""
Builds the full-window friction_proxies_daily.csv for Phase 5 regressions. 
Three friction proxies, all over the full common sample window:

  1. CME futures basis:  (F_CME - S) / S  (annualized)
  2. Deribit PCP basis:  (F_PCP - S) / S  (annualized)
  3. Deribit perpetual funding rate (annualized, from API)

Merged with the matched-day cumulant-premium differences from Phase 4, producing the regression-ready panel

"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import get_path, SAMPLE

CLEAN_DIR = Path(get_path("cleaned_cme")).parent
PHASE4_DIR = Path("results") / "phase4"
OUTPUT_DIR = Path("results") / "phase5" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS_PER_YEAR = 365.25
TAU_DAYS = 27
THETA = 2.0

def build():
    print("\n" + "=" * 60)
    print("  Full-Window Friction Proxy Builder")
    print(f"  Window: {SAMPLE['start_date']} -> {SAMPLE['end_date']}")
    print("=" * 60)

    # 1. CME futures basis
    print("\n  [1/4] CME futures basis...")
    cme = pd.read_parquet(get_path("cleaned_cme"))
    cme['date'] = pd.to_datetime(cme['date'])

    # Daily mean CME forward price
    cme_daily = cme.groupby('date').agg(cme_forward=('forward_price', 'mean'),).reset_index()

    # Load Deribit spot for the basis denominator
    der_basis_path = CLEAN_DIR / "deribit_daily_basis.parquet"
    if not der_basis_path.exists():
        print(f"  ERROR: {der_basis_path} not found.")
        print(f"  Run clean_deribit.py (PCP version) first.")
        return None

    der_spot = pd.read_parquet(der_basis_path)
    der_spot['date'] = pd.to_datetime(der_spot['date'])

    basis_df = cme_daily.merge(
        der_spot[['date', 'spot_price']], on='date', how='inner'
    )
    basis_df['cme_basis'] = (
        (basis_df['cme_forward'] - basis_df['spot_price'])
        / basis_df['spot_price']
    )
    basis_df['cme_basis_ann'] = basis_df['cme_basis'] * (DAYS_PER_YEAR / TAU_DAYS)
    print(f"    {len(basis_df):,} days")
    print(f"    Median basis: {basis_df['cme_basis'].median():.4%} "
          f"(ann: {basis_df['cme_basis_ann'].median():.2%})")

    # 2. Deribit PCP basis
    print("\n  [2/4] Deribit PCP basis...")
    der_basis = der_spot[['date', 'deribit_basis_mean']].rename(
        columns={'deribit_basis_mean': 'der_basis'}
    )
    der_basis['der_basis_ann'] = der_basis['der_basis'] * (DAYS_PER_YEAR / TAU_DAYS)
    print(f"    {len(der_basis):,} days")
    print(f"    Median basis: {der_basis['der_basis'].median():.4%} "
          f"(ann: {der_basis['der_basis_ann'].median():.2%})")

    # 3. Deribit perpetual funding rate
    print("\n  [3/4] Deribit perpetual funding...")
    funding_path = CLEAN_DIR / "funding_deribit.parquet"
    if funding_path.exists():
        funding = pd.read_parquet(funding_path)
        funding['date'] = pd.to_datetime(funding['date'])
        funding = funding.rename(columns={'funding_der_annual': 'der_funding'})
        # Trim to common window
        funding = funding[
            (funding['date'] >= SAMPLE['start_date']) &
            (funding['date'] <= SAMPLE['end_date'])
        ]
        print(f"    {len(funding):,} days in common window")
    else:
        print("    [WARN] funding_deribit.parquet not found. "
              "Run funding_diff.py first. Proceeding without funding proxy.")
        funding = pd.DataFrame(columns=['date', 'der_funding'])

    # 4. Merge with matched-day cumulant-premium differences
    print("\n  [4/4] Merging with matched-day ΔΠ_k...")

    # Load daily BKM moments per venue
    bkm_cme_path = PHASE4_DIR / "bkm_moments_CME.parquet"
    bkm_der_path = PHASE4_DIR / "bkm_moments_DER.parquet"

    if not bkm_cme_path.exists() or not bkm_der_path.exists():
        # Try alternative paths (tables subdirectory)
        bkm_cme_path = PHASE4_DIR / "tables" / "bkm_moments_CME.parquet"
        bkm_der_path = PHASE4_DIR / "tables" / "bkm_moments_DER.parquet"

    if not bkm_cme_path.exists() or not bkm_der_path.exists():
        print(f"  ERROR: BKM moment files not found at {PHASE4_DIR}")
        print(f"  Run run_phase4.py first.")
        return None

    bkm_cme = pd.read_parquet(bkm_cme_path)
    bkm_der = pd.read_parquet(bkm_der_path)
    for d in (bkm_cme, bkm_der):
        d['date'] = pd.to_datetime(d['date'])

    # Filter to the target maturity
    if 'tau_days' in bkm_cme.columns:
        bkm_cme = bkm_cme[bkm_cme['tau_days'] == TAU_DAYS]
        bkm_der = bkm_der[bkm_der['tau_days'] == TAU_DAYS]

    # Compute CL20 contributions (theta=2: lambda = 1, -1, 1)
    for d in (bkm_cme, bkm_der):
        d['Pi_2'] = 1.0 * d['V']
        d['Pi_3'] = -1.0 * d['W']
        d['Pi_4'] = 1.0 * d['X']

    matched = bkm_cme[['date', 'Pi_2', 'Pi_3', 'Pi_4']].merge(
        bkm_der[['date', 'Pi_2', 'Pi_3', 'Pi_4']],
        on='date', suffixes=('_cme', '_der'), how='inner'
    )
    matched['d_Pi_2'] = matched['Pi_2_der'] - matched['Pi_2_cme']
    matched['d_Pi_3'] = matched['Pi_3_der'] - matched['Pi_3_cme']
    matched['d_Pi_4'] = matched['Pi_4_der'] - matched['Pi_4_cme']
    print(f"    Matched BKM days: {len(matched):,}")

    # Merge all proxies onto matched days
    panel = matched[['date', 'd_Pi_2', 'd_Pi_3', 'd_Pi_4']].copy()
    panel = panel.merge(
        basis_df[['date', 'cme_basis', 'cme_basis_ann']],
        on='date', how='left'
    )
    panel = panel.merge(
        der_basis[['date', 'der_basis', 'der_basis_ann']],
        on='date', how='left'
    )
    panel = panel.merge(
        funding[['date', 'der_funding']],
        on='date', how='left'
    )

    # Standardized versions for regression
    for col in ['cme_basis', 'der_basis', 'der_funding']:
        if col in panel.columns and panel[col].notna().sum() > 1:
            panel[f'{col}_z'] = (
                (panel[col] - panel[col].mean()) / panel[col].std()
            )

    out_path = OUTPUT_DIR / "friction_proxies_daily.csv"
    panel.to_csv(out_path, index=False)

    print(f"\n  {'=' * 50}")
    print(f"  FRICTION PROXY PANEL COMPLETE")
    print(f"  {'=' * 50}")
    print(f"  Saved: {out_path}")
    print(f"  Total matched days: {len(panel):,}")
    print(f"  CME basis coverage:       "
          f"{panel['cme_basis'].notna().sum():>5,} / {len(panel):,}")
    print(f"  Deribit PCP basis coverage: "
          f"{panel['der_basis'].notna().sum():>5,} / {len(panel):,}")
    print(f"  Deribit funding coverage:   "
          f"{panel['der_funding'].notna().sum():>5,} / {len(panel):,}")

    # Quick correlation check
    print(f"\n  Proxy correlations:")
    proxy_cols = [c for c in ['cme_basis', 'der_basis', 'der_funding']
                  if c in panel.columns and panel[c].notna().sum() > 10]
    if len(proxy_cols) > 1:
        print(panel[proxy_cols].corr().round(3).to_string())

    return panel

if __name__ == "__main__":
    build()
