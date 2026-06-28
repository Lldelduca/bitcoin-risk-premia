"""
Builds the full-window friction_proxies_daily.csv for Phase 5 regressions. 
Three friction proxies, all over the full common sample window:
 
  1. CME futures basis:  (F_CME - S) / S  (annualized)
  2. Deribit PCP basis:  (F_PCP - S) / S  (annualized)
  3. Deribit perpetual funding rate (annualized, from API)
 
Merged with the matched-day cumulant-premium differences from Phase 4, producing the regression-ready panel.
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import get_path, SAMPLE
 
CLEAN_DIR = Path(get_path("cleaned_cme")).parent
PHASE4_DIR = Path("results") / "phase4" / "tables"
OUTPUT_DIR = Path("results") / "phase5" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
DAYS_PER_YEAR = 365.25
 
 
def build():
    print("\n" + "=" * 60)
    print("  Full-Window Friction Proxy Builder")
    print("=" * 60)
 
    # 1. CME futures basis
    print("\n  [1/4] CME futures basis...")
    cme = pd.read_parquet(get_path("cleaned_cme"))
    cme['date'] = pd.to_datetime(cme['date'])
    der_spot = pd.read_parquet(CLEAN_DIR / "deribit_daily_basis.parquet")
    der_spot['date'] = pd.to_datetime(der_spot['date'])
 
    cme_daily = cme.groupby('date').agg(
        cme_forward=('forward_price', 'mean'),
    ).reset_index()
 
    basis_df = cme_daily.merge(
        der_spot[['date', 'spot_price']], on='date', how='inner'
    )
    basis_df['cme_basis'] = (
        (basis_df['cme_forward'] - basis_df['spot_price'])
        / basis_df['spot_price']
    )
    # Annualize (assuming 27-day tenor)
    basis_df['cme_basis_ann'] = basis_df['cme_basis'] * (DAYS_PER_YEAR / 27)
    print(f"    {len(basis_df):,} days, median basis: "
          f"{basis_df['cme_basis'].median():.4%} "
          f"(ann: {basis_df['cme_basis_ann'].median():.2%})")
 
    # 2. Deribit PCP basis (already computed in cleaning)
    print("\n  [2/4] Deribit PCP basis...")
    der_basis = der_spot[['date', 'deribit_basis_mean']].rename(
        columns={'deribit_basis_mean': 'der_basis'}
    )
    der_basis['der_basis_ann'] = der_basis['der_basis'] * (DAYS_PER_YEAR / 27)
    print(f"    {len(der_basis):,} days, median basis: "
          f"{der_basis['der_basis'].median():.4%} "
          f"(ann: {der_basis['der_basis_ann'].median():.2%})")
 
    # 3. Deribit perpetual funding rate
    print("\n  [3/4] Deribit perpetual funding...")
    funding_path = CLEAN_DIR / "funding_deribit.parquet"
    if funding_path.exists():
        funding = pd.read_parquet(funding_path)
        funding['date'] = pd.to_datetime(funding['date'])
        funding = funding.rename(columns={'funding_der_annual': 'der_funding'})
        print(f"    {len(funding):,} days available")
    else:
        print("    [WARN] funding_deribit.parquet not found. "
              "Run funding_diff.py first.")
        funding = pd.DataFrame(columns=['date', 'der_funding'])
 
    # 4. Merge with matched cumulant-premium differences
    print("\n  [4/4] Merging with matched-day ΔΠ_k...")
    cyl_matched = pd.read_csv(
        PHASE4_DIR / "cyl_decomposition_matched.csv"
    )

    bkm_cme = pd.read_parquet(
        Path("results") / "phase4" / "bkm_moments_CME.parquet"
    )
    bkm_der = pd.read_parquet(
        Path("results") / "phase4" / "bkm_moments_DER.parquet"
    )
    for d in (bkm_cme, bkm_der):
        d['date'] = pd.to_datetime(d['date'])
 
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
 
    # Merge all proxies onto matched days
    panel = matched[['date', 'd_Pi_2', 'd_Pi_3', 'd_Pi_4']].copy()
    panel = panel.merge(
        basis_df[['date', 'cme_basis', 'cme_basis_ann']], on='date', how='left'
    )
    panel = panel.merge(
        der_basis[['date', 'der_basis', 'der_basis_ann']], on='date', how='left'
    )
    panel = panel.merge(
        funding[['date', 'der_funding']], on='date', how='left'
    )
 
    # Standardized versions for regression
    for col in ['cme_basis', 'der_basis', 'der_funding']:
        if col in panel.columns and panel[col].notna().sum() > 0:
            panel[f'{col}_z'] = (
                (panel[col] - panel[col].mean()) / panel[col].std()
            )
 
    out_path = OUTPUT_DIR / "friction_proxies_daily.csv"
    panel.to_csv(out_path, index=False)
 
    print(f"\n  Saved: {out_path}")
    print(f"  Total matched days: {len(panel):,}")
    print(f"  CME basis coverage:     "
          f"{panel['cme_basis'].notna().sum():,} / {len(panel):,}")
    print(f"  Deribit basis coverage: "
          f"{panel['der_basis'].notna().sum():,} / {len(panel):,}")
    print(f"  Deribit funding coverage: "
          f"{panel['der_funding'].notna().sum():,} / {len(panel):,}")
 
    return panel
 
if __name__ == "__main__":
    build()
    