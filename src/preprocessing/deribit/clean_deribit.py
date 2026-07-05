import sys
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import get_path, FILTERS_DERIBIT
from src.preprocessing import filters

def load_scraped_trades() -> pd.DataFrame:
    path = get_path('raw_deribit_dir') / "btc_options_trades.parquet"
    if not path.exists():
        print(f"\n  ERROR: Scraped trades not found at {path}")
        print(f"  Run `python -m src.preprocessing.scrape_deribit` first.\n")
        sys.exit(1)
    return pd.read_parquet(path)


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure proper types
    for c in ['price_btc', 'price_usd', 'mark_price_btc', 'mark_price_usd',
              'index_price_usd', 'iv', 'amount']:
        if c in df.columns:
            df[c] = df[c].astype(float)

    # Pre-compute the volume-weighted numerator for vectorized aggregation.
    df['_w'] = df['amount']
    df['_w_price_btc'] = df['_w'] * df['price_btc']
    df['_w_price_usd'] = df['_w'] * df['price_usd']
    df['_w_iv'] = df['_w'] * df['iv']

    group_cols = ['date', 'expiration', 'strike', 'callput']

    agg = df.groupby(group_cols).agg(
        sum_w=('_w', 'sum'),
        sum_w_price_btc=('_w_price_btc', 'sum'),
        sum_w_price_usd=('_w_price_usd', 'sum'),
        sum_w_iv=('_w_iv', 'sum'),
        mark_price_btc=('mark_price_btc', 'last'),
        mark_price_usd=('mark_price_usd', 'last'),
        spot_price=('index_price_usd', 'last'),
        volume=('amount', 'sum'),
        n_trades=('amount', 'count'),
        days_to_expiry=('days_to_expiry', 'first'),
        tau=('tau', 'first'),
    ).reset_index()

    # Compute VWAPs 
    agg['price_btc'] = agg['sum_w_price_btc'] / agg['sum_w']
    agg['price_usd'] = agg['sum_w_price_usd'] / agg['sum_w']
    agg['iv'] = agg['sum_w_iv'] / agg['sum_w']
    agg['settlementprice'] = agg['mark_price_usd']

    # Drop intermediate weighted-sum columns
    agg = agg.drop(columns=['sum_w', 'sum_w_price_btc', 'sum_w_price_usd', 'sum_w_iv'])

    return agg

def compute_deribit_forward_pcp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the synthetic forward price via put-call parity for
    Deribit, with fallback to the spot index price.

    F = K_ATM + (C_ATM - P_ATM)

    Since Deribit is BTC-margined with r = 0, the discount factor
    e^(r*tau) = 1 and the PCP formula simplifies. For (date, exp)
    groups where no matched call-put pair exists at the ATM strike,
    falls back to F = spot_price (the previous convention).

    The Deribit basis (F_PCP - S) / S is stored as a column for
    downstream friction-proxy analysis.
    """
    df = df.copy()

    # For each (date, expiration), find the strike closest to spot
    df['moneyness_abs'] = (df['strike'] - df['spot_price']).abs()

    # Get ATM strike per group
    atm_idx = df.groupby(['date', 'expiration'])['moneyness_abs'].idxmin()
    atm_strikes = df.loc[atm_idx, ['date', 'expiration', 'strike']].rename(
        columns={'strike': 'atm_strike'}
    )

    df = df.merge(atm_strikes, on=['date', 'expiration'], how='left')

    # Get matched ATM call and put prices (using mark prices)
    atm_options = df[df['strike'] == df['atm_strike']].copy()

    calls = atm_options[atm_options['callput'] == 'C'][
        ['date', 'expiration', 'strike', 'settlementprice']
    ].rename(columns={'settlementprice': 'call_price'})

    puts = atm_options[atm_options['callput'] == 'P'][
        ['date', 'expiration', 'strike', 'settlementprice']
    ].rename(columns={'settlementprice': 'put_price'})

    pcp = calls.merge(puts, on=['date', 'expiration', 'strike'], how='inner')

    # F = K + (C - P)  since r = 0 for Deribit
    pcp['forward_pcp'] = pcp['strike'] + (pcp['call_price'] - pcp['put_price'])

    # Merge back to main dataframe
    forward_map = pcp[['date', 'expiration', 'forward_pcp']]
    df = df.merge(forward_map, on=['date', 'expiration'], how='left')

    # Coverage diagnostics
    n_pcp = df['forward_pcp'].notna().sum()
    n_total = len(df)
    n_slices_pcp = pcp[['date', 'expiration']].drop_duplicates().shape[0]
    n_slices_total = df[['date', 'expiration']].drop_duplicates().shape[0]

    # Use PCP forward where available, fallback to spot
    df['forward_price'] = df['forward_pcp'].fillna(df['spot_price'])

    # Compute the Deribit basis: (F_PCP - S) / S
    df['deribit_basis'] = (df['forward_price'] - df['spot_price']) / df['spot_price']

    # Recompute log-moneyness with the improved forward
    df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])

    # Risk-free rate remains zero
    df['risk_free_rate'] = 0.0

    # Clean up temporary columns
    df.drop(columns=['moneyness_abs', 'atm_strike', 'forward_pcp'],
            inplace=True, errors='ignore')

    print(f"\n  PCP forward computed:")
    print(f"    Slices with matched call-put pair: "
          f"{n_slices_pcp:,} / {n_slices_total:,} "
          f"({100 * n_slices_pcp / max(n_slices_total, 1):.1f}%)")
    print(f"    Rows with PCP forward: "
          f"{n_pcp:,} / {n_total:,} "
          f"({100 * n_pcp / max(n_total, 1):.1f}%)")
    basis = df.groupby('date')['deribit_basis'].mean()
    print(f"    Deribit basis (F_PCP - S) / S:")
    print(f"      Median: {basis.median():.4%}")
    print(f"      Mean:   {basis.mean():.4%}")
    print(f"      Std:    {basis.std():.4%}")
    print(f"      Range:  [{basis.min():.4%}, {basis.max():.4%}]")

    return df

def process_deribit_data():
    print("\n" + "=" * 60)
    print("  Deribit Bitcoin Options Cleaning Pipeline")
    print("=" * 60)

    # Step 1: Load scraped trades
    print("\n  Loading scraped trades...")
    df = load_scraped_trades()
    n_raw = len(df)
    print(f"  Raw trades loaded: {n_raw:,}")

    # Step 2: Grith et al. (2026) transaction-level filters
    print("\n  --- Sequential Filter Attrition ---")
    print(f"  {'Step':<40} {'Rows':>12} {'Dropped':>12}")
    print(f"  {'-' * 64}")
    print(f"  {'Raw trades':<40} {len(df):>12,}")

    # Filter 1: Remove zero/negative IV
    n_before = len(df)
    df = df[(df['iv'] > 0) & (df['iv'].notna())]
    print(f"  {'Remove IV ≤ 0 or missing':<40} {len(df):>12,} {n_before - len(df):>12,}")

    # Filter 2: Remove zero-quantity transactions
    n_before = len(df)
    df = df[df['amount'] > FILTERS_DERIBIT['min_amount']]
    print(f"  {'Remove zero-quantity trades':<40} {len(df):>12,} {n_before - len(df):>12,}")

    # Filter 3: Remove non-positive prices (basic no-arbitrage)
    n_before = len(df)
    df = df[(df['price_btc'] > 0) & (df['price_usd'] > 0)]
    print(f"  {'Remove non-positive prices':<40} {len(df):>12,} {n_before - len(df):>12,}")

    # Step 3: Aggregate to daily cross-sections (VWAP)
    print(f"\n  Aggregating to daily cross-sections (VWAP)...")
    df = aggregate_to_daily(df)
    print(f"  {'Daily cross-section rows':<40} {len(df):>12,}")

    # Drop slices with too few trades
    min_trades = FILTERS_DERIBIT['min_n_trades_per_slice']
    if min_trades > 1:
        n_before = len(df)
        df = df[df['n_trades'] >= min_trades]
        print(f"  {'Drop slices with <%d trades' % min_trades:<40} {len(df):>12,} {n_before - len(df):>12,}")

    # Step 4: Compute forward via put-call parity (with fallback to spot)
    df = compute_deribit_forward_pcp(df)

    # Step 5: Consistency filters (matching CME pipeline)
    df = df.rename(columns={'iv': 'impliedvolatility'})

    steps = [
        ("Maturity filter", lambda d: filters.filter_maturity(
            d,
            min_dte=FILTERS_DERIBIT['min_tau_days'],
            max_dte=FILTERS_DERIBIT['max_tau_days']
        )),
        ("OTM only (K≥F calls, K≤F puts)", filters.filter_out_of_the_money),
        ("Moneyness trim |κ| ≤ 3√τ", lambda d: filters.trim_extreme_moneyness(
            d, coeff=FILTERS_DERIBIT['moneyness_trim']
        )),
        ("IV bounds", lambda d: filters.filter_iv_bounds(
            d,
            min_iv=FILTERS_DERIBIT['min_iv'],
            max_iv=FILTERS_DERIBIT['max_iv']
        )),
    ]

    for step_name, step_fn in steps:
        n_before = len(df)
        df = step_fn(df)
        print(f"  {step_name:<40} {len(df):>12,} {n_before - len(df):>12,}")

    # Step 6: Final cleanup and save
    df = df.sort_values(['date', 'expiration', 'strike']).reset_index(drop=True)

    output_cols = [
        'date', 'expiration', 'days_to_expiry', 'tau',
        'strike', 'callput', 'forward_price', 'log_moneyness',
        'settlementprice', 'impliedvolatility',
        'price_btc', 'price_usd', 'mark_price_btc', 'mark_price_usd',
        'volume', 'n_trades', 'risk_free_rate', 'spot_price',
        'deribit_basis'
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols]

    out_path = get_path('cleaned_deribit')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Summary statistics
    print(f"\n  {'=' * 60}")
    print(f"  CLEANING COMPLETE")
    print(f"  {'=' * 60}")
    print(f"  Final rows:      {len(df):>12,}")
    print(f"  Date range:       {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique dates:     {df['date'].nunique():>12,}")
    print(f"  Saved to:         {out_path}")

    print(f"\n  Rows per year:")
    yearly = df.groupby(df['date'].dt.year).size()
    for year, count in yearly.items():
        print(f"    {year}: {count:>10,}")

    daily = df.groupby('date').size()
    print(f"\n  Options per day:")
    print(f"    Median: {daily.median():.0f}")
    print(f"    Min:    {daily.min()}")
    print(f"    Max:    {daily.max()}")
    print(f"    Mean:   {daily.mean():.1f}")

    # Slice density (relevant for SSVI fitting)
    per_slice = df.groupby(['date', 'expiration']).size().reset_index(name='n_options')
    print(f"\n  Options per (date, expiration) slice:")
    print(f"    Median: {per_slice['n_options'].median():.0f}")
    print(f"    Min:    {per_slice['n_options'].min()}")
    print(f"    Max:    {per_slice['n_options'].max()}")
    print(f"    Slices with < 4 options: "
          f"{(per_slice['n_options'] < 4).sum():,} "
          f"({(per_slice['n_options'] < 4).mean():.1%})")

    # Export daily Deribit basis for friction-proxy analysis
    daily_basis = df.groupby('date').agg(
        deribit_basis_mean=('deribit_basis', 'mean'),
        deribit_basis_median=('deribit_basis', 'median'),
        spot_price=('spot_price', 'last'),
        forward_price_mean=('forward_price', 'mean'),
        n_slices=('expiration', 'nunique'),
    ).reset_index()
    basis_path = out_path.parent / "deribit_daily_basis.parquet"
    daily_basis.to_parquet(basis_path, index=False)
    print(f"\n  Deribit daily basis saved to: {basis_path}")
    print(f"  ({len(daily_basis):,} days, "
          f"median basis: {daily_basis['deribit_basis_mean'].median():.4%})")

    return df

if __name__ == "__main__":
    process_deribit_data()