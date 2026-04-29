import sys
import pandas as pd
import numpy as np
from src.config import get_path, FILTERS
from src.preprocessing import filters

def convert_csv_to_parquet():
    """One-time conversion of the 740MB CSV to Parquet for fast loading."""
    csv_path = get_path('raw_cme_csv')
    parquet_path = get_path('raw_cme_parquet')

    if parquet_path.exists():
        print(f"  Parquet already exists at {parquet_path.name}. Skipping conversion.")
        return

    if not csv_path.exists():
        print(f"\n{'='*60}")
        print(f"  ERROR: Raw CSV not found at: {csv_path}")
        print(f"  Place the IvyDB file in: {csv_path.parent}")
        print(f"{'='*60}\n")
        sys.exit(1)

    print(f"  Converting CSV → Parquet (this takes ~30s for 740MB)...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, index=False)
    print(f"  Done. Parquet saved to {parquet_path.name}")


def load_zero_curve() -> pd.DataFrame:
    """Loads the OptionMetrics USD zero-coupon yield curve."""
    zc_path = get_path('zero_curve')
    if not zc_path.exists():
        print(f"\n  ERROR: Zero curve not found at {zc_path}\n")
        sys.exit(1)

    df_zc = pd.read_csv(zc_path, parse_dates=['date'])
    df_zc = df_zc[df_zc['currency'] == 'USD'].copy()
    return df_zc.sort_values(['date', 'days'])


def process_cme_data():
    """Main cleaning pipeline for CME Bitcoin futures options."""

    print("\n" + "="*60)
    print("  CME Bitcoin Options Cleaning Pipeline")
    print("="*60)

    # ── Step 0: CSV → Parquet ──────────────────────────────────
    convert_csv_to_parquet()

    print("\n  Loading CME Parquet into memory...")
    df = pd.read_parquet(get_path('raw_cme_parquet'))
    n_raw = len(df)
    print(f"  Raw rows loaded: {n_raw:,}")

    # ── Step 1: Parse dates, compute DTE and τ ─────────────────
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['days_to_expiry'] = (df['expiration'] - df['date']).dt.days
    df['tau'] = df['days_to_expiry'] / 365.25

    # Note: strikemultiplier is always 1.0 for BTC futures options,
    # but we apply it for correctness in case of future data changes.
    df['strike'] = df['strike'] * df['strikemultiplier']

    # Preliminary forward and moneyness (refined after PCP in Step 4)
    df['forward_price'] = df['futuresettlementprice']
    df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])

    # ── Step 2: Merge risk-free rate from zero curve ───────────
    print("  Merging zero curve...")
    df_zc = load_zero_curve()

    df = df.sort_values('days_to_expiry')
    df_zc = df_zc.sort_values('days')

    df = pd.merge_asof(
        df,
        df_zc[['date', 'days', 'rate']],
        by='date',
        left_on='days_to_expiry',
        right_on='days',
        direction='nearest'
    )
    df.rename(columns={'rate': 'risk_free_rate'}, inplace=True)
    df.drop(columns=['days'], inplace=True, errors='ignore')

    # ── Step 3: Remove IvyDB sentinel values ───────────────────
    # Per IvyDB manual: -99.99 = missing/uncalculated for IV,
    # delta, bid, offer, etc. Must be removed BEFORE any filter
    # that references these fields.
    print("\n  --- Sequential Filter Attrition ---")
    print(f"  {'Step':<40} {'Rows':>10} {'Dropped':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Raw data':<40} {len(df):>10,}")

    n_before = len(df)
    df = filters.remove_ivydb_sentinels(df)
    print(f"  {'Remove IvyDB sentinels (-99.99)':<40} {len(df):>10,} {n_before - len(df):>10,}")

    # ── Step 4: Compute forward via put-call parity ────────────
    # Uses ATM call-put pairs: F = K_ATM + e^(rτ)(C_ATM - P_ATM)
    # Falls back to futures settlement where ATM pairs unavailable.
    df = filters.compute_forward_via_put_call_parity(df)

    # ── Step 5: Apply sequential filters ───────────────────────
    steps = [
        ("Maturity filter", lambda d: filters.filter_maturity(
            d, min_dte=FILTERS['min_tau_days'], max_dte=FILTERS['max_tau_days']
        )),
        ("Static no-arbitrage bounds", filters.filter_static_arbitrage),
        ("Liquidity (OI≥1 or Vol≥1)", lambda d: filters.filter_liquidity(
            d, min_oi=FILTERS['min_open_interest'], min_vol=FILTERS.get('min_volume', 0)
        )),
        ("OTM only (K≥F calls, K≤F puts)", filters.filter_out_of_the_money),
        ("Moneyness trim |κ| ≤ 3√τ", lambda d: filters.trim_extreme_moneyness(
            d, coeff=FILTERS['moneyness_trim']
        )),
        ("IV bounds [0.01, 5.0]", filters.filter_iv_bounds),
    ]

    for step_name, step_fn in steps:
        n_before = len(df)
        df = step_fn(df)
        print(f"  {step_name:<40} {len(df):>10,} {n_before - len(df):>10,}")

    # ── Step 6: Final cleanup and save ─────────────────────────
    df = df.sort_values(['date', 'expiration', 'strike']).reset_index(drop=True)

    # Select columns for output
    output_cols = [
        'date', 'expiration', 'days_to_expiry', 'tau',
        'strike', 'callput', 'forward_price', 'log_moneyness',
        'settlementprice', 'bid', 'offer',
        'impliedvolatility', 'delta', 'gamma', 'vega', 'theta',
        'volume', 'openinterest', 'risk_free_rate',
        'futuresettlementprice', 'optionid', 'exercisestyle'
    ]
    # Keep only columns that exist (in case some were dropped)
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols]

    out_path = get_path('cleaned_cme')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # ── Summary statistics ─────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  CLEANING COMPLETE")
    print(f"  {'='*60}")
    print(f"  Final rows:      {len(df):>10,}")
    print(f"  Retention rate:   {len(df)/n_raw:>9.1%}")
    print(f"  Date range:       {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique dates:     {df['date'].nunique():>10,}")
    print(f"  Saved to:         {out_path}")

    # Per-year breakdown
    print(f"\n  Rows per year:")
    yearly = df.groupby(df['date'].dt.year).size()
    for year, count in yearly.items():
        print(f"    {year}: {count:>8,}")

    # Daily stats
    daily = df.groupby('date').size()
    print(f"\n  Contracts per day:")
    print(f"    Median: {daily.median():.0f}")
    print(f"    Min:    {daily.min()}")
    print(f"    Max:    {daily.max()}")
    print(f"    Mean:   {daily.mean():.1f}")

    return df


if __name__ == "__main__":
    process_cme_data()