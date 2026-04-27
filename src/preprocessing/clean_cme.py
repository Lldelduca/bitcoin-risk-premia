# src/preprocessing/clean_cme.py
import pandas as pd
import numpy as np
import sys
from src.config import get_path, FILTERS
from src.preprocessing import filters

def convert_csv_to_parquet():
    """Converts CSV to Parquet and forces loud errors if files are missing."""
    csv_path = get_path('raw_cme_csv')
    parquet_path = get_path('raw_cme_parquet')
    
    print(f"Checking for Parquet file at: {parquet_path}")
    
    if not parquet_path.exists():
        print(f"Parquet not found. Looking for CSV at: {csv_path}")
        if csv_path.exists():
            print(f"Found CSV! Converting to Parquet (this takes ~30 seconds)...")
            df = pd.read_csv(csv_path, low_memory=False)
            df.to_parquet(parquet_path, index=False)
            print("Conversion complete!")
        else:
            print("\n" + "="*50)
            print("❌ ERROR: RAW CSV FILE NOT FOUND!")
            print(f"Please make sure your CSV is named exactly: {csv_path.name}")
            print(f"And is located exactly in this folder: {csv_path.parent}")
            print("="*50 + "\n")
            sys.exit(1) # Stops the script immediately
    else:
        print("Parquet file already exists. Skipping conversion.")

def load_and_prep_zero_curve():
    """Loads the FRED/OptionMetrics zero curve."""
    zc_path = get_path('zero_curve')
    if not zc_path.exists():
        print(f"\n❌ ERROR: ZERO CURVE CSV NOT FOUND AT {zc_path}\n")
        sys.exit(1)
        
    df_zc = pd.read_csv(zc_path, parse_dates=['date'])
    df_zc = df_zc[df_zc['currency'] == 'USD'].copy()
    df_zc = df_zc.sort_values(['date', 'days'])
    return df_zc

def process_cme_data():
    # 1. Convert CSV to Parquet (Will stop the script if CSV is missing)
    convert_csv_to_parquet()
    
    print("Loading CME Parquet into memory...")
    df = pd.read_parquet(get_path('raw_cme_parquet'))
    
    # 2. Date and Time to Maturity calculations
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['days_to_expiry'] = (df['expiration'] - df['date']).dt.days
    df['tau'] = df['days_to_expiry'] / 365.25  # Annualized time to maturity

    # 3. Extract true strike and forward price based on IvyDB schema
    df['strike'] = df['strike'] / df['strikemultiplier']
    df['forward_price'] = df['futuresettlementprice'] 
    
    # Calculate log-moneyness (κ = ln(K / F_t))
    df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])

    # 4. Merge Risk-Free Rate
    print("Merging Zero Curve...")
    df_zc = load_and_prep_zero_curve()
    df_zc = df_zc.sort_values('days')
    df = df.sort_values('days_to_expiry')
    
    df = pd.merge_asof(
        df, 
        df_zc[['date', 'days', 'rate']], 
        by='date', 
        left_on='days_to_expiry', 
        right_on='days', 
        direction='nearest'
    )
    df.rename(columns={'rate': 'risk_free_rate'}, inplace=True)
    df.drop(columns=['days'], inplace=True)

    df = df.sort_values(['date', 'days_to_expiry', 'strike']).reset_index(drop=True)

    # 5. Apply Filters sequentially and track attrition
    print("\n--- Applying Mathematical Filters ---")
    print(f"Initial raw rows: {len(df):,}")
    
    df = filters.filter_maturity(df, min_tau_days=FILTERS['min_tau_days'], max_tau_days=FILTERS['max_tau_days'])
    print(f"After Maturity filter: {len(df):,}")

    df = filters.filter_static_arbitrage(df)
    print(f"After Arbitrage filter: {len(df):,}")
    
    df = filters.apply_liquidity_filters(df, min_oi=FILTERS['min_open_interest'], min_vol=FILTERS['min_volume'])
    print(f"After Liquidity filter: {len(df):,}")

    df = filters.filter_out_of_the_money(df)
    print(f"After OTM filter: {len(df):,}")

    df = filters.trim_extreme_moneyness(df, multiplier=FILTERS['moneyness_trim'])
    print(f"After Extreme Moneyness Trim: {len(df):,}")

    # 6. Save cleaned dataset
    out_path = get_path('cleaned_cme')
    df.to_parquet(out_path, index=False)
    print(f"\n✅ SUCCESS: Cleaned CME data saved to {out_path}")

if __name__ == "__main__":
    process_cme_data()