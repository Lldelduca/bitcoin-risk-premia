import pandas as pd
import numpy as np
import yfinance as yf
import requests
from src.config import get_path

def fetch_vix():
    """Fetches daily VIX close prices."""
    print("Fetching VIX data...")
    vix = yf.download("^VIX", start="2019-12-20", end="2025-12-31", progress=False)
    
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    vix = vix[['Close']].reset_index()
    vix.columns = ['date', 'vix']
    vix['date'] = pd.to_datetime(vix['date']).dt.tz_localize(None)
    return vix

def fetch_btc_and_calculate_rv(window=30):
    """Fetches daily BTC spot and calculates 30-day annualized Realized Variance."""
    print("Fetching BTC Spot and calculating Realized Variance...")
    btc = yf.download("BTC-USD", start="2019-11-01", end="2025-12-31", progress=False)
    
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
        
    btc = btc[['Close']].reset_index()
    btc.columns = ['date', 'btc_spot']
    btc['date'] = pd.to_datetime(btc['date']).dt.tz_localize(None)
    
    btc['log_ret'] = np.log(btc['btc_spot'] / btc['btc_spot'].shift(1))
    btc['rv'] = btc['log_ret'].rolling(window=window).apply(
        lambda x: (365.25 / window) * np.sum(x**2), raw=True
    )
    
    btc = btc[btc['date'] >= '2019-12-20'].copy()
    return btc[['date', 'btc_spot', 'rv']]

def fetch_deribit_dvol():
    """Fetches DVOL index in chunks to bypass the 1000-record API limit."""
    print("Fetching Deribit DVOL index (multichunk)...")
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Define two chunks to cover ~5 years without hitting the 1000-day limit
    chunks = [
        ("2021-03-29", "2023-12-31"),
        ("2024-01-01", "2025-12-31")
    ]
    
    all_data = []
    for start, end in chunks:
        params = {
            "currency": "BTC",
            "start_timestamp": int(pd.Timestamp(start).timestamp() * 1000),
            "end_timestamp": int(pd.Timestamp(end).timestamp() * 1000),
            "resolution": "1D"
        }
        response = requests.get(url, params=params, headers=headers)
        chunk_data = response.json().get("result", {}).get("data", [])
        all_data.extend(chunk_data)
    
    df_dvol = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "dvol"])
    df_dvol['date'] = pd.to_datetime(df_dvol['timestamp'], unit='ms').dt.normalize()
    # Drop duplicates in case chunks overlap
    return df_dvol[['date', 'dvol']].drop_duplicates()

def build_auxiliary_panel():
    """Compiles all macro and crypto conditioning variables into one panel."""
    print("\n--- Building Auxiliary Panel ---")
    
    df_btc = fetch_btc_and_calculate_rv()
    df_vix = fetch_vix()
    df_dvol = fetch_deribit_dvol()
    
    # 1. Base merge: BTC Spot/RV with VIX
    panel = pd.merge(df_btc, df_vix, on='date', how='left')
    panel['vix'] = panel['vix'].ffill()
    
    # 2. Merge DVOL
    panel = pd.merge(panel, df_dvol, on='date', how='left')
    
    # 3. Trim to the official study window (Fixes the Jan 1st NaN issue)
    panel = panel[panel['date'] >= '2020-01-01'].copy()
    
    # Save to Parquet
    out_path = get_path('cleaned_auxiliary')
    panel.to_parquet(out_path, index=False)
    
    print(f"✅ Auxiliary panel saved to {out_path}\n")
    
    # Print a sanity check on the data
    print("Recent rows (showing valid DVOL):")
    print(panel.tail())
    print("\nTotal Missing Values:")
    print(panel.isna().sum())

if __name__ == "__main__":
    build_auxiliary_panel()