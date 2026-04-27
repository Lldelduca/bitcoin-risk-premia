import pandas as pd
import requests
import time
from src.config import get_path

def fetch_hl_funding():
    """Fetches full BTC funding history from Hyperliquid using pagination."""
    print("Fetching Hyperliquid BTC funding history...")
    url = "https://api.hyperliquid.xyz/info"
    all_data = []
    
    # HL launched in 2023; we start from Jan 1st
    current_start = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
    
    while True:
        payload = {
            "type": "fundingHistory",
            "coin": "BTC",
            "startTime": current_start
        }
        response = requests.post(url, json=payload)
        data = response.json()
        
        if not data or len(data) == 0:
            break
            
        all_data.extend(data)
        
        # Move start time to 1ms after the last received timestamp
        last_ts = data[-1]['time']
        if last_ts == current_start: # Safety break if no new data
            break
        current_start = last_ts + 1
        
        # Respect rate limits
        time.sleep(0.1)

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['time'], unit='ms').dt.normalize()
    df['funding_hl'] = df['fundingRate'].astype(float)
    
    # Daily average of hourly funding
    return df.groupby('date')['funding_hl'].mean().reset_index()

def fetch_deribit_perp_funding():
    """Fetches Deribit BTC-PERPETUAL funding with chunking."""
    print("Fetching Deribit perpetual funding...")
    url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
    all_data = []
    
    # Match the HL start date
    start_ts = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
    end_ts = int(pd.Timestamp("2025-12-31").timestamp() * 1000)
    
    params = {
        "instrument_name": "BTC-PERPETUAL",
        "start_timestamp": start_ts,
        "end_timestamp": end_ts
    }
    
    response = requests.get(url, params=params)
    data = response.json().get("result", [])
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    
    # Aggregate interest_8h to daily average
    return df.groupby('date')['interest_8h'].mean().reset_index().rename(columns={'interest_8h': 'funding_der'})

def process_funding_differential():
    """Calculates Δf = HL_funding - DER_funding (The Decentralized Margin Pulse)."""
    hl = fetch_hl_funding()
    der = fetch_deribit_perp_funding()
    
    # Inner join ensures we only have dates where both venues were active (2023+)
    df = pd.merge(hl, der, on='date', how='inner')
    df['delta_f_hl_der'] = df['funding_hl'] - df['funding_der']
    
    out_path = get_path('cleaned_auxiliary').parent / "funding_diff.parquet"
    df.to_parquet(out_path, index=False)
    
    print(f"✅ Phase 0 Complete: Funding differential saved to {out_path}")
    print(f"Sample range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.tail(3))

if __name__ == "__main__":
    process_funding_differential()