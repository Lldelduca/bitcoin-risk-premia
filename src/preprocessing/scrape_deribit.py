import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.config import get_path

HISTORY_API = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
MAX_COUNT = 10000       # Maximum trades per request
CHUNK_HOURS = 1         # Query in 1-hour chunks
RATE_LIMIT_DELAY = 0.2  # Seconds between requests
RETRY_DELAY = 5.0       # Seconds to wait on error before retry
MAX_RETRIES = 5         # Maximum retries per chunk

START_DATE = "2017-07-01"
END_DATE = "2025-12-31"

def fetch_trades_chunk(start_ms: int, end_ms: int, retries: int = 0) -> list:
    """
    Fetches up to MAX_COUNT trades in a single time window.
    Returns a list of trade dicts.
    """
    params = {
        "currency": "BTC",
        "kind": "option",
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": MAX_COUNT,
        "sorting": "asc"
    }

    try:
        resp = requests.get(HISTORY_API, params=params, timeout=30)

        if resp.status_code == 429:
            # Rate limited — back off and retry
            wait = RATE_LIMIT_DELAY * (2 ** retries)
            print(f"    Rate limited. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if retries < MAX_RETRIES:
                return fetch_trades_chunk(start_ms, end_ms, retries + 1)
            return []

        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            if retries < MAX_RETRIES:
                return fetch_trades_chunk(start_ms, end_ms, retries + 1)
            return []

        data = resp.json()
        result = data.get("result", {})
        trades = result.get("trades", [])
        has_more = result.get("has_more", False)

        if has_more and trades:
            # Recursively fetch remaining trades starting after the last returned trade
            last_ts = trades[-1]["timestamp"]
            more_trades = fetch_trades_chunk(last_ts + 1, end_ms)
            trades.extend(more_trades)

        return trades

    except requests.exceptions.RequestException as e:
        print(f"    Request error: {e}. Retrying in {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
        if retries < MAX_RETRIES:
            return fetch_trades_chunk(start_ms, end_ms, retries + 1)
        return []


def parse_instrument_name(name: str) -> dict:
    """
    Parses Deribit instrument names like 'BTC-28MAR25-80000-C'.
    Returns dict with expiration, strike, callput.
    """
    parts = name.split("-")
    if len(parts) != 4:
        return {"expiration": None, "strike": None, "callput": None}

    try:
        exp_str = parts[1]  # e.g., "28MAR25"
        expiration = pd.to_datetime(exp_str, format="%d%b%y")
        strike = float(parts[2])
        callput = parts[3]  # "C" or "P"
        return {"expiration": expiration, "strike": strike, "callput": callput}
    except (ValueError, IndexError):
        return {"expiration": None, "strike": None, "callput": None}


def scrape_all_trades():
    """
    Main scraping loop. Iterates over 1-hour chunks from START_DATE
    to END_DATE, collecting all BTC option trades.
    """
    print("\n" + "=" * 60)
    print("  Deribit Historical BTC Options Trade Scraper")
    print("=" * 60)

    start_dt = pd.Timestamp(START_DATE)
    end_dt = pd.Timestamp(END_DATE)
    chunk_delta = timedelta(hours=CHUNK_HOURS)

    # Calculate total chunks for progress tracking
    total_hours = int((end_dt - start_dt).total_seconds() / 3600)
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Total chunks: ~{total_hours:,} ({CHUNK_HOURS}h each)")
    print(f"  Estimated time: {total_hours * RATE_LIMIT_DELAY / 60:.0f} min minimum\n")

    all_trades = []
    current = start_dt
    chunk_count = 0
    empty_streak = 0

    # Output directory
    out_dir = Path(get_path('raw_deribit_dir'))
    out_dir.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_INTERVAL = 500_000

    while current < end_dt:
        chunk_end = min(current + chunk_delta, end_dt)

        start_ms = int(current.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        trades = fetch_trades_chunk(start_ms, end_ms)
        chunk_count += 1

        if trades:
            all_trades.extend(trades)
            empty_streak = 0

            # Progress update every 100 chunks
            if chunk_count % 100 == 0:
                print(f"  [{current.date()}] Chunk {chunk_count:>6,}/{total_hours:,} | "
                      f"Trades so far: {len(all_trades):>10,}")
        else:
            empty_streak += 1
            # Reduce logging for empty periods (pre-2017 options launch)
            if empty_streak == 100:
                print(f"  [{current.date()}] 100 empty chunks in a row. "
                      f"Skipping ahead...")

        # Checkpoint save
        if len(all_trades) > 0 and len(all_trades) % CHECKPOINT_INTERVAL < len(trades):
            checkpoint_path = out_dir / "btc_options_trades_checkpoint.parquet"
            pd.DataFrame(all_trades).to_parquet(checkpoint_path, index=False)
            print(f"  ** Checkpoint saved: {len(all_trades):,} trades → {checkpoint_path.name}")

        current = chunk_end
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\n  Scraping complete. Total trades: {len(all_trades):,}")

    if not all_trades:
        print("  WARNING: No trades fetched. Check API connectivity.")
        return None

    print("  Parsing instrument names and cleaning...")
    df = pd.DataFrame(all_trades)

    # Parse instrument names into expiration, strike, callput
    parsed = df['instrument_name'].apply(parse_instrument_name).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Core columns
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    df['price_btc'] = df['price'].astype(float)       # Price in BTC
    df['mark_price_btc'] = df['mark_price'].astype(float)
    df['index_price_usd'] = df['index_price'].astype(float)
    df['iv'] = df['iv'].astype(float) / 100.0          # Convert from % to decimal
    df['amount'] = df['amount'].astype(float)
    df['trade_id_raw'] = df['trade_id']

    # Convert BTC price to USD equivalent
    df['price_usd'] = df['price_btc'] * df['index_price_usd']
    df['mark_price_usd'] = df['mark_price_btc'] * df['index_price_usd']

    # Compute days to expiry and tau
    df['days_to_expiry'] = (df['expiration'] - df['date']).dt.days
    df['tau'] = df['days_to_expiry'] / 365.25

    # Drop rows with parse failures
    n_before = len(df)
    df = df.dropna(subset=['expiration', 'strike', 'callput'])
    print(f"  Dropped {n_before - len(df):,} rows with unparseable instrument names")

    # Select output columns
    output_cols = [
        'date', 'timestamp', 'instrument_name',
        'expiration', 'strike', 'callput',
        'days_to_expiry', 'tau',
        'price_btc', 'price_usd', 'mark_price_btc', 'mark_price_usd',
        'iv', 'index_price_usd', 'amount',
        'direction', 'trade_id_raw'
    ]
    df = df[output_cols]

    out_path = out_dir / "btc_options_trades.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n  {'=' * 60}")
    print(f"  SCRAPING COMPLETE")
    print(f"  {'=' * 60}")
    print(f"  Total trades:     {len(df):>12,}")
    print(f"  Date range:       {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique dates:     {df['date'].nunique():>12,}")
    print(f"  Calls:            {(df['callput'] == 'C').sum():>12,}")
    print(f"  Puts:             {(df['callput'] == 'P').sum():>12,}")
    print(f"  Saved to:         {out_path}")

    # Per-year summary
    print(f"\n  Trades per year:")
    yearly = df.groupby(df['date'].dt.year).size()
    for year, count in yearly.items():
        print(f"    {year}: {count:>10,}")

    return df

if __name__ == "__main__":
    scrape_all_trades()