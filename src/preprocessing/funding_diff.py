import pandas as pd
import requests
import time
from src.config import get_path

# Constants
HL_PERIODS_PER_DAY = 24      # Hyperliquid: hourly funding
DER_PERIODS_PER_DAY = 3      # Deribit: 8-hour funding
DAYS_PER_YEAR = 365

HL_ANNUAL_FACTOR = HL_PERIODS_PER_DAY * DAYS_PER_YEAR
DER_ANNUAL_FACTOR = DER_PERIODS_PER_DAY * DAYS_PER_YEAR

# Date bounds
START_DATE = "2023-05-12"
END_DATE = "2025-12-31"


def fetch_hl_funding() -> pd.DataFrame:
    print("Fetching Hyperliquid BTC funding history...")
    url = "https://api.hyperliquid.xyz/info"
    all_data = []

    current_start = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_DATE).timestamp() * 1000)

    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 12
    SKIP_AHEAD_MS = 30 * 24 * 3600 * 1000 

    while current_start < end_ts:
        payload = {
            "type": "fundingHistory",
            "coin": "BTC",
            "startTime": current_start,
            "endTime": end_ts,
        }
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code != 200:
            print(f"  HTTP {response.status_code}. Retrying in 5s...")
            time.sleep(5)
            continue

        data = response.json()

        if not data or len(data) == 0:
            consecutive_empty += 1
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"  {consecutive_empty} consecutive empty responses. Stopping.")
                break
            current_start += SKIP_AHEAD_MS
            time.sleep(0.1)
            continue

        consecutive_empty = 0
        all_data.extend(data)

        last_ts = max(d["time"] for d in data)
        if last_ts <= current_start:
            break  
        current_start = last_ts + 1

        time.sleep(0.1)

    if not all_data:
        raise RuntimeError("No Hyperliquid funding data retrieved.")

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()

    # Hourly periodic rate → annualized
    df["funding_hl_annual"] = df["fundingRate"].astype(float) * HL_ANNUAL_FACTOR
    daily = df.groupby("date")["funding_hl_annual"].mean().reset_index()

    print(f"  Fetched {len(df):,} hourly records across {daily['date'].nunique()} days")
    return daily

def fetch_deribit_perp_funding() -> pd.DataFrame:
    print("Fetching Deribit BTC-PERPETUAL funding (paginated)...")
    url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"

    overall_start = pd.Timestamp(START_DATE)
    overall_end = pd.Timestamp(END_DATE)
    chunk = pd.Timedelta(days=30)

    all_records = []
    cur = overall_start
    chunk_idx = 0

    while cur < overall_end:
        chunk_end = min(cur + chunk, overall_end)
        params = {
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": int(cur.timestamp() * 1000),
            "end_timestamp": int(chunk_end.timestamp() * 1000),
        }

        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} on chunk {chunk_idx}. Retrying in 5s...")
            time.sleep(5)
            continue

        records = response.json().get("result", [])
        if records:
            all_records.extend(records)
        chunk_idx += 1
        cur = chunk_end
        time.sleep(0.1) 

    if not all_records:
        raise RuntimeError("No Deribit funding data retrieved.")

    df = pd.DataFrame(all_records).drop_duplicates(subset=["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()

    # 8-hour periodic rate → annualized
    df["funding_der_annual"] = df["interest_8h"].astype(float) * DER_ANNUAL_FACTOR
    daily = df.groupby("date")["funding_der_annual"].mean().reset_index()

    print(f"  Fetched {len(df):,} hourly records across {daily['date'].nunique()} days")
    return daily

def process_funding_differential():
    """
    Computes Δf_t^{HL-DER} = funding_HL,t - funding_DER,t (annualized).
    """
    print("\n" + "=" * 60)
    print("  Funding Rate Differential (HL - DER, annualized)")
    print("=" * 60)

    hl = fetch_hl_funding()
    der = fetch_deribit_perp_funding()

    df = pd.merge(hl, der, on="date", how="inner")
    df["delta_f_hl_der"] = df["funding_hl_annual"] - df["funding_der_annual"]

    out_path = get_path("cleaned_auxiliary").parent / "funding_diff.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n  Saved to {out_path}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Total days: {len(df):,}")
    print(f"\n  Annualized funding rate statistics:")
    print(df[["funding_hl_annual", "funding_der_annual", "delta_f_hl_der"]].describe())
    print(f"\n  Recent observations:")
    print(df.tail(3))

if __name__ == "__main__":
    process_funding_differential()