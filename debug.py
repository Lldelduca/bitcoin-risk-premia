"""
Standalone Hyperliquid pagination debugger.

Run this directly to see exactly when and why the pagination loop exits.
Run from project root: python debug_hl_pagination.py
"""

import requests
import time
import pandas as pd

URL = "https://api.hyperliquid.xyz/info"
START_DATE = "2023-03-01"  # HL BTC perp launched mid-2023
END_DATE = "2025-12-31"


def main():
    current_start = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_DATE).timestamp() * 1000)

    all_data = []
    iteration = 0

    print(f"Target range: {START_DATE} to {END_DATE}")
    print(f"start_ts={current_start}, end_ts={end_ts}")
    print(f"Expected duration: {(end_ts - current_start)/1000/86400:.0f} days\n")

    while current_start < end_ts:
        iteration += 1
        payload = {
            "type": "fundingHistory",
            "coin": "BTC",
            "startTime": current_start,
            "endTime": end_ts,
        }

        try:
            r = requests.post(URL, json=payload, timeout=30)
        except Exception as e:
            print(f"Iter {iteration}: EXCEPTION: {e}")
            break

        if r.status_code != 200:
            print(f"Iter {iteration}: HTTP {r.status_code}: {r.text[:200]}")
            break

        try:
            data = r.json()
        except Exception as e:
            print(f"Iter {iteration}: JSON parse error: {e}")
            break

        if not isinstance(data, list):
            print(f"Iter {iteration}: unexpected response type {type(data)}: {data}")
            break

        n = len(data)
        if n == 0:
            print(f"Iter {iteration}: empty response. STOPPING.")
            break

        min_ts = min(d["time"] for d in data)
        max_ts = max(d["time"] for d in data)
        start_str = pd.Timestamp(current_start, unit="ms").strftime("%Y-%m-%d %H:%M")
        min_str = pd.Timestamp(min_ts, unit="ms").strftime("%Y-%m-%d %H:%M")
        max_str = pd.Timestamp(max_ts, unit="ms").strftime("%Y-%m-%d %H:%M")

        print(f"Iter {iteration:>3}: req start={start_str} | "
              f"got {n:>3} records | "
              f"window [{min_str} → {max_str}]")

        all_data.extend(data)

        if max_ts <= current_start:
            print(f"  ! max_ts ({max_ts}) <= current_start ({current_start}). STOPPING.")
            break

        current_start = max_ts + 1
        time.sleep(0.1)

        # Safety: don't run forever during debugging
        if iteration >= 200:
            print(f"Hit iteration cap of 200. Stopping.")
            break

    print(f"\n========= SUMMARY =========")
    print(f"Total iterations:  {iteration}")
    print(f"Total records:     {len(all_data):,}")
    if all_data:
        all_min = min(d["time"] for d in all_data)
        all_max = max(d["time"] for d in all_data)
        print(f"Records min time:  {pd.Timestamp(all_min, unit='ms')}")
        print(f"Records max time:  {pd.Timestamp(all_max, unit='ms')}")
        print(f"Reached end_ts?    {all_max >= end_ts - 24*3600*1000}")


if __name__ == "__main__":
    main()