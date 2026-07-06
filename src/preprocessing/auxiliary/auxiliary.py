"""
Builds the complete auxiliary panel from raw sources in a single pass. 

Variables:
  Base (yfinance + Deribit API):
    - btc_spot  : BTC-USD daily close (yfinance)
    - rv        : 30-day annualized realized variance of log-returns
    - vix       : CBOE VIX daily close (yfinance)
    - dvol      : Deribit BTC DVOL index, daily (Deribit public API)

  Macro (FRED official API, requires FRED_API_KEY):
    - baa_spread: Moody's Baa corporate bond yield minus 10Y Treasury (FRED: BAA10Y)
    - dgs2      : 2-year Treasury constant maturity yield (FRED: DGS2)
    - dxy       : Trade-weighted US Dollar Index, broad (FRED: TWEXBGSMTH)

  Crypto sentiment:
    - fng       : Crypto Fear & Greed Index, 0-100 (alternative.me, daily JSON)

"""

import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config import get_path

try:
    from src.config import API_KEYS
    _FRED_KEY = API_KEYS.get("fred")
except (ImportError, AttributeError):
    _FRED_KEY = None

PANEL_PATH = get_path("cleaned_auxiliary")
COVERAGE_PATH = PANEL_PATH.with_name("auxiliary_panel_coverage.csv")

FRED_SERIES = {
    "baa_spread": "BAA10Y",
    "dgs2": "DGS2",
    "dxy": "TWEXBGSMTH",
}

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"

FETCH_START = "2019-11-01"
FETCH_END = "2025-12-31"
PANEL_START = "2020-01-01"

# Base variables (yfinance + Deribit)
def fetch_vix() -> pd.DataFrame:
    print("  Fetching VIX...")
    vix = yf.download("^VIX", start=FETCH_START, end=FETCH_END, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].reset_index()
    vix.columns = ["date", "vix"]
    vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
    return vix

def fetch_btc_spot_and_rv(window: int = 30) -> pd.DataFrame:
    print("  Fetching BTC spot + computing RV...")
    btc = yf.download("BTC-USD", start=FETCH_START, end=FETCH_END, progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    btc = btc[["Close"]].reset_index()
    btc.columns = ["date", "btc_spot"]
    btc["date"] = pd.to_datetime(btc["date"]).dt.tz_localize(None)
    btc["log_ret"] = np.log(btc["btc_spot"] / btc["btc_spot"].shift(1))
    btc["rv"] = btc["log_ret"].rolling(window=window).apply(
        lambda x: (365.25 / window) * np.sum(x ** 2), raw=True
    )
    return btc[["date", "btc_spot", "rv"]]

def fetch_deribit_dvol() -> pd.DataFrame:
    print("  Fetching Deribit DVOL (multi-chunk)...")
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    headers = {"User-Agent": "Mozilla/5.0"}
    chunks = [
        ("2021-03-29", "2023-12-31"),
        ("2024-01-01", "2025-12-31"),
    ]
    all_data = []
    for start, end in chunks:
        params = {
            "currency": "BTC",
            "start_timestamp": int(pd.Timestamp(start).timestamp() * 1000),
            "end_timestamp": int(pd.Timestamp(end).timestamp() * 1000),
            "resolution": "1D",
        }
        response = requests.get(url, params=params, headers=headers, timeout=30)
        chunk_data = response.json().get("result", {}).get("data", [])
        all_data.extend(chunk_data)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "dvol"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    return df[["date", "dvol"]].drop_duplicates(subset="date")

# FRED macro variables
def get_fred_api_key() -> str:
    if _FRED_KEY:
        return _FRED_KEY
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "FRED API key not found. Add 'api_keys: fred: your_key' to config.yaml, "
        "or set the FRED_API_KEY environment variable."
    )

def fetch_fred_series(series_id: str, start: str, end: str, api_key: str, max_retries: int = 3) -> pd.Series:
    print(f"  Fetching FRED:{series_id} ({start} -> {end})...")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(FRED_API_BASE, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if "observations" not in payload:
                raise ValueError(
                    f"Unexpected FRED response for {series_id}: "
                    f"{payload.get('error_message', payload)}"
                )
            obs = payload["observations"]
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df[series_id] = pd.to_numeric(df["value"], errors="coerce")
            return df.set_index("date")[series_id]

        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** (attempt + 1)
                print(f"    [WARN] Attempt {attempt + 1} failed: {e}. "
                      f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"    [ERROR] Exhausted {max_retries} retries for {series_id}.")
                raise

# Crypto sentiment
def fetch_fear_greed_index() -> pd.DataFrame:
    """Fetch the full historical Crypto Fear & Greed Index from alternative.me."""
    print("  Fetching Crypto Fear & Greed Index...")
    r = requests.get(FNG_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()
    rows = payload["data"]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.normalize()
    df["fng"] = pd.to_numeric(df["value"])
    return df[["date", "fng"]].sort_values("date").reset_index(drop=True)

# Main builder
def build_auxiliary_panel() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  Auxiliary Panel Construction")
    print("=" * 60)

    # Base: BTC spot + RV + VIX + DVOL
    print("\n  [1/4] Base variables (yfinance + Deribit)...")
    df_btc = fetch_btc_spot_and_rv()
    df_vix = fetch_vix()
    df_dvol = fetch_deribit_dvol()

    panel = df_btc.merge(df_vix, on="date", how="left")
    panel["vix"] = panel["vix"].ffill()
    panel = panel.merge(df_dvol, on="date", how="left")

    # FRED macro series
    print("\n  [2/4] FRED macro variables...")
    try:
        api_key = get_fred_api_key()
        start = panel["date"].min().strftime("%Y-%m-%d")
        end = panel["date"].max().strftime("%Y-%m-%d")
        for col, series_id in FRED_SERIES.items():
            try:
                s = fetch_fred_series(series_id, start=start, end=end, api_key=api_key)
                df_s = s.to_frame(col).reset_index()
                panel = panel.merge(df_s, on="date", how="left")
            except Exception as e:
                print(f"    [WARN] Failed to fetch {series_id}: {e}")
                panel[col] = np.nan
        # Forward-fill business-day-only FRED data across weekends/holidays.
        for col in FRED_SERIES.keys():
            panel[col] = panel[col].ffill()
    except RuntimeError as e:
        print(f"    [WARN] Skipping FRED series: {e}")
        for col in FRED_SERIES.keys():
            panel[col] = np.nan

    # ── Crypto sentiment ─────────────────────────────────────────────────────
    print("\n  [3/4] Crypto sentiment...")
    try:
        fng = fetch_fear_greed_index()
        panel = panel.merge(fng, on="date", how="left")
    except Exception as e:
        print(f"    [WARN] Failed to fetch Fear & Greed Index: {e}")
        panel["fng"] = np.nan

    # ── Trim to study window and save ────────────────────────────────────────
    print("\n  [4/4] Finalizing...")
    panel = panel[panel["date"] >= PANEL_START].copy().reset_index(drop=True)

    # Coverage report
    print(f"\n  Coverage summary ({len(panel):,} rows):")
    ALL_VARS = ["btc_spot", "rv", "vix", "dvol"] + list(FRED_SERIES.keys()) + ["fng"]
    summary_rows = []
    for col in ALL_VARS:
        if col not in panel.columns:
            continue
        non_null = panel[col].notna().sum()
        first = panel.loc[panel[col].notna(), "date"].min()
        last = panel.loc[panel[col].notna(), "date"].max()
        coverage = non_null / len(panel)
        summary_rows.append({
            "variable": col,
            "non_null": int(non_null),
            "total": int(len(panel)),
            "coverage_pct": round(100 * coverage, 2),
            "first_valid": first.date() if pd.notna(first) else None,
            "last_valid": last.date() if pd.notna(last) else None,
        })
        print(f"    {col:<12s}: {non_null:>5d} / {len(panel)} "
              f"({coverage:.1%}), "
              f"{first.date() if pd.notna(first) else 'N/A'} -> "
              f"{last.date() if pd.notna(last) else 'N/A'}")

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(COVERAGE_PATH, index=False)
    print(f"\n  Saved coverage: {COVERAGE_PATH}")

    panel.to_parquet(PANEL_PATH, index=False)
    print(f"  Saved panel:    {PANEL_PATH}")
    print(f"  Columns:        {list(panel.columns)}")

    return panel


if __name__ == "__main__":
    build_auxiliary_panel()