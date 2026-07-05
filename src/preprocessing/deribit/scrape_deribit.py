# src/preprocessing/scrape_deribit.py
"""
Deribit Historical BTC Options Trade Scraper (memory-safe version).

Uses the public historical API endpoint:
  https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time

This endpoint provides ALL historical trades since Deribit's inception (2017),
no authentication required.

MEMORY MANAGEMENT:
  - Buffers trades in memory only between checkpoints (default 500K trades)
  - Writes each checkpoint as an incremental Parquet shard
  - Frees the buffer after each shard write
  - Concatenates all shards at the end via streaming read
  - Maximum memory footprint: ~one shard (~150 MB) instead of the full dataset

RESUMABILITY:
  - On restart, scans existing shards in `data/raw/deribit/shards/`
  - Resumes from the timestamp of the last successfully written shard
  - Run multiple times if interrupted; nothing is lost

Usage:
  python -m src.preprocessing.scrape_deribit

Output:
  data/raw/deribit/shards/btc_trades_shard_NNNN.parquet  (incremental raw)
  data/raw/deribit/shards/parsed/btc_trades_shard_NNNN.parquet  (parsed)
  data/raw/deribit/btc_options_trades.parquet            (final concatenated)
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.config import get_path


# ── API Configuration ──────────────────────────────────────────────
HISTORY_API = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
MAX_COUNT = 10000        # Maximum trades per request
CHUNK_HOURS = 1          # Query in 1-hour chunks
RATE_LIMIT_DELAY = 0.2   # Seconds between requests
RETRY_DELAY = 5.0
MAX_RETRIES = 5

# ── Memory Management ─────────────────────────────────────────────
# Write a shard every CHECKPOINT_TRADES trades to keep memory bounded.
# 250K trades × ~200 bytes ≈ 50 MB per shard, well within 16 GB RAM.
CHECKPOINT_TRADES = 250_000

# ── Date Range ─────────────────────────────────────────────────────
START_DATE = "2017-07-01"
END_DATE = "2025-12-31"


def fetch_trades_chunk(start_ms: int, end_ms: int, retries: int = 0) -> list:
    """Fetches up to MAX_COUNT trades in a single time window."""
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


def write_shard(trades: list, shards_dir: Path, shard_index: int) -> int:
    """
    Writes a list of trade dicts to a numbered Parquet shard.
    Returns the number of trades written.
    """
    if not trades:
        return 0

    df = pd.DataFrame(trades)
    shard_path = shards_dir / f"btc_trades_shard_{shard_index:04d}.parquet"
    df.to_parquet(shard_path, index=False, compression='snappy')
    return len(df)


def find_resume_point(shards_dir: Path) -> tuple:
    """
    Inspects existing shards to determine the resume timestamp and
    next shard index. Returns (resume_timestamp_ms, next_shard_idx).
    Returns (None, 0) if no shards exist.
    """
    if not shards_dir.exists():
        return None, 0

    shards = sorted(shards_dir.glob("btc_trades_shard_*.parquet"))
    if not shards:
        return None, 0

    last_shard = shards[-1]
    print(f"  Found {len(shards)} existing shards. Resuming from {last_shard.name}...")

    df_last = pd.read_parquet(last_shard, columns=['timestamp'])
    max_ts = int(df_last['timestamp'].max())

    next_idx = len(shards)
    return max_ts, next_idx


def scrape_all_trades():
    """Main scraping loop with incremental shard writes."""
    print("\n" + "=" * 60)
    print("  Deribit Historical BTC Options Trade Scraper")
    print("=" * 60)

    out_dir = Path(get_path('raw_deribit_dir'))
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume
    resume_ts_ms, shard_index = find_resume_point(shards_dir)

    if resume_ts_ms is not None:
        start_dt = pd.Timestamp(resume_ts_ms, unit='ms') + timedelta(milliseconds=1)
        print(f"  Resuming from: {start_dt}")
    else:
        start_dt = pd.Timestamp(START_DATE)

    end_dt = pd.Timestamp(END_DATE)

    if start_dt >= end_dt:
        print(f"  Start date {start_dt} is past end date {end_dt}. Nothing to scrape.")
        print(f"  Running consolidation step...")
        consolidate_shards(shards_dir, out_dir)
        return

    chunk_delta = timedelta(hours=CHUNK_HOURS)
    total_hours = int((end_dt - start_dt).total_seconds() / 3600)
    print(f"  Date range: {start_dt.date()} to {end_dt.date()}")
    print(f"  Total chunks: ~{total_hours:,} ({CHUNK_HOURS}h each)")
    print(f"  Estimated time: {total_hours * RATE_LIMIT_DELAY / 60:.0f} min minimum")
    print(f"  Checkpoint every {CHECKPOINT_TRADES:,} trades\n")

    buffer = []
    current = start_dt
    chunk_count = 0
    total_trades = 0
    empty_streak = 0

    while current < end_dt:
        chunk_end = min(current + chunk_delta, end_dt)

        start_ms = int(current.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        trades = fetch_trades_chunk(start_ms, end_ms)
        chunk_count += 1

        if trades:
            buffer.extend(trades)
            empty_streak = 0

            if chunk_count % 100 == 0:
                print(f"  [{current.date()}] Chunk {chunk_count:>6,}/{total_hours:,} | "
                      f"Buffer: {len(buffer):>8,} | "
                      f"Total: {total_trades + len(buffer):>10,}")
        else:
            empty_streak += 1
            if empty_streak == 100:
                print(f"  [{current.date()}] 100 empty chunks. Skipping ahead...")

        # Write shard when buffer hits checkpoint size
        if len(buffer) >= CHECKPOINT_TRADES:
            written = write_shard(buffer, shards_dir, shard_index)
            total_trades += written
            print(f"  ** Shard {shard_index:04d} written: {written:,} trades | "
                  f"Total on disk: {total_trades:,}")
            buffer.clear()           # Free memory
            shard_index += 1

        current = chunk_end
        time.sleep(RATE_LIMIT_DELAY)

    # Final flush
    if buffer:
        written = write_shard(buffer, shards_dir, shard_index)
        total_trades += written
        print(f"  ** Final shard {shard_index:04d} written: {written:,} trades")
        buffer.clear()
        shard_index += 1

    print(f"\n  Scraping complete. Total trades on disk: {total_trades:,}")
    print(f"  Shards saved to: {shards_dir}")

    consolidate_shards(shards_dir, out_dir)


def consolidate_shards(shards_dir: Path, out_dir: Path):
    """
    Reads all raw shards, parses instrument names, computes derived fields,
    writes parsed shards, then streams them into a single final Parquet.

    Memory-safe: processes shards one at a time.
    """
    print("\n" + "=" * 60)
    print("  Consolidating shards → final Parquet")
    print("=" * 60)

    shards = sorted(shards_dir.glob("btc_trades_shard_*.parquet"))
    if not shards:
        print("  No shards found. Nothing to consolidate.")
        return

    print(f"  Found {len(shards)} raw shards.")

    parsed_dir = shards_dir / "parsed"
    parsed_dir.mkdir(exist_ok=True)

    total_kept = 0
    total_dropped = 0

    for i, shard_path in enumerate(shards):
        df = pd.read_parquet(shard_path)
        n_in = len(df)

        # Parse instrument names
        parsed = df['instrument_name'].apply(parse_instrument_name).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)

        # Drop unparseable
        df = df.dropna(subset=['expiration', 'strike', 'callput'])
        n_out = len(df)
        total_kept += n_out
        total_dropped += (n_in - n_out)

        # Compute derived fields
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
        df['price_btc'] = df['price'].astype(float)
        df['mark_price_btc'] = df['mark_price'].astype(float)
        df['index_price_usd'] = df['index_price'].astype(float)
        df['iv'] = df['iv'].astype(float) / 100.0
        df['amount'] = df['amount'].astype(float)
        df['price_usd'] = df['price_btc'] * df['index_price_usd']
        df['mark_price_usd'] = df['mark_price_btc'] * df['index_price_usd']
        df['days_to_expiry'] = (df['expiration'] - df['date']).dt.days
        df['tau'] = df['days_to_expiry'] / 365.25

        output_cols = [
            'date', 'timestamp', 'instrument_name',
            'expiration', 'strike', 'callput',
            'days_to_expiry', 'tau',
            'price_btc', 'price_usd', 'mark_price_btc', 'mark_price_usd',
            'iv', 'index_price_usd', 'amount',
            'direction', 'trade_id'
        ]
        output_cols = [c for c in output_cols if c in df.columns]
        df = df[output_cols]

        out_path = parsed_dir / shard_path.name
        df.to_parquet(out_path, index=False, compression='snappy')

        print(f"  Shard {i+1}/{len(shards)}: {n_in:,} in → {n_out:,} out")

        del df  # Explicit free

    print(f"\n  Total: {total_kept:,} kept, {total_dropped:,} dropped (unparseable)")

    # Stream parsed shards into final Parquet via PyArrow
    print(f"\n  Building final Parquet via streaming concatenation...")
    final_path = out_dir / "btc_options_trades.parquet"

    parsed_shards = sorted(parsed_dir.glob("*.parquet"))

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(parsed_shards[0])

    with pq.ParquetWriter(final_path, schema, compression='snappy') as writer:
        for shard_path in parsed_shards:
            table = pq.read_table(shard_path)
            writer.write_table(table)

    # Summary using minimal column read
    print(f"\n  {'=' * 60}")
    print(f"  CONSOLIDATION COMPLETE")
    print(f"  {'=' * 60}")

    summary_df = pd.read_parquet(final_path, columns=['date', 'callput'])
    print(f"  Total trades:     {len(summary_df):>12,}")
    print(f"  Date range:       {summary_df['date'].min().date()} to "
          f"{summary_df['date'].max().date()}")
    print(f"  Unique dates:     {summary_df['date'].nunique():>12,}")
    print(f"  Calls:            {(summary_df['callput'] == 'C').sum():>12,}")
    print(f"  Puts:             {(summary_df['callput'] == 'P').sum():>12,}")
    print(f"  Saved to:         {final_path}")

    print(f"\n  Trades per year:")
    yearly = summary_df.groupby(summary_df['date'].dt.year).size()
    for year, count in yearly.items():
        print(f"    {year}: {count:>10,}")


def parse_instrument_name(name: str) -> dict:
    """Parses 'BTC-28MAR25-80000-C' → {expiration, strike, callput}."""
    parts = name.split("-")
    if len(parts) != 4:
        return {"expiration": None, "strike": None, "callput": None}

    try:
        exp_str = parts[1]
        expiration = pd.to_datetime(exp_str, format="%d%b%y")
        strike = float(parts[2])
        callput = parts[3]
        return {"expiration": expiration, "strike": strike, "callput": callput}
    except (ValueError, IndexError):
        return {"expiration": None, "strike": None, "callput": None}


if __name__ == "__main__":
    scrape_all_trades()