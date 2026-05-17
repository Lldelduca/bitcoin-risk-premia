"""
SSVI Surface Fitting Orchestrator.

Loads cleaned CME and Deribit data, trims to the common sample window, fits SSVI per (venue, date) and saves:
  1. Fitted parameters per (venue, date, expiration)
  2. Per-day fit diagnostics (RMSE, n_slices, n_options)

"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import get_path, SAMPLE, SSVI_CONFIG
from src.surfaces.ssvi import SSVI


SAMPLE_START = SAMPLE['start_date']
SAMPLE_END = SAMPLE['end_date']
MIN_OPTIONS_PER_SLICE = SSVI_CONFIG['min_options_per_slice']

SURFACES_DIR = Path(get_path('cleaned_cme')).parent.parent / "surfaces"

def load_venue(venue: str) -> pd.DataFrame:
    if venue == "CME":
        path = get_path('cleaned_cme')
    elif venue == "DER":
        path = get_path('cleaned_deribit')
    else:
        raise ValueError(f"Unknown venue: {venue}")

    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= SAMPLE_START) & (df['date'] <= SAMPLE_END)
    df = df[mask].copy()

    return df


def fit_venue(venue: str, df: pd.DataFrame):
    """Fits SSVI on every trading day for a single venue"""
    dates = np.sort(df['date'].unique())
    n_days = len(dates)
    print(f"\n  [{venue}] {n_days} trading days to fit")

    params_list = []
    eval_list = []
    n_success = 0
    n_skip = 0
    start_time = time.time()

    for idx, d in enumerate(dates):
        df_day = df[df['date'] == d]

        try:
            ssvi = SSVI(
                df_day,
                venue=venue,
                date=d,
                min_options_per_slice=MIN_OPTIONS_PER_SLICE
            )
            ssvi.fit()

            # Collect parameters
            params_list.extend(ssvi.get_fitted_params())

            # Collect diagnostics
            eval_list.append(ssvi.evaluate_fit())

            n_success += 1

        except ValueError as e:
            n_skip += 1

            if n_skip <= 5 or n_skip % 50 == 0:
                print(f"    [{venue}] Day {idx+1}/{n_days} "
                      f"({pd.Timestamp(d).date()}) skipped: {e}")

        if (idx + 1) % 100 == 0:
            elapsed = (time.time() - start_time) / 60
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_days - idx - 1) / rate if rate > 0 else 0
            print(f"    [{venue}] Day {idx+1}/{n_days} | "
                  f"Success: {n_success} | Skip: {n_skip} | "
                  f"Elapsed: {elapsed:.1f}m | ETA: {eta:.1f}m")

    elapsed = (time.time() - start_time) / 60
    print(f"  [{venue}] Complete: {n_success} fitted, {n_skip} skipped "
          f"in {elapsed:.1f} min")

    return params_list, eval_list


def fit_all_venues():
    print("\n" + "=" * 60)
    print("  SSVI Surface Fitting Pipeline")
    print("=" * 60)
    print(f"  Sample window: {SAMPLE_START} → {SAMPLE_END}")

    SURFACES_DIR.mkdir(parents=True, exist_ok=True)

    all_params = []
    all_evals = []

    for venue in ["CME", "DER"]:
        print(f"\n  Loading {venue} data...")
        df = load_venue(venue)
        print(f"  Loaded {len(df):,} rows, {df['date'].nunique()} days")

        params, evals = fit_venue(venue, df)
        all_params.extend(params)
        all_evals.extend(evals)

    # Save fitted parameters
    df_params = pd.DataFrame(all_params)
    df_params['date'] = pd.to_datetime(df_params['date'])
    params_path = SURFACES_DIR / "ssvi_params.parquet"
    df_params.to_parquet(params_path, index=False)

    # Save diagnostics
    for e in all_evals:
        e['slice_rmses_str'] = str(e.pop('slice_rmses', {}))
    df_evals = pd.DataFrame(all_evals)
    df_evals['date'] = pd.to_datetime(df_evals['date'])
    evals_path = SURFACES_DIR / "ssvi_diagnostics.parquet"
    df_evals.to_parquet(evals_path, index=False)

    # Summary
    print(f"\n  {'=' * 60}")
    print(f"  SSVI FITTING COMPLETE")
    print(f"  {'=' * 60}")
    print(f"  Parameters saved to: {params_path}")
    print(f"  Diagnostics saved to: {evals_path}")
    print(f"  Total fitted slices: {len(df_params):,}")
    print(f"  Total fitted days:   {df_evals[df_evals['rmse'].notna()].shape[0]:,}")

    # Per-venue RMSE summary
    for venue in ["CME", "DER"]:
        v = df_evals[df_evals['venue'] == venue]
        if len(v) > 0:
            rmse_vals = v['rmse'].dropna()
            print(f"\n  {venue} RMSE summary ({len(rmse_vals)} days):")
            print(f"    Median: {rmse_vals.median():.4f} ({rmse_vals.median()*100:.2f}%)")
            print(f"    Mean:   {rmse_vals.mean():.4f} ({rmse_vals.mean()*100:.2f}%)")
            print(f"    p5:     {rmse_vals.quantile(0.05):.4f}")
            print(f"    p95:    {rmse_vals.quantile(0.95):.4f}")
            print(f"    Max:    {rmse_vals.max():.4f}")

    return df_params, df_evals

if __name__ == "__main__":
    np.random.seed(42)
    fit_all_venues()