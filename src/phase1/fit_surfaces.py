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
from src.config import get_path, get_sample_window, get_ssvi_config
from src.phase1.ssvi import SSVI

SAMPLE_START, SAMPLE_END = get_sample_window()
MIN_OPTIONS_PER_SLICE = get_ssvi_config().get('min_options_per_slice', 4)

DATA_DIR = get_path('data_phase1')
RESULTS_DIR = get_path('results_phase1')
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

def fit_venue(venue: str, df: pd.DataFrame) -> tuple:
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
        df_day = df[df['date'] == d].copy()

        try:
            ssvi = SSVI(
                df_day,
                venue=venue,
                date=d,
                min_options_per_slice=MIN_OPTIONS_PER_SLICE
            )
            ssvi.fit()

            params_list.extend(ssvi.get_fitted_params())
            eval_list.append(ssvi.evaluate_fit())
            n_success += 1

        except Exception as e:
            n_skip += 1
            if n_skip <= 5 or n_skip % 50 == 0:
                print(f"    [{venue}] Day {idx+1}/{n_days} "
                      f"({pd.Timestamp(d).date()}) skipped: {e}")

        if (idx + 1) % 100 == 0:
            elapsed = (time.time() - start_time) / 60
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_days - idx - 1) / rate if rate > 0 else 0
            print(f"    [{venue}] Day {idx+1:4d}/{n_days} | "
                  f"Success: {n_success} | Skip: {n_skip} | "
                  f"Elapsed: {elapsed:.1f}m | ETA: {eta:.1f}m")

    elapsed = (time.time() - start_time) / 60
    print(f"  [{venue}] Complete: {n_success} fitted, {n_skip} skipped "
          f"in {elapsed:.1f} min")

    return params_list, eval_list

def fit_all_venues():
    print("\n" + "=" * 60)
    print("  Phase 1a: SSVI Surface Fitting")
    print("=" * 60)

    all_params = []
    all_evals = []

    for venue in ["CME", "DER"]:
        df = load_venue(venue)
        print(f"  {venue}: {len(df):,} options, {df['date'].nunique()} days")
        params, evals = fit_venue(venue, df)
        all_params.extend(params)
        all_evals.extend(evals)

    # Save parameters (Data)
    df_params = pd.DataFrame(all_params)
    df_params['date'] = pd.to_datetime(df_params['date'])
    params_path = DATA_DIR / "ssvi_params.parquet"
    df_params.to_parquet(params_path, index=False)

    # Save diagnostics (Results)
    for e in all_evals:
        e['slice_rmses_str'] = str(e.pop('slice_rmses', {}))
    df_evals = pd.DataFrame(all_evals)
    df_evals['date'] = pd.to_datetime(df_evals['date'])
    evals_path = RESULTS_DIR / "ssvi_diagnostics.parquet"
    df_evals.to_parquet(evals_path, index=False)

    # Summary
    print(f"\n  {'=' * 60}")
    print(f"  SSVI FITTING COMPLETE")
    print(f"  {'=' * 60}")
    print(f"  Parameters saved to:  {params_path}")
    print(f"  Diagnostics saved to: {evals_path}")
    print(f"  Total fitted slices: {len(df_params):,}")
    print(f"  Total fitted days:   {df_evals[df_evals['rmse'].notna()].shape[0]:,}")

# Per-venue RMSE summary
    for venue in ["CME", "DER"]:
        v = df_evals[df_evals['venue'] == venue]
        if len(v) > 0:
            rmse_vals = v['rmse'].dropna()
            print(f"\n  {venue} RMSE Fit Summary ({len(rmse_vals)} days):")
            print(f"    Mean/Median: {rmse_vals.mean():.4%} / {rmse_vals.median():.4%}")
            print(f"    p05/p95:     {rmse_vals.quantile(0.05):.4%} / {rmse_vals.quantile(0.95):.4%}")
            print(f"    Max RMSE:    {rmse_vals.max():.4%}")

if __name__ == "__main__":
    np.random.seed(42)
    fit_all_venues()