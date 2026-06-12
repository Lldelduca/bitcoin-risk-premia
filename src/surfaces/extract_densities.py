"""
Risk-Neutral Density Extraction Orchestrator.

Loads the fitted SSVI parameters, reconstructs the surface for each (venue, date), and extracts the risk-neutral density at the primary
maturity (27 days) and robustness maturities (14, 60 days).

"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import get_path, SAMPLE
from src.surfaces.ssvi import SSVI
from src.surfaces.breeden_litzenberger import (extract_rnd_with_gpd_tails, validate_rnd)

SAMPLE_START = SAMPLE['start_date']
SAMPLE_END = SAMPLE['end_date']
TARGET_MATURITIES_DAYS = [14, 27, 60]

N_STRIKES = 500
MIN_OPTIONS_PER_SLICE = 4

SURFACES_DIR = Path(get_path('cleaned_cme')).parent.parent / "surfaces"

def load_venue_data(venue: str) -> pd.DataFrame:
    if venue == "CME":
        path = get_path('cleaned_cme')
    elif venue == "DER":
        path = get_path('cleaned_deribit')
    else:
        raise ValueError(f"Unknown venue: {venue}")

    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= SAMPLE_START) & (df['date'] <= SAMPLE_END)
    return df[mask].copy()

def load_ssvi_params() -> pd.DataFrame:
    """Loads the fitted SSVI parameters saved by fit_surfaces (Phase 1a)."""
    params_path = SURFACES_DIR / "ssvi_params.parquet"
    if not params_path.exists():
        raise FileNotFoundError(
            f"{params_path} not found. Run fit_surfaces first; densities are "
            f"extracted from the saved surfaces, never by refitting."
        )
    params = pd.read_parquet(params_path)
    params["date"] = pd.to_datetime(params["date"])
    return params

def extract_venue_densities(venue: str, df: pd.DataFrame, params: pd.DataFrame):
    params_v = params[params["venue"] == venue]
    dates = np.sort(params_v["date"].unique())
    n_days = len(dates)
    has_forward_col = "forward" in params_v.columns and params_v["forward"].notna().all()
    print(f"\n  [{venue}] Reconstructing surfaces for {n_days} fitted days × "
          f"{len(TARGET_MATURITIES_DAYS)} maturities "
          f"({'forwards from parquet' if has_forward_col else 'forwards from cleaned data'})")

    results = []
    n_success = 0
    n_skip = 0
    start_time = time.time()

    for idx, d in enumerate(dates):
        params_day = params_v[params_v["date"] == d]
        df_day = df[df["date"] == d]

        # Reconstruct the fitted surface (no refitting)
        try:
            forward_map = None
            if not has_forward_col:
                if df_day.empty:
                    n_skip += 1
                    continue
                forward_map = df_day.groupby("tau")["forward_price"].mean()
            ssvi = SSVI.from_params(params_day, forward_map=forward_map,
                                    venue=venue, date=d)
        except ValueError:
            n_skip += 1
            continue

        # Extract RND at each target maturity
        fitted_taus = ssvi.res['maturities']
        r = (df_day['risk_free_rate'].iloc[0]
             if ('risk_free_rate' in df_day.columns and not df_day.empty) else 0.0)

        for tau_days in TARGET_MATURITIES_DAYS:
            tau = tau_days / 365.25

            # Check that the target maturity is within the fitted range
            if tau < fitted_taus.min() * 0.8 or tau > fitted_taus.max() * 1.2:
                continue

            try:
                rnd = extract_rnd_with_gpd_tails(ssvi, tau, n_strikes=N_STRIKES, r=r)
                val = validate_rnd(rnd)
                tail = rnd.get('tail_diagnostics', {})

                results.append({
                    'date': d,
                    'venue': venue,
                    'tau_days': tau_days,
                    'tau': tau,
                    'forward': rnd['forward'],
                    'returns': rnd['returns'],
                    'density': rnd['density'],
                    'integral': val['integral'],
                    'mean_return': val['mean_return'],
                    'std_return': val['std_return'],
                    'valid': val['valid'],

                    # Tail-splice diagnostics: xi is the option-implied GPD
                    'xi_left': tail.get('xi_left', np.nan),
                    'xi_right': tail.get('xi_right', np.nan),
                    'splice_left': tail.get('splice_left', np.nan),
                    'splice_right': tail.get('splice_right', np.nan),
                    'mass_renorm': tail.get('mass_renorm', np.nan),
                })

            except Exception as e:
                if n_skip < 10:
                    print(f"    [{venue}] {pd.Timestamp(d).date()} τ={tau_days}d: {e}")
                continue

        n_success += 1

        # Progress
        if (idx + 1) % 100 == 0:
            elapsed = (time.time() - start_time) / 60
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_days - idx - 1) / rate if rate > 0 else 0
            print(f"    [{venue}] Day {idx+1}/{n_days} | "
                  f"RNDs: {len(results)} | "
                  f"Elapsed: {elapsed:.1f}m | ETA: {eta:.1f}m")

    elapsed = (time.time() - start_time) / 60
    print(f"  [{venue}] Complete: {n_success} days, {len(results)} RNDs, "
          f"{n_skip} skipped in {elapsed:.1f} min")

    return results

def save_rnd_results(results, venue):
    if not results:
        print(f"  [{venue}] No results to save.")
        return

    scalar_rows = []
    density_rows = []

    for r in results:
        scalar_rows.append({
            'date': r['date'],
            'venue': r['venue'],
            'tau_days': r['tau_days'],
            'tau': r['tau'],
            'forward': r['forward'],
            'integral': r['integral'],
            'mean_return': r['mean_return'],
            'std_return': r['std_return'],
            'valid': r['valid'],
            'xi_left': r.get('xi_left', np.nan),
            'xi_right': r.get('xi_right', np.nan),
            'splice_left': r.get('splice_left', np.nan),
            'splice_right': r.get('splice_right', np.nan),
            'mass_renorm': r.get('mass_renorm', np.nan),
        })
        density_rows.append({
            'date': r['date'],
            'tau_days': r['tau_days'],
            'returns': r['returns'].tolist(),
            'density': r['density'].tolist(),
        })

    # Save scalar summary
    df_scalar = pd.DataFrame(scalar_rows)
    scalar_path = SURFACES_DIR / f"rnd_{venue}_summary.parquet"
    df_scalar.to_parquet(scalar_path, index=False)

    # Save full densities
    df_density = pd.DataFrame(density_rows)
    df_density['date'] = pd.to_datetime(df_density['date'])
    density_path = SURFACES_DIR / f"rnd_{venue}_densities.parquet"
    df_density.to_parquet(density_path, index=False)

    print(f"  [{venue}] Saved: {scalar_path.name} ({len(df_scalar)} rows)")
    print(f"  [{venue}] Saved: {density_path.name} ({len(df_density)} rows)")

    return df_scalar

def extract_all_densities():
    print("\n" + "=" * 60)
    print("  Risk-Neutral Density Extraction Pipeline")
    print("=" * 60)
    print(f"  Sample: {SAMPLE_START} → {SAMPLE_END}")
    print(f"  Target maturities: {TARGET_MATURITIES_DAYS} days")

    SURFACES_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    print("\n  Loading fitted SSVI parameters...")
    params = load_ssvi_params()
    print(f"  Loaded {len(params):,} parameter rows "
          f"({params['date'].nunique()} dates, venues: {sorted(params['venue'].unique())})")

    for venue in ["CME", "DER"]:
        df = load_venue_data(venue)
        print(f"\n  {venue}: {len(df):,} options, {df['date'].nunique()} days")

        results = extract_venue_densities(venue, df, params)
        summary = save_rnd_results(results, venue)
        if summary is not None:
            all_summaries.append(summary)

    # Summary statistics
    if all_summaries:
        df_all = pd.concat(all_summaries, ignore_index=True)

        print(f"\n  {'=' * 60}")
        print(f"  RND EXTRACTION COMPLETE")
        print(f"  {'=' * 60}")

        for venue in ["CME", "DER"]:
            v = df_all[df_all['venue'] == venue]
            print(f"\n  {venue}:")
            print(f"    Total RNDs extracted: {len(v):,}")
            for tau_d in TARGET_MATURITIES_DAYS:
                vt = v[v['tau_days'] == tau_d]
                valid = vt['valid'].sum()
                print(f"    τ={tau_d}d: {len(vt)} extracted, "
                      f"{valid} valid ({valid/max(len(vt),1):.0%})")
            print(f"    Mean return (should ≈ 1.0): "
                  f"{v['mean_return'].mean():.4f}")
            print(f"    Mean std return: {v['std_return'].mean():.4f}")

if __name__ == "__main__":
    np.random.seed(42)
    extract_all_densities()