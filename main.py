"""
Master Pipeline Runner

Runs the full empirical pipeline in dependency order with checkpoints, wall-clock timing, and verification guards at each stage. 
Designed to be kicked off once and left to run; each phase prints a clear banner so you can scroll through the log afterward.

Usage:
    python main.py                    # full pipeline, all phases
    python main.py --from 1a          # resume from Phase 1a onward
    python main.py --only 0b 4        # run only Phase 0b and Phase 4
    python main.py --skip-bootstrap   # skip the heavy Phase 3b bootstrap
    python main.py --bootstrap-B 200  # set bootstrap replicates (default 200)
    python main.py --bootstrap-workers 4  # parallel workers (default 6)

Phase dependency graph:
    0a  Data Scraping (Deribit)       (reads: Deribit API)
    0b  Data Cleaning & Auxiliary     (reads: raw CME/Deribit, FRED/yfinance APIs)
    1a  SSVI surface fitting          (reads: cleaned options)
    1b  RND extraction                (reads: ssvi_params.parquet)
    1c  CP tensor decomposition       (reads: ssvi_params.parquet)
    1d  Conditioning vectors          (reads: CP factors, auxiliary panel)
    2   Physical density + EP decomp  (reads: RNDs, spot prices)
    3   Conditional pricing kernel    (reads: RNDs, physical density, Z vectors)
    3b  Phase 3 bootstrap [optional]  (reads: Phase 3 theta, RNDs, Z vectors)
    4   BKM / CL20 decomposition      (reads: ssvi_params.parquet, Z vectors)
    5   Cross-venue analysis          (reads: RNDs, cumulant premia, Z vectors)
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_path

# Helpers
LOG_DIR = Path("results") / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

class PhaseTimer:

    def __init__(self, name, manifest):
        self.name = name
        self.manifest = manifest

    def __enter__(self):
        self.t0 = time.time()
        hdr = f"  {self.name}  "
        print(f"\n{'=' * 70}")
        print(f"{'=' * 70}")
        print(f"{hdr:=^70s}")
        print(f"{'=' * 70}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.time() - self.t0
        status = "OK" if exc_type is None else "FAILED"
        self.manifest.append({
            "phase": self.name,
            "status": status,
            "seconds": round(dt, 1),
            "error": str(exc_val) if exc_val else "",
        })
        if exc_type is None:
            print(f"\n  [{self.name}] completed in {_fmt_time(dt)}")
        else:
            print(f"\n  [{self.name}] FAILED after {_fmt_time(dt)}")
            traceback.print_exception(exc_type, exc_val, exc_tb)
        # Never suppress exceptions — the caller decides whether to continue
        return False

def _fmt_time(s):
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"

def _check_file(path, label=""):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"  [CHECKPOINT FAILED] {label}: {p} not found")
    sz = p.stat().st_size
    if sz == 0:
        raise ValueError(f"  [CHECKPOINT FAILED] {label}: {p} is empty")
    if sz < 1024:
        print(f"  [checkpoint] {label}: {p.name} ({sz} B)")
    else:
        print(f"  [checkpoint] {label}: {p.name} ({sz / 1024:.1f} KB)")

def _check_parquet_rows(path, label="", min_rows=1):
    p = Path(path)
    _check_file(p, label)
    n = len(pd.read_parquet(p))
    if n < min_rows:
        raise ValueError(f"  [CHECKPOINT FAILED] {label}: {p.name} has {n} rows, expected >= {min_rows}")
    print(f"  [checkpoint] {label}: {n:,} rows")
    return n

# Phase runners
def phase_0a_scraping():
    """Scrape raw historical trades from Deribit API."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.preprocessing.deribit.scrape_deribit import scrape_all_trades
    
    scrape_all_trades()

    from src.config import get_path
    _check_parquet_rows(get_path("raw_deribit_dir") / "btc_options_trades.parquet", 
                        "Deribit raw trades", min_rows=1000)

def phase_0b_preprocessing():
    """Clean CME/Deribit data, build auxiliary panel, and compute funding diffs."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.preprocessing.cme.clean_cme import process_cme_data
    from src.preprocessing.deribit.clean_deribit import process_deribit_data
    from src.preprocessing.auxiliary.auxiliary import build_auxiliary_panel
    from src.preprocessing.auxiliary.funding_diff import process_funding_differential
    
    process_cme_data()
    process_deribit_data()
    build_auxiliary_panel()
    process_funding_differential()

    from src.config import get_path
    _check_parquet_rows(get_path("cleaned_cme"), "CME cleaned", min_rows=1000)
    _check_parquet_rows(get_path("cleaned_deribit"), "Deribit cleaned", min_rows=1000)
    _check_parquet_rows(get_path("cleaned_auxiliary"), "Auxiliary panel", min_rows=100)
    _check_parquet_rows(get_path("funding_diff"), "Funding differential", min_rows=100)

def phase_1a_surfaces():
    """Fit SSVI surfaces for both venues; saves ssvi_params.parquet."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase1.fit_surfaces import fit_all_venues
    fit_all_venues()

    data_dir = get_path('data_phase1')
    _check_parquet_rows(data_dir / "ssvi_params.parquet", "SSVI params", min_rows=500)

    params = pd.read_parquet(data_dir / "ssvi_params.parquet")
    n_fwd = params["forward"].notna().sum()
    print(f"  [checkpoint] 'forward' column present: {n_fwd:,}/{len(params):,} non-null")

def phase_1b_densities():
    """Extract RNDs from saved surfaces; saves rnd_*.parquet."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase1.extract_densities import extract_all_densities
    extract_all_densities()

    data_dir = get_path('data_phase1')
    for venue in ["CME", "DER"]:
        _check_parquet_rows(data_dir / f"rnd_{venue}_summary.parquet",
                            f"{venue} RND summary", min_rows=100)

def phase_1c_tensor():
    """CP tensor decomposition on both maturity grids with sign convention."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase1.run_tensor_pca import run_tensor_decomposition
    for grid in ["almeida", "broad"]:
        print(f"\n{'#' * 60}")
        print(f"# Grid: {grid}")
        print(f"{'#' * 60}")
        run_tensor_decomposition(grid_name=grid)

    data_dir = get_path('data_phase1')
    results_dir = get_path('results_phase1')
    _check_file(data_dir / "tensor_pca_diagnostics_almeida.csv", "Almeida diagnostics")
    _check_file(data_dir / "tensor_pca_state_almeida.parquet", "Z_IVS state vector")

def phase_1d_conditioning():
    """Build conditioning vectors (Z_crypto, Z_macro, Z_full)."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase1.build_conditioning_vectors import build_conditioning_vectors
    build_conditioning_vectors()

    data_dir = get_path('data_phase1')
    _check_parquet_rows(data_dir / "Z_crypto.parquet", "Z_crypto", min_rows=100)
    
    Z = pd.read_parquet(data_dir / "Z_crypto.parquet")
    rho = Z[["Z_IVS_1", "rv"]].dropna().corr().iloc[0, 1]
    print(f"  [checkpoint] corr(Z_IVS_1, RV) = {rho:+.3f} (must be > 0)")
    assert rho > 0, "Sign convention violated — re-run Phase 1c"

def phase_2_ep(spot_data=None):
    """Physical density estimation and equity premium decomposition."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase2.run_phase2 import run_phase2
    run_phase2()

    phase2_dir = Path("results") / "phase2" / "tables"
    _check_file(phase2_dir / "ep_decomposition_summary.csv", "EP summary")
    _check_file(phase2_dir / "ep_bootstrap_ci.csv", "EP bootstrap CIs")

    # Headline check: raw-moment anchor should be in the table
    boot = pd.read_csv(phase2_dir / "ep_bootstrap_ci.csv")
    raw = boot[boot["estimator"] == "raw_moment"]
    if len(raw) == 0:
        print("  [WARN] No raw_moment row in ep_bootstrap_ci.csv")
    else:
        r = raw.iloc[0]
        print(f"  [checkpoint] raw-moment EP: {r['point']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")

def phase_3_kernel():
    """Conditional pricing kernel estimation (new (b,c,d) parameterization)."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase3.run_phase3 import run_phase3
    run_phase3()

    from src.config import get_path
    phase3_dir = Path(get_path("cleaned_cme")).parent.parent / "phase3"
    tab_dir = Path("results") / "phase3" / "tables"

    for venue in ["CME", "DER"]:
        f = phase3_dir / f"phase3_{venue}_crypto.npz"
        _check_file(f, f"{venue} kernel (crypto)")
        d = np.load(f)
        n_params = len(d["theta"])
        expected = 3 * (1 + 3)  # (b,c,d) x (const + 3 Z vars for crypto)
        if n_params != expected:
            print(f"  [WARN] {venue} theta has {n_params} params, "
                  f"expected {expected}. Old (a,b,c,d) parameterization?")
        else:
            print(f"  [checkpoint] {venue} theta: {n_params} params "
                  f"(b,c,d parameterization ✓)")

    _check_file(phase3_dir / "mfk_unconditional.npz", "MFK (block-bootstrap bands)")

def phase_3b_kernel_bootstrap(B=200, workers=6):
    """Block-bootstrap CIs for the tercile kernel coefficients."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase3.run_phase3_bootstrap import run_bootstrap
    run_bootstrap(venues=["CME", "DER"], spec_name="crypto", B=B, workers=workers)

    tab_dir = Path("results") / "phase3" / "tables"
    _check_file(tab_dir / "phase3_bootstrap_ci_crypto.csv", "kernel bootstrap CIs")

    ci = pd.read_csv(tab_dir / "phase3_bootstrap_ci_crypto.csv")
    c_rows = ci[ci["coef"] == "c"]
    if len(c_rows):
        print("\n  Curvature c by tercile [95% CI], P(c < 0):")
        for _, r in c_rows.iterrows():
            print(f"    {r['venue']:>4s} {r['tercile']:>4s}: "
                  f"{r['point']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"P(c<0) = {r['frac_negative']:.3f}")

def phase_4_bkm():
    """BKM moment extraction and CL20/CL24 cumulant decomposition."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase4.run_phase4 import run_phase4
    run_phase4()

    phase4_dir = Path("results") / "phase4" / "tables"
    _check_file(phase4_dir / "cyl_decomposition_matched.csv",
                "CL20 matched-day decomposition (headline)")
    _check_file(phase4_dir / "kappa_sensitivity.csv",
                "κ-bound sensitivity")

    # Headline kurtosis share check
    dec = pd.read_csv(phase4_dir / "cyl_decomposition_matched.csv")
    for venue in ["CME", "DER"]:
        row = dec[(dec["venue"] == venue) & (dec["regime"] == "unconditional")]
        if len(row):
            sk = row.iloc[0].get("share_kurt", None)
            if sk is not None:
                print(f"  [checkpoint] {venue} kurtosis share = {sk:.3f}")

    # κ sensitivity: kurtosis share range across truncation bounds
    ks = pd.read_csv(phase4_dir / "kappa_sensitivity.csv")
    for venue in ["CME", "DER"]:
        v = ks[ks["venue"] == venue]
        if "share_kurt" in v.columns and len(v) >= 2:
            lo, hi = v["share_kurt"].min(), v["share_kurt"].max()
            print(f"  [checkpoint] {venue} kurtosis share range "
                  f"across κ bounds: [{lo:.3f}, {hi:.3f}]")

def phase_5_cross_venue():
    """Cross-venue MFK, panel regressions, regional decomposition."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from src.phase5.run_phase5 import run_phase5
    run_phase5()

    tab_dir = Path("results") / "phase5" / "tables"
    _check_file(tab_dir / "matched_difference_regressions.csv",
                "Matched-diff regressions (headline)")
    _check_file(tab_dir / "panel_regressions_dk.csv",
                "Driscoll-Kraay panel (secondary)")

    # Headline wedge verification
    try:
        diff = pd.read_csv(tab_dir / "matched_difference_regressions.csv")
        wedge_rows = diff[diff["regressor"] == "const (venue wedge)"]
        if len(wedge_rows):
            print("\n  Venue wedge (matched-difference, headline):")
            for _, row in wedge_rows.iterrows():
                sig = row.get("stars", "")
                print(f"    {row['dep_var']}: β = {row['coef']:+.5f} "
                      f"(t = {row['t_stat']:+.3f}){sig}")
    except Exception as e:
        print(f"  [WARN] Could not parse wedge table: {e}")


# Orchestrator
PHASE_ORDER = [
    ("0a", "Data Scraping (Deribit)",          phase_0a_scraping),
    ("0b", "Data Cleaning & Auxiliary",        phase_0b_preprocessing),
    ("1a", "SSVI Surface Fitting",             phase_1a_surfaces),
    ("1b", "RND Extraction (Figlewski tails)", phase_1b_densities),
    ("1c", "CP Tensor Decomposition",          phase_1c_tensor),
    ("1d", "Conditioning Vectors",             phase_1d_conditioning),
    ("2",  "Physical Density & EP Decomp",     phase_2_ep),
    ("3",  "Conditional Pricing Kernel",       phase_3_kernel),
    ("3b", "Kernel Bootstrap (heavy)",         None),  # special handling
    ("4",  "BKM / CL20 Decomposition",         phase_4_bkm),
    ("5",  "Cross-Venue Analysis",             phase_5_cross_venue),
]

PHASE_LABELS = {tag: name for tag, name, _ in PHASE_ORDER}

def main():
    parser = argparse.ArgumentParser(
        description="Run the full Bitcoin risk-premia pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # full pipeline
  python main.py --from 1a              # resume from Phase 1a
  python main.py --only 0b 4            # run only Phases 0b and 4
  python main.py --skip-bootstrap       # skip the heavy kernel bootstrap
  python main.py --skip 0a              # skip the long API scraping step
        """,
    )
    parser.add_argument("--from", dest="from_phase", default=None,
                        help="Start from this phase (inclusive)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these phases")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Skip these phases")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip the Phase 3b kernel bootstrap")
    parser.add_argument("--bootstrap-B", type=int, default=200,
                        help="Number of bootstrap replicates (default 200)")
    parser.add_argument("--bootstrap-workers", type=int, default=6,
                        help="Parallel workers for bootstrap (default 6)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to the next phase on failure "
                             "(default: abort)")
    args = parser.parse_args()

    # Resolve which phases to run
    all_tags = [tag for tag, _, _ in PHASE_ORDER]
    if args.only:
        run_tags = set(args.only)
    elif args.from_phase:
        try:
            idx = all_tags.index(args.from_phase)
        except ValueError:
            print(f"Unknown phase '{args.from_phase}'. "
                  f"Valid: {', '.join(all_tags)}")
            sys.exit(1)
        run_tags = set(all_tags[idx:])
    else:
        run_tags = set(all_tags)

    if args.skip:
        run_tags -= set(args.skip)
    if args.skip_bootstrap:
        run_tags.discard("3b")

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   Bitcoin Risk Premia — Full Pipeline Run".ljust(69) + "#")
    print(f"#   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(69) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\n  Phases to run: {', '.join(t for t in all_tags if t in run_tags)}")
    if args.skip_bootstrap:
        print("  (kernel bootstrap skipped)")
    print()

    manifest = []
    t_total = time.time()

    for tag, name, runner in PHASE_ORDER:
        if tag not in run_tags:
            continue

        # Phase 3b (bootstrap) needs special argument forwarding
        if tag == "3b":
            with PhaseTimer(f"Phase {tag}: {name}", manifest) as _:
                phase_3b_kernel_bootstrap(
                    B=args.bootstrap_B,
                    workers=args.bootstrap_workers,
                )
            if manifest[-1]["status"] == "FAILED" and not args.continue_on_error:
                break
            continue

        try:
            with PhaseTimer(f"Phase {tag}: {name}", manifest):
                runner()
        except Exception:
            if not args.continue_on_error:
                print(f"\n  Pipeline aborted at Phase {tag}. "
                      f"Use --continue-on-error to proceed past failures, "
                      f"or --from {tag} to resume after fixing the issue.")
                break

    # Summary
    dt_total = time.time() - t_total
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  {'Phase':<45s} {'Status':>8s} {'Time':>10s}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    for row in manifest:
        status_str = "✓" if row["status"] == "OK" else "✗ FAIL"
        print(f"  {row['phase']:<45s} {status_str:>8s} {_fmt_time(row['seconds']):>10s}")
        if row["error"]:
            print(f"    → {row['error'][:80]}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    print(f"  {'Total':<45s} {'':>8s} {_fmt_time(dt_total):>10s}")

    n_ok = sum(1 for r in manifest if r["status"] == "OK")
    n_fail = sum(1 for r in manifest if r["status"] == "FAILED")
    if n_fail == 0:
        print(f"\n  All {n_ok} phases completed successfully.")
    else:
        print(f"\n  {n_ok} phases OK, {n_fail} FAILED.")

    # Save manifest
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = LOG_DIR / f"run_{ts}.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"  Run manifest saved to {manifest_path}")

    # Post-run verification checklist
    print("\n" + "=" * 70)
    print("  POST-RUN CHECKLIST (manual)")
    print("=" * 70)
    print("""
  After a successful full run, verify these locked facts before updating
  the LaTeX draft. Each must be checked against the new CSVs — do not
  carry forward old numbers.

  1. CP rank: R = 1 survives all four genuine seeds? (tensor_pca_diagnostics_seeds.csv)
  2. Almeida-grid R = 2 CORCONDIA still ~ -66.8? (tensor_pca_diagnostics.csv)
  3. Lambda weights at θ = 2: (1, -1, 1)? (cyl_decomposition_matched.csv header)
  4. Kurtosis share ≈ 30%? Stable across κ bounds? (kappa_sensitivity.csv)
  5. Low-vol kernel: c < 0? Bootstrap P(c < 0) > 0.95? (phase3_bootstrap_ci_crypto.csv)
  6. Variance wedge β₂ significant? β₃, β₄ insignificant? (matched_difference_regressions.csv)
  7. EP totals: compare raw-moment vs density-based rows (ep_bootstrap_ci.csv)
  8. MFK tent shape: significant under block-bootstrap bands? (fig_mfk_unconditional.png)
  9. Macro/full specs: did they converge with the new (b,c,d) parameterization?
     (run_phase3 log — if yes, update the non-convergence narrative)
    """)

    sys.exit(1 if n_fail > 0 else 0)

if __name__ == "__main__":
    main()