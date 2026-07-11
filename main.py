"""
Master Pipeline Runner — Bitcoin Risk Premia Across Venues.

Runs the full empirical pipeline in dependency order with checkpoints, wall-clock timing, and verification guards at each stage. 
Designed to be kicked off once and left to run; each phase prints a clear banner. A supervisor reproducing the thesis runs:

    python main.py

and then executes the notebooks in notebooks/ top to bottom.

Usage:
    python main.py                    # full pipeline, all phases
    python main.py --from 1a          # resume from Phase 1a onward
    python main.py --only 0b 4        # run only Phase 0b and Phase 4
    python main.py --skip 0a          # skip the long API scraping step
    python main.py --skip-bootstrap   # skip Phase 3b (and therefore 3c)
    python main.py --skip-diagnostics # skip Phases 2b and 2c (appendix only)
    python main.py --bootstrap-B 200  # bootstrap replicates (default 200)
    python main.py --bootstrap-workers 6

Phase dependency graph:
    0a  Data Scraping (Deribit)         (reads: Deribit API)
    0b  Data Cleaning & Auxiliary       (reads: raw CME/Deribit, FRED/yfinance)
    1a  SSVI surface fitting            (reads: cleaned options)
    1b  RND extraction                  (reads: ssvi_params.parquet)
    1c  CP tensor decomposition         (reads: ssvi_params.parquet)
    1d  Conditioning vectors            (reads: CP factors, auxiliary panel)
    2   Physical density + EP decomp    (reads: RNDs, spot prices)
    2b  NB-sweep diagnostic [appendix]  (reads: Phase 2 loaders)
    2c  Grid-sensitivity diag [appendix](reads: RNDs, spot prices)
    3   Conditional pricing kernel      (reads: RNDs, p-hat, Z vectors)
    3b  Kernel bootstrap [heavy]        (reads: Phase 3 theta, RNDs, Z)
    3c  Joint regime test               (reads: Phase 3b draws)
    4   BKM / CL20 decomposition        (reads: ssvi_params, Z vectors)
    4b  CL24 regional decomposition     (reads: RNDs, Z vectors; Q-only)
    5   Cross-venue analysis            (reads: RNDs, cumulant premia, Z)
    5b  Friction-proxy regressions      (reads: premia, basis, funding)
    5c  Inverse-contract extension      (reads: RNDs, cumulant premia)
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
        print(f"{hdr:=^70s}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.time() - self.t0
        status = "OK" if exc_type is None else "FAILED"
        self.manifest.append({
            "phase": self.name, "status": status,
            "seconds": round(dt, 1),
            "error": str(exc_val) if exc_val else "",
        })
        if exc_type is None:
            print(f"\n  [{self.name}] completed in {_fmt_time(dt)}")
        else:
            print(f"\n  [{self.name}] FAILED after {_fmt_time(dt)}")
            traceback.print_exception(exc_type, exc_val, exc_tb)
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
    print(f"  [checkpoint] {label}: {p.name} "
          f"({sz} B)" if sz < 1024 else
          f"  [checkpoint] {label}: {p.name} ({sz / 1024:.1f} KB)")


def _check_parquet_rows(path, label="", min_rows=1):
    p = Path(path)
    _check_file(p, label)
    n = len(pd.read_parquet(p))
    if n < min_rows:
        raise ValueError(f"  [CHECKPOINT FAILED] {label}: {p.name} has "
                         f"{n} rows, expected >= {min_rows}")
    print(f"  [checkpoint] {label}: {n:,} rows")
    return n


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def phase_0a_scraping():
    from src.preprocessing.deribit.scrape_deribit import scrape_all_trades
    scrape_all_trades()
    _check_parquet_rows(get_path("raw_deribit_dir") / "btc_options_trades.parquet",
                        "Deribit raw trades", min_rows=1000)


def phase_0b_preprocessing():
    from src.preprocessing.cme.clean_cme import process_cme_data
    from src.preprocessing.deribit.clean_deribit import process_deribit_data
    from src.preprocessing.auxiliary.auxiliary import build_auxiliary_panel
    from src.preprocessing.auxiliary.funding_diff import process_funding_differential
    process_cme_data()
    process_deribit_data()
    build_auxiliary_panel()
    process_funding_differential()
    _check_parquet_rows(get_path("cleaned_cme"), "CME cleaned", min_rows=1000)
    _check_parquet_rows(get_path("cleaned_deribit"), "Deribit cleaned", min_rows=1000)
    _check_parquet_rows(get_path("cleaned_auxiliary"), "Auxiliary panel", min_rows=100)
    _check_parquet_rows(get_path("funding_diff"), "Funding differential", min_rows=100)


def phase_1a_surfaces():
    from src.phase1.fit_surfaces import fit_all_venues
    fit_all_venues()
    data_dir = get_path("data_phase1")
    _check_parquet_rows(data_dir / "ssvi_params.parquet", "SSVI params", min_rows=500)
    params = pd.read_parquet(data_dir / "ssvi_params.parquet")
    n_fwd = params["forward"].notna().sum()
    print(f"  [checkpoint] 'forward' column: {n_fwd:,}/{len(params):,} non-null")


def phase_1b_densities():
    from src.phase1.extract_densities import extract_all_densities
    extract_all_densities()
    data_dir = get_path("data_phase1")
    for venue in ["CME", "DER"]:
        _check_parquet_rows(data_dir / f"rnd_{venue}_summary.parquet",
                            f"{venue} RND summary", min_rows=100)


def phase_1c_tensor():
    from src.phase1.run_tensor_pca import run_tensor_decomposition
    for grid in ["almeida", "broad"]:
        print(f"\n{'#' * 60}\n# Grid: {grid}\n{'#' * 60}")
        run_tensor_decomposition(grid_name=grid)
    data_dir = get_path("data_phase1")
    _check_file(data_dir / "tensor_pca_diagnostics_almeida.csv", "Almeida diagnostics")
    _check_file(data_dir / "tensor_pca_state_almeida.parquet", "Z_IVS state vector")


def phase_1d_conditioning():
    from src.phase1.build_conditioning_vectors import build_conditioning_vectors
    build_conditioning_vectors()
    data_dir = get_path("data_phase1")
    cond_dir = Path(get_path("cleaned_cme")).parent.parent / "conditioning"
    zpath = cond_dir / "Z_crypto.parquet" if (cond_dir / "Z_crypto.parquet").exists() \
        else data_dir / "Z_crypto.parquet"
    _check_parquet_rows(zpath, "Z_crypto", min_rows=100)
    Z = pd.read_parquet(zpath)
    rho = Z[["Z_IVS_1", "rv"]].dropna().corr().iloc[0, 1]
    print(f"  [checkpoint] corr(Z_IVS_1, RV) = {rho:+.3f} (must be > 0)")
    assert rho > 0, "Sign convention violated — re-run Phase 1c"


def phase_2_ep():
    from src.phase2.run_phase2 import run_phase2
    run_phase2()
    tab = get_path("results_phase2") / "tables"
    _check_file(tab / "ep_decomposition_summary.csv", "EP summary")
    _check_file(tab / "ep_bootstrap_ci.csv", "EP bootstrap CIs")
    boot = pd.read_csv(tab / "ep_bootstrap_ci.csv")
    raw = boot[boot["estimator"] == "raw_moment"]
    if len(raw):
        r = raw.iloc[0]
        print(f"  [checkpoint] raw-moment EP: {r['point']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    summ = pd.read_csv(tab / "ep_decomposition_summary.csv")
    for est in ["almeida", "vanilla", "kde"]:
        assert est in set(summ["estimator"]), f"estimator '{est}' missing"
    print("  [checkpoint] three-estimator lineup present "
          "(enhanced / AGMW vanilla / KDE)")


def phase_2b_nb_sweep():
    """Appendix diagnostic: bin-count stability, enhanced vs AGMW vanilla."""
    from src.phase2.run_nb_sweep import run_nb_sweep
    run_nb_sweep()
    tab = get_path("results_phase2") / "tables"
    _check_file(tab / "nb_sweep_summary.csv", "NB-sweep summary")
    _check_file(tab / "sigma_binding_check.csv", "Published-bound binding check")
    summary = pd.read_csv(tab / "nb_sweep_summary.csv")
    for _, r in summary.iterrows():
        print(f"    {r['estimator']:>7s} {r['venue']}: "
              f"density={r['density_range_pct']:.3f}, ep={r['ep_range_pct']:.3f}")
    bind = pd.read_csv(tab / "sigma_binding_check.csv")
    print(f"  [checkpoint] published bounds bind in "
          f"{int(bind['van_any_bound_binds'].sum())}/{len(bind)} NB settings")


def phase_2c_grid_sensitivity():
    """Appendix diagnostic: integration-window dependence of the EP total."""
    from src.phase2.run_grid_sensitivity import run_grid_sensitivity
    run_grid_sensitivity()
    tab = get_path("results_phase2") / "tables"
    _check_file(tab / "grid_sensitivity.csv", "Grid sensitivity table")
    gs = pd.read_csv(tab / "grid_sensitivity.csv")
    w2 = gs[(gs.grid == "wide_2") & (gs.estimator == "almeida")]
    for _, r in w2.iterrows():
        print(f"  [checkpoint] {r['venue']} enhanced on [0.30, 2.60]: "
              f"{r['total_ep']:+.4f}, gap closed "
              f"{r['gap_closed_vs_headline']:+.1%}")


def phase_3_kernel():
    from src.phase3.run_phase3 import run_phase3
    run_phase3()
    phase3_dir = Path(get_path("cleaned_cme")).parent.parent / "phase3"
    for venue in ["CME", "DER"]:
        f = phase3_dir / f"phase3_{venue}_crypto.npz"
        _check_file(f, f"{venue} kernel (crypto)")
        d = np.load(f, allow_pickle=True)
        expected = 3 * (1 + 3)
        if len(d["theta"]) != expected:
            print(f"  [WARN] {venue} theta has {len(d['theta'])} params, "
                  f"expected {expected} — old parameterization?")
        else:
            print(f"  [checkpoint] {venue} theta: {expected} params "
                  f"((b,c,d) parameterization)")
        if "tercile_labels" not in d:
            print(f"  [WARN] {venue} npz lacks tercile_labels — pre-fix run?")
    _check_file(phase3_dir / "mfk_unconditional.npz", "MFK (bootstrap bands)")


def phase_3b_kernel_bootstrap(B=200, workers=6):
    from src.phase3.run_phase3_bootstrap import run_bootstrap
    run_bootstrap(venues=["CME", "DER"], spec_name="crypto", B=B, workers=workers)
    tab = Path("results") / "phase3" / "tables"
    _check_file(tab / "phase3_bootstrap_ci_crypto.csv", "kernel bootstrap CIs")
    ci = pd.read_csv(tab / "phase3_bootstrap_ci_crypto.csv")
    c_rows = ci[ci["coef"] == "c"]
    for _, r in c_rows.iterrows():
        print(f"    {r['venue']:>4s} {r['tercile']:>4s}: "
              f"c = {r['point']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
              f"P(c<0) = {r['frac_negative']:.3f}")


def phase_3c_joint_regime_test():
    """Post-process: joint Wald tests + curvature-at-money from 3b draws."""
    phase3_dir = Path(get_path("cleaned_cme")).parent.parent / "phase3"
    draws = list(phase3_dir.glob("phase3_bootstrap_draws_*_crypto.parquet"))
    if len(draws) < 2:
        print("  [SKIPPED] no bootstrap draws found — run Phase 3b first "
              "(or drop --skip-bootstrap).")
        return
    from src.phase3.joint_regime_test import run_all
    run_all(venues=("CME", "DER"), spec="crypto")
    tab = Path("results") / "phase3" / "tables"
    _check_file(tab / "joint_regime_test_crypto.csv", "Joint regime test")
    jt = pd.read_csv(tab / "joint_regime_test_crypto.csv")
    for _, r in jt.iterrows():
        print(f"  [checkpoint] {r.get('venue', '?')}: "
              f"Wald = {r.get('wald_stat', float('nan')):.2f}, "
              f"p = {r.get('p_value', float('nan')):.4f}")


def phase_4_bkm():
    from src.phase4.run_phase4 import run_phase4
    run_phase4()
    tab = Path("results") / "phase4" / "tables"
    _check_file(tab / "cyl_decomposition_matched.csv",
                "CL20 matched-day decomposition (headline)")
    _check_file(tab / "kappa_sensitivity.csv", "kappa-bound sensitivity")
    dec = pd.read_csv(tab / "cyl_decomposition_matched.csv")
    for venue in ["CME", "DER"]:
        row = dec[(dec["venue"] == venue) & (dec["regime"] == "unconditional")]
        if len(row) and "share_kurt" in row.columns:
            print(f"  [checkpoint] {venue} kurtosis share = "
                  f"{row.iloc[0]['share_kurt']:.3f}")


def phase_4b_cl24_regional():
    """CL24 Table-4 analog: regional decomposition of the CL20 bound."""
    from src.phase4.run_cl24_regional import run_cl24_regional
    run_cl24_regional()
    tab = Path("results") / "phase4" / "tables"
    _check_file(tab / "cl24_regional.csv", "CL24 regional table")
    _check_file(tab / "cl24_regional_wedge.csv", "CL24 regional wedge")
    w = pd.read_csv(tab / "cl24_regional_wedge.csv")
    for _, r in w[w["order"] == "LB"].iterrows():
        print(f"  [checkpoint] LB wedge {r['region']:>6s}: "
              f"{r['wedge']:+.5f} (t = {r['t_stat']:+.2f}){r['stars'] or ''}")


def phase_5_cross_venue():
    from src.phase5.run_phase5 import run_phase5
    run_phase5()
    tab = Path("results") / "phase5" / "tables"
    _check_file(tab / "matched_difference_regressions.csv",
                "Matched-diff regressions (headline)")
    _check_file(tab / "panel_regressions_dk.csv", "Driscoll-Kraay panel")
    diff = pd.read_csv(tab / "matched_difference_regressions.csv")
    wedge_rows = diff[diff["regressor"] == "const (venue wedge)"]
    for _, row in wedge_rows.iterrows():
        print(f"    {row['dep_var']}: beta = {row['coef']:+.5f} "
              f"(t = {row['t_stat']:+.3f}){row.get('stars', '')}")


def phase_5b_frictions():
    from src.phase5.run_friction_regressions import run_friction_regressions
    run_friction_regressions()
    tab = Path("results") / "phase5" / "tables"
    _check_file(tab / "friction_regressions.csv", "Friction-proxy regressions")


def phase_5c_inverse_contract():
    from src.phase5.run_inverse_contract import run_inverse_contract
    run_inverse_contract()
    tab = Path("results") / "phase5" / "tables"
    _check_file(tab / "inverse_contract_wedge.csv", "Inverse-contract wedge")
    w = pd.read_csv(tab / "inverse_contract_wedge.csv")
    if {"measured_wedge", "predicted_wedge"}.issubset(w.columns):
        agree = int(np.sign(w["measured_wedge"])
                    .eq(np.sign(w["predicted_wedge"])).sum())
        print(f"  [checkpoint] sign agreement measured vs predicted: "
              f"{agree}/{len(w)}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

PHASE_ORDER = [
    ("0a", "Data Scraping (Deribit)",             phase_0a_scraping),
    ("0b", "Data Cleaning & Auxiliary",           phase_0b_preprocessing),
    ("1a", "SSVI Surface Fitting",                phase_1a_surfaces),
    ("1b", "RND Extraction (Figlewski tails)",    phase_1b_densities),
    ("1c", "CP Tensor Decomposition",             phase_1c_tensor),
    ("1d", "Conditioning Vectors",                phase_1d_conditioning),
    ("2",  "Physical Density & EP Decomp",        phase_2_ep),
    ("2b", "NB-Sweep Diagnostic (appendix)",      phase_2b_nb_sweep),
    ("2c", "Grid-Sensitivity Diagnostic (appx)",  phase_2c_grid_sensitivity),
    ("3",  "Conditional Pricing Kernel",          phase_3_kernel),
    ("3b", "Kernel Bootstrap (heavy)",            None),   # special args
    ("3c", "Joint Regime Test",                   phase_3c_joint_regime_test),
    ("4",  "BKM / CL20 Decomposition",            phase_4_bkm),
    ("4b", "CL24 Regional Decomposition",         phase_4b_cl24_regional),
    ("5",  "Cross-Venue Analysis",                phase_5_cross_venue),
    ("5b", "Friction-Proxy Regressions",          phase_5b_frictions),
    ("5c", "Inverse-Contract Extension",          phase_5c_inverse_contract),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Bitcoin risk-premia pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_phase", default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip", nargs="+", default=None)
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip Phase 3b (Phase 3c then auto-skips)")
    parser.add_argument("--skip-diagnostics", action="store_true",
                        help="Skip the appendix diagnostics (Phases 2b, 2c)")
    parser.add_argument("--bootstrap-B", type=int, default=200)
    parser.add_argument("--bootstrap-workers", type=int, default=6)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    all_tags = [tag for tag, _, _ in PHASE_ORDER]
    if args.only:
        run_tags = set(args.only)
    elif args.from_phase:
        if args.from_phase not in all_tags:
            print(f"Unknown phase '{args.from_phase}'. Valid: {', '.join(all_tags)}")
            sys.exit(1)
        run_tags = set(all_tags[all_tags.index(args.from_phase):])
    else:
        run_tags = set(all_tags)
    if args.skip:
        run_tags -= set(args.skip)
    if args.skip_bootstrap:
        run_tags.discard("3b")
    if args.skip_diagnostics:
        run_tags -= {"2b", "2c"}

    print("\n" + "#" * 70)
    print("#   Bitcoin Risk Premia — Full Pipeline Run".ljust(69) + "#")
    print(f"#   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(69) + "#")
    print("#" * 70)
    print(f"\n  Phases to run: {', '.join(t for t in all_tags if t in run_tags)}\n")

    manifest = []
    t_total = time.time()
    for tag, name, runner in PHASE_ORDER:
        if tag not in run_tags:
            continue
        try:
            with PhaseTimer(f"Phase {tag}: {name}", manifest):
                if tag == "3b":
                    phase_3b_kernel_bootstrap(B=args.bootstrap_B,
                                              workers=args.bootstrap_workers)
                else:
                    runner()
        except Exception:
            if not args.continue_on_error:
                print(f"\n  Pipeline aborted at Phase {tag}. Use "
                      f"--continue-on-error to proceed past failures, or "
                      f"--from {tag} to resume after fixing the issue.")
                break

    dt_total = time.time() - t_total
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  {'Phase':<45s} {'Status':>8s} {'Time':>10s}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    for row in manifest:
        status_str = "OK" if row["status"] == "OK" else "FAIL"
        print(f"  {row['phase']:<45s} {status_str:>8s} "
              f"{_fmt_time(row['seconds']):>10s}")
        if row["error"]:
            print(f"    -> {row['error'][:80]}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    print(f"  {'Total':<45s} {'':>8s} {_fmt_time(dt_total):>10s}")

    n_ok = sum(1 for r in manifest if r["status"] == "OK")
    n_fail = sum(1 for r in manifest if r["status"] == "FAILED")
    print(f"\n  {n_ok} phases OK, {n_fail} FAILED."
          if n_fail else f"\n  All {n_ok} phases completed successfully.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(manifest).to_csv(LOG_DIR / f"run_{ts}.csv", index=False)
    print(f"  Run manifest saved to {LOG_DIR / f'run_{ts}.csv'}")

    print("\n" + "=" * 70)
    print("  POST-RUN CHECKLIST (manual, before updating the LaTeX)")
    print("=" * 70)
    print("""
  Verify against the NEW CSVs — never carry numbers forward from memory.

   1. CP rank: R = 1 across all seeds (tensor_pca_diagnostics_almeida.csv)
   2. CL20 weights at theta = 2: (1, -1, 1) (cyl tables)
   3. Kurtosis share ~ 30%, stable across kappa (kappa_sensitivity.csv)
   4. EP totals: enhanced ~ vanilla; both bracketed by KDE and the
      raw-moment anchor (ep_bootstrap_ci.csv, ep_decomposition_summary.csv)
   5. NB sweep: enhanced strictly stabler than vanilla
      (nb_sweep_summary.csv); published bounds bind (sigma_binding_check.csv)
   6. Grid sensitivity: how much of the anchor gap closes on [0.30, 2.60]
      (grid_sensitivity.csv)
   7. Kernel: (b,c,d) params, tercile_labels present in npz, mean-one
      normalization diagnostics printed by Phase 3
   8. Joint regime test: Wald + curvature-at-money (joint_regime_test_*.csv)
      — the formal basis for any hump-regime claim
   9. Macro/full convergence status: if the fixed p-hat lets them converge,
      the non-convergence narrative must be rewritten
  10. Venue wedges: variance significant; skew/kurt per new run; no state
      interactions (matched_difference_regressions.csv)
  11. CL24 regional: shares sum to 1 per venue; regional LB wedge pattern
      (cl24_regional.csv, cl24_regional_wedge.csv)
  12. Frictions all insignificant under the 1/5/10 star convention
      (friction_regressions.csv)
  13. Inverse contract: sign agreement count (inverse_contract_wedge.csv)
  14. MFK: tent shape within block-bootstrap bands (fig_mfk_unconditional)
    """)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()