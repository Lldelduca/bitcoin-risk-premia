"""
No-Trade Band Test

The friction regressions show the cross-venue wedge does not COVARY with basis/funding proxies. 
This script tests the complementary LEVEL hypothesis: is the wedge smaller than the round-trip cost of the
cheapest strategy that harvests it? If the daily variance wedge dPi_2 sits inside the transaction-cost band, 
segmentation persists because it is unprofitable to close (the Gromb-Vayanos limits-to-arbitrage
conclusion) turning two negative results into one positive mechanism.

Strategy costed: the BKM V-contract replicating strip on both venues (long the cheap venue, short the expensive one). 
Daily cost per venue:

    C_t^j = kappa_cost * SUM_i |w(K_i)| * half_spread_i ,
    w(K) = 2 (1 - ln(K/F)) / K^2   (the V-contract weights),

discretized over that day's available strikes at tau ~ 27d, with kappa_cost = 1 for entry-only and 2 for a full round trip. 
The band is B_t = C_t^CME + C_t^DER. Report: fraction of days |wedge_t| <= B_t, the mean band vs the mean wedge, and 
sensitivity over spread assumptions.

"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DEFAULT_HALF_SPREAD_REL = {"CME": 0.010, "DER": 0.005} 
SPREAD_MULTS = [0.5, 1.0, 2.0]
KAPPAS = [1.0, 2.0]          
TAU_TARGET = 27

# ---------------------------------------------------------------------------
# Pure cores (unit-testable)
# ---------------------------------------------------------------------------

def v_contract_weights(K, F):
    """|w(K)| * dK of the BKM variance contract, aligned to the INPUT
    order of K. Sorting and spacing are handled internally."""
    K = np.asarray(K, dtype=float)
    order = np.argsort(K)
    K_s = K[order]
    w_s = np.abs(2.0 * (1.0 - np.log(K_s / F)) / K_s ** 2)
    dK_s = np.gradient(K_s)
    out = np.empty_like(K_s)
    out[order] = w_s * dK_s          # map back to input positions
    return out


def strip_cost(K, F, half_spreads_abs, kappa=1.0):
    """Cost of trading the V-contract strip once (kappa=1) or round trip
    (kappa=2): kappa * sum_i |w_i| dK_i * half_spread_i, with
    half_spread_i in PRICE (USD) units. Dimensionless, comparable to
    Pi_2 (both are option prices times 1/K^2-type weights times dK)."""
    wts = v_contract_weights(K, F)
    return float(kappa * np.sum(wts * np.asarray(half_spreads_abs,
                                                 dtype=float)))


def roll_half_spread(prices):
    """Roll (1984) effective half-spread from a trade-price series."""
    p = np.asarray(prices, dtype=float)
    if len(p) < 4:
        return np.nan
    dp = np.diff(p)
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    return float(np.sqrt(-cov)) if cov < 0 else np.nan


def band_table(wedge, band):
    """Summary stats for |wedge| vs band on aligned daily series."""
    wedge = np.asarray(wedge, dtype=float)
    band = np.asarray(band, dtype=float)
    ok = np.isfinite(wedge) & np.isfinite(band)
    w, b = np.abs(wedge[ok]), band[ok]
    return {
        "n_days": int(ok.sum()),
        "mean_abs_wedge": float(w.mean()),
        "mean_band": float(b.mean()),
        "median_band": float(np.median(b)),
        "frac_inside_band": float((w <= b).mean()),
        "frac_wedge_exceeds_2x_band": float((w > 2 * b).mean()),
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _daily_strip_costs(df, venue, kappa, rel_spread=None, quote_mult=1.0):
    """Series(date -> strip cost). Uses per-row ABSOLUTE half-spreads
    from quoted bid/offer when available (CME), else rel_spread times the
    settlement price (Deribit, calibrated). One expiration per day: the
    one closest to TAU_TARGET, so strike spacing is never mixed across
    maturities."""
    price_col = next((c for c in ["settlementprice", "mid_price",
                                  "price_usd", "mark_price_usd", "price"]
                      if c in df.columns), None)
    if price_col is None:
        raise KeyError(f"[{venue}] no option-price column found; "
                       f"columns: {list(df.columns)}")
    ask_col = next((c for c in ["ask", "offer"] if c in df.columns), None)
    has_quotes = ("bid" in df.columns) and (ask_col is not None)

    tau_col = next(c for c in ["days_to_expiry", "tau_days", "dte"]
                   if c in df.columns)
    fwd_col = next((c for c in ["forward_price", "forward"]
                    if c in df.columns), None)

    out = {}
    for date, g_day in df.groupby("date"):
        # single maturity: closest to target
        g_day = g_day.copy()
        g_day["_gap"] = (g_day[tau_col] - TAU_TARGET).abs()
        exp_pick = g_day.loc[g_day["_gap"].idxmin(), "expiration"]
        g = g_day[g_day["expiration"] == exp_pick]
        g = g.drop_duplicates(subset="strike")
        if g["strike"].nunique() < 5:
            continue
        F = float(g[fwd_col].median()) if fwd_col else float(g["strike"].median())
        if has_quotes:
            hs = quote_mult * 0.5 * (g[ask_col] - g["bid"]).clip(lower=0).values
        else:
            hs = rel_spread * g[price_col].values
        out[date] = strip_cost(g["strike"].values, F, hs, kappa)
    return pd.Series(out).sort_index(), has_quotes


def run_notrade_band():
    from src.config import get_path

    # Corrected: Replaced hardcoded results folders with centralized config getters
    TAB = get_path("results_phase5") / "tables"
    TAB.mkdir(parents=True, exist_ok=True)

    # wedge series from the Phase 4 daily panel
    panel = pd.read_parquet(get_path("data_phase4")
                            / "cumulant_premia_daily.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    piv = panel.pivot_table(index="date", columns="venue",
                            values="Pi_2").dropna()
    wedge = piv["DER"] - piv["CME"]
    print(f"  Daily variance wedge: {len(wedge)} matched days, "
          f"mean {wedge.mean():+.5f}")

    # cleaned options near tau = 27 per venue
    opts = {}
    for venue, key in [("CME", "cleaned_cme"), ("DER", "cleaned_deribit")]:
        df = pd.read_parquet(get_path(key))
        df["date"] = pd.to_datetime(df["date"])
        tau_col = next(c for c in ["tau_days", "days_to_expiry", "dte"]
                       if c in df.columns)
        df = df[(df[tau_col] >= TAU_TARGET - 7)
                & (df[tau_col] <= TAU_TARGET + 7)]
        has_quotes = ("bid" in df.columns) and ("ask" in df.columns)
        opts[venue] = (df, has_quotes)
        print(f"  [{venue}] {len(df):,} options near tau=27; "
              f"quoted spreads available: {has_quotes}")

    rows, daily = [], {}
    for mult in SPREAD_MULTS:
        for kappa in KAPPAS:
            bands = {}
            for venue, (df, _) in opts.items():
                costs, has_quotes = _daily_strip_costs(
                    df, venue, kappa,
                    rel_spread=mult * DEFAULT_HALF_SPREAD_REL[venue],
                    quote_mult=mult)
                bands[venue] = costs
                if mult == 1.0 and kappa == 2.0:
                    src = "quoted (per-row bid/offer)" if has_quotes \
                        else (f"calibrated rel="
                              f"{DEFAULT_HALF_SPREAD_REL[venue]:.4f} "
                              f"(ASSUMPTION)")
                    print(f"    [{venue}] half-spread source: {src}")
            band = (bands["CME"].reindex(wedge.index)
                    + bands["DER"].reindex(wedge.index))
            stats = band_table(wedge.values, band.values)
            stats.update({"spread_mult": mult, "kappa_roundtrip": kappa})
            rows.append(stats)
            if mult == 1.0:
                daily[f"band_k{int(kappa)}"] = band

    summary = pd.DataFrame(rows)
    summary.to_csv(TAB / "notrade_band_summary.csv", index=False)
    dd = pd.DataFrame({"wedge": wedge, **daily})
    dd["inside_band_k2"] = dd["wedge"].abs() <= dd.get("band_k2")
    dd.to_parquet(TAB / "notrade_band_daily.parquet")

    print("\n  |wedge| vs round-trip band (kappa = 2, spread x1):")
    r = summary[(summary.spread_mult == 1.0)
                & (summary.kappa_roundtrip == 2.0)].iloc[0]
    print(f"    mean |wedge| = {r['mean_abs_wedge']:.5f}  "
          f"mean band = {r['mean_band']:.5f}  "
          f"inside band on {r['frac_inside_band']:.1%} of days")
    print(f"  Saved: {TAB / 'notrade_band_summary.csv'}")
    return summary


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run_notrade_band()