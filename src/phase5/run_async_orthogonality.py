"""
Asynchronicity Orthogonality Check

The two venues' effective observation windows differ by construction: 

Deribit option quotes are volume-weighted aggregates over the full 24h trading day, while CME settlement
prices are anchored to the 14:59:00-15:00:00 Central Time futures settlement window, with option settlements set by CME 
staff from day-informed implied-volatility skews. 

The K/F normalization with PCP-recovered forwards makes each venue's density invariant to pure price-LEVEL moves between 
these windows (prices and the forward move coherently within each venue's own aggregate). The residual channel is
second-order: intraday changes in the IV SHAPE smear into the Deribit day-VWAP but not into the CME settlement anchor.

This script closes that channel empirically with one regression. For each matched day it builds, from RAW Deribit trades 
(timestamps + index prices the pipeline already stores) and estimates, for each cumulant order k in {2, 3, 4},

    dPi_k,t = alpha + b1 * async_move_t + b2 * intraday_rv_t + e_t

with Newey-West(27) errors. Orthogonality holds if b1 is insignificant and alpha reproduces the unconditional wedge.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

NW_LAGS = 27
MAX_GAP_MIN = 90           # max distance to the nearest trade at a query time
RV_RESAMPLE = "15min"

TRADES_COL_CANDIDATES = {
    "timestamp": ["timestamp", "time", "trade_time"],
    "index_price": ["index_price_usd", "index_price", "underlying_price"],
    "amount": ["amount", "size", "quantity", "volume"],
}

def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

# ---------------------------------------------------------------------------
# Pure cores (unit-testable without the repo)
# ---------------------------------------------------------------------------
def cme_settle_anchor_utc(date) -> pd.Timestamp:
    """15:00 America/Chicago on `date`, converted to naive UTC.

    Uses zoneinfo so the CST/CDT switch is exact across the sample
    (21:00 UTC in winter, 20:00 UTC in summer)."""
    local = pd.Timestamp(date).replace(hour=15, minute=0, second=0)
    local = local.tz_localize(ZoneInfo("America/Chicago"))
    return local.tz_convert("UTC").tz_localize(None)


def daily_async_proxies(ts, index_price, amount, anchor_utc,
                        max_gap_min=MAX_GAP_MIN):
    """Proxies for one day of trades.

    ts          : datetime64 array (UTC, naive), any order
    index_price : float array, same length
    amount      : float array (volume weights), same length
    anchor_utc  : pd.Timestamp (naive UTC)

    Returns dict(async_move, intraday_rv, t_vw, S_anchor, S_vw, n_trades)
    with NaNs where a price cannot be located within max_gap_min."""
    order = np.argsort(ts)
    ts = np.asarray(ts)[order]
    px = np.asarray(index_price, dtype=float)[order]
    w = np.asarray(amount, dtype=float)[order]
    t_sec = ts.astype("datetime64[s]").astype(np.int64)

    w_sum = w.sum()
    w_n = w / w_sum if w_sum > 0 else np.full_like(w, 1.0 / len(w))
    t_vw_sec = int(np.round((w_n * t_sec).sum()))

    def price_at(t0_sec):
        i = int(np.searchsorted(t_sec, t0_sec))
        cands = [j for j in (i - 1, i) if 0 <= j < len(t_sec)]
        if not cands:
            return np.nan
        j = min(cands, key=lambda j: abs(t_sec[j] - t0_sec))
        if abs(t_sec[j] - t0_sec) > max_gap_min * 60:
            return np.nan
        return px[j]

    anchor_sec = int(pd.Timestamp(anchor_utc).timestamp())
    S_anchor = price_at(anchor_sec)
    S_vw = price_at(t_vw_sec)
    async_move = (abs(np.log(S_anchor / S_vw))
                  if np.isfinite(S_anchor) and np.isfinite(S_vw)
                  and S_anchor > 0 and S_vw > 0 else np.nan)

    s = pd.Series(px, index=pd.DatetimeIndex(ts))
    r = np.log(s.resample(RV_RESAMPLE).last().dropna()).diff().dropna()
    intraday_rv = float(r.std(ddof=1)) if len(r) > 3 else np.nan

    return {"async_move": async_move, "intraday_rv": intraday_rv,
            "t_vw": pd.Timestamp(t_vw_sec, unit="s"),
            "S_anchor": S_anchor, "S_vw": S_vw, "n_trades": len(ts)}


def orthogonality_table(wedges: pd.DataFrame, proxies: pd.DataFrame,
                        nw_lags=NW_LAGS) -> pd.DataFrame:
    """wedges: DataFrame indexed by date with columns dPi_2, dPi_3, dPi_4.
    proxies: DataFrame indexed by date with async_move, intraday_rv.
    Returns the stacked NW regression table."""
    import statsmodels.api as sm
    df = wedges.join(proxies[["async_move", "intraday_rv"]], how="inner")
    df = df.dropna()
    rows = []
    for dep in [c for c in wedges.columns if c.startswith("dPi_")]:
        X = sm.add_constant(df[["async_move", "intraday_rv"]].values.astype(float))
        res = sm.OLS(df[dep].values.astype(float), X).fit(
            cov_type="HAC", cov_kwds={"maxlags": nw_lags})

        res0 = sm.OLS(df[dep].values.astype(float),
                      np.ones((len(df), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        for name, j in [("const (alpha)", 0), ("async_move", 1),
                        ("intraday_rv", 2)]:
            rows.append({
                "dep_var": dep, "regressor": name,
                "coef": float(res.params[j]),
                "t_stat": float(res.tvalues[j]),
                "p_value": float(res.pvalues[j]),
                "stars": _stars(float(res.pvalues[j])),
                "uncond_wedge_same_sample": float(res0.params[0]),
                "n_days": int(len(df)), "nw_lags": nw_lags,
                "r_squared": float(res.rsquared),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def _pick(df, key):
    for c in TRADES_COL_CANDIDATES[key]:
        if c in df.columns:
            return c
    raise KeyError(f"None of {TRADES_COL_CANDIDATES[key]} found for "
                   f"'{key}'; available: {list(df.columns)}")


def run_async_orthogonality(trades_path=None):
    from src.config import get_path

    TAB = get_path("results_phase5") / "tables"
    TAB.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Asynchronicity Orthogonality Check")
    print("=" * 60)

    # 1) daily wedge from the Phase 4 panel
    panel = pd.read_parquet(get_path("data_phase4") / "cumulant_premia.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    pi_cols = [c for c in panel.columns
               if c.lower().replace("pi_", "Pi_").startswith("Pi_")
               and c[-1] in "234"]
    if not pi_cols:
        pi_cols = [c for c in panel.columns if c.lower() in
                   ("pi2", "pi3", "pi4", "pi_2", "pi_3", "pi_4")]
    assert len(pi_cols) == 3, \
        f"Could not identify Pi_2..4 columns in {list(panel.columns)}"
    pi_cols = sorted(pi_cols, key=lambda c: c[-1])
    piv = {k: panel.pivot_table(index="date", columns="venue", values=c)
           for k, c in zip((2, 3, 4), pi_cols)}
    wedges = pd.DataFrame({
        f"dPi_{k}": (piv[k]["DER"] - piv[k]["CME"]).dropna()
        for k in (2, 3, 4)}).dropna()
    print(f"  Daily wedges: {len(wedges)} matched days "
          f"({wedges.index.min().date()} -> {wedges.index.max().date()})")

    # 2) raw Deribit trades -> daily proxies (single pass, 3 columns only)
    path = (Path(trades_path) if trades_path else
            Path(get_path("raw_deribit_dir")) / "btc_options_trades.parquet")
    # resolve column names from a one-row probe, then read minimal columns
    head = pd.read_parquet(path).head(1)
    cols = {k: _pick(head, k) for k in TRADES_COL_CANDIDATES}
    trades = pd.read_parquet(path, columns=list(cols.values()))
    ts = trades[cols["timestamp"]]
    if np.issubdtype(ts.dtype, np.number):
        trades["_ts"] = pd.to_datetime(ts, unit="ms")
    else:
        trades["_ts"] = pd.to_datetime(ts)
    trades["_d"] = trades["_ts"].dt.normalize()
    trades = trades[trades["_d"].isin(wedges.index)]
    print(f"  Raw trades on matched days: {len(trades):,} "
          f"(columns mapped: {cols})")

    prox_rows = {}
    for d, g in trades.groupby("_d"):
        prox_rows[d] = daily_async_proxies(
            g["_ts"].values, g[cols["index_price"]].values,
            g[cols["amount"]].values, cme_settle_anchor_utc(d))
    proxies = pd.DataFrame(prox_rows).T
    proxies.index.name = "date"
    cov = proxies["async_move"].notna().mean()
    print(f"  Proxies built for {len(proxies)} days; "
          f"async_move coverage {cov:.1%}; "
          f"mean |move| = {proxies['async_move'].mean():.4%}")
    proxies.to_parquet(TAB / "async_proxies_daily.parquet")

    # 3) regressions
    table = orthogonality_table(wedges, proxies)
    table.to_csv(TAB / "async_orthogonality.csv", index=False)
    print(f"\n  Saved: {TAB / 'async_orthogonality.csv'}")
    print("\n  Orthogonality regressions (NW-27):")
    for dep in ["dPi_2", "dPi_3", "dPi_4"]:
        sub = table[table.dep_var == dep]
        a = sub[sub.regressor == "const (alpha)"].iloc[0]
        b = sub[sub.regressor == "async_move"].iloc[0]
        print(f"    {dep}: alpha = {a['coef']:+.5f} (t={a['t_stat']:+.2f}) "
              f"[uncond wedge {a['uncond_wedge_same_sample']:+.5f}]; "
              f"beta_async = {b['coef']:+.4f} "
              f"(t={b['t_stat']:+.2f}){b['stars']}  R2={b['r_squared']:.3f}")
    return table


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-path", default=None)
    run_async_orthogonality(ap.parse_args().trades_path)