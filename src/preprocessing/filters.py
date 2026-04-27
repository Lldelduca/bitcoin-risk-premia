import pandas as pd
import numpy as np

def apply_liquidity_filters(df, min_oi=1, min_vol=0):
    """
    Excludes severe illiquidity based strictly on volume and open interest,
    mirroring Almeida et al. (2026).
    """
    mask = (
        (df['openinterest'] >= min_oi) &
        (df['volume'] >= min_vol)
    )
    return df[mask].copy()

def filter_static_arbitrage(df):
    """
    Removes options violating basic static no-arbitrage bounds.
    - Negative time value (for OTM options, price > 0)
    - Bid > Offer
    """
    mask = (df['settlementprice'] > 0) & (df['offer'] >= df['bid'])
    return df[mask].copy()

def filter_out_of_the_money(df):
    """
    Restricts to out-of-the-money (OTM) options to neutralize early-exercise premia.
    Calls where K >= F, Puts where K <= F.
    Assumes 'forward_price' and 'strike' columns exist.
    """
    is_otm_call = (df['callput'] == 'C') & (df['strike'] >= df['forward_price'])
    is_otm_put = (df['callput'] == 'P') & (df['strike'] <= df['forward_price'])
    
    return df[is_otm_call | is_otm_put].copy()

def trim_extreme_moneyness(df, multiplier=3.0):
    """
    Trims extreme observations using a maturity-dependent bound: |k| <= 3 * sqrt(tau).
    k = ln(K/F)
    """
    # Bound increases with maturity to account for volatility expansion
    bound = multiplier * np.sqrt(df['tau'])
    mask = df['log_moneyness'].abs() <= bound
    return df[mask].copy()

def filter_maturity(df, min_tau_days=7, max_tau_days=180):
    """
    Filters extremely short and extremely long maturities.
    """
    mask = (df['days_to_expiry'] >= min_tau_days) & (df['days_to_expiry'] <= max_tau_days)
    return df[mask].copy()