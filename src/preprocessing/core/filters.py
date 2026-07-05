import pandas as pd
import numpy as np


def remove_ivydb_sentinels(df: pd.DataFrame, sentinel: float = -99.99) -> pd.DataFrame:
    """
    Removes rows with IvyDB sentinel values (-99.99) in critical fields.
    Per IvyDB Futures Reference Manual v3.0, -99.99 indicates
    missing/uncalculated values for IV, delta, gamma, vega, theta,
    rho, drho, bid, and offer.
    """
    mask = (
        (df['impliedvolatility'] != sentinel) &
        (df['bid'] != sentinel) &
        (df['offer'] != sentinel) &
        (df['delta'] != sentinel)
    )
    return df[mask].copy()


def filter_maturity(df: pd.DataFrame, min_dte: int = 10, max_dte: int = 365) -> pd.DataFrame:
    """
    Removes options with fewer than min_dte or more than max_dte
    calendar days to expiration.
    
    min_dte >= 10: avoids near-term microstructure frictions
    max_dte <= 365: focuses on liquid maturities
    """
    mask = (df['days_to_expiry'] >= min_dte) & (df['days_to_expiry'] <= max_dte)
    return df[mask].copy()


def filter_static_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes options violating basic static no-arbitrage bounds:
      - Option settlement price must be positive
      - Offer must be >= Bid (after sentinel removal, this catches data errors)
      - Bid must be non-negative
    
    Note: Butterfly (d²C/dK² < 0) and calendar (dw/dτ < 0) violations
    are checked after SSVI fitting in Phase 1, not here.
    """
    mask = (
        (df['settlementprice'] > 0) &
        (df['bid'] >= 0) &
        (df['offer'] >= df['bid'])
    )
    return df[mask].copy()



def filter_liquidity(df: pd.DataFrame, min_oi: int = 1, min_vol: int = 0) -> pd.DataFrame:
    """
    Excludes illiquid contracts based on open interest and volume.
    Uses OR logic: a contract is retained if it has sufficient OI
    *or* non-zero volume (capturing newly traded contracts where
    OI hasn't updated yet).
    
    For Deribit (24/7 market with immediate OI updates), min_vol=0
    and min_oi=1 is appropriate.
    For CME (with delayed OI reporting), the OR condition prevents
    dropping contracts that traded today but show OI=0.
    """
    mask = (df['openinterest'] >= min_oi) | (df['volume'] >= max(min_vol, 1))
    return df[mask].copy()


def filter_out_of_the_money(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restricts to out-of-the-money (OTM) options to neutralize
    early-exercise premia on CME American-style contracts.
    
    Calls where K >= F, Puts where K <= F.
    
    Per proposal: "the dataset is restricted to out-of-the-money
    options (calls where K >= F and puts where K <= F) to neutralize
    early-exercise premia on CME American-style contracts."
    """
    is_otm_call = (df['callput'] == 'C') & (df['strike'] >= df['forward_price'])
    is_otm_put = (df['callput'] == 'P') & (df['strike'] <= df['forward_price'])
    return df[is_otm_call | is_otm_put].copy()


def trim_extreme_moneyness(df: pd.DataFrame, coeff: float = 3.0) -> pd.DataFrame:
    """
    Trims extreme moneyness observations using the maturity-dependent bound:
        |κ| <= coeff * sqrt(τ)
    where κ = log(K/F) and τ = days_to_expiry / 365.25.
    
    This bound widens with maturity to account for volatility expansion,
    retaining core smile dynamics while excluding sparsely observed
    deep-tail quotes. Following the Optiver seminar convention (coeff=3).
    """
    bound = coeff * np.sqrt(df['tau'])
    mask = df['log_moneyness'].abs() <= bound
    return df[mask].copy()


def filter_iv_bounds(df: pd.DataFrame, min_iv: float = 0.01, max_iv: float = 5.0) -> pd.DataFrame:
    """
    Removes options with implausible implied volatilities.
    
    min_iv = 0.01 (1% annualized): below this is numerical noise
    max_iv = 5.0 (500% annualized): Bitcoin can exhibit extreme IV,
    but values above 500% are likely data errors or illiquid quotes.
    """
    mask = (df['impliedvolatility'] >= min_iv) & (df['impliedvolatility'] <= max_iv)
    return df[mask].copy()


def compute_forward_via_put_call_parity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the synthetic forward price via put-call parity at the
    ATM strike for each (date, expiration) group.
    
    F = K_ATM + e^(rτ) * (C_ATM - P_ATM)
    
    For groups where ATM put-call parity cannot be computed
    (missing matched pairs), falls back to the futures settlement price.
    
    This approach follows the Optiver seminar methodology and is more
    precise than using the futures settlement price directly.
    """
    df = df.copy()
    
    # For each (date, expiration), find the strike closest to the futures price
    df['moneyness_abs'] = (df['strike'] - df['futuresettlementprice']).abs()
    
    # Get ATM strike per group
    atm_idx = df.groupby(['date', 'expiration'])['moneyness_abs'].idxmin()
    atm_strikes = df.loc[atm_idx, ['date', 'expiration', 'strike']].rename(
        columns={'strike': 'atm_strike'}
    )
    
    df = df.merge(atm_strikes, on=['date', 'expiration'], how='left')
    
    # Get matched ATM call and put prices at the ATM strike
    atm_options = df[df['strike'] == df['atm_strike']].copy()
    
    calls = atm_options[atm_options['callput'] == 'C'][
        ['date', 'expiration', 'strike', 'settlementprice']
    ].rename(columns={'settlementprice': 'call_price'})
    
    puts = atm_options[atm_options['callput'] == 'P'][
        ['date', 'expiration', 'strike', 'settlementprice']
    ].rename(columns={'settlementprice': 'put_price'})
    
    pcp = calls.merge(puts, on=['date', 'expiration', 'strike'], how='inner')
    
    # F = K + e^(rτ) * (C - P)
    # Need risk_free_rate and tau — merge from main df
    rate_info = df.groupby(['date', 'expiration']).agg(
        risk_free_rate=('risk_free_rate', 'first'),
        tau=('tau', 'first')
    ).reset_index()
    
    pcp = pcp.merge(rate_info, on=['date', 'expiration'], how='left')
    pcp['forward_pcp'] = pcp['strike'] + np.exp(pcp['risk_free_rate'] * pcp['tau']) * (
        pcp['call_price'] - pcp['put_price']
    )
    
    # Merge back to main dataframe
    forward_map = pcp[['date', 'expiration', 'forward_pcp']]
    df = df.merge(forward_map, on=['date', 'expiration'], how='left')
    
    # Use PCP forward where available, fallback to futures settlement
    df['forward_price'] = df['forward_pcp'].fillna(df['futuresettlementprice'])
    
    # Recompute log-moneyness with the improved forward
    df['log_moneyness'] = np.log(df['strike'] / df['forward_price'])
    
    # Clean up temporary columns
    df.drop(columns=['moneyness_abs', 'atm_strike', 'forward_pcp'], inplace=True)
    
    return df