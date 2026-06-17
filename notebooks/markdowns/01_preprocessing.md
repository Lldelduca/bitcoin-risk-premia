```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

PROJECT_ROOT = Path.cwd().parent
CLEAN = PROJECT_ROOT / 'data' / 'cleaned'
FIG_DIR = PROJECT_ROOT / 'results' / 'data' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

cme = pd.read_parquet(CLEAN / 'cme_options_clean.parquet')
der = pd.read_parquet(CLEAN / 'deribit_options_clean.parquet')
aux = pd.read_parquet(CLEAN / 'auxiliary_panel.parquet')

# Funding differential may not exist yet if HL scrape is still running
funding_path = CLEAN / 'funding_diff.parquet'
funding = pd.read_parquet(funding_path) if funding_path.exists() else None

print(f"CME:       {len(cme):>10,} rows, {cme['date'].nunique()} days")
print(f"Deribit:   {len(der):>10,} rows, {der['date'].nunique()} days")
print(f"Auxiliary: {len(aux):>10,} rows, {aux['date'].nunique()} days")
if funding is not None:
    print(f"Funding:   {len(funding):>10,} rows, {funding['date'].nunique()} days")
```

    CME:           65,598 rows, 802 days
    Deribit:      327,202 rows, 3105 days
    Auxiliary:      2,191 rows, 2191 days
    Funding:          964 rows, 964 days
    

#### 1. Cross-Venue Comparison: Daily Option Counts


```python
daily_cme = cme.groupby('date').size().rename('CME')
daily_der = der.groupby('date').size().rename('Deribit')

fig, ax = plt.subplots(figsize=(13, 5))
daily_der.plot(ax=ax, label='Deribit', alpha=0.7)
daily_cme.plot(ax=ax, label='CME', alpha=0.7)
ax.axvline(pd.Timestamp('2024-01-10'), color='red', linestyle='--',
           label='Spot ETF approval (Jan 2024)')
ax.set_ylabel('Cleaned options per day')
ax.set_title('Daily option cross-section size by venue')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_daily_option_counts.png', dpi=150)
plt.show()

print("Median per day:")
print(f"  CME:     {daily_cme.median():>6.0f}")
print(f"  Deribit: {daily_der.median():>6.0f}")
```


    
![png](01_preprocessing_files/01_preprocessing_2_0.png)
    


    Median per day:
      CME:         82
      Deribit:    100
    

#### 2. ATM Implied Volatility Time Series (Thesis Figure 1)


```python
def atm_iv_series(df, venue_label):
    """
    Extract daily ATM IV: for each (date, expiration), pick the strike
    closest to the forward, then average across expirations.
    """
    df = df.copy()
    df['abs_kappa'] = df['log_moneyness'].abs()
    # Closest-to-ATM option per (date, expiration)
    idx = df.groupby(['date', 'expiration'])['abs_kappa'].idxmin()
    atm = df.loc[idx]
    # Average across expirations per day
    return atm.groupby('date')['impliedvolatility'].mean()

atm_cme = atm_iv_series(cme, 'CME')
atm_der = atm_iv_series(der, 'Deribit')

fig, ax = plt.subplots(figsize=(13, 5))
(atm_der * 100).plot(ax=ax, label='Deribit', alpha=0.85)
(atm_cme * 100).plot(ax=ax, label='CME', alpha=0.85)
ax.axvline(pd.Timestamp('2024-01-10'), color='red', linestyle='--',
           label='Spot ETF approval')
ax.axvline(pd.Timestamp('2022-11-08'), color='gray', linestyle=':',
           label='FTX collapse')
ax.set_ylabel('ATM Implied Volatility (%)')
ax.set_title('Daily ATM IV across venues')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_atm_iv_timeseries.png', dpi=150)
plt.show()
```


    
![png](01_preprocessing_files/01_preprocessing_4_0.png)
    


#### 3. Slice Density Diagnostics (Critical for SSVI)


```python
def slice_diagnostics(df, name):
    per_slice = df.groupby(['date', 'expiration']).size().reset_index(name='n')
    print(f"\n=== {name} slice density ===")
    print(per_slice['n'].describe())
    print(f"Slices with < 4 options: {(per_slice['n'] < 4).sum():,} "
          f"({(per_slice['n'] < 4).mean():.1%})")
    print(f"Slices with < 3 options: {(per_slice['n'] < 3).sum():,} "
          f"({(per_slice['n'] < 3).mean():.1%})")
    return per_slice

slices_cme = slice_diagnostics(cme, 'CME')
slices_der = slice_diagnostics(der, 'Deribit')
```

    
    === CME slice density ===
    count    2804.000000
    mean       23.394437
    std        17.850788
    min         1.000000
    25%         9.000000
    50%        21.000000
    75%        34.000000
    max       148.000000
    Name: n, dtype: float64
    Slices with < 4 options: 353 (12.6%)
    Slices with < 3 options: 283 (10.1%)
    
    === Deribit slice density ===
    count    13490.000000
    mean        24.255152
    std         11.012241
    min          1.000000
    25%         16.000000
    50%         22.000000
    75%         30.000000
    max         70.000000
    Name: n, dtype: float64
    Slices with < 4 options: 44 (0.3%)
    Slices with < 3 options: 26 (0.2%)
    


```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(slices_cme['n'].clip(upper=50), bins=40, color='C0', edgecolor='black', linewidth=0.3)
axes[0].axvline(4, color='red', linestyle='--', label='SSVI min (4)')
axes[0].set_title('CME: options per (date, expiration) slice')
axes[0].set_xlabel('Options per slice')
axes[0].legend()

axes[1].hist(slices_der['n'].clip(upper=50), bins=40, color='C1', edgecolor='black', linewidth=0.3)
axes[1].axvline(4, color='red', linestyle='--', label='SSVI min (4)')
axes[1].set_title('Deribit: options per (date, expiration) slice')
axes[1].set_xlabel('Options per slice')
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_slice_density.png', dpi=150)
plt.show()
```


    
![png](01_preprocessing_files/01_preprocessing_7_0.png)
    


#### 4. Moneyness Coverage by Venue


```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(cme['log_moneyness'], bins=60, color='C0', edgecolor='black', linewidth=0.3)
axes[0].set_title('CME: log-moneyness distribution')
axes[0].set_xlabel('κ = log(K/F)')

axes[1].hist(der['log_moneyness'], bins=60, color='C1', edgecolor='black', linewidth=0.3)
axes[1].set_title('Deribit: log-moneyness distribution')
axes[1].set_xlabel('κ = log(K/F)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_moneyness_coverage.png', dpi=150)
plt.show()
```


    
![png](01_preprocessing_files/01_preprocessing_9_0.png)
    


#### 5. Auxiliary Panel: Conditioning Variables


```python
print("Auxiliary panel missing-value summary:")
print(aux.isna().sum())
print(f"\nDVOL first valid date: {aux.dropna(subset=['dvol'])['date'].min().date()}")
print(f"DVOL last valid date:  {aux.dropna(subset=['dvol'])['date'].max().date()}")
```

    Auxiliary panel missing-value summary:
    date            0
    btc_spot        0
    rv              0
    vix             0
    dvol          461
    baa_spread      0
    dgs2            0
    dxy             0
    fng             1
    dtype: int64
    
    DVOL first valid date: 2021-04-06
    DVOL last valid date:  2025-12-30
    


```python
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
aux.set_index('date')['vix'].plot(ax=axes[0], color='C0')
axes[0].set_ylabel('VIX')
axes[0].set_title('Conditioning variables (Z_t^{(1)})')

(aux.set_index('date')['rv'] * 100).plot(ax=axes[1], color='C1')
axes[1].set_ylabel('30-day RV (annualized %)')

aux.set_index('date')['dvol'].plot(ax=axes[2], color='C2')
axes[2].set_ylabel('DVOL')
axes[2].set_xlabel('Date')
axes[2].axvline(pd.Timestamp('2021-03-29'), color='red', linestyle='--', alpha=0.5,
                label='DVOL inception')
axes[2].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_conditioning_variables.png', dpi=150)
plt.show()
```


    
![png](01_preprocessing_files/01_preprocessing_12_0.png)
    


#### 6. Funding Rate Differential (HL - DER, Annualized)


```python
if funding is not None:
    fig, ax = plt.subplots(figsize=(13, 4))
    (funding.set_index('date')['funding_hl_annual'] * 100).plot(
        ax=ax, label='Hyperliquid (annualized %)', alpha=0.8)
    (funding.set_index('date')['funding_der_annual'] * 100).plot(
        ax=ax, label='Deribit (annualized %)', alpha=0.8)
    ax.set_ylabel('Funding rate (annualized %)')
    ax.set_title('Perpetual funding rates by venue')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_funding_rates.png', dpi=150)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(13, 4))
    (funding.set_index('date')['delta_f_hl_der'] * 100).plot(ax=ax, color='C3')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Δf (HL - DER, annualized %)')
    ax.set_title('Decentralized vs. centralized funding rate differential')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_funding_differential.png', dpi=150)
    plt.show()
    
    print("\nFunding rate statistics (annualized):")
    print(funding[['funding_hl_annual', 'funding_der_annual', 'delta_f_hl_der']].describe())
else:
    print('Funding panel not yet available — run clean_hyperliquid.py first.')
```


    
![png](01_preprocessing_files/01_preprocessing_14_0.png)
    



    
![png](01_preprocessing_files/01_preprocessing_14_1.png)
    


    
    Funding rate statistics (annualized):
           funding_hl_annual  funding_der_annual  delta_f_hl_der
    count         964.000000          964.000000      964.000000
    mean            0.147445            0.083242        0.064203
    std             0.348048            0.114546        0.319185
    min            -6.354915           -0.278013       -6.353183
    25%             0.084278            0.006842        0.008307
    50%             0.109500            0.044849        0.070942
    75%             0.208380            0.124771        0.126461
    max             2.237398            0.717332        1.640086
    

#### 7. Thesis Data Summary Table


```python
summary = pd.DataFrame({
    'CME': [
        len(cme),
        cme['date'].nunique(),
        cme['date'].min().date(),
        cme['date'].max().date(),
        round(daily_cme.median()),
        round(slices_cme['n'].median()),
        (cme['callput']=='C').sum(),
        (cme['callput']=='P').sum(),
    ],
    'Deribit': [
        len(der),
        der['date'].nunique(),
        der['date'].min().date(),
        der['date'].max().date(),
        round(daily_der.median()),
        round(slices_der['n'].median()),
        (der['callput']=='C').sum(),
        (der['callput']=='P').sum(),
    ],
}, index=['Cleaned rows', 'Trading days', 'Sample start', 'Sample end',
         'Median options/day', 'Median options/slice', 'Calls', 'Puts'])
print(summary)
summary.to_csv(PROJECT_ROOT / 'results' / 'data' / 'tables' / 'data_summary.csv')
```

                                 CME     Deribit
    Cleaned rows               65598      327202
    Trading days                 802        3105
    Sample start          2020-01-13  2017-07-01
    Sample end            2023-08-31  2025-12-30
    Median options/day            82         100
    Median options/slice          21          22
    Calls                      34414      180908
    Puts                       31184      146294
    

#### 8. Grid parameters optimization


```python
project_root = Path.cwd().parent
for venue, fname in [('CME', 'cme_options_clean.parquet'), ('Deribit', 'deribit_options_clean.parquet')]:
    df = pd.read_parquet(project_root / 'data' / 'cleaned' / fname)
    
    # Trim Deribit to CME window
    df = df[(df['date'] >= '2020-01-13') & (df['date'] <= '2023-08-31')]
    
    print(f"\n{'='*60}")
    print(f"  {venue} Summary Statistics")
    print(f"{'='*60}")
    
    # DTE distribution
    print(f"\n  Days to expiry:")
    print(f"  {df['days_to_expiry'].describe().to_string()}")
    
    # Unique DTE values (what maturities actually exist?)
    dte_counts = df.groupby('days_to_expiry').size()
    print(f"\n  Most common DTEs (top 15):")
    print(f"  {dte_counts.nlargest(15).to_string()}")
    
    # Log-moneyness distribution
    print(f"\n  Log-moneyness (κ):")
    print(f"  {df['log_moneyness'].describe().to_string()}")
    
    # κ percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    p{p}: {df['log_moneyness'].quantile(p/100):.4f}")
    
    # τ (in years) distribution
    print(f"\n  τ (years):")
    print(f"  {df['tau'].describe().to_string()}")
    
    # How many expirations per day?
    exp_per_day = df.groupby('date')['expiration'].nunique()
    print(f"\n  Expirations per day: median={exp_per_day.median():.0f}, "
          f"min={exp_per_day.min()}, max={exp_per_day.max()}")
    
    # IV distribution
    print(f"\n  Implied Volatility:")
    print(f"  {df['impliedvolatility'].describe().to_string()}")
```

    
    ============================================================
      CME Summary Statistics
    ============================================================
    
      Days to expiry:
      count    65598.000000
    mean        50.016052
    std         34.181982
    min         10.000000
    25%         25.000000
    50%         41.000000
    75%         67.000000
    max        180.000000
    
      Most common DTEs (top 15):
      days_to_expiry
    19    1709
    18    1708
    21    1650
    25    1614
    27    1551
    26    1532
    12    1450
    20    1424
    10    1423
    11    1415
    28    1391
    13    1371
    24    1336
    32    1332
    34    1330
    
      Log-moneyness (κ):
      count    65598.000000
    mean         0.037603
    std          0.379596
    min         -1.774611
    25%         -0.214765
    50%          0.029816
    75%          0.266741
    max          1.762756
        p1: -0.8098
        p5: -0.5578
        p25: -0.2148
        p50: 0.0298
        p75: 0.2667
        p95: 0.6704
        p99: 1.0711
    
      τ (years):
      count    65598.000000
    mean         0.136936
    std          0.093585
    min          0.027379
    25%          0.068446
    50%          0.112252
    75%          0.183436
    max          0.492813
    
      Expirations per day: median=3, min=1, max=7
    
      Implied Volatility:
      count    65598.000000
    mean         0.753809
    std          0.210055
    min          0.255184
    25%          0.603057
    50%          0.729975
    75%          0.873750
    max          1.956072
    
    ============================================================
      Deribit Summary Statistics
    ============================================================
    
      Days to expiry:
      count    135276.000000
    mean         56.815858
    std          46.589947
    min          10.000000
    25%          18.000000
    50%          42.000000
    75%          83.000000
    max         180.000000
    
      Most common DTEs (top 15):
      days_to_expiry
    14    4442
    11    4308
    10    4289
    15    4159
    13    4156
    12    4070
    21    3702
    16    3688
    17    3636
    18    3630
    19    3389
    20    3387
    22    3358
    35     957
    31     950
    
      Log-moneyness (κ):
      count    135276.000000
    mean          0.053830
    std           0.442408
    min          -2.023430
    25%          -0.193888
    50%           0.046415
    75%           0.291989
    max           2.068223
        p1: -1.1003
        p5: -0.6696
        p25: -0.1939
        p50: 0.0464
        p75: 0.2920
        p95: 0.8108
        p99: 1.2463
    
      τ (years):
      count    135276.000000
    mean          0.155553
    std           0.127556
    min           0.027379
    25%           0.049281
    50%           0.114990
    75%           0.227242
    max           0.492813
    
      Expirations per day: median=5, min=2, max=6
    
      Implied Volatility:
      count    135276.000000
    mean          0.789986
    std           0.233552
    min           0.246100
    25%           0.624000
    50%           0.758287
    75%           0.929310
    max           2.820061
    
