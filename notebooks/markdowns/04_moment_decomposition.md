```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
```


```python
from src.config import get_path, get_sample_window

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

DATA_P4 = get_path("data_phase4")
RES_P4 = get_path("results_phase4")
FIG_DIR = RES_P4 / "figures"
TAB_DIR = RES_P4 / "tables"

# Load precomputed outputs
bkm = pd.read_parquet(DATA_P4 / "bkm_moments.parquet")
premia = pd.read_parquet(DATA_P4 / "cumulant_premia.parquet")
decomp_matched = pd.read_csv(TAB_DIR / "cyl_decomposition_matched.csv")  # headline
decomp_all = pd.read_csv(TAB_DIR / "cyl_decomposition.csv")              # supplementary
theta_rob = pd.read_csv(TAB_DIR / "theta_robustness.csv")
moment_summary = pd.read_csv(TAB_DIR / "moment_summary.csv", header=[0, 1], index_col=0)
bkm["date"] = pd.to_datetime(bkm["date"])
premia["date"] = pd.to_datetime(premia["date"])

# Matched-day masks (headline convention: intersected sample)
cme_days = set(premia.loc[premia["venue"] == "CME", "date"])
der_days = set(premia.loc[premia["venue"] == "DER", "date"])
matched_days = cme_days & der_days
premia_matched = premia[premia["date"].isin(matched_days)].copy()

print(f"BKM moments: {len(bkm)} day-venue pairs")
print(f"  CME: {(bkm['venue']=='CME').sum()} days")
print(f"  DER: {(bkm['venue']=='DER').sum()} days")
print(f"Cumulant premia: {len(premia)} rows")
print(f"  Date range: {premia['date'].min().date()} -> {premia['date'].max().date()}")
print(f"Matched CME-Deribit days: {len(matched_days)}")

def show_table(path, cols=None, title=None, rounding=4, sort=None, produced_by=None):
    """Display a results CSV defensively."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        msg = f"[not found] {path}"
        if produced_by:
            msg += f"\n              run {produced_by} to produce it"
        print(msg)
        return None
    df = pd.read_csv(path)
    if title:
        print(f"--- {title} ---")
    if sort:
        keep = [c for c in sort if c in df.columns]
        if keep:
            df = df.sort_values(keep)
    if cols:
        avail = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[note] columns absent from file: {missing}")
        if avail:
            df_show = df[avail]
        else:
            print("[note] none of the requested columns present — showing all")
            df_show = df
    else:
        df_show = df
    print(df_show.round(rounding).to_string(index=False))
    print(f"\n[{len(df)} rows]  source: {path.name}")
    return df
```

    BKM moments: 1946 day-venue pairs
      CME: 629 days
      DER: 1317 days
    Cumulant premia: 1946 rows
      Date range: 2020-01-13 -> 2023-08-31
    Matched CME-Deribit days: 624
    

#### 1. Risk-Neutral Moment Summary


```python
display(moment_summary)

# Matched-day cross-venue correlations
cme = bkm[bkm["venue"] == "CME"].set_index("date")
der = bkm[bkm["venue"] == "DER"].set_index("date")
matched = cme.join(der, lsuffix="_cme", rsuffix="_der", how="inner")
print(f"\nMatched CME-DER days: {len(matched)}")
for col in ["var_Q", "skew_Q", "kurt_Q"]:
    rho = matched[[f"{col}_cme", f"{col}_der"]].corr().iloc[0, 1]
    print(f"  Cross-venue correlation ({col}): rho = {rho:.3f}")
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">var_Q</th>
      <th colspan="2" halign="left">skew_Q</th>
      <th colspan="2" halign="left">kurt_Q</th>
      <th colspan="2" halign="left">Pi_2</th>
      <th colspan="2" halign="left">Pi_3</th>
      <th colspan="2" halign="left">Pi_4</th>
      <th colspan="2" halign="left">vrp</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>venue</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CME</th>
      <td>0.0496</td>
      <td>0.0293</td>
      <td>-1.0032</td>
      <td>0.5131</td>
      <td>9.3063</td>
      <td>2.6664</td>
      <td>0.0504</td>
      <td>0.0303</td>
      <td>0.0174</td>
      <td>0.0200</td>
      <td>0.0291</td>
      <td>0.0366</td>
      <td>0.0119</td>
      <td>0.0269</td>
    </tr>
    <tr>
      <th>DER</th>
      <td>0.0531</td>
      <td>0.0325</td>
      <td>-0.9897</td>
      <td>0.5162</td>
      <td>9.0730</td>
      <td>2.5307</td>
      <td>0.0540</td>
      <td>0.0338</td>
      <td>0.0194</td>
      <td>0.0268</td>
      <td>0.0324</td>
      <td>0.0485</td>
      <td>0.0153</td>
      <td>0.0289</td>
    </tr>
  </tbody>
</table>
</div>


    
    Matched CME-DER days: 624
      Cross-venue correlation (var_Q): rho = 0.964
      Cross-venue correlation (skew_Q): rho = 0.853
      Cross-venue correlation (kurt_Q): rho = 0.792
    

#### 2. Risk-Neutral Moment Time Series


```python
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for venue, color in [("CME", "C0"), ("DER", "C1")]:
    v = bkm[bkm["venue"] == venue].set_index("date").sort_index()
    axes[0].plot(v.index, v["var_Q"], color=color, lw=0.8, alpha=0.8, label=venue)
    axes[1].plot(v.index, v["skew_Q"], color=color, lw=0.8, alpha=0.8, label=venue)
    axes[2].plot(v.index, v["kurt_Q"], color=color, lw=0.8, alpha=0.8, label=venue)
axes[0].set_ylabel(r"$V_t^{\mathbb{Q},j}$")
axes[0].set_title("Risk-Neutral Moments at 27-day Horizon (BKM Extraction)")
axes[0].legend()
axes[1].set_ylabel(r"Skew$_t^{\mathbb{Q},j}$")
axes[1].axhline(0, color="black", lw=0.4)
axes[2].set_ylabel(r"Kurt$_t^{\mathbb{Q},j}$")
axes[2].axhline(3, color="black", lw=0.4, ls="--", label="Gaussian ref.")
axes[2].legend(loc="upper center")
axes[2].set_xlabel("Date")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_rnd_moments_ts", dpi=300)
plt.show()
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_5_0.png)
    


#### 3. CL20 Lower-Bound Decomposition (Unconditional)


```python
print("=== HEADLINE: matched-day sample (thesis numbers) ===")
display(decomp_matched.round(4))
uncond = decomp_matched[decomp_matched["regime"] == "unconditional"]
for _, row in uncond.iterrows():
    print(f"  {row['venue']}: var {row['share_var']:.1%}, "
          f"skew {row['share_skew']:.1%}, kurt {row['share_kurt']:.1%} "
          f"| total 27d = {row['lb_total']:.4f} "
          f"(ann. {100*row['lb_total']*365/27:.0f}%)")

print("\n=== Supplementary: all available days per venue ===")
display(decomp_all[decomp_all["regime"] == "unconditional"].round(4))
```

    === HEADLINE: matched-day sample (thesis numbers) ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>venue</th>
      <th>regime</th>
      <th>n_days</th>
      <th>Pi_2</th>
      <th>Pi_3</th>
      <th>Pi_4</th>
      <th>lb_total</th>
      <th>share_var</th>
      <th>share_skew</th>
      <th>share_kurt</th>
      <th>mean_vrp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CME</td>
      <td>unconditional</td>
      <td>624</td>
      <td>0.0504</td>
      <td>0.0174</td>
      <td>0.0291</td>
      <td>0.0969</td>
      <td>0.5200</td>
      <td>0.1798</td>
      <td>0.3002</td>
      <td>0.0119</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DER</td>
      <td>unconditional</td>
      <td>624</td>
      <td>0.0540</td>
      <td>0.0194</td>
      <td>0.0324</td>
      <td>0.1057</td>
      <td>0.5106</td>
      <td>0.1834</td>
      <td>0.3060</td>
      <td>0.0153</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CME</td>
      <td>low</td>
      <td>224</td>
      <td>0.0257</td>
      <td>0.0053</td>
      <td>0.0074</td>
      <td>0.0384</td>
      <td>0.6689</td>
      <td>0.1388</td>
      <td>0.1923</td>
      <td>-0.0033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CME</td>
      <td>mid</td>
      <td>193</td>
      <td>0.0456</td>
      <td>0.0144</td>
      <td>0.0210</td>
      <td>0.0810</td>
      <td>0.5635</td>
      <td>0.1776</td>
      <td>0.2589</td>
      <td>0.0073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CME</td>
      <td>high</td>
      <td>207</td>
      <td>0.0818</td>
      <td>0.0335</td>
      <td>0.0604</td>
      <td>0.1757</td>
      <td>0.4656</td>
      <td>0.1906</td>
      <td>0.3438</td>
      <td>0.0319</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DER</td>
      <td>low</td>
      <td>224</td>
      <td>0.0275</td>
      <td>0.0062</td>
      <td>0.0087</td>
      <td>0.0424</td>
      <td>0.6476</td>
      <td>0.1468</td>
      <td>0.2056</td>
      <td>-0.0017</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DER</td>
      <td>mid</td>
      <td>193</td>
      <td>0.0486</td>
      <td>0.0156</td>
      <td>0.0229</td>
      <td>0.0871</td>
      <td>0.5579</td>
      <td>0.1787</td>
      <td>0.2633</td>
      <td>0.0102</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DER</td>
      <td>high</td>
      <td>207</td>
      <td>0.0877</td>
      <td>0.0372</td>
      <td>0.0667</td>
      <td>0.1916</td>
      <td>0.4577</td>
      <td>0.1941</td>
      <td>0.3482</td>
      <td>0.0375</td>
    </tr>
  </tbody>
</table>
</div>


      CME: var 52.0%, skew 18.0%, kurt 30.0% | total 27d = 0.0969 (ann. 131%)
      DER: var 51.1%, skew 18.3%, kurt 30.6% | total 27d = 0.1057 (ann. 143%)
    
    === Supplementary: all available days per venue ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>venue</th>
      <th>regime</th>
      <th>n_days</th>
      <th>Pi_2</th>
      <th>Pi_3</th>
      <th>Pi_4</th>
      <th>lb_total</th>
      <th>share_var</th>
      <th>share_skew</th>
      <th>share_kurt</th>
      <th>mean_vrp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CME</td>
      <td>unconditional</td>
      <td>629</td>
      <td>0.0506</td>
      <td>0.0176</td>
      <td>0.0295</td>
      <td>0.0977</td>
      <td>0.5178</td>
      <td>0.1805</td>
      <td>0.3017</td>
      <td>0.0119</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DER</td>
      <td>unconditional</td>
      <td>1317</td>
      <td>0.0558</td>
      <td>0.0205</td>
      <td>0.0348</td>
      <td>0.1111</td>
      <td>0.5018</td>
      <td>0.1847</td>
      <td>0.3135</td>
      <td>0.0151</td>
    </tr>
  </tbody>
</table>
</div>



```python
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(uncond)); width = 0.25
ax.bar(x - width, uncond["Pi_2"].values, width, label=r"$\Pi_2$ (variance)", color="C0")
ax.bar(x, uncond["Pi_3"].values, width, label=r"$\Pi_3$ (skewness)", color="C1")
ax.bar(x + width, uncond["Pi_4"].values, width, label=r"$\Pi_4$ (kurtosis)", color="C2")
ax.set_xticks(x); ax.set_xticklabels(uncond["venue"].values)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("27-day contribution")
ax.set_title("CL20 Lower-Bound Decomposition (Unconditional, θ = 2)")
ax.legend()
plt.tight_layout()
plt.show()
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_8_0.png)
    


#### 4. CL24 Conditional Decomposition by Volatility Tercile


```python
# Tercile decomposition: stacked bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, venue in zip(axes, ["CME", "DER"]):
    t_data = decomp_matched[(decomp_matched["venue"] == venue) & (decomp_matched["regime"] != "unconditional")]
    terciles = ["low", "mid", "high"]
    x = np.arange(3)
    ax.bar(x, t_data.set_index("regime").loc[terciles, "Pi_2"].values,
           width=0.25, label=r"$\Pi_2$ var", color="C0")
    ax.bar(x + 0.25, t_data.set_index("regime").loc[terciles, "Pi_3"].values,
           width=0.25, label=r"$\Pi_3$ skew", color="C1")
    ax.bar(x + 0.5, t_data.set_index("regime").loc[terciles, "Pi_4"].values,
           width=0.25, label=r"$\Pi_4$ kurt", color="C2")
    ax.set_xticks(x + 0.25); ax.set_xticklabels(terciles)
    ax.set_title(venue); ax.legend(fontsize=9)
    ax.set_xlabel("Volatility tercile"); ax.set_ylabel("27-day contribution")
fig.suptitle("CL24 Conditional Decomposition by Volatility Tercile (θ = 2)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_rnd_moments_tercile", dpi=300)
plt.show()

# Print shares by tercile
print("=== Moment shares by venue and tercile (matched days) ===")
for venue in ["CME", "DER"]:
    print(f"\n  {venue}:")
    for regime in ["low", "mid", "high"]:
        row = decomp_matched[(decomp_matched["venue"] == venue) & (decomp_matched["regime"] == regime)].iloc[0]
        print(f"    {regime}: var {row['share_var']:.1%}, skew {row['share_skew']:.1%}, "
              f"kurt {row['share_kurt']:.1%} | total = {row['lb_total']:.4f}")
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_10_0.png)
    


    === Moment shares by venue and tercile (matched days) ===
    
      CME:
        low: var 66.9%, skew 13.9%, kurt 19.2% | total = 0.0384
        mid: var 56.3%, skew 17.8%, kurt 25.9% | total = 0.0810
        high: var 46.6%, skew 19.1%, kurt 34.4% | total = 0.1757
    
      DER:
        low: var 64.8%, skew 14.7%, kurt 20.6% | total = 0.0424
        mid: var 55.8%, skew 17.9%, kurt 26.3% | total = 0.0871
        high: var 45.8%, skew 19.4%, kurt 34.8% | total = 0.1916
    

#### CL24 Regional Bound by Regime



```python
show_table(
    TAB_DIR / "cl24_regional.csv",
    cols=["venue", "regime", "region", "n_days", "Pi_2", "Pi_3", "Pi_4",
          "LB_region", "LB_grid_total", "share_of_grid_total"],
    title="CL24 regional lower-bound contributions (matched days)",
    sort=["venue", "regime", "region"],
    produced_by="run_cl24_regional.py",
)

show_table(
    TAB_DIR / "cl24_route_diagnostic.csv",
    cols=["venue", "n_days", "mean_lb_grid_total", "mean_lb_spanning_total",
          "mean_truncation_gap", "pct_truncation_gap"],
    title="Grid-route vs spanning-route truncation diagnostic",
    produced_by="run_cl24_regional.py",
)

show_table(
    TAB_DIR / "cl24_regional_wedge.csv",
    cols=["dep_var", "region", "order", "wedge", "t_stat", "p_value",
          "stars", "n_days", "nw_lags"],
    title="CL24 regional cross-venue wedge (DER - CME), NW(27)",
    sort=["region", "order"],
    produced_by="run_cl24_regional.py",
)
```

    --- CL24 regional lower-bound contributions (matched days) ---
    venue        regime region  n_days   Pi_2    Pi_3   Pi_4  LB_region  LB_grid_total  share_of_grid_total
      CME          high center     205 0.0013  0.0000 0.0000     0.0013         0.0816               0.0162
      CME          high   down     205 0.0361  0.0184 0.0110     0.0655         0.0816               0.8020
      CME          high  total     205 0.0561  0.0114 0.0141     0.0816         0.0816               1.0000
      CME          high     up     205 0.0188 -0.0070 0.0031     0.0148         0.0816               0.1819
      CME           low center     223 0.0016  0.0000 0.0000     0.0017         0.0303               0.0546
      CME           low   down     223 0.0137  0.0057 0.0030     0.0224         0.0303               0.7388
      CME           low  total     223 0.0230  0.0034 0.0039     0.0303         0.0303               1.0000
      CME           low     up     223 0.0077 -0.0023 0.0009     0.0063         0.0303               0.2066
      CME           mid center     191 0.0015  0.0000 0.0000     0.0015         0.0513               0.0297
      CME           mid   down     191 0.0230  0.0107 0.0061     0.0398         0.0513               0.7764
      CME           mid  total     191 0.0369  0.0067 0.0077     0.0513         0.0513               1.0000
      CME           mid     up     191 0.0124 -0.0041 0.0016     0.0100         0.0513               0.1940
      CME unconditional center     619 0.0015  0.0000 0.0000     0.0015         0.0538               0.0279
      CME unconditional   down     619 0.0240  0.0115 0.0066     0.0420         0.0538               0.7816
      CME unconditional  total     619 0.0383  0.0071 0.0084     0.0538         0.0538               1.0000
      CME unconditional     up     619 0.0128 -0.0044 0.0018     0.0102         0.0538               0.1904
      DER          high center     205 0.0012  0.0000 0.0000     0.0013         0.0873               0.0145
      DER          high   down     205 0.0385  0.0197 0.0118     0.0700         0.0873               0.8027
      DER          high  total     205 0.0599  0.0121 0.0152     0.0873         0.0873               1.0000
      DER          high     up     205 0.0202 -0.0076 0.0033     0.0159         0.0873               0.1828
      DER           low center     223 0.0016  0.0000 0.0000     0.0017         0.0321               0.0517
      DER           low   down     223 0.0144  0.0061 0.0032     0.0238         0.0321               0.7422
      DER           low  total     223 0.0242  0.0037 0.0042     0.0321         0.0321               1.0000
      DER           low     up     223 0.0082 -0.0025 0.0009     0.0066         0.0321               0.2061
      DER           mid center     191 0.0015  0.0000 0.0000     0.0015         0.0545               0.0276
      DER           mid   down     191 0.0244  0.0115 0.0065     0.0424         0.0545               0.7788
      DER           mid  total     191 0.0391  0.0071 0.0082     0.0545         0.0545               1.0000
      DER           mid     up     191 0.0132 -0.0044 0.0017     0.0105         0.0545               0.1936
      DER unconditional center     619 0.0015  0.0000 0.0000     0.0015         0.0572               0.0259
      DER unconditional   down     619 0.0255  0.0123 0.0071     0.0449         0.0572               0.7835
      DER unconditional  total     619 0.0406  0.0075 0.0091     0.0572         0.0572               1.0000
      DER unconditional     up     619 0.0137 -0.0047 0.0020     0.0109         0.0572               0.1907
    
    [32 rows]  source: cl24_regional.csv
    --- Grid-route vs spanning-route truncation diagnostic ---
    venue  n_days  mean_lb_grid_total  mean_lb_spanning_total  mean_truncation_gap  pct_truncation_gap
      CME     619              0.0538                  0.0970               0.0432             44.5589
      DER     619              0.0572                  0.1056               0.0484             45.8050
    
    [2 rows]  source: cl24_route_diagnostic.csv
    --- CL24 regional cross-venue wedge (DER - CME), NW(27) ---
         dep_var region order   wedge  t_stat  p_value stars  n_days  nw_lags
    dPi_2_center center     2 -0.0000 -3.3112   0.0009   ***     619       27
    dPi_3_center center     3  0.0000  2.1382   0.0325    **     619       27
    dPi_4_center center     4 -0.0000 -1.6916   0.0907     *     619       27
      dLB_center center    LB -0.0000 -3.2097   0.0013   ***     619       27
      dPi_2_down   down     2  0.0015  6.3505   0.0000   ***     619       27
      dPi_3_down   down     3  0.0008  5.9394   0.0000   ***     619       27
      dPi_4_down   down     4  0.0005  5.6315   0.0000   ***     619       27
        dLB_down   down    LB  0.0028  6.1457   0.0000   ***     619       27
     dPi_2_total  total     2  0.0023  6.7348   0.0000   ***     619       27
     dPi_3_total  total     3  0.0005  5.1442   0.0000   ***     619       27
     dPi_4_total  total     4  0.0007  5.8614   0.0000   ***     619       27
       dLB_total  total    LB  0.0035  6.4132   0.0000   ***     619       27
        dPi_2_up     up     2  0.0009  7.1467   0.0000   ***     619       27
        dPi_3_up     up     3 -0.0003 -6.5342   0.0000   ***     619       27
        dPi_4_up     up     4  0.0002  6.1015   0.0000   ***     619       27
          dLB_up     up    LB  0.0007  7.2214   0.0000   ***     619       27
    
    [16 rows]  source: cl24_regional_wedge.csv
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dep_var</th>
      <th>region</th>
      <th>order</th>
      <th>wedge</th>
      <th>t_stat</th>
      <th>p_value</th>
      <th>stars</th>
      <th>n_days</th>
      <th>nw_lags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>dPi_2_center</td>
      <td>center</td>
      <td>2</td>
      <td>-2.274029e-05</td>
      <td>-3.311223</td>
      <td>9.288924e-04</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dPi_3_center</td>
      <td>center</td>
      <td>3</td>
      <td>5.254723e-07</td>
      <td>2.138216</td>
      <td>3.249923e-02</td>
      <td>**</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dPi_4_center</td>
      <td>center</td>
      <td>4</td>
      <td>-7.200125e-08</td>
      <td>-1.691623</td>
      <td>9.071792e-02</td>
      <td>*</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dLB_center</td>
      <td>center</td>
      <td>LB</td>
      <td>-2.228682e-05</td>
      <td>-3.209684</td>
      <td>1.328810e-03</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>0</th>
      <td>dPi_2_down</td>
      <td>down</td>
      <td>2</td>
      <td>1.506400e-03</td>
      <td>6.350511</td>
      <td>2.146006e-10</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dPi_3_down</td>
      <td>down</td>
      <td>3</td>
      <td>8.102563e-04</td>
      <td>5.939377</td>
      <td>2.861078e-09</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dPi_4_down</td>
      <td>down</td>
      <td>4</td>
      <td>4.969854e-04</td>
      <td>5.631495</td>
      <td>1.786538e-08</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dLB_down</td>
      <td>down</td>
      <td>LB</td>
      <td>2.813642e-03</td>
      <td>6.145670</td>
      <td>7.962667e-10</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>12</th>
      <td>dPi_2_total</td>
      <td>total</td>
      <td>2</td>
      <td>2.346175e-03</td>
      <td>6.734811</td>
      <td>1.641434e-11</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>13</th>
      <td>dPi_3_total</td>
      <td>total</td>
      <td>3</td>
      <td>4.664692e-04</td>
      <td>5.144183</td>
      <td>2.686875e-07</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>14</th>
      <td>dPi_4_total</td>
      <td>total</td>
      <td>4</td>
      <td>6.524317e-04</td>
      <td>5.861449</td>
      <td>4.588452e-09</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>15</th>
      <td>dLB_total</td>
      <td>total</td>
      <td>LB</td>
      <td>3.465075e-03</td>
      <td>6.413213</td>
      <td>1.424840e-10</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dPi_2_up</td>
      <td>up</td>
      <td>2</td>
      <td>8.625144e-04</td>
      <td>7.146666</td>
      <td>8.891088e-13</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dPi_3_up</td>
      <td>up</td>
      <td>3</td>
      <td>-3.443126e-04</td>
      <td>-6.534203</td>
      <td>6.394909e-11</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>10</th>
      <td>dPi_4_up</td>
      <td>up</td>
      <td>4</td>
      <td>1.555183e-04</td>
      <td>6.101493</td>
      <td>1.050820e-09</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
    <tr>
      <th>11</th>
      <td>dLB_up</td>
      <td>up</td>
      <td>LB</td>
      <td>6.737200e-04</td>
      <td>7.221436</td>
      <td>5.144127e-13</td>
      <td>***</td>
      <td>619</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>



#### 5. Cumulant Contribution Time Series and Boxplots


```python
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for venue, color in [("CME", "C0"), ("DER", "C1")]:
    v = premia[premia["venue"] == venue].set_index("date").sort_index()
    axes[0].plot(v.index, v["Pi_2"], color=color, lw=0.8, alpha=0.8, label=venue)
    axes[1].plot(v.index, v["Pi_3"], color=color, lw=0.8, alpha=0.8, label=venue)
    axes[2].plot(v.index, v["Pi_4"], color=color, lw=0.8, alpha=0.8, label=venue)
for ax in axes:
    ax.axhline(0, color="black", lw=0.4)
axes[0].set_ylabel(r"$\Pi_{2,t}^j$ (variance)")
axes[0].set_title("CL20 Cumulant Premium Contributions (θ = 2)")
axes[0].legend()
axes[1].set_ylabel(r"$\Pi_{3,t}^j$ (skewness)")
axes[2].set_ylabel(r"$\Pi_{4,t}^j$ (kurtosis)")
axes[2].set_xlabel("Date")
plt.tight_layout()
plt.show()
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_14_0.png)
    



```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, label) in zip(axes, [
    ("Pi_2", r"$\Pi_2$ (variance)"),
    ("Pi_3", r"$\Pi_3$ (skewness)"),
    ("Pi_4", r"$\Pi_4$ (kurtosis)")
]):
    data, labels_x = [], []
    for venue in ["CME", "DER"]:
        for tercile in ["low", "mid", "high"]:
            mask = (premia_matched["venue"] == venue) & (premia_matched["tercile"] == tercile)
            data.append(premia_matched.loc[mask, col].dropna().values)
            labels_x.append(f"{venue}\n{tercile}")
    ax.boxplot(data, tick_labels=labels_x, showfliers=False)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel(label); ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Cumulant Premium Contributions by Venue and Volatility Tercile (matched days)", fontsize=13)
plt.tight_layout()
plt.show()
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_15_0.png)
    


#### 6. Cross-Venue Moment Agreement and VRP Diagnostic


```python
# Cross-venue scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, label) in zip(axes, [
    ("var_Q", "Variance"), ("skew_Q", "Skewness"), ("kurt_Q", "Kurtosis")
]):
    ax.scatter(matched[f"{col}_cme"], matched[f"{col}_der"],
               s=3, alpha=0.4, color="C0")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel(f"CME {label}"); ax.set_ylabel(f"Deribit {label}")
    rho = matched[[f"{col}_cme", f"{col}_der"]].corr().iloc[0, 1]
    ax.set_title(f"{label} (ρ = {rho:.3f})")
fig.suptitle("Cross-Venue RN Moment Agreement (matched days)", fontsize=13)
plt.tight_layout()
plt.show()
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_17_0.png)
    



```python
# VRP time series
fig, ax = plt.subplots(figsize=(14, 5))
for venue, color in [("CME", "C0"), ("DER", "C1")]:
    v = premia[premia["venue"] == venue].set_index("date").sort_index()
    ax.plot(v.index, v["vrp"], color=color, lw=0.8, alpha=0.8, label=venue)
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Date")
ax.set_ylabel(r"$\mathrm{VRP}_t^j = V_t^{\mathbb{Q},j} - V_t^{\mathbb{P}}$")
ax.set_title("Variance Risk Premium: CME vs Deribit (27-day)")
ax.legend()
plt.tight_layout()
plt.show()

print("=== VRP Summary (matched days) ===")
for venue in ["CME", "DER"]:
    v = premia_matched[premia_matched["venue"] == venue]["vrp"].dropna()
    print(f"  {venue}: mean = {v.mean():.4f}, std = {v.std():.4f}, "
          f"median = {v.median():.4f}, positive share = {(v > 0).mean():.1%}")
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_18_0.png)
    


    === VRP Summary (matched days) ===
      CME: mean = 0.0119, std = 0.0269, median = 0.0068, positive share = 71.1%
      DER: mean = 0.0153, std = 0.0289, median = 0.0106, positive share = 77.7%
    

#### 7. Preference Parameter Robustness ($\theta$ sweep)


```python
print("=== Theta Robustness Sweep ===\n")
display(theta_rob.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
for venue, color in [("CME", "C0"), ("DER", "C1")]:
    v = theta_rob[theta_rob["venue"] == venue].sort_values("theta")
    ax.plot(v["theta"], v["lb_annualized_pct"], "o-", color=color, label=venue, markersize=8)
ax.set_xlabel(r"Preference parameter $\theta$")
ax.set_ylabel("Annualized lower bound (%)")
ax.set_title("CL20 Lower Bound: Sensitivity to Preference Parameter")
ax.legend()
plt.tight_layout()
plt.show()
```

    === Theta Robustness Sweep ===
    
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>venue</th>
      <th>theta</th>
      <th>lambda_1</th>
      <th>lambda_2</th>
      <th>lambda_3</th>
      <th>mean_Pi_2</th>
      <th>mean_Pi_3</th>
      <th>mean_Pi_4</th>
      <th>mean_lb_total</th>
      <th>lb_annualized_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CME</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.6667</td>
      <td>0.5000</td>
      <td>0.0504</td>
      <td>0.0116</td>
      <td>0.0145</td>
      <td>0.0765</td>
      <td>103.4653</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DER</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.6667</td>
      <td>0.5000</td>
      <td>0.0540</td>
      <td>0.0129</td>
      <td>0.0162</td>
      <td>0.0831</td>
      <td>112.3251</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CME</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>-1.0000</td>
      <td>1.0000</td>
      <td>0.0504</td>
      <td>0.0174</td>
      <td>0.0291</td>
      <td>0.0969</td>
      <td>130.9770</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DER</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>-1.0000</td>
      <td>1.0000</td>
      <td>0.0540</td>
      <td>0.0194</td>
      <td>0.0324</td>
      <td>0.1057</td>
      <td>142.9301</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CME</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>-1.3333</td>
      <td>1.6667</td>
      <td>0.0504</td>
      <td>0.0232</td>
      <td>0.0485</td>
      <td>0.1221</td>
      <td>165.0423</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DER</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>-1.3333</td>
      <td>1.6667</td>
      <td>0.0540</td>
      <td>0.0258</td>
      <td>0.0539</td>
      <td>0.1338</td>
      <td>180.8249</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](04_moment_decomposition_files/04_moment_decomposition_20_2.png)
    


#### 8. Kappa Truncation Sensitivity (matched days)


```python
show_table(
    TAB_DIR / "kappa_sensitivity.csv",
    cols=["venue", "kappa_bound", "n_days", "Pi_2", "Pi_3", "Pi_4",
          "lb_total", "share_kurt", "lb_annualized_pct"],
    title="CL20 decomposition by strike-domain truncation",
    sort=["venue", "kappa_bound"],
    produced_by="run_phase4.py",
)
```

    --- CL20 decomposition by strike-domain truncation ---
    venue  kappa_bound  n_days   Pi_2   Pi_3   Pi_4  lb_total  share_kurt  lb_annualized_pct
      CME         1.00     624 0.0491 0.0141 0.0213    0.0844      0.2526           114.1592
      CME         1.25     624 0.0499 0.0161 0.0257    0.0917      0.2805           123.9851
      CME         1.50     624 0.0504 0.0174 0.0291    0.0969      0.3002           130.9770
      DER         1.00     624 0.0525 0.0154 0.0233    0.0912      0.2558           123.3341
      DER         1.25     624 0.0534 0.0178 0.0284    0.0996      0.2849           134.6288
      DER         1.50     624 0.0540 0.0194 0.0324    0.1057      0.3060           142.9301
    
    [6 rows]  source: kappa_sensitivity.csv
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>venue</th>
      <th>kappa_bound</th>
      <th>n_days</th>
      <th>Pi_2</th>
      <th>Pi_3</th>
      <th>Pi_4</th>
      <th>lb_total</th>
      <th>share_var</th>
      <th>share_skew</th>
      <th>share_kurt</th>
      <th>lb_annualized_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CME</td>
      <td>1.00</td>
      <td>624</td>
      <td>0.049059</td>
      <td>0.014060</td>
      <td>0.021327</td>
      <td>0.084447</td>
      <td>0.580953</td>
      <td>0.166492</td>
      <td>0.252555</td>
      <td>114.159203</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CME</td>
      <td>1.25</td>
      <td>624</td>
      <td>0.049913</td>
      <td>0.016080</td>
      <td>0.025722</td>
      <td>0.091715</td>
      <td>0.544224</td>
      <td>0.175324</td>
      <td>0.280452</td>
      <td>123.985054</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CME</td>
      <td>1.50</td>
      <td>624</td>
      <td>0.050377</td>
      <td>0.017422</td>
      <td>0.029087</td>
      <td>0.096887</td>
      <td>0.519960</td>
      <td>0.179822</td>
      <td>0.300218</td>
      <td>130.976966</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DER</td>
      <td>1.00</td>
      <td>624</td>
      <td>0.052450</td>
      <td>0.015441</td>
      <td>0.023342</td>
      <td>0.091233</td>
      <td>0.574899</td>
      <td>0.169251</td>
      <td>0.255850</td>
      <td>123.334087</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DER</td>
      <td>1.25</td>
      <td>624</td>
      <td>0.053435</td>
      <td>0.017782</td>
      <td>0.028372</td>
      <td>0.099588</td>
      <td>0.536556</td>
      <td>0.178554</td>
      <td>0.284891</td>
      <td>134.628815</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DER</td>
      <td>1.50</td>
      <td>624</td>
      <td>0.053988</td>
      <td>0.019386</td>
      <td>0.032355</td>
      <td>0.105729</td>
      <td>0.510629</td>
      <td>0.183356</td>
      <td>0.306014</td>
      <td>142.930086</td>
    </tr>
  </tbody>
</table>
</div>


