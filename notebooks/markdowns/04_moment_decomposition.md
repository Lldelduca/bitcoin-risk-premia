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
theta_rob = pd.read_csv(RES_P4 / "tables" / "theta_robustness.csv")
moment_summary = pd.read_csv(TAB_DIR / "moment_summary.csv", header=[0, 1], index_col=0)
bkm["date"] = pd.to_datetime(bkm["date"])
premia["date"] = pd.to_datetime(premia["date"])

bkm["date"] = pd.to_datetime(bkm["date"])
premia["date"] = pd.to_datetime(premia["date"])

print(f"BKM moments: {len(bkm)} day-venue pairs")
print(f"  CME: {(bkm['venue']=='CME').sum()} days")
print(f"  DER: {(bkm['venue']=='DER').sum()} days")
print(f"Cumulant premia: {len(premia)} rows")
print(f"  Date range: {premia['date'].min().date()} -> {premia['date'].max().date()}")
```

    BKM moments: 1946 day-venue pairs
      CME: 629 days
      DER: 1317 days
    Cumulant premia: 1946 rows
      Date range: 2020-01-13 -> 2023-08-31
    

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
      <td>0.0498</td>
      <td>0.0297</td>
      <td>-1.0051</td>
      <td>0.5152</td>
      <td>9.3185</td>
      <td>2.6815</td>
      <td>0.0506</td>
      <td>0.0308</td>
      <td>0.0176</td>
      <td>0.0205</td>
      <td>0.0295</td>
      <td>0.0375</td>
      <td>0.0119</td>
      <td>0.0270</td>
    </tr>
    <tr>
      <th>DER</th>
      <td>0.0549</td>
      <td>0.0332</td>
      <td>-0.9959</td>
      <td>0.5329</td>
      <td>9.1733</td>
      <td>2.6080</td>
      <td>0.0558</td>
      <td>0.0345</td>
      <td>0.0205</td>
      <td>0.0257</td>
      <td>0.0348</td>
      <td>0.0464</td>
      <td>0.0151</td>
      <td>0.0306</td>
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
axes[2].legend(loc="upper right")
axes[2].set_xlabel("Date")
plt.tight_layout()
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
# Bar chart: unconditional lower-bound contributions
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
plt.show()

# Print shares by tercile
print("=== Moment shares by venue and tercile ===")
for venue in ["CME", "DER"]:
    print(f"\n  {venue}:")
    for regime in ["low", "mid", "high"]:
        row = decomp_matched[(decomp_matched["venue"] == venue) & (decomp_matched["regime"] == regime)].iloc[0]
        print(f"    {regime}: var {row['share_var']:.1%}, skew {row['share_skew']:.1%}, "
              f"kurt {row['share_kurt']:.1%} | total = {row['lb_total']:.4f}")
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_10_0.png)
    


    === Moment shares by venue and tercile ===
    
      CME:
        low: var 66.9%, skew 13.9%, kurt 19.2% | total = 0.0384
        mid: var 56.3%, skew 17.8%, kurt 25.9% | total = 0.0810
        high: var 46.6%, skew 19.1%, kurt 34.4% | total = 0.1757
    
      DER:
        low: var 64.8%, skew 14.7%, kurt 20.6% | total = 0.0424
        mid: var 55.8%, skew 17.9%, kurt 26.3% | total = 0.0871
        high: var 45.8%, skew 19.4%, kurt 34.8% | total = 0.1916
    

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


    
![png](04_moment_decomposition_files/04_moment_decomposition_12_0.png)
    



```python
# Boxplots by venue x tercile
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, label) in zip(axes, [
    ("Pi_2", r"$\Pi_2$ (variance)"),
    ("Pi_3", r"$\Pi_3$ (skewness)"),
    ("Pi_4", r"$\Pi_4$ (kurtosis)")
]):
    data, labels_x = [], []
    for venue in ["CME", "DER"]:
        for tercile in ["low", "mid", "high"]:
            mask = (premia["venue"] == venue) & (premia["tercile"] == tercile)
            data.append(premia.loc[mask, col].dropna().values)
            labels_x.append(f"{venue}\n{tercile}")
    ax.boxplot(data, tick_labels=labels_x, showfliers=False)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel(label); ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Cumulant Premium Contributions by Venue and Volatility Tercile", fontsize=13)
plt.tight_layout()
plt.show()
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_13_0.png)
    


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


    
![png](04_moment_decomposition_files/04_moment_decomposition_15_0.png)
    



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

print("=== VRP Summary ===")
for venue in ["CME", "DER"]:
    v = premia[premia["venue"] == venue]["vrp"].dropna()
    print(f"  {venue}: mean = {v.mean():.4f}, std = {v.std():.4f}, "
          f"median = {v.median():.4f}, positive share = {(v > 0).mean():.1%}")
```


    
![png](04_moment_decomposition_files/04_moment_decomposition_16_0.png)
    


    === VRP Summary ===
      CME: mean = 0.0119, std = 0.0270, median = 0.0068, positive share = 71.0%
      DER: mean = 0.0151, std = 0.0306, median = 0.0101, positive share = 73.8%
    

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



    
![png](04_moment_decomposition_files/04_moment_decomposition_18_2.png)
    


 ## Summary



 **Key findings (Phase 4, common window, $\theta = 2$):**



 1. **Kurtosis is co-equal with variance** in the CL20 lower bound (~30% vs ~52%

    unconditionally), distinguishing Bitcoin from the variance-dominated equity premium.



 2. **Composition shift across regimes**: variance share drops from ~67% (low-vol)

    to ~46% (high-vol) while kurtosis share rises from ~19% to ~35%, indicating

    that tail risk pricing intensifies under stress.



 3. **Cross-venue consistency**: Deribit exceeds CME at every moment order, but the

    *shares* are nearly identical (within 2 pp), indicating the MFK tent-shape

    operates uniformly across moment orders.



 4. **VRP diagnostic**: positive on both venues (CME 1.2%, DER 1.5% per 27 days),

    confirming the risk-neutral variance exceeds the physical variance.



 5. **Theta robustness**: the cross-venue ordering and composition shift are

    invariant to the preference parameter.

*(Note: the bullet numbers above are hard-coded prose and must be re-read against the post-rerun CSVs before quoting - the tables above are the source of truth.)*
