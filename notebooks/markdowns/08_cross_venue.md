```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Image

# Paths (house convention: notebook lives in notebooks/, root is parent)
PROJECT_ROOT = Path(os.getcwd()).parent
DATA_DIR = PROJECT_ROOT / "data"
PHASE3_DIR = DATA_DIR / "phase3"
PHASE5_FIG = PROJECT_ROOT / "results" / "phase5" / "figures"
PHASE5_TAB = PROJECT_ROOT / "results" / "phase5" / "tables"
PHASE3_TAB = PROJECT_ROOT / "results" / "phase3" / "tables"

# Cross-venue outputs (Phase 6 of main.py)
wedge_md   = pd.read_csv(PHASE5_TAB / "matched_difference_regressions.csv")
panel_dk   = pd.read_csv(PHASE5_TAB / "panel_regressions_dk.csv")
regional   = pd.read_csv(PHASE5_TAB / "regional_mfk_summary.csv")

# Extension 1 (inverse-contract) + joint regime test
inv_wedge  = pd.read_csv(PHASE5_TAB / "inverse_contract_wedge.csv")
inv_daily  = pd.read_csv(PHASE5_TAB / "inverse_contract_daily.csv")
joint      = pd.read_csv(PHASE3_TAB / "joint_regime_test_crypto.csv")
joint_det  = pd.read_csv(PHASE3_TAB / "joint_regime_detail_crypto.csv")
```

#### 1. Cross-venue cumulant-premium wedge (matched-difference)



```python
venue_wedge = wedge_md[wedge_md["regressor"] == "const (venue wedge)"].copy()
display(venue_wedge[["dep_var", "coef", "se", "t_stat", "p_value", "stars"]]
        .reset_index(drop=True))

# State interactions: confirm none significant (the wedge is a level effect)
inter = wedge_md[wedge_md["regressor"].isin(["Z_IVS_1", "rv", "fng"])]
print("State interactions (should all be insignificant -> constant level wedge):")
for dv in ["Pi_2", "Pi_3", "Pi_4"]:
    sub = inter[inter["dep_var"] == dv]
    sig = (sub["p_value"] < 0.05).any()
    print(f"  {dv}: any state interaction significant at 5%? {bool(sig)}")
```


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
      <th>coef</th>
      <th>se</th>
      <th>t_stat</th>
      <th>p_value</th>
      <th>stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pi_2</td>
      <td>0.003567</td>
      <td>0.000417</td>
      <td>8.563973</td>
      <td>1.090429e-17</td>
      <td>***</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pi_3</td>
      <td>0.001957</td>
      <td>0.000573</td>
      <td>3.417222</td>
      <td>6.326374e-04</td>
      <td>***</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pi_4</td>
      <td>0.003260</td>
      <td>0.001039</td>
      <td>3.139037</td>
      <td>1.695042e-03</td>
      <td>***</td>
    </tr>
  </tbody>
</table>
</div>


    State interactions (should all be insignificant -> constant level wedge):
      Pi_2: any state interaction significant at 5%? False
      Pi_3: any state interaction significant at 5%? False
      Pi_4: any state interaction significant at 5%? False
    

#### 2. Conditional / regional MFK by volatility tercile (95% bootstrap bands)


```python
display(regional[["regime", "n_days", "mean_down", "lo_down", "hi_down",
                  "mean_mid", "lo_mid", "hi_mid",
                  "mean_up", "lo_up", "hi_up"]].round(4))

fig, ax = plt.subplots(figsize=(8, 5))
regimes = regional["regime"].tolist()
x = np.arange(len(regimes))
w = 0.25
for k, (reg, col) in enumerate([("down", "C3"), ("mid", "C0"), ("up", "C2")]):
    m = regional[f"mean_{reg}"].values
    lo = regional[f"lo_{reg}"].values
    hi = regional[f"hi_{reg}"].values
    ax.bar(x + (k-1)*w, m, w, color=col, alpha=0.8, label=f"{reg} region")
    ax.errorbar(x + (k-1)*w, m, yerr=[m-lo, hi-m], fmt="none",
                ecolor="0.3", capsize=3, lw=1)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(regimes)
ax.set_ylabel(r"Regional $\Psi$ (mass)"); ax.set_xlabel("Volatility regime")
ax.set_title("Regional MFK by volatility tercile (95% block-bootstrap)")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()
```


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
      <th>regime</th>
      <th>n_days</th>
      <th>mean_down</th>
      <th>lo_down</th>
      <th>hi_down</th>
      <th>mean_mid</th>
      <th>lo_mid</th>
      <th>hi_mid</th>
      <th>mean_up</th>
      <th>lo_up</th>
      <th>hi_up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>unconditional</td>
      <td>619</td>
      <td>-0.0307</td>
      <td>-0.0416</td>
      <td>-0.0209</td>
      <td>0.0069</td>
      <td>0.0045</td>
      <td>0.0093</td>
      <td>-0.0826</td>
      <td>-0.1180</td>
      <td>-0.0565</td>
    </tr>
    <tr>
      <th>1</th>
      <td>low</td>
      <td>223</td>
      <td>-0.0296</td>
      <td>-0.0553</td>
      <td>-0.0104</td>
      <td>0.0023</td>
      <td>-0.0004</td>
      <td>0.0049</td>
      <td>-0.1037</td>
      <td>-0.2052</td>
      <td>-0.0352</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mid</td>
      <td>191</td>
      <td>-0.0318</td>
      <td>-0.0458</td>
      <td>-0.0198</td>
      <td>0.0062</td>
      <td>0.0045</td>
      <td>0.0082</td>
      <td>-0.0733</td>
      <td>-0.0969</td>
      <td>-0.0500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>high</td>
      <td>205</td>
      <td>-0.0309</td>
      <td>-0.0376</td>
      <td>-0.0230</td>
      <td>0.0124</td>
      <td>0.0099</td>
      <td>0.0145</td>
      <td>-0.0682</td>
      <td>-0.0826</td>
      <td>-0.0481</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](08_cross_venue_files/08_cross_venue_4_1.png)
    


#### 3. Extension 1 — Inverse-contract numeraire prediction

Deribit is coin-margined (inverse) -> BTC-numeraire measure = unit Esscher tilt of the CME density. Parameter-free predicted wedge vs measured wedge. **Predicted negative, measured positive at every order: contract design predicts the wrong sign.**


```python
display(inv_wedge[["cumulant", "predicted_wedge", "pred_ci_lo", "pred_ci_hi",
                   "measured_wedge", "meas_ci_lo", "meas_ci_hi", "residual"]].round(5))

agree = int(np.sum(np.sign(inv_wedge["predicted_wedge"]) ==
                   np.sign(inv_wedge["measured_wedge"])))
print(f"Sign agreement: {agree}/{len(inv_wedge)} cumulants.")
print("Contract design predicts the WRONG sign -> the friction is real and "
      "amplified (residual > measured at every order).")

# The Psi overlay decomposition figure (mechanical vs residual)
img = PHASE5_FIG / "fig_inverse_contract_psi.png"
if img.exists():
    display(Image(filename=str(img)))
else:
    print(f"[figure not found: run run_inverse_contract.py] {img}")
```


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
      <th>cumulant</th>
      <th>predicted_wedge</th>
      <th>pred_ci_lo</th>
      <th>pred_ci_hi</th>
      <th>measured_wedge</th>
      <th>meas_ci_lo</th>
      <th>meas_ci_hi</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pi_2</td>
      <td>-0.00367</td>
      <td>-0.00512</td>
      <td>-0.00230</td>
      <td>0.00292</td>
      <td>0.00204</td>
      <td>0.00384</td>
      <td>0.00659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pi_3</td>
      <td>-0.01599</td>
      <td>-0.02068</td>
      <td>-0.01200</td>
      <td>0.00091</td>
      <td>0.00053</td>
      <td>0.00131</td>
      <td>0.01690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pi_4</td>
      <td>-0.00374</td>
      <td>-0.00529</td>
      <td>-0.00222</td>
      <td>0.00139</td>
      <td>0.00075</td>
      <td>0.00203</td>
      <td>0.00513</td>
    </tr>
  </tbody>
</table>
</div>


    Sign agreement: 0/3 cumulants.
    Contract design predicts the WRONG sign -> the friction is real and amplified (residual > measured at every order).
    


    
![png](08_cross_venue_files/08_cross_venue_6_2.png)
    


#### 4. Joint regime test (Phase 3 follow-up)

The marginal curvature c is collinear (insignificant per-tercile), so the joint test of H₀: (b,c,d)_low = (b,c,d)_high is the correct question. **Significant at both venues; the low-vol kernel is more concave at the money (2c+6d).**


```python
display(joint[["venue", "wald_stat", "p_value",
               "delta_b", "delta_c", "delta_d", "frac_vector_consistent"]].round(4))

curv = joint_det[joint_det["test"] == "curv_at_money_diff"]
print("Curvature at money (2c+6d at R=1), low - high:")
for _, r in curv.iterrows():
    print(f"  {r['venue']}: {r['diff_point']:+.2f} "
          f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  P(<0) = {r['frac_negative']:.3f}")
print("\nThe kernel significantly depends on the volatility regime; the low-vol "
      "kernel is significantly more concave at the money (P<0 = 1.000 both venues).")
```


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
      <th>wald_stat</th>
      <th>p_value</th>
      <th>delta_b</th>
      <th>delta_c</th>
      <th>delta_d</th>
      <th>frac_vector_consistent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CME</td>
      <td>32.9016</td>
      <td>0.005</td>
      <td>15.8425</td>
      <td>-7.6535</td>
      <td>0.0587</td>
      <td>0.460</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DER</td>
      <td>20.3802</td>
      <td>0.010</td>
      <td>17.1939</td>
      <td>-9.3011</td>
      <td>0.6724</td>
      <td>0.655</td>
    </tr>
  </tbody>
</table>
</div>


    Curvature at money (2c+6d at R=1), low - high:
      CME: -14.95 [-24.27, -10.41]  P(<0) = 1.000
      DER: -14.57 [-22.78, -9.65]  P(<0) = 1.000
    
    The kernel significantly depends on the volatility regime; the low-vol kernel is significantly more concave at the money (P<0 = 1.000 both venues).
    

---
**Summary.** The cross-venue wedge is positive and significant at every cumulant
order (a constant level effect, no state interactions); contract design predicts
the opposite sign, so the friction is real and amplified; the kernel's
regime-dependence is jointly significant via the at-money curvature. The recurring
theme: cross-venue *differences* are precisely estimated where single-venue *levels*
are not.
