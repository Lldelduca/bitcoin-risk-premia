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
from src.config import get_path

plt.rcParams['figure.figsize'] = (13, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Centralized Path Registry Resolution
DATA_P1 = get_path("data_phase1")
DATA_P3 = get_path("data_phase3")
DATA_P5 = get_path("data_phase5")
RES_P3 = get_path("results_phase3")
RES_P5 = get_path("results_phase5")

PHASE5_FIG = RES_P5 / "figures"
PHASE5_TAB = RES_P5 / "tables"
PHASE3_TAB = RES_P3 / "tables"

# Cross-venue outputs
wedge_md   = pd.read_csv(PHASE5_TAB / "matched_difference_regressions.csv")
panel_dk   = pd.read_csv(PHASE5_TAB / "panel_regressions_dk.csv")
regional   = pd.read_csv(PHASE5_TAB / "regional_mfk_summary.csv")

# Extension 1 (inverse-contract) + joint regime test
inv_wedge  = pd.read_csv(PHASE5_TAB / "inverse_contract_wedge.csv")
inv_daily  = pd.read_csv(PHASE5_TAB / "inverse_contract_daily.csv")
joint      = pd.read_csv(PHASE3_TAB / "joint_regime_test_crypto.csv")
joint_det  = pd.read_csv(PHASE3_TAB / "joint_regime_detail_crypto.csv")
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[2], line 26
         22 
         23 # Extension 1 (inverse-contract) + joint regime test
         24 inv_wedge  = pd.read_csv(PHASE5_TAB / "inverse_contract_wedge.csv")
         25 inv_daily  = pd.read_csv(PHASE5_TAB / "inverse_contract_daily.csv")
    ---> 26 joint      = pd.read_csv(PHASE3_TAB / "joint_regime_test_crypto.csv")
         27 joint_det  = pd.read_csv(PHASE3_TAB / "joint_regime_detail_crypto.csv")
    

    File c:\Projects\bitcoin-risk-premia\.venv\Lib\site-packages\pandas\io\parsers\readers.py:873, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, skip_blank_lines, parse_dates, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, low_memory, memory_map, float_precision, storage_options, dtype_backend)
        861 kwds_defaults = _refine_defaults_read(
        862     dialect,
        863     delimiter,
       (...)    869     dtype_backend=dtype_backend,
        870 )
        871 kwds.update(kwds_defaults)
    --> 873 return _read(filepath_or_buffer, kwds)
    

    File c:\Projects\bitcoin-risk-premia\.venv\Lib\site-packages\pandas\io\parsers\readers.py:300, in _read(filepath_or_buffer, kwds)
        297 _validate_names(kwds.get("names", None))
        299 # Create the parser.
    --> 300 parser = TextFileReader(filepath_or_buffer, **kwds)
        302 if chunksize or iterator:
        303     return parser
    

    File c:\Projects\bitcoin-risk-premia\.venv\Lib\site-packages\pandas\io\parsers\readers.py:1645, in TextFileReader.__init__(self, f, engine, **kwds)
       1642     self.options["has_index_names"] = kwds["has_index_names"]
       1644 self.handles: IOHandles | None = None
    -> 1645 self._engine = self._make_engine(f, self.engine)
    

    File c:\Projects\bitcoin-risk-premia\.venv\Lib\site-packages\pandas\io\parsers\readers.py:1904, in TextFileReader._make_engine(self, f, engine)
       1902     if "b" not in mode:
       1903         mode += "b"
    -> 1904 self.handles = get_handle(
       1905     f,
       1906     mode,
       1907     encoding=self.options.get("encoding", None),
       1908     compression=self.options.get("compression", None),
       1909     memory_map=self.options.get("memory_map", False),
       1910     is_text=is_text,
       1911     errors=self.options.get("encoding_errors", "strict"),
       1912     storage_options=self.options.get("storage_options", None),
       1913 )
       1914 assert self.handles is not None
       1915 f = self.handles.handle
    

    File c:\Projects\bitcoin-risk-premia\.venv\Lib\site-packages\pandas\io\common.py:926, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        921 elif isinstance(handle, str):
        922     # Check whether the filename is to be opened in binary mode.
        923     # Binary mode does not support 'encoding' and 'newline'.
        924     if ioargs.encoding and "b" not in ioargs.mode:
        925         # Encoding
    --> 926         handle = open(
        927             handle,
        928             ioargs.mode,
        929             encoding=ioargs.encoding,
        930             errors=errors,
        931             newline="",
        932         )
        933     else:
        934         # Binary mode
        935         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Projects\\bitcoin-risk-premia\\results\\phase3\\tables\\joint_regime_test_crypto.csv'


> **Note on sample sizes.** Cell counts differ by data requirement, not by filtering choices: the CL20 wedge regressions require BKM extractions on both venues; the MFK requires a valid daily RND on both venues; the friction-proxy regressions additionally require CME basis and Deribit funding data. Each analysis uses the largest sample its inputs permit - the exact counts are printed by each table below and should be quoted from there.

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
      <td>0.003571</td>
      <td>0.000417</td>
      <td>8.557052</td>
      <td>1.157904e-17</td>
      <td>***</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pi_3</td>
      <td>0.001959</td>
      <td>0.000573</td>
      <td>3.419211</td>
      <td>6.280297e-04</td>
      <td>***</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pi_4</td>
      <td>0.003264</td>
      <td>0.001039</td>
      <td>3.140777</td>
      <td>1.685005e-03</td>
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
      <td>-0.0308</td>
      <td>-0.0417</td>
      <td>-0.0210</td>
      <td>0.0069</td>
      <td>0.0045</td>
      <td>0.0093</td>
      <td>-0.0827</td>
      <td>-0.1182</td>
      <td>-0.0565</td>
    </tr>
    <tr>
      <th>1</th>
      <td>low</td>
      <td>223</td>
      <td>-0.0298</td>
      <td>-0.0554</td>
      <td>-0.0106</td>
      <td>0.0024</td>
      <td>-0.0004</td>
      <td>0.0049</td>
      <td>-0.1038</td>
      <td>-0.2055</td>
      <td>-0.0353</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mid</td>
      <td>191</td>
      <td>-0.0318</td>
      <td>-0.0458</td>
      <td>-0.0198</td>
      <td>0.0062</td>
      <td>0.0046</td>
      <td>0.0082</td>
      <td>-0.0735</td>
      <td>-0.0969</td>
      <td>-0.0501</td>
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



    
![png](05_cross_venue_files/05_cross_venue_6_1.png)
    


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
      <td>0.00205</td>
      <td>0.00384</td>
      <td>0.00659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pi_3</td>
      <td>-0.01599</td>
      <td>-0.02068</td>
      <td>-0.01199</td>
      <td>0.00092</td>
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
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[5], line 13
          9 
         10 # The Psi overlay decomposition figure (mechanical vs residual)
         11 img = PHASE5_FIG / "fig_inverse_contract_psi.png"
         12 if img.exists():
    ---> 13     display(Image(filename=str(img)))
         14 else:
         15     print(f"[figure not found: run run_inverse_contract.py] {img}")
    

    NameError: name 'Image' is not defined


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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[6], line 1
    ----> 1 display(joint[["venue", "wald_stat", "p_value",
          2                "delta_b", "delta_c", "delta_d", "frac_vector_consistent"]].round(4))
          3 
          4 curv = joint_det[joint_det["test"] == "curv_at_money_diff"]
    

    NameError: name 'joint' is not defined


---
**Summary.** The cross-venue wedge is positive and significant at every cumulant
order (a constant level effect, no state interactions); contract design predicts
the opposite sign, so the friction is real and amplified; the kernel's
regime-dependence is jointly significant via the at-money curvature. The recurring
theme: cross-venue *differences* are precisely estimated where single-venue *levels*
are not.

#### 5. Regional Probability-Mass Wedge (appendix diagnostic)

The regional MFK integrates the log-density ratio; this cell reports the companion object in probability units: the difference in region mass $\int q^{CME} - \int q^{DER}$ over the downside ($R<0.90$), mid ($0.90$-$1.10$), and upside ($R>1.10$) regions, with 95% block-bootstrap bands and the same full-sample tercile stratification as the regional MFK. Positive values mean CME assigns more probability to that region.


```python
# --- CELL 5: APPENDIX MASS WEDGE DIAGNOSTIC ---
import sys, warnings
from src.config import get_return_grid
from src.phase3.bootstrap_inference import (block_bootstrap_mean_bands,
                                            block_bootstrap_group_mean_bands)

R_GRID = get_return_grid()
down_m = R_GRID < 0.90
mid_m = (R_GRID >= 0.90) & (R_GRID <= 1.10)
up_m = R_GRID > 1.10

def _load_rnds(venue):
    df = pd.read_parquet(DATA_P1 / f'rnd_{venue}_densities.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df[df['tau_days'] == 27].set_index('date')

cme_d, der_d = _load_rnds('CME'), _load_rnds('DER')
matched_dates = cme_d.index.intersection(der_d.index).sort_values()

Z_crypto = pd.read_parquet(DATA_P1 / 'Z_crypto.parquet')
Z_crypto['date'] = pd.to_datetime(Z_crypto['date'])
Z_crypto['tercile'] = pd.qcut(Z_crypto['Z_IVS_1'], q=3, labels=['low', 'mid', 'high'])
terc_map = Z_crypto.set_index('date')['tercile']

def _interp(row):
    q = np.interp(R_GRID, np.array(row['returns']), np.array(row['density']),
                  left=0, right=0)
    m = np.trapezoid(q, R_GRID)
    return q / m if m > 0 else q

recs = []
for date in matched_dates:
    dq = _interp(cme_d.loc[date]) - _interp(der_d.loc[date])
    recs.append({'date': date, 'tercile': terc_map.get(date, np.nan),
                 'dm_down': np.trapezoid(dq[down_m], R_GRID[down_m]),
                 'dm_mid': np.trapezoid(dq[mid_m], R_GRID[mid_m]),
                 'dm_up': np.trapezoid(dq[up_m], R_GRID[up_m])})
mass_df = pd.DataFrame(recs).sort_values('date').reset_index(drop=True)
vals = mass_df[['dm_down', 'dm_mid', 'dm_up']].values.astype(float)
labels = mass_df['tercile'].astype(object).values

def _row(regime, band):
    return {'regime': regime, 'n_days': band['n_days'],
            'mean_down': band['mean'][0], 'lo_down': band['lo'][0], 'hi_down': band['hi'][0],
            'mean_mid': band['mean'][1], 'lo_mid': band['lo'][1], 'hi_mid': band['hi'][1],
            'mean_up': band['mean'][2], 'lo_up': band['lo'][2], 'hi_up': band['hi'][2]}

rows = [_row('unconditional', block_bootstrap_mean_bands(vals, block_length=27, B=1000, seed=42))]\n",
g = block_bootstrap_group_mean_bands(vals, labels, ['low', 'mid', 'high'],
                                     block_length=27, B=1000, seed=42)
for terc in ['low', 'mid', 'high']:
    if terc in g:
        rows.append(_row(terc, g[terc]))
mass_wedge = pd.DataFrame(rows)
display(mass_wedge.round(4))
mass_wedge.to_csv(PHASE5_TAB / 'regional_mass_wedge_summary.csv', index=False)
print(f"Sign: positive = CME assigns more mass to the region ({len(mass_df)} matched days).")
```
