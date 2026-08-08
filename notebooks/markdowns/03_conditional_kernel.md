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

DATA_P1 = get_path('data_phase1')
DATA_P2 = get_path('data_phase2')
DATA_P3 = get_path('data_phase3')
RES_P3 = get_path('results_phase3')

TAB = RES_P3 / 'tables'
FIG = RES_P3 / 'figures'

summary = pd.read_csv(TAB / 'phase3_summary.csv')
print(summary[['venue','spec','n_days','n_params','kl_mean','converged',
               'mean_c','c_low','c_mid','c_high']].to_string(index=False))

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

    venue   spec  n_days  n_params   kl_mean  converged     mean_c      c_low      c_mid     c_high
      CME crypto     619        12 -0.281161       True -13.029793 -17.187712 -11.898411  -9.560905
      CME  macro     619        15 -0.277692       True -10.540581 -11.199003  -8.402117 -11.816768
      CME   full     619        24 -0.288710       True -12.260382 -16.030331 -11.754511  -8.630736
      DER crypto     619        12 -0.251333       True -12.380871 -16.602760 -11.522762  -8.587787
      DER  macro     619        15 -0.249922       True  -9.825687 -10.872588  -7.979342 -10.407116
      DER   full     619        24 -0.260586       True -11.346126 -15.520960 -10.765121  -7.346047
    

#### 1. Estimation Diagnostics

Key checks: convergence status, KL mean (lower = better fit), and the
curvature coefficient across terciles.

Notes: (i) the reported `kl_mean` is a cross-entropy (not a true KL
divergence), so negative values are expected and do not indicate a
numerical problem; (ii) do NOT read hump presence off the marginal `c`
in the cubic family — the economically meaningful object is the
curvature at the money (2c + 6d at R = 1), tested formally by
`joint_regime_test.py`.


```python
display_cols = ["venue", "spec", "n_days", "n_params", "kl_total", "kl_mean",
                "converged", "grad_inf", "c_low", "c_mid", "c_high"]
avail = [c for c in display_cols if c in summary.columns]
print(summary[avail].round(4).to_string(index=False))
```

    venue   spec  n_days  n_params  kl_total  kl_mean  converged  grad_inf    c_low    c_mid   c_high
      CME crypto     619        12 -174.0386  -0.2812       True       0.0 -17.1877 -11.8984  -9.5609
      CME  macro     619        15 -171.8916  -0.2777       True       0.0 -11.1990  -8.4021 -11.8168
      CME   full     619        24 -178.7115  -0.2887       True       0.0 -16.0303 -11.7545  -8.6307
      DER crypto     619        12 -155.5751  -0.2513       True       0.0 -16.6028 -11.5228  -8.5878
      DER  macro     619        15 -154.7014  -0.2499       True       0.0 -10.8726  -7.9793 -10.4071
      DER   full     619        24 -161.3029  -0.2606       True       0.0 -15.5210 -10.7651  -7.3460
    

#### 2. Kernel at Volatility Terciles (crypto spec)

Kernels are normalized **mean-one under the Phase 2 physical density**
(the model's own normalization, m = q~/p), so they are directly
comparable to the Phase 2 unconditional q/p kernel. Tercile labels are
loaded from the saved `.npz` (full-sample Z_IVS_1 `pd.qcut(q=3)`,
shared with Phases 4 and 5); if the `.npz` predates the fix, labels are
recomputed by date-aligning Z_crypto with the saved dates.


```python
from src.phase3.conditional_kernel import (
    ConditionalKernelResult, evaluate_kernel_at_terciles
)
from src.config import get_return_grid

R_GRID = get_return_grid()
colors = {'low': 'C2', 'mid': 'C7', 'high': 'C3'}

p_phys = np.load(DATA_P2 / 'phase2_densities.npz')['p_almeida']
Z_crypto = pd.read_parquet(DATA_P1 / 'Z_crypto.parquet')

Z_crypto['date'] = pd.to_datetime(Z_crypto['date'])
z_cols = [c for c in Z_crypto.columns if c != 'date' and not c.endswith('_raw')]
Z_crypto['tercile'] = pd.qcut(Z_crypto['Z_IVS_1'], q=3, labels=['low', 'mid', 'high'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, venue in zip(axes, ['CME', 'DER']):
    key = f'{venue}_crypto'
    data = np.load(DATA_P3 / f'phase3_{key}.npz', allow_pickle=True)
    row = summary[(summary['venue']==venue) & (summary['spec']=='crypto')].iloc[0]
    dates_run = pd.to_datetime(data['dates'])

    if 'tercile_labels' in data:
        tercile_labels = data['tercile_labels'].astype(object)
    else:
        run_df = pd.DataFrame({'date': dates_run})
        merged = run_df.merge(Z_crypto[['date', 'tercile']], on='date', how='inner')
        tercile_labels = merged['tercile'].astype(object).values
        print(f"  [{venue}] WARNING: npz has no tercile_labels — recomputed "
              f"from date-aligned Z_crypto ({len(merged)} days)")

    run_df = pd.DataFrame({'date': dates_run})
    merged_z = run_df.merge(Z_crypto[['date'] + z_cols], on='date', how='inner')
    Z_mat = merged_z[z_cols].values

    result = ConditionalKernelResult(
        theta=data['theta'], n_params=int(row['n_params']),
        n_days=int(row['n_days']), n_Z=len(z_cols), venue=venue,
        spec_name='crypto', kl_total=row['kl_total'],
        kl_mean=row['kl_mean'], converged=row['converged'],
        grad_inf=float(row.get('grad_inf', np.nan)),
        hessian_inv=None, message="", status=0)

    terciles = evaluate_kernel_at_terciles(result, R_GRID, Z_mat,
                                           tercile_labels=tercile_labels,
                                           p_phys=p_phys)
    for name in ['low', 'mid', 'high']:
        if name not in terciles:
            continue
        t = terciles[name]
        ax.plot(R_GRID, t['kernel'], color=colors[name], lw=1.5,
                label=f"{name}-vol (c={t['c']:.3f}, n={t['n_days']})")
        if 'p_mean_check' in t:
            assert abs(t['p_mean_check'] - 1.0) < 1e-6
    ax.axhline(1.0, color='gray', lw=0.5, ls=':')
    ax.axvline(1.0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Gross return $R$')
    ax.set_xlim(0.5, 1.6)
    ax.set_ylim(0, 2)
    ax.set_title(f'{venue} — crypto spec')
    ax.legend(fontsize=9)
axes[0].set_ylabel(r'$\hat{m}^j(R \mid Z_t)$  (mean-one under $\hat{p}$)')
fig.suptitle('Conditional Pricing Kernel by Volatility Tercile', fontsize=13)
plt.tight_layout()
plt.savefig(FIG / 'fig_conditional_kernel_terciles_crypto.png', dpi=300)
plt.show()
```


    
![png](03_conditional_kernel_files/03_conditional_kernel_5_0.png)
    


#### 2b. Joint Regime Test: H0 (b,c,d)_low = (b,c,d)_high



```python
joint = show_table(
    TAB / "joint_regime_test_crypto.csv",
    cols=["venue", "wald_stat", "p_value", "frac_vector_consistent",
          "delta_b", "delta_c", "delta_d", "B_effective"],
    title="Joint bootstrap Wald test (crypto spec, low vs high vol)",
    produced_by="joint_regime_test.py",
)

detail = show_table(
    TAB / "joint_regime_detail_crypto.csv",
    cols=["venue", "test", "coef", "diff_point", "ci_lo", "ci_hi",
          "frac_negative", "se_boot", "B_effective"],
    title="Per-coefficient and curvature-at-money contrasts (low minus high)",
    produced_by="joint_regime_test.py",
)

if detail is not None and "test" in detail.columns:
    curv = detail[detail["test"] == "curv_at_money_diff"]
    if len(curv):
        print("\n--- Curvature at money, 2c + 6d at R = 1 (low minus high) ---")
        for _, r in curv.iterrows():
            print(f"  {r['venue']}: {r['diff_point']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]   "
                  f"P(diff < 0) = {r['frac_negative']:.3f}")
```

    --- Joint bootstrap Wald test (crypto spec, low vs high vol) ---
    venue  wald_stat  p_value  frac_vector_consistent  delta_b  delta_c  delta_d  B_effective
      CME     18.876    0.019                   0.917  12.8816  -7.6268   0.9115         1000
      DER     10.663    0.045                   0.975  13.0792  -8.0150   1.0874         1000
    
    [2 rows]  source: joint_regime_test_crypto.csv
    --- Per-coefficient and curvature-at-money contrasts (low minus high) ---
    venue               test      coef  diff_point    ci_lo   ci_hi  frac_negative  se_boot  B_effective
      CME      per_coef_diff         b     12.8816   7.4503 30.2500          0.000   6.2547         1000
      CME      per_coef_diff         c     -7.6268 -20.0204 -2.8669          0.994   4.6328         1000
      CME      per_coef_diff         d      0.9115  -0.6638  3.7451          0.083   1.0909         1000
      CME curv_at_money_diff 2c+6d@R=1     -9.7845 -20.1178 -7.4497          1.000   3.6017         1000
      DER      per_coef_diff         b     13.0792   8.3867 29.3116          0.000   5.8277         1000
      DER      per_coef_diff         c     -8.0150 -19.2889 -4.2148          0.999   4.1606         1000
      DER      per_coef_diff         d      1.0874   0.0071  3.6493          0.025   0.9107         1000
      DER curv_at_money_diff 2c+6d@R=1     -9.5056 -19.4454 -7.0894          1.000   3.5103         1000
    
    [8 rows]  source: joint_regime_detail_crypto.csv
    
    --- Curvature at money, 2c + 6d at R = 1 (low minus high) ---
      CME: -9.784 [-20.118, -7.450]   P(diff < 0) = 1.000
      DER: -9.506 [-19.445, -7.089]   P(diff < 0) = 1.000
    

#### 2c. Bootstrap Intervals on Tercile Coefficients


```python
ci = show_table(
    TAB / "phase3_bootstrap_ci_crypto.csv",
    cols=["venue", "spec", "tercile", "coef", "point", "ci_lo", "ci_hi",
          "se_boot", "frac_negative", "B_effective", "block_length"],
    title="Percentile intervals at fixed full-sample tercile-mean states",
    sort=["venue", "coef", "tercile"],
    produced_by="run_phase3_bootstrap.py",
)

if ci is not None and "B_effective" in ci.columns:
    print(f"\nEffective replicates: {sorted(ci['B_effective'].unique())}")
```

    --- Percentile intervals at fixed full-sample tercile-mean states ---
    venue   spec tercile coef    point    ci_lo    ci_hi  se_boot  frac_negative  B_effective  block_length
      CME crypto    high    b   7.5940  -0.8115  14.9738   4.0513          0.038         1000            27
      CME crypto     low    b  20.4756  14.5511  33.0260   4.4038          0.000         1000            27
      CME crypto     mid    b  13.3931   8.7613  20.5756   2.9712          0.000         1000            27
      CME crypto    high    c  -9.5609 -15.4768  -2.8897   3.1493          0.994         1000            27
      CME crypto     low    c -17.1877 -26.6180 -12.2773   3.5145          1.000         1000            27
      CME crypto     mid    c -11.8984 -18.1221  -7.2429   2.6328          1.000         1000            27
      CME crypto    high    d   3.3565   1.7269   4.9633   0.7785          0.000         1000            27
      CME crypto     low    d   4.2681   2.5675   6.5374   0.9516          0.000         1000            27
      CME crypto     mid    d   3.1096   1.5609   4.8278   0.7954          0.001         1000            27
      DER crypto    high    b   5.8832  -2.6309  11.5631   3.7223          0.078         1000            27
      DER crypto     low    b  18.9624  13.4595  29.4775   3.9487          0.000         1000            27
      DER crypto     mid    b  12.1665   7.7881  17.9431   2.6011          0.000         1000            27
      DER crypto    high    c  -8.5878 -13.1770  -2.3661   2.8173          0.991         1000            27
      DER crypto     low    c -16.6028 -24.2751 -12.2965   3.0216          1.000         1000            27
      DER crypto     mid    c -11.5228 -16.5670  -7.5334   2.2515          1.000         1000            27
      DER crypto    high    d   3.2656   1.7424   4.4581   0.6665          0.000         1000            27
      DER crypto     low    d   4.3530   2.9628   6.1470   0.7840          0.000         1000            27
      DER crypto     mid    d   3.2421   1.9161   4.5902   0.6734          0.000         1000            27
    
    [18 rows]  source: phase3_bootstrap_ci_crypto.csv
    
    Effective replicates: [np.int64(1000)]
    

#### 3. Coefficient Time Series (crypto spec)


```python
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for venue, col_offset in [('CME', 0), ('DER', 1)]:
    data = np.load(DATA_P3 / f'phase3_{venue}_crypto.npz', allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    for row_idx, (coeff, label) in enumerate([
        ('coeffs_b', '$b_t$ (slope)'), ('coeffs_c', '$c_t$ (curvature)')
    ]):
        ax = axes[row_idx, col_offset]
        ax.plot(dates, data[coeff], lw=0.7, color='C0')
        ax.axhline(0, color='black', lw=0.4)
        ax.set_ylabel(label)
        if row_idx == 0:
            ax.set_title(f'{venue} — crypto spec')
for ax in axes[-1]:
    ax.set_xlabel('Date')
plt.suptitle('Key Kernel Coefficients Over Time', fontsize=13)
plt.tight_layout()
plt.savefig(FIG / 'fig_kernel_coeffs_over_time_crypto.png', dpi=300)
plt.show()
```


    
![png](03_conditional_kernel_files/03_conditional_kernel_11_0.png)
    


#### 4. Unconditional Microstructure Friction Kernel

$\Psi(R) = \log(\hat{q}^{\mathrm{CME}}(R) / \hat{q}^{\mathrm{DER}}(R))$.
Positive values mean CME prices that return state more expensively than
Deribit. `mfk_std` in the npz is the bootstrap SE of the mean (std of B
bootstrap means), NOT the raw daily std — it is used directly, never
divided by sqrt(n) again. The saved `mfk_lo`/`mfk_hi` percentile bands
are preferred when available.


```python
mfk = np.load(DATA_P3 / 'mfk_unconditional.npz')
R_mfk = mfk['R_grid']
psi = mfk['mfk_mean']

if 'mfk_lo' in mfk and 'mfk_hi' in mfk:
    ci_lo, ci_hi = mfk['mfk_lo'], mfk['mfk_hi']
    ci_label = '95% block-bootstrap CI'
else:
    psi_se = mfk['mfk_std']   # already SE of the mean
    ci_lo, ci_hi = psi - 1.96 * psi_se, psi + 1.96 * psi_se
    ci_label = '95% CI (normal approx)'

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(R_mfk, psi, 'k-', lw=1.5, label=r'$\bar{\Psi}(R)$')
ax.fill_between(R_mfk, ci_lo, ci_hi, alpha=0.2, color='gray', label=ci_label)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(1.0, color='gray', lw=0.5, ls=':')
ax.axvspan(R_mfk[0], 0.90, alpha=0.05, color='red')
ax.axvspan(1.10, R_mfk[-1], alpha=0.05, color='green')
ax.set_xlabel('Gross return $R$')
ax.set_ylabel(r'$\Psi(R)=\log(\hat{q}^{\mathrm{DER}}/\hat{q}^{\mathrm{CME}})$')
ax.set_xlim(0.5, 1.6)
ax.set_ylim(-0.2, 0.2)
ax.set_title(f'Unconditional Microstructure Friction Kernel ({int(mfk["n_days"])} matched days)')
ax.legend()
plt.tight_layout()
plt.show()
```


    
![png](03_conditional_kernel_files/03_conditional_kernel_13_0.png)
    


#### 5. Specification Comparison: KL Fit Quality


```python
kl_pivot = summary.pivot(index='venue', columns='spec', values='kl_mean')
kl_pivot = kl_pivot[['macro', 'crypto', 'full']]

fig, ax = plt.subplots(figsize=(8, 4))
kl_pivot.plot(kind='bar', ax=ax)
ax.set_ylabel('Mean cross-entropy (lower = better fit)')
ax.set_title('Conditioning Specification Comparison')
ax.legend(title='Specification')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG / 'fig_kl_comparison.png', dpi=300)
plt.show()

print('\nConvergence status:')
print(summary.pivot(index='venue', columns='spec', values='converged').to_string())
```


    
![png](03_conditional_kernel_files/03_conditional_kernel_15_0.png)
    


    
    Convergence status:
    spec   crypto  full  macro
    venue                     
    CME      True  True   True
    DER      True  True   True
    

#### 6. Martingale Check


```python
import numpy as np
from src.config import get_return_grid
from src.phase3.conditional_kernel import (
    _compute_coefficients, _log_kernel, _normalize_kernel)
from src.phase3.run_phase3 import (
    load_daily_rnds_from_parquet, load_conditioning_spec, align_rnds_and_Z)
from src.phase2.run_phase2 import intersect_venue_dates

R = get_return_grid()
p = np.load(DATA_P2 / "phase2_densities.npz")["p_almeida"]
z_dates, Z_full, _ = load_conditioning_spec("crypto")

# Intersect to the common matched days (same as estimation)
_raw = {v: load_daily_rnds_from_parquet(v, tau_days=27) for v in ["CME", "DER"]}
cme_d, cme_r, der_d, der_r = intersect_venue_dates(*_raw["CME"], *_raw["DER"])
per_venue = {"CME": (cme_d, cme_r), "DER": (der_d, der_r)}

for venue in ["CME", "DER"]:
    theta = np.load(DATA_P3 / f"phase3_{venue}_crypto.npz")["theta"]
    rnd_dates, rnds = per_venue[venue]
    _, arnds, Z, _ = align_rnds_and_Z(rnd_dates, rnds, z_dates, Z_full)
    n_Z = Z.shape[1]

    mm, mo = [], []
    for t in range(Z.shape[0]):
        b, c, d = _compute_coefficients(theta, Z[t], n_Z)
        m = _normalize_kernel(_log_kernel(R, b, c, d), R, p_phys=p)
        qt = p * m
        qt = qt / np.trapezoid(qt, R)
        mm.append(np.trapezoid(R * qt, R))
        mo.append(np.trapezoid(R * arnds[t], R))

    mm, mo = np.array(mm), np.array(mo)
    print(f"{venue}:  model-implied E^Q[R]  mean={mm.mean():.4f}  "
          f"[{mm.min():.4f}, {mm.max():.4f}]")
    print(f"       extracted q_t E^Q[R]  mean={mo.mean():.4f}  "
          f"[{mo.min():.4f}, {mo.max():.4f}]")
    print(f"       mean absolute gap: {np.abs(mm - mo).mean():.4f}\n")
```

      [CME] Loaded 624 daily RNDs at tau=27d
      [DER] Loaded 1317 daily RNDs at tau=27d
      Intersected to 619 common matched days (CME 624 -> 619, DER 1317 -> 619)
    CME:  model-implied E^Q[R]  mean=1.0000  [0.8349, 1.0255]
           extracted q_t E^Q[R]  mean=1.0000  [0.9561, 1.0269]
           mean absolute gap: 0.0039
    
    DER:  model-implied E^Q[R]  mean=1.0002  [0.8535, 1.0265]
           extracted q_t E^Q[R]  mean=1.0002  [0.9467, 1.0399]
           mean absolute gap: 0.0042
    
    

#### 7. Robustness: KDE Tilting Density



```python
kde = show_table(
    TAB / "kde_tilt_robustness.csv",
    cols=["venue", "tercile", "n_days",
          "b_enh", "c_enh", "d_enh", "curv_enh",
          "b_kde", "c_kde", "d_kde", "curv_kde",
          "curv_delta", "curv_sign_agrees"],
    title="Tercile coefficients under enhanced vs KDE tilting density",
    sort=["venue", "tercile"],
    produced_by="run_phase3_kde_robustness.py",
)

if kde is not None and "curv_sign_agrees" in kde.columns:
    n_agree = int(kde["curv_sign_agrees"].sum())
    print(f"\nCurvature sign agreement: {n_agree}/{len(kde)} venue-tercile cells")
```

    --- Tercile coefficients under enhanced vs KDE tilting density ---
    venue tercile  n_days   b_enh    c_enh  d_enh  curv_enh    b_kde    c_kde   d_kde  curv_kde  curv_delta  curv_sign_agrees
      CME    high     205  7.5940  -9.5609 3.3565    1.0174  -7.7660   5.4022 -1.3274    2.8399      1.8225              True
      CME     low     223 20.4756 -17.1877 4.2681   -8.7670  15.1398 -11.6849  2.4168   -8.8688     -0.1019              True
      CME     mid     191 13.3931 -11.8984 3.1097   -5.1389   4.8562  -3.4937  0.4303   -4.4056      0.7333              True
      DER    high     205  5.8832  -8.5878 3.2656    2.4181 -11.0033   7.8847 -1.8947    4.4010      1.9829              True
      DER     low     223 18.9624 -16.6028 4.3530   -7.0874  11.7531  -9.1980  1.8819   -7.1047     -0.0173              True
      DER     mid     191 12.1665 -11.5228 3.2421   -3.5928   1.7375  -1.2065 -0.0567   -2.7531      0.8397              True
    
    [6 rows]  source: kde_tilt_robustness.csv
    
    Curvature sign agreement: 6/6 venue-tercile cells
    

#### 8. Episode-Based Regime Robustness (exact intervals)


```python
ep = show_table(
    TAB / "regime_episodes_crypto.csv",
    cols=["venue", "episode", "start", "end", "n_days_state",
          "curv_at_money", "curv_lo", "curv_hi", "interval_method"],
    title="Curvature at money by volatility episode",
    produced_by="run_regime_episodes.py",
)

contrasts = show_table(
    TAB / "regime_episode_contrasts_crypto.csv",
    cols=["venue", "high_episode", "curv_high", "curv_calm",
          "curv_diff_calm_minus_high", "diff_lo", "diff_hi", "P_diff_lt_0"],
    title="Separation test: calm minus high curvature contrast",
    produced_by="run_regime_episodes.py",
)
```

    --- Curvature at money by volatility episode ---
    venue episode      start        end  n_days_state  curv_at_money  curv_lo  curv_hi interval_method
      CME  high_2 2020-03-16 2020-06-02            33         3.5628   0.9188  17.4969     exact_theta
      CME  high_1 2020-11-24 2021-12-23           226        -0.2643  -3.4343   3.5665     exact_theta
      CME    calm 2022-10-04 2023-08-31           219        -8.4593 -14.5316  -5.5421     exact_theta
      DER  high_2 2020-03-16 2020-06-02            33         4.8515   2.2642  18.4665     exact_theta
      DER  high_1 2020-11-24 2021-12-23           226         1.1931  -1.5308   5.0660     exact_theta
      DER    calm 2022-10-04 2023-08-31           219        -6.7861 -12.5745  -3.8300     exact_theta
    
    [6 rows]  source: regime_episodes_crypto.csv
    --- Separation test: calm minus high curvature contrast ---
    venue high_episode  curv_high  curv_calm  curv_diff_calm_minus_high  diff_lo  diff_hi  P_diff_lt_0
      CME       high_2     3.5628    -8.4593                   -12.0221 -30.5914  -9.2404          1.0
      CME       high_1    -0.2643    -8.4593                    -8.1950 -16.6669  -6.1356          1.0
      DER       high_2     4.8515    -6.7861                   -11.6376 -29.4642  -8.7586          1.0
      DER       high_1     1.1931    -6.7861                    -7.9793 -16.4047  -5.9065          1.0
    
    [4 rows]  source: regime_episode_contrasts_crypto.csv
    
