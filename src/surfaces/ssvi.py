"""
Surface Stochastic Volatility Inspired (SSVI) Model.

Implements the Power-Law parameterization of Gatheral & Jacquier (2014)
with strict enforcement of Butterfly and Calendar no-arbitrage constraints.

"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, brentq
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class SSVI:
    def __init__(self, df, venue='', date='', min_options_per_slice=4):
        if df.empty:
            raise ValueError(f"No data for {venue} on {date}")
        self.df = df.reset_index(drop=True)
        self.venue = venue
        self.date = date
        self.min_opts = min_options_per_slice
        self.res = None
        self._surface = None

    # Core SSVI functions based on Gatheral & Jacquier (2014)
    def _phi_power_law(self, theta, eta, gamma):
        """Power-law: phi(theta) = eta / (theta^gamma * (1+theta)^(1-gamma))."""
        theta_safe = np.maximum(theta, 1e-8)
        return eta / (np.power(theta_safe, gamma) * np.power(1 + theta_safe, 1 - gamma))

    def _ssvi_total_variance(self, k, theta, rho, phi):
        """Total variance w(k, theta) = (theta/2)(1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1-rho^2))."""
        p = phi * k
        inner = np.maximum(np.square(p + rho) + (1 - rho**2), 0)
        return (theta / 2.0) * (1 + rho * p + np.sqrt(inner))

    # No-arbitrage constraints from Gatheral & Jacquier (2014)
    def _no_butterfly_constraint(self, theta, params):
        rho, eta, gamma = params
        phi = self._phi_power_law(theta, eta, gamma)
        term1 = theta * phi * (1.0 + abs(rho))
        term2 = theta * phi**2 * (1.0 + abs(rho))
        return 4.0 - max(term1, term2)

    def _no_calendar_shape_constraint(self, params):
        rho, _, gamma = params
        if abs(rho) >= 1.0:
            return -1.0
        return (1.0 + np.sqrt(1.0 - rho**2)) - rho**2 * (1.0 - gamma)

    def _objective_smile(self, params, theta, k, T, iv_mkt):
        rho, eta, gamma = params
        phi = self._phi_power_law(theta, eta, gamma)
        w_model = self._ssvi_total_variance(k, theta, rho, phi)
        iv_model = np.sqrt(np.maximum(w_model, 1e-9) / T)
        return np.sum((iv_model - iv_mkt)**2) * 1e4

    # Calibration: Fit slice-by-slice across maturities, enforcing constraints
    def fit(self, max_iter=10000):
        k_all = self.df['log_moneyness'].values
        T_all = self.df['tau'].values
        iv_all = self.df['impliedvolatility'].values
        unique_T = np.sort(self.df['tau'].unique())

        # Step 1: Extract raw ATM total variances
        theta_map_raw = {}
        for T in unique_T:
            mask = T_all == T
            k = k_all[mask]
            if len(k) >= self.min_opts:
                iv_mkt = iv_all[mask]
                w_mkt_slice = (iv_mkt ** 2) * T

                sort_idx = np.argsort(k)
                k_sorted = k[sort_idx]
                w_sorted = w_mkt_slice[sort_idx]

                if (k_sorted <= 0).any() and (k_sorted > 0).any():
                    theta = np.interp(0.0, k_sorted, w_sorted)
                else:
                    theta = w_sorted[np.argmin(np.abs(k_sorted))]
                theta_map_raw[T] = theta

        if not theta_map_raw:
            raise ValueError(
                f"SSVI: no valid slice for {self.venue} on {self.date} "
                f"(all slices < {self.min_opts} options)"
            )

        # Step 2: Enforce ATM variance monotonicity
        Ts = np.array(sorted(theta_map_raw.keys()))
        thetas_raw = np.array([theta_map_raw[T] for T in Ts])
        iso = IsotonicRegression(increasing=True)
        thetas_mono = iso.fit_transform(Ts, thetas_raw)
        theta_map = {T: thetas_mono[i] for i, T in enumerate(Ts)}

        # Step 3: Calibrate smile parameters per slice
        rho_map, eta_map, gamma_map = {}, {}, {}
        last_params = np.array([-0.5, 0.5, 0.5])

        for i, T in enumerate(Ts):
            mask = T_all == T
            k = k_all[mask]
            iv_mkt = iv_all[mask]
            theta = theta_map[T]

            bounds = [
                (-1 + 1e-6, 1 - 1e-6),
                (1e-6, 4.0),
                (1e-6, 1.0),
            ]
            constraints = [
                {"type": "ineq",
                 "fun": lambda p, t=theta: self._no_butterfly_constraint(t, p)},
                {"type": "ineq",
                 "fun": lambda p: self._no_calendar_shape_constraint(p)},
            ]

            if i == 0:
                res = differential_evolution(
                    self._objective_smile,
                    bounds=bounds,
                    args=(theta, k, T, iv_mkt),
                    seed=42,
                )
            else:
                res = minimize(
                    self._objective_smile,
                    last_params,
                    args=(theta, k, T, iv_mkt),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": max_iter},
                )

            if res.success and abs(res.x[0]) < 0.99:
                last_params = res.x
                rho_map[T] = res.x[0]
                eta_map[T] = res.x[1]
                gamma_map[T] = res.x[2]

        if not rho_map:
            raise ValueError(
                f"SSVI: no slice converged for {self.venue} on {self.date}"
            )

        self.res = {
            "theta": theta_map,
            "rho": rho_map,
            "eta": eta_map,
            "gamma": gamma_map,
            "maturities": np.array(sorted(rho_map.keys())),
        }
        self._compile_surface()
        return self

    @classmethod
    def from_params(cls, params_day: pd.DataFrame, forward_map=None, venue: str = "", date=""):
        if params_day.empty:
            raise ValueError(f"from_params: empty parameter frame for {venue} on {date}")

        obj = cls.__new__(cls)
        obj.df = None
        obj.venue = venue or (str(params_day["venue"].iloc[0]) if "venue" in params_day.columns else "")
        obj.date = date if date != "" else (params_day["date"].iloc[0] if "date" in params_day.columns else "")
        obj.min_opts = None

        p = params_day.sort_values("tau")
        Ts = p["tau"].to_numpy(dtype=float)
        if len(np.unique(Ts)) != len(Ts):
            raise ValueError(f"from_params: duplicate maturities for {obj.venue} on {obj.date}")

        obj.res = {
            "theta": dict(zip(Ts, p["theta"].astype(float))),
            "rho": dict(zip(Ts, p["rho"].astype(float))),
            "eta": dict(zip(Ts, p["eta"].astype(float))),
            "gamma": dict(zip(Ts, p["gamma"].astype(float))),
            "maturities": Ts,
        }

        # Forward prices: prefer the saved column, else use the provided map
        if "forward" in p.columns and p["forward"].notna().all():
            F_vals = p["forward"].to_numpy(dtype=float)
        elif forward_map is not None:
            fm = pd.Series(dict(forward_map)).sort_index()
            fm_taus = fm.index.to_numpy(dtype=float)
            F_vals = np.empty(len(Ts))
            for i, T in enumerate(Ts):
                j = int(np.argmin(np.abs(fm_taus - T)))
                if abs(fm_taus[j] - T) > 1e-6:
                    raise ValueError(
                        f"from_params: no forward within 1e-6 of tau={T} "
                        f"for {obj.venue} on {obj.date}"
                    )
                F_vals[i] = float(fm.iloc[j])
        else:
            raise ValueError(
                "from_params requires a 'forward' column in params_day or a "
                "forward_map (e.g. df_day.groupby('tau')['forward_price'].mean())"
            )

        obj._compile_surface(F_vals=F_vals)
        return obj

    def _compile_surface(self, F_vals=None):
        Ts = self.res["maturities"]
        self._surface = {
            "Ts": Ts,
            "theta": np.array([self.res["theta"][T] for T in Ts]),
            "rho": np.array([self.res["rho"][T] for T in Ts]),
            "eta": np.array([self.res["eta"][T] for T in Ts]),
            "gamma": np.array([self.res["gamma"][T] for T in Ts]),
        }
        self._surface["sqrt_theta"] = np.sqrt(self._surface["theta"])

        # Forward price per fitted maturity: from the raw data when fitting,
        if F_vals is None:
            F_map = self.df.groupby("tau")["forward_price"].mean()
            F_vals = np.array([F_map.get(T, F_map.iloc[-1]) for T in Ts])
        else:
            F_vals = np.asarray(F_vals, dtype=float)
            if len(F_vals) != len(Ts):
                raise ValueError("_compile_surface: F_vals length mismatch")
        self._surface["F"] = F_vals
        self._forward_interp = interp1d(
            Ts, F_vals, kind="linear", fill_value="extrapolate"
        )
        self._surface["phi"] = np.array([
            self._phi_power_law(
                self._surface["theta"][i],
                self._surface["eta"][i],
                self._surface["gamma"][i]
            ) for i in range(len(Ts))
        ])

    # Evaluation
    def total_variance(self, T, k):
        """Returns w(T, k) using Lemma 5.1 convex interpolation."""
        s = self._surface
        Ts = s["Ts"]

        if T in Ts:
            idx = np.where(Ts == T)[0][0]
            return self._ssvi_total_variance(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])

        i = np.searchsorted(Ts, T)
        if i == 0 or i == len(Ts):
            idx = 0 if i == 0 else -1
            return self._ssvi_total_variance(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])

        T1, T2 = Ts[i - 1], Ts[i]
        theta_t = np.interp(T, [T1, T2], [s["theta"][i - 1], s["theta"][i]])
        denom = s["sqrt_theta"][i] - s["sqrt_theta"][i - 1]
        if abs(denom) < 1e-10:
            alpha = (T2 - T) / (T2 - T1)
        else:
            alpha = (s["sqrt_theta"][i] - np.sqrt(theta_t)) / denom

        def get_slice_price(idx):
            F = s["F"][idx]
            K = F * np.exp(k)
            w = self._ssvi_total_variance(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])
            vol_sqrt_t = np.sqrt(np.maximum(w, 1e-12))
            d1 = (np.log(F / K) + 0.5 * w) / vol_sqrt_t
            d2 = d1 - vol_sqrt_t
            return F * norm.cdf(d1) - K * norm.cdf(d2), K

        C1, K1 = get_slice_price(i - 1)
        C2, K2 = get_slice_price(i)
        F_t = float(self._forward_interp(T))
        K_t = F_t * np.exp(k)
        Ct = np.clip(
            K_t * (alpha * (C1 / K1) + (1 - alpha) * (C2 / K2)),
            max(F_t - K_t, 0.0) + 1e-9, F_t - 1e-9
        )

        def obj(w):
            vol_sqrt_t = np.sqrt(np.maximum(w, 1e-12))
            d1 = (np.log(F_t / K_t) + 0.5 * w) / vol_sqrt_t
            d2 = d1 - vol_sqrt_t
            return (F_t * norm.cdf(d1) - K_t * norm.cdf(d2)) - Ct

        return brentq(obj, 1e-9, 5.0)

    def get_iv(self, K, T, F, r=0.0):
        """Returns interpolated IV at (K, T) given forward F."""
        if F <= 1e-8 or K <= 1e-8 or T < 1e-6:
            return 0.0
        k = np.log(K / F)
        w = self.total_variance(T, k)
        return np.sqrt(max(w, 0.0) / T)

    def get_variance_grid(self, t_flat, k_flat):
        """Evaluates total variance on a flat grid (for Tensor PCA)."""
        w_out = np.zeros_like(t_flat)
        for i, (t, k) in enumerate(zip(t_flat, k_flat)):
            try:
                w_out[i] = self.total_variance(t, k)
            except Exception:
                w_out[i] = np.nan
        return w_out

    # Diagnostics
    def evaluate_fit(self):
        """Computes per-slice and aggregate RMSE."""
        if self.res is None:
            raise ValueError("Call .fit() before .evaluate_fit()")

        k_all = self.df['log_moneyness'].values
        T_all = self.df['tau'].values
        iv_all = self.df['impliedvolatility'].values
        Ts = self.res["maturities"]

        slice_rmses = {}
        total_se, total_n = 0.0, 0

        for T in Ts:
            mask = T_all == T
            k, iv_mkt = k_all[mask], iv_all[mask]
            theta = self.res["theta"][T]
            phi = self._phi_power_law(theta, self.res["eta"][T], self.res["gamma"][T])
            w_model = self._ssvi_total_variance(k, theta, self.res["rho"][T], phi)
            iv_model = np.sqrt(np.maximum(w_model, 1e-9) / T)
            se = np.sum((iv_model - iv_mkt) ** 2)
            n = len(iv_mkt)
            slice_rmses[round(T * 365.25)] = np.sqrt(se / n) if n > 0 else np.nan
            total_se += se
            total_n += n

        return {
            "date": str(self.date),
            "venue": self.venue,
            "rmse": np.sqrt(total_se / total_n) if total_n > 0 else np.nan,
            "n_slices": len(Ts),
            "n_options": total_n,
            "slice_rmses": slice_rmses,
        }

    def get_fitted_params(self):
        """Returns fitted parameters as list of dicts for DataFrame construction."""
        if self.res is None:
            raise ValueError("Call .fit() before .get_fitted_params()")
        rows = []
        for i, T in enumerate(self.res["maturities"]):
            rows.append({
                "date": self.date,
                "venue": self.venue,
                "tau": T,
                "days_to_expiry": round(T * 365.25),
                "theta": self.res["theta"][T],
                "rho": self.res["rho"][T],
                "eta": self.res["eta"][T],
                "gamma": self.res["gamma"][T],
                "phi": self._phi_power_law(
                    self.res["theta"][T], self.res["eta"][T], self.res["gamma"][T]
                ),
                # Saved so SSVI.from_params can reconstruct the surface
                # without access to the raw option data.
                "forward": float(self._surface["F"][i]),
            })
        return rows
    