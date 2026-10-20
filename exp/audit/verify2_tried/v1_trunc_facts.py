"""Independent check (verify2_tried): is train gas hard-truncated at 1e6 kWh, and how sharp is the cut?
Reads only a few columns of the cached parquet files (light). Writes nothing except its log (stdout)."""
import numpy as np
import pandas as pd
from scipy import stats

GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
tr = pd.read_parquet("cache/train.parquet", columns=["id", GAS, EL, EUI, "bld_dim_06", "bld_dim_12"])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
n_tr, n_te = len(tr), len(te)
print(f"rows train {n_tr} test {n_te} total {n_tr + n_te} train share {n_tr / (n_tr + n_te):.7f} "
      f"0.7*total {0.7 * (n_tr + n_te):.3f} ceil(0.3*total) {int(np.ceil(0.3 * (n_tr + n_te)))}")
g = tr[GAS].values
A = tr["bld_dim_06"].values
M = tr["bld_dim_12"].values.astype(float)
print(f"gas max {g.max():.1f}  n>=1e6 {np.sum(g >= 1e6)}  n>=9.9e5 {np.sum(g >= 9.9e5)}  n>=9.8e5 {np.sum(g >= 9.8e5)}")
rel = np.abs(tr[EUI].values * A - g) / g
print(f"identity eui*A==gas: max rel diff {rel.max():.2e}")
# 10k bins 7e5..1.1e6
edges = np.arange(7e5, 1.10001e6, 1e4)
cnt, _ = np.histogram(g, edges)
print("gas counts per 10k bin from 700k:", cnt.tolist())
# log-linear fit of bin counts on [7e5, 1e6) (Poisson GLM via weighted lstsq on log counts), extrapolate above 1e6
mid = (edges[:-1] + edges[1:]) / 2
fit = mid < 1e6
X = np.column_stack([np.ones(fit.sum()), (mid[fit] - 1e6) / 1e5])
coef = np.linalg.lstsq(X * np.sqrt(cnt[fit])[:, None], np.log(np.maximum(cnt[fit], 0.5)) * np.sqrt(cnt[fit]),
                       rcond=None)[0]
exp_above = np.exp(coef[0] + coef[1] * (mid[~fit] - 1e6) / 1e5)
print(f"log-linear tail fit: slope per 100k {coef[1]:+.3f}; expected counts in the 10 bins above 1e6 if untruncated:",
      np.round(exp_above, 1).tolist(), f"sum {exp_above.sum():.0f}, observed {cnt[~fit].sum()}, "
      f"Poisson P(0 | mean) = {stats.poisson.pmf(0, exp_above.sum()):.2e}")
# same view for elec and EUI (should show no hard cut)
for name, v, e0, st in [("elec", tr[EL].values, 2.5e5, 1e4), ("eui", tr[EUI].values, 400.0, 20.0)]:
    ed = np.arange(e0, v.max() + 2 * st, st)
    c, _ = np.histogram(v, ed)
    print(f"{name} max {v.max():.2f}; counts per bin from {e0} step {st}:", c.tolist())
# per-meter view: is the cut on gas itself (not gas per meter / EUI)? share above 9e5 by M
for lo, hi in [(2, 30), (30, 50), (50, 70), (70, 90), (90, 200)]:
    m = (M >= lo) & (M < hi)
    gm = g[m] / M[m]
    # if the cut is on gas, the max of gas is ~1e6 in every M band where 1e6/M is inside the gas-per-meter range
    print(f"M in [{lo},{hi}): n={m.sum():6d} max gas {g[m].max():9.0f}  q99.9 gas {np.quantile(g[m], 0.999):9.0f} "
          f" max gas/M {gm.max():7.0f}  rows gas>=9.5e5 {np.sum(g[m] >= 9.5e5)}")
# test-side exposure: how many test rows have exposure such that the train rate distribution would put mass above 1e6
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
print(f"test A max {At.max():.0f} (train {A.max():.0f}); test M max {Mt.max():.0f} (train {M.max():.0f}); "
      f"test rows with M>=80: {np.sum(Mt >= 80)} (train {np.sum(M >= 80)})")
