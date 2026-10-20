"""Identifiability of per-meter twins (equal gas/M, different gas) and gas twins (equal gas):
rank of the true partner among all train rows by distance in several feature spaces, and among rows whose
gas/M is within +-5% of the partner's (an optimistic 'oracle-narrowed' candidate set)."""
import numpy as np, pandas as pd
cols = ["bld_dim_12", "bld_dim_01", "bld_dim_06", "bld_dim_03", "bld_dim_02", "bld_dim_17", "env_climate_01", "env_climate_03", "env_climate_05", "subzone_id"]
tr = pd.read_parquet("cache/train.parquet", columns=cols + ["target_macro_gas_kwh"])
g = tr.target_macro_gas_kwh.values; M = tr.bld_dim_12.values
sub_n = tr.subzone_id.str[4:].astype(int).values
q = np.round(g / M, 9); s = pd.Series(q); vc = s.map(s.value_counts()).values
gn = pd.Series(g).map(pd.Series(g).value_counts()).values
pm = np.where((vc == 2) & (gn == 1))[0]
cnt = pd.Series(q[pm]).map(pd.Series(q[pm]).value_counts()).values; pm = pm[cnt == 2]
pm = pm[np.argsort(q[pm], kind="stable")]; pa, pb = pm[0::2], pm[1::2]
assert np.all(q[pa] == q[pb])
X = np.column_stack([np.log(tr.bld_dim_06 / M), np.log(tr.bld_dim_01 / M), np.log(M), np.log(tr.bld_dim_03 / tr.bld_dim_06),
                     tr.bld_dim_02 / tr.bld_dim_01, tr.bld_dim_17, tr.env_climate_01, tr.env_climate_03, tr.env_climate_05, sub_n / 1e4])
X = (X - X.mean(0)) / X.std(0)
rng = np.random.default_rng(0); sel = rng.choice(len(pa), min(300, len(pa)), replace=False)
lq = np.log(g / M)
for name, cidx in [("per-meter stock", [0, 1, 2, 3, 4, 5]), ("climate+sub_n", [6, 7, 8, 9]), ("all", list(range(10)))]:
    rk, rk_n = [], []
    for i in sel:
        a, b = pa[i], pb[i]
        dd = ((X[:, cidx] - X[a, cidx]) ** 2).sum(1); dd[a] = np.inf
        rk.append((dd < dd[b]).sum())
        win = np.abs(lq - lq[a]) < 0.05; win[a] = False
        rk_n.append((dd[win] < dd[b]).sum())
    rk, rk_n = np.array(rk), np.array(rk_n)
    print(f"{name}: partner rank median {np.median(rk):.0f}, top-1 {np.mean(rk == 0):.3f}; within +-5% gas/M window: median rank {np.median(rk_n):.0f}, top-1 {np.mean(rk_n == 0):.3f}, window size median {np.median([np.sum(np.abs(lq - lq[pa[i]]) < 0.05) for i in sel[:50]]):.0f}")
