"""Twin pairs (exact equal gas): how similar are partners in each feature vs random pairs? Are they perturbed copies?"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm", "target_clean_building_ratio", "target_electric_heating_adoption"]
names = pq.read_schema("cache/train.parquet").names
num = [c for c in names if c not in T + ["id", "subzone_id", "zone_region_id", "bld_dim_18", "socio_demog_016"]]
tr = pd.read_parquet("cache/train.parquet", columns=num + ["subzone_id", "zone_region_id"] + T)
tr["sub_n"] = tr.subzone_id.str[4:].astype(int); tr["zone_n"] = tr.zone_region_id.str[4:].astype(int)
gas = tr[T[0]]
vc = gas.value_counts(); grp_id = gas.map(vc)
tw = tr[grp_id == 2].copy()
tw = tw.sort_values(T[0]); a = tw.iloc[0::2].reset_index(); b = tw.iloc[1::2].reset_index()
assert (a[T[0]].values == b[T[0]].values).all()
print("pairs", len(a))
rng = np.random.default_rng(0)
r = tr.sample(len(a), random_state=1).reset_index(); r2 = tr.sample(len(a), random_state=2).reset_index()
res = []
for c in num + ["sub_n", "zone_n", T[1], T[2], T[3], T[4], "index"]:
    x, y = a[c].values.astype(float), b[c].values.astype(float)
    xr, yr = r[c].values.astype(float), r2[c].values.astype(float)
    eq = np.mean(x == y); eqr = np.mean(xr == yr)
    cc = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else np.nan
    res.append((c, eq, eqr, cc))
df = pd.DataFrame(res, columns=["col", "eq_twin", "eq_rand", "corr_twin"]).sort_values("corr_twin", ascending=False)
df.to_csv("exp/audit/structure/s03_twin_featsim.csv", index=False)
print(df.head(25).to_string()); print(df.tail(8).to_string())
d = np.abs(a.sub_n - b.sub_n); print("sub_n |diff| quantiles", np.percentile(d, [5, 25, 50, 75, 95]))
dr = np.abs(r.sub_n - r2.sub_n); print("random sub_n |diff| quantiles", np.percentile(dr, [5, 25, 50, 75, 95]))
print("ratio M quantiles", np.percentile(a.bld_dim_12 / b.bld_dim_12, [5, 25, 50, 75, 95]))
print("elec ratio quantiles", np.percentile(a[T[1]] / b[T[1]], [5, 25, 50, 75, 95]))
print("index diff quantiles", np.percentile(np.abs(a["index"] - b["index"]), [5, 50, 95]))
