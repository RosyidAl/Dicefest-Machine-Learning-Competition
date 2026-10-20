"""Per-meter twins: different train rows with exactly equal gas/M (but different gas). Characterise partners:
same subzone/zone/cell? M, E, A relation? Do they also share elec/M or EUI? Is the partner identifiable from features?"""
import numpy as np, pandas as pd
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
cols = K0 + ["bld_dim_03", "bld_dim_02", "zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"]
T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
tr = pd.read_parquet("cache/train.parquet", columns=cols + T)
tr["sub_n"] = tr.subzone_id.str[4:].astype(int)
g = tr[T[0]].values; M = tr.bld_dim_12.values
q = pd.Series(np.round(g / M, 9)); tr["q"] = q.values
grp = tr[q.map(q.value_counts()).values > 1].copy()
grp = grp[grp.groupby("q")[T[0]].transform("nunique") > 1]   # keep groups where gas differs (not plain twins/dups)
print("groups", grp.q.nunique(), "rows", len(grp), "group size dist", grp.groupby("q").size().value_counts().to_dict())
G = grp.groupby("q")
for c in ["subzone_id", "zone_region_id", "env_climate_01", "bld_dim_12", "bld_dim_01", "bld_dim_06", "bld_dim_03", T[1], T[2]]:
    print(f"  identical {c}: {(G[c].nunique() == 1).mean():.3f}")
ex = grp.sort_values(["q", "sub_n"]).head(16)[["q", "sub_n", "zone_region_id", "bld_dim_12", "bld_dim_01", "bld_dim_06", T[0], T[1]]]
print(ex.round(2).to_string())
# M ratio within pairs
pairs = grp.groupby("q").filter(lambda x: len(x) == 2).sort_values("q")
a, b = pairs.iloc[0::2], pairs.iloc[1::2]
mr = a.bld_dim_12.values / b.bld_dim_12.values
print("M ratio a/b (sorted pair) distinct values head:", pd.Series(np.round(np.maximum(mr, 1 / mr), 4)).value_counts().head(8).to_dict())
print("sub_n |diff| quantiles", np.percentile(np.abs(a.sub_n.values - b.sub_n.values), [10, 50, 90]))
print("gas ratio = M ratio check", np.allclose(a[T[0]].values / b[T[0]].values, mr))
# also equal gas/E? equal E?
print("E equal in pairs", np.mean(a.bld_dim_01.values == b.bld_dim_01.values).round(3), "M==E rows share in pairs", np.mean(pairs.bld_dim_12 == pairs.bld_dim_01).round(3), "overall", np.mean(tr.bld_dim_12 == tr.bld_dim_01).round(3))
