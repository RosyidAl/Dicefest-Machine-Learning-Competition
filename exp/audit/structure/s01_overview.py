"""Overview: what socio_demog_005 is, twins structure, groups of identical feature vectors."""
import numpy as np, pandas as pd
T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
B = [f"bld_dim_{i:02d}" for i in range(1, 44)]
import pyarrow.parquet as pq
names = pq.read_schema("cache/train.parquet").names
B = [b for b in B if b in names]
cols = sorted(B + ["socio_demog_005", "subzone_id", "zone_region_id", "env_climate_01", "env_climate_03", "env_climate_05"])
tr = pd.read_parquet("cache/train.parquet", columns=cols + T)
te = pd.read_parquet("cache/test.parquet", columns=cols)
print(tr.dtypes.value_counts())
for c in ["socio_demog_005", "bld_dim_12", "bld_dim_01", "bld_dim_06"]:
    print(c, "nunique train", tr[c].nunique(), "test", te[c].nunique(), "sample", tr[c].head(5).tolist())
# socio_demog_005 constant within subzone?
g = tr.groupby("subzone_id")["socio_demog_005"].nunique()
print("socio_demog_005 nunique per subzone: mean", g.mean(), "share==1", (g == 1).mean())
# twins on gas
gas = tr[T[0]]
vc = gas.value_counts()
tw = gas.map(vc) > 1
print("gas twins rows", tw.sum(), "share", tw.mean(), "groups", (vc > 1).sum(), "max group", vc.max())
print("top repeated gas values", vc.head(10).to_dict())
el = tr[T[1]]; ve = el.value_counts(); print("elec repeated rows", (el.map(ve) > 1).sum(), "top", ve.head(5).to_dict())
# twin pairs: compare features
t2 = tr[tw].copy(); t2["g"] = gas[tw]
grp = t2.groupby("g")
for c in ["bld_dim_12", "bld_dim_01", "bld_dim_06", "subzone_id", "zone_region_id", "socio_demog_005", T[1], "env_climate_01"]:
    print("twin groups with identical", c, round((grp[c].nunique() == 1).mean(), 4))
# M equal, gas/M equal
print("gas min", gas.min(), "share gas<1e5", (gas < 1e5).mean())
# full building vector identical within subzone
key = ["subzone_id"] + B
gb = tr.groupby(key, sort=False)
sz = gb.size(); mult = sz[sz > 1]
print("identical subzone+all bld groups>1:", len(mult))
m = te[key].merge(sz.rename("n").reset_index(), on=key, how="left")
print("test rows matching on subzone+all bld:", int(m.n.notna().sum()))
key2 = B
sz2 = tr.groupby(key2, sort=False).size()
print("identical all-bld groups>1 (any subzone):", int((sz2 > 1).sum()))
m2 = te[key2].merge(sz2.rename("n").reset_index(), on=key2, how="left")
print("test rows matching on all bld:", int(m2.n.notna().sum()))
