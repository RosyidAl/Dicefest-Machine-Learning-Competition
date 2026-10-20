"""Adoption is constant within subzone on train. Check (1) exactly constant on train; (2) submission: seen-subzone rows equal the
train value; (3) unseen subzones with >=2 test rows: is the predicted adoption identical inside each subzone? If not,
averaging inside the subzone is a guaranteed SSE drop (true value constant). (4) clean ratio identity on train and in the submission."""
import numpy as np, pandas as pd
tr = pd.read_parquet("cache/train.parquet", columns=["subzone_id", "bld_dim_01", "bld_dim_02", "target_electric_heating_adoption", "target_clean_building_ratio"])
te = pd.read_parquet("cache/test.parquet", columns=["subzone_id", "bld_dim_01", "bld_dim_02"])
sb = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
a = "target_electric_heating_adoption"; c = "target_clean_building_ratio"
print("train adoption constant within every subzone:", bool((tr.groupby("subzone_id")[a].nunique() == 1).all()))
ref = tr.groupby("subzone_id")[a].first()
seen = te.subzone_id.isin(ref.index).values
print("test seen rows", seen.sum(), "max |sub - train value|", np.abs(sb.loc[seen, a].values - te.subzone_id[seen].map(ref).values).max())
u = te[~seen].assign(p=sb.loc[~seen, a].values)
sz = u.groupby("subzone_id").p.agg(["size", "nunique", "std"])
multi = sz[sz["size"] > 1]
print("unseen test rows", len(u), "unseen subzones", len(sz), "with >=2 rows", len(multi), "rows in them", int(multi["size"].sum()),
      "subzones with differing predictions", int((multi["nunique"] > 1).sum()))
if (multi["nunique"] > 1).any():
    u["avg"] = u.groupby("subzone_id").p.transform("mean")
    dsse = ((u.p - u.avg) ** 2).sum()
    base = 0.0414 * 7.18  # approx adoption RMSE (memory: NRMSE ~0.0414 overall)
    dn = np.sqrt(base ** 2 - dsse / len(te)) / 7.18 - base / 7.18
    print(f"guaranteed SSE drop {dsse:.3f}, NRMSE change ~{dn:+.7f}, total ~{dn / 5:+.7f}")
r = np.round(100 * tr.bld_dim_02 / tr.bld_dim_01, 2)
print("clean identity train exact:", bool(np.all(r.values == tr[c].values)), "| submission max diff vs identity:",
      np.abs(np.round(100 * te.bld_dim_02 / te.bld_dim_01, 2).values - sb[c].values).max())
