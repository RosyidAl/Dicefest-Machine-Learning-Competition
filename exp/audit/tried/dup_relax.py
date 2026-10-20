"""Could a relaxed duplicate key copy more test rows? For several keys: train groups with > 1 row, share of groups with
identical gas / elec, and number of test rows matching a train row (beyond the 77 exact duplicates)."""
import numpy as np
import pandas as pd

T = ["target_macro_gas_kwh", "target_macro_elec_kwh"]
K0 = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
cols = sorted(set(K0 + ["bld_dim_03", "bld_dim_02", "bld_dim_17", "zone_region_id"]))
tr = pd.read_parquet("cache/train.parquet", columns=cols + T)
te = pd.read_parquet("cache/test.parquet", columns=cols)
for key in (K0, ["subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"], ["bld_dim_12", "bld_dim_01", "bld_dim_06", "bld_dim_03"],
            ["socio_demog_005", "subzone_id"], ["socio_demog_005"], ["bld_dim_06", "bld_dim_03", "bld_dim_17"]):
    g = tr.groupby(key, sort=False)
    size = g.size()
    multi = size[size > 1]
    same = (g[T].nunique() == 1).all(axis=1)[multi.index]
    m = te[key].merge(size.rename("n").reset_index(), on=key, how="left")
    print(f"key {key}: train groups >1 row {len(multi)}, identical gas+elec {same.mean():.3f}; test rows matching train {int(m.n.notna().sum())}",
          flush=True)
