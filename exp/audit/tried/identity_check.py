"""Quick integrity checks of sub_v19vce: clean ratio identity (train + submitted test values), adoption copy for seen
subzones (train constant within subzone), duplicate rows carry the train targets."""
import numpy as np
import pandas as pd

T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm", "target_clean_building_ratio",
     "target_electric_heating_adoption"]
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_01", "bld_dim_02", "subzone_id", T[3], T[4]])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_01", "bld_dim_02", "subzone_id"])
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
ok = np.round(100 * tr.bld_dim_02 / tr.bld_dim_01, 2) == tr[T[3]]
print("train clean identity exact:", ok.mean(), "min b01 train/test:", tr.bld_dim_01.min(), te.bld_dim_01.min())
print("test clean == identity:", (np.round(100 * te.bld_dim_02 / te.bld_dim_01, 2).values == sub[T[3]].values).mean())
nun = tr.groupby("subzone_id")[T[4]].nunique()
print("train subzones with >1 adoption value:", int((nun > 1).sum()), "of", len(nun))
ad = tr.groupby("subzone_id")[T[4]].first()
seen = te.subzone_id.isin(ad.index).values
print("test rows in seen subzones:", seen.mean().round(4), "adoption == train value:",
      (sub[T[4]].values[seen] == te.subzone_id[seen].map(ad).values).mean())
u = sub.loc[~seen, T[4]].groupby(te.subzone_id[~seen].values).nunique()
print("unseen subzones with >1 predicted value:", int((u > 1).sum()), "of", len(u))
