"""Does the exact dup key K0 miss duplicates whose float columns differ only by rounding noise?
Rounded versions of the key (A to 6 dp, socio_demog_005 to 10 dp) and partial keys; count train groups / identical targets / test matches."""
import numpy as np, pandas as pd
T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
cols = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=cols + T); te = pd.read_parquet("cache/test.parquet", columns=cols)
for df in (tr, te):
    df["A6"] = df.bld_dim_06.round(6); df["A2"] = df.bld_dim_06.round(2); df["s10"] = df.socio_demog_005.round(10); df["s6"] = df.socio_demog_005.round(6)
for key in (["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"], ["s10", "subzone_id", "bld_dim_12", "bld_dim_01", "A6"],
            ["s6", "subzone_id", "A2"], ["s6", "subzone_id"], ["subzone_id", "bld_dim_12", "bld_dim_01", "A2"]):
    sub = tr[(tr.socio_demog_005 != 1.0)] if "s6" in key else tr
    gb = sub.groupby(key); sz = gb.size(); multi = sz[sz > 1]
    same = (gb[T].nunique() == 1).all(axis=1)[multi.index].mean() if len(multi) else np.nan
    m = te[key].merge(sz.rename("n").reset_index(), on=key, how="left")
    extra = ""
    # gas near-equality (rel 1e-9) inside groups
    print(f"{key}: train groups>1 {len(multi)}, all-target identical share {same:.3f}, test rows matching {int(m.n.notna().sum())}")
