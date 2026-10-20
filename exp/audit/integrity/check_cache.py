"""Integrity audit 12: parquet cache == raw CSV (columns used by identities/dup key); subzone/zone id -> int one-to-one.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_cache.py
"""
import json
import numpy as np
import pandas as pd
HV = [f"hvac_dist_{i:02d}" for i in range(1, 11)]
cols = ["id", "subzone_id", "zone_region_id", "socio_demog_005", "bld_dim_01", "bld_dim_02", "bld_dim_06", "bld_dim_12"] + HV
out = {}
for part, extra in (("test", []), ("train", ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm",
                                             "target_clean_building_ratio", "target_electric_heating_adoption"])):
    raw = pd.read_csv(f"data/raw/{part}.csv", usecols=cols + extra, float_precision="round_trip")
    pq = pd.read_parquet(f"data/cache/{part}.parquet", columns=cols + extra)
    raw = raw[cols + extra]
    res = {}
    for c in cols + extra:
        a, b = raw[c].values, pq[c].values
        if a.dtype.kind in "fi":
            res[c] = int(np.sum(~((a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float))))))
        else:
            res[c] = int(np.sum(a.astype(str) != b.astype(str)))
    out[f"{part}_raw_vs_cache_mismatches"] = res
    del raw
tr = pd.read_parquet("data/cache/train.parquet", columns=["subzone_id", "zone_region_id"])
te = pd.read_parquet("data/cache/test.parquet", columns=["subzone_id", "zone_region_id"])
for c in ("subzone_id", "zone_region_id"):
    s = pd.concat([tr[c], te[c]]).drop_duplicates()
    n = s.str[4:].astype(int)
    out[f"{c}_unique_strings"] = int(len(s))
    out[f"{c}_unique_ints"] = int(n.nunique())
    out[f"{c}_prefixes"] = sorted(s.str[:4].unique().tolist())
    out[f"{c}_len_values"] = sorted(s.str.len().unique().tolist())
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_cache.json", "w"), indent=1)
