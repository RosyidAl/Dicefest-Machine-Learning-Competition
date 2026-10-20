"""Integrity audit 4: duplicate rows. Is src/dup_fix.py KEY the widest safe key?

For several candidate keys (full feature row, row-level features only, dup_fix KEY, relaxed subsets):
  train-train: groups with >= 2 rows, share of those groups with identical targets (purity)
  test: rows matching >= 1 train row, and how many of them map to a pure train group
Also: train pairs sharing BOTH exact gas and exact elec (same postcode record by construction) -> which features differ?
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_dup.py
"""
import json

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm", "target_clean_building_ratio",
     "target_electric_heating_adoption"]
tr = pd.read_parquet("data/cache/train.parquet")
te = pd.read_parquet("data/cache/test.parquet")
feat = sorted(c for c in te.columns if c != "id")
nun = tr.groupby("subzone_id")[feat].nunique()
rowlvl = sorted(c for c in feat if (nun[c] > 1).any())  # varies within a subzone (postcode level)
bld = sorted(c for c in feat if c.startswith("bld_"))
DUPKEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
keys = {
    "full": feat,
    "rowlevel": rowlvl,
    "rowlevel+subzone": rowlvl + ["subzone_id"],
    "bld_all": bld,
    "bld_all+subzone": bld + ["subzone_id"],
    "dup_fix_KEY": DUPKEY,
    "s005": ["socio_demog_005"],
    "s005+subzone": ["socio_demog_005", "subzone_id"],
    "M+A+E+subzone": ["bld_dim_12", "bld_dim_06", "bld_dim_01", "subzone_id"],
    "M+A+E": ["bld_dim_12", "bld_dim_06", "bld_dim_01"],
    "M+A+E+zone": ["bld_dim_12", "bld_dim_06", "bld_dim_01", "zone_region_id"],
    "s005+M+A+E": ["socio_demog_005", "bld_dim_12", "bld_dim_06", "bld_dim_01"],
    "climate+M+A": ["env_climate_01", "env_climate_03", "env_climate_05", "bld_dim_12", "bld_dim_06"],
}
ntr = len(tr)
allx = pd.concat([tr[feat], te[feat]], ignore_index=True)
out = {"n_rowlevel": len(rowlvl), "n_bld": len(bld)}
tgt = tr[T].values
for name, cols in keys.items():
    h = hash_pandas_object(allx[cols], index=False).values
    htr, hte = h[:ntr], h[ntr:]
    s = pd.Series(htr)
    size = s.map(s.value_counts()).values
    multi = size > 1
    df = pd.DataFrame({"h": htr[multi]})
    for j, c in enumerate(T):
        df[c] = tgt[multi, j]
    g = df.groupby("h")
    nu = g[T].nunique()
    pure_all = (nu == 1).all(axis=1)
    pure_gas = nu[T[0]] == 1
    # test rows matching a train row
    match = np.isin(hte, htr)
    pure_set = set(nu.index[pure_all]) | set(s[size == 1].values)  # singletons are trivially "pure"
    pure_match = np.array([x in pure_set for x in hte[match]])
    out[name] = dict(n_cols=len(cols), train_rows_in_multi=int(multi.sum()), train_groups_multi=int(len(nu)),
                     groups_pure_all_targets=int(pure_all.sum()), groups_pure_gas=int(pure_gas.sum()),
                     purity_all=float(pure_all.mean()) if len(nu) else None,
                     test_rows_matching_train=int(match.sum()),
                     test_rows_matching_pure=int(pure_match.sum()))
    print(name, out[name], flush=True)

# exact (gas, elec) twins in train
ge = tr[[T[0], T[1]]]
hge = hash_pandas_object(ge, index=False).values
sg = pd.Series(hge)
sz = sg.map(sg.value_counts()).values
idx = np.where(sz > 1)[0]
out["gas_elec_twin_rows"] = int(len(idx))
hg = hash_pandas_object(tr[[T[0]]], index=False).values
sgg = pd.Series(hg)
out["gas_twin_rows"] = int((sgg.map(sgg.value_counts()).values > 1).sum())
he = hash_pandas_object(tr[[T[1]]], index=False).values
sge = pd.Series(he)
out["elec_twin_rows"] = int((sge.map(sge.value_counts()).values > 1).sum())
# for (gas, elec) twin groups: how many differ in the dup_fix key, and which features differ
hk = hash_pandas_object(tr[DUPKEY], index=False).values
grp = pd.DataFrame({"g": hge[idx], "i": idx})
diffcount = {}
n_same_key, n_groups = 0, 0
for _, ii in grp.groupby("g")["i"]:
    ii = ii.values
    n_groups += 1
    if len(set(hk[ii])) == 1:
        n_same_key += 1
    sub = tr.iloc[ii][feat]
    for c in feat:
        if sub[c].nunique(dropna=False) > 1:
            diffcount[c] = diffcount.get(c, 0) + 1
out["gas_elec_twin_groups"] = n_groups
out["gas_elec_twin_groups_same_dupkey"] = n_same_key
out["gas_elec_twin_cols_differing_top"] = dict(sorted(diffcount.items(), key=lambda kv: -kv[1])[:25])
print(json.dumps({k: v for k, v in out.items() if k.startswith("gas") or k.startswith("elec")}, indent=1))
json.dump(out, open("exp/audit/integrity/check_dup.json", "w"), indent=1)
