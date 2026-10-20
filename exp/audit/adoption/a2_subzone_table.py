"""Adoption lens, check 2: subzone-level table (labelled + unlabelled) with exact household counts.

Writes exp/audit/adoption/sz.parquet: one row per subzone (train or test), the 126 subzone-constant columns (float64,
raw values), means of the postcode-level columns, row counts, N, K, rest, count_r, r, q (labelled only).
usage: .venv/bin/python exp/audit/adoption/a2_subzone_table.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import HVAC  # noqa: E402
from adopt_v2 import household_count  # noqa: E402

AD = "target_electric_heating_adoption"
OUT = "exp/audit/adoption"
const = json.load(open(f"{OUT}/const_cols.json"))["const_cols"]
tr = pd.read_parquet("cache/train.parquet")
te = pd.read_parquet("cache/test.parquet")
cols = [c for c in te.columns if c not in ("id", "subzone_id", "zone_region_id")]
pc = [c for c in cols if c not in const]
al = pd.concat([tr[["subzone_id", "zone_region_id"] + cols], te[["subzone_id", "zone_region_id"] + cols]],
               ignore_index=True)
al["bld_dim_18"] = al["bld_dim_18"].astype("category").cat.codes
al["socio_demog_016"] = (al["socio_demog_016"] == "Urban").astype(float)
g = al.groupby("subzone_id", sort=True)
S = g[const].first()
num_pc = [c for c in pc if c not in ("bld_dim_18", "socio_demog_016")]
S = S.join(g[num_pc].mean().add_suffix("_m"))
S["nrow"] = g.size()
S["nrow_tr"] = tr.groupby("subzone_id").size().reindex(S.index).fillna(0).astype(int)
S["nrow_te"] = te.groupby("subzone_id").size().reindex(S.index).fillna(0).astype(int)
S["zone"] = g["zone_region_id"].first().str[4:].astype(int)
S["sub_n"] = S.index.str[4:].astype(int)
S["N"] = household_count(S[HVAC].values)
S["rest"] = 100 - S[HVAC].sum(axis=1)
S["K_raw"] = S.rest * S.N / 100
S["K"] = np.round(S.K_raw).astype(int)
S["ad"] = tr.groupby("subzone_id")[AD].first().reindex(S.index)
S["cr_raw"] = S.K_raw - S.ad * S.N / 100
S["cr"] = np.round(S.cr_raw)
S["r"] = S.rest - S.ad
S["q"] = S.cr / S.K.where(S.K > 0)
S.to_parquet(f"{OUT}/sz.parquet")
lab = S.ad.notna()
print("subzones", len(S), "labelled", int(lab.sum()), "unlabelled", int((~lab).sum()))
print("N==0 (not found)", int((S.N == 0).sum()), "N quantiles", S.N.quantile([0, .01, .1, .5, .9, .99, 1]).tolist())
print("K_raw max dev from int", float(np.abs(S.K_raw - S.K).max()))
print("cr_raw max dev from int (labelled)", float(np.abs(S.cr_raw - S.cr)[lab].max()),
      "n dev>1e-6", int((np.abs(S.cr_raw - S.cr)[lab] > 1e-6).sum()))
print("cr<0", int((S.cr[lab] < 0).sum()), "cr>K", int((S.cr[lab] > S.K[lab]).sum()))
print("mean ad (subzone)", S.ad[lab].mean(), "mean rest", S.rest[lab].mean(), "mean r", S.r[lab].mean())
print("pooled q", S.cr[lab].sum() / S.K[lab].sum())
print("hvac means (subzone-level):", S[HVAC].mean().round(3).to_dict())
# raw precision of hvac values: number of decimals
v = S[HVAC].values.ravel()
for d in range(0, 8):
    print("decimals<=", d, float(np.mean(np.abs(v * 10 ** d - np.round(v * 10 ** d)) < 1e-6)))
