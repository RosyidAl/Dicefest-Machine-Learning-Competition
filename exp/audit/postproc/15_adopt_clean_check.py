"""Audit postproc 15: numeric rules on the adoption and clean-ratio columns of sub_v19vce.csv.
- adoption <= rest = 100 - sum(hvac_dist) (exact on train), adoption = 0 where rest = 0, within [0, 100];
- seen subzones: submitted adoption equals the (constant) train value of the subzone;
- unseen subzones: distribution of predictions vs rest;
- clean ratio: identity round(100 b02 / b01, 2) for every test row.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/15_adopt_clean_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
HV = [f"hvac_dist_{i:02d}" for i in range(1, 11)]
tr = pd.read_parquet("cache/train.parquet", columns=["subzone_id", "bld_dim_01", "bld_dim_02", AD] + HV)
te = pd.read_parquet("cache/test.parquet", columns=["subzone_id", "bld_dim_01", "bld_dim_02"] + HV)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
rest_tr = 100 - tr[HV].sum(axis=1).values
rest_te = 100 - te[HV].sum(axis=1).values
a = sub[AD].values
print("train: adoption > rest + 1e-9:", int((tr[AD].values > rest_tr + 1e-9).sum()),
      "| rows rest<=1e-9:", int((rest_tr <= 1e-9).sum()), "of which adoption==0:", int(((rest_tr <= 1e-9) & (tr[AD].values == 0)).sum()))
nun = tr.groupby("subzone_id")[AD].nunique()
print("train subzones with >1 adoption value:", int((nun > 1).sum()), "of", len(nun))
zmap = tr.groupby("subzone_id")[AD].first()
seen = te["subzone_id"].isin(zmap.index).values
cp = te["subzone_id"].map(zmap).values
print("test rows seen subzone:", int(seen.sum()), "unseen:", int((~seen).sum()))
print("seen: submitted == train subzone value:", int((a[seen] == cp[seen]).sum()), "max abs diff", float(np.nanmax(np.abs(a[seen] - cp[seen]))))
# hvac consistency between test rows and train rows of the same subzone (the copy assumes the same OA)
hv_tr = tr.groupby("subzone_id")[HV].first()
d = np.abs(te.loc[seen, HV].values - hv_tr.loc[te.loc[seen, "subzone_id"]].values).max(axis=1)
print("seen: test rows whose hvac_dist differs from the train subzone (max abs > 1e-9):", int((d > 1e-9).sum()))
u = ~seen
print("unseen: adoption > rest + 1e-6:", int((a[u] > rest_te[u] + 1e-6).sum()), "max excess", float((a[u] - rest_te[u]).max()))
print("unseen: rest<=1e-9 rows:", int((rest_te[u] <= 1e-9).sum()), "with adoption != 0:", int(((rest_te[u] <= 1e-9) & (a[u] != 0)).sum()))
print("all: adoption > rest + 1e-9:", int((a > rest_te + 1e-9).sum()), "| min", a.min(), "max", a.max())
print("unseen: adoption / rest quantiles", np.round(np.quantile(a[u][rest_te[u] > 0] / rest_te[u][rest_te[u] > 0], [0, .01, .5, .99, 1]), 4))
cl = np.round(100 * te["bld_dim_02"].values / te["bld_dim_01"].values, 2)
print("clean: submitted == identity:", int((sub[CL].values == cl).sum()), "of", len(cl))
