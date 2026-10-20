"""Integrity audit 10: do gas-only twin partners share any feature (corr across pairs vs random pairs)?

If some feature links partners strongly, a twin of a test row could be located in train and its exact gas copied.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_twins2.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T = ["target_macro_gas_kwh", "target_macro_elec_kwh"]
te_cols = pd.read_parquet("data/cache/test.parquet").columns
feat = sorted(c for c in te_cols if c not in ("id", "bld_dim_18", "socio_demog_016", "zone_region_id", "subzone_id"))
tr = pd.read_parquet("data/cache/train.parquet", columns=feat + T + ["subzone_id", "zone_region_id"])
g = tr[T[0]]
vc = g.value_counts()
# exclude the 92 true duplicate groups (same elec too)
ge = tr.groupby(T)[T[0]].transform("size")
tw = tr[(g.map(vc).values > 1) & (ge.values == 1)]
pairs = tw.groupby(T[0]).apply(lambda d: d.index[:2].tolist(), include_groups=False)
pairs = np.array([p for p in pairs if len(p) == 2])
a, b = pairs[:, 0], pairs[:, 1]
rng = np.random.default_rng(0)
ra = rng.choice(len(tr), len(a))
rb = rng.choice(len(tr), len(a))
res = {}
for c in feat:
    x = tr[c].values.astype(float)
    ok = ~np.isnan(x[a]) & ~np.isnan(x[b])
    rho = spearmanr(x[a][ok], x[b][ok]).statistic
    ok2 = ~np.isnan(x[ra]) & ~np.isnan(x[rb])
    rho0 = spearmanr(x[ra][ok2], x[rb][ok2]).statistic
    eq = float(np.mean(x[a][ok] == x[b][ok]))
    res[c] = dict(rho_pairs=float(rho), rho_random=float(rho0), share_equal=eq)
top = dict(sorted(res.items(), key=lambda kv: -abs(kv[1]["rho_pairs"]))[:15])
topeq = dict(sorted(res.items(), key=lambda kv: -kv[1]["share_equal"])[:8])
out = dict(n_pairs=int(len(a)), same_zone=float(np.mean(tr["zone_region_id"].values[a] == tr["zone_region_id"].values[b])),
           top_by_rho=top, top_by_share_equal=topeq,
           elec_rho=float(spearmanr(tr[T[1]].values[a], tr[T[1]].values[b]).statistic))
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_twins2.json", "w"), indent=1)
