"""Power of the tail-count test: if test were NOT truncated at 1e6, how many extra test rows would the OOF gas blend
put above 900k / 950k / 1e6? Uses train rows with true gas in the top 50k bin below 1e6 as a (conservative) proxy
for the prediction ratio p / y of untruncated rows, and the geometric tail extrapolation (decay 0.6457 per 50k).
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/verify_trunc_power.py
"""
import json

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
tr = pd.read_parquet("data/cache/train.parquet", columns=[GAS, EUI, "bld_dim_06"])
A, y = tr["bld_dim_06"].values, tr[GAS].values
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
Po = np.column_stack([np.load(f"exp/oof/{k}.npy") for k in mem])
w, _ = nnls(Po * A[:, None], tr[EUI].values * A)
go = (Po @ w) * A
del Po
rng = np.random.default_rng(1)
top = (y >= 8.5e5) & (y < 1e6)
ratio = go[top] / y[top]  # prediction / truth for large rows
r = 0.6456986404236604  # decay per 50k from verify_clip.json
n_test_untrunc = 150.74
# sample untruncated truths: exponential tail above 1e6 with the same decay (scale = 50k / -ln r)
scale = 5e4 / -np.log(r)
sims = {t: [] for t in (9e5, 9.5e5, 1e6)}
for _ in range(2000):
    k = rng.poisson(n_test_untrunc)
    yt = 1e6 + rng.exponential(scale, k)
    pt = yt * rng.choice(ratio, k)
    for t in sims:
        sims[t].append(int((pt > t).sum()))
out = dict(n_proxy_rows=int(top.sum()), proxy_ratio_median=float(np.median(ratio)), proxy_ratio_mean=float(ratio.mean()),
           tail_scale=scale,
           extra_test_rows_if_untruncated={f"{t:.0f}": dict(mean=float(np.mean(v)), p05=float(np.percentile(v, 5)))
                                           for t, v in sims.items()})
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/verify_integrity/verify_trunc_power.json", "w"), indent=1)
