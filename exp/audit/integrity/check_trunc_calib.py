"""Integrity audit 13: calibration of the truncation adjustment on H (mean residual by prediction band, before/after).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_trunc_calib.py
"""
import json
import numpy as np
from scipy.stats import norm
p2 = json.load(open("exp/audit/integrity/check_trunc2.json"))
a0, b0 = p2["var_normal"]; c = p2["normal"]["gas"]["c_P"]
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, M = z["H"], z["M"]
y, p = z["gas_y"], np.clip(z["gas_corrected"], 0, None)
s = np.maximum(p * np.sqrt(np.maximum(a0 + b0 / M[H], 1e-6)), 1.0)
a = (1e6 - p) / s
pa = p - c * s * np.exp(norm.logpdf(a) - norm.logcdf(a))
out = {}
for lo, hi in ((6e5, 7e5), (7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 2e6)):
    m = (p >= lo) & (p < hi)
    out[f"{lo:.0f}-{hi:.0f}"] = dict(n=int(m.sum()), resid_before=float((y - p)[m].mean()), resid_after=float((y - pa)[m].mean()),
                                     se=float((y - p)[m].std() / np.sqrt(m.sum())))
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_trunc_calib.json", "w"), indent=1)
