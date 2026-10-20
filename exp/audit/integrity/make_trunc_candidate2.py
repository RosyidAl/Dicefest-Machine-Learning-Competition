"""Integrity audit 15: final truncation candidate (k and c fitted on P, check_trunc3.py) -> H bootstrap + test file.

gas: p += c_gas * adj(p);  EUI: p = (p*A + c_eui * adj(p*A)) / A;  adj(pg) = -s*phi(a)/Phi(a), a = (1e6 - pg)/s,
s = k * pg * sqrt(a0 + b0/M). Duplicate rows (77, exact train targets) untouched. Writes
exp/audit/integrity/sub_v19vce_trunc2.csv (NOT submitted).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/make_trunc_candidate2.py
"""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
SIG = dict(gas=151750.04, eui=40.65)
a0, b0 = json.load(open("exp/audit/integrity/check_trunc2.json"))["var_normal"]
t3 = json.load(open("exp/audit/integrity/check_trunc3.json"))
k = t3["gas"]["k_best_on_P"]; assert k == t3["eui"]["k_best_on_P"]
c_gas, c_eui = t3["gas"]["grid"][str(k)]["c"], t3["eui"]["grid"][str(k)]["c"]
def adj(pg, m):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / m, 1e-6)), 1.0)
    a = (1e6 - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))
out = dict(params=dict(a0=a0, b0=b0, k=k, c_gas=c_gas, c_eui=c_eui))
# H bootstrap of the final parameters
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, A, M = z["H"], z["A"], z["M"]
rng = np.random.default_rng(2)
B = [rng.integers(0, len(H), len(H)) for _ in range(300)]
tot = np.zeros(300); d0 = 0.0
for m, c in (("gas", c_gas), ("eui", c_eui)):
    sc = A[H] if m == "eui" else 1.0
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    pa = (p * sc + c * adj(p * sc, M[H])) / sc
    e0, e1 = (y - p) ** 2, (y - pa) ** 2
    d = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m] / 5
    bs = np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] / 5 for b in B])
    out[f"H_{m}"] = dict(delta_total=float(d), boot_sd=float(bs.std()), boot_p_worse=float((bs > 0).mean()))
    tot += bs; d0 += d
out["H_total"] = dict(delta_total=float(d0), boot_sd=float(tot.std()), boot_p_worse=float((tot > 0).mean()))
# test candidate
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
tr = pd.read_parquet("data/cache/train.parquet", columns=KEY + [GAS])
te = pd.read_parquet("data/cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
ref = tr.groupby(KEY, sort=False)[GAS].first().reset_index()
hit = ~te[KEY].merge(ref, on=KEY, how="left")[GAS].isna().values
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
g, e = sub[GAS].values, sub[EUI].values
new = sub.copy()
new[GAS] = np.where(hit, g, np.clip(g + c_gas * adj(g, Mt), 0, None))
new[EUI] = np.where(hit, e, np.clip((e * At + c_eui * adj(e * At, Mt)) / At, 0, None))
dg, de = new[GAS].values - g, new[EUI].values - e
out["test"] = dict(dup_rows_untouched=int(hit.sum()), gas_rows_changed_gt_1pct=int((np.abs(dg) > 0.01 * g).sum()),
                   gas_rms_change_over_sigma=float(np.sqrt(np.mean(dg ** 2)) / SIG["gas"]),
                   eui_rows_changed_gt_1pct=int((np.abs(de) > 0.01 * e).sum()),
                   eui_rms_change_over_sigma=float(np.sqrt(np.mean(de ** 2)) / SIG["eui"]),
                   gas_max_after=float(new[GAS].max()), gas_min_change=float(dg.min()))
path = "exp/audit/integrity/sub_v19vce_trunc2.csv"
new.to_csv(path, index=False)
out["test"]["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/make_trunc_candidate2.json", "w"), indent=1)
