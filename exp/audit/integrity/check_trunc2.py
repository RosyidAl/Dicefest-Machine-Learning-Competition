"""Integrity audit 9: truncation adjustment variants, all fitted on P (teacher P-OOF), scored on H (honest chain).

normal   : E[y|y<T] - p = -s*phi(a)/Phi(a), a=(T-p)/s, s = p*sqrt(a0+b0/M)
lognormal: log y ~ N(log mu, v), v = a0+b0/M, mu*exp(v/2) = p (mean-preserving):
           E[y|y<T] = p * Phi((log T - log mu - v)/sqrt v) / Phi((log T - log mu)/sqrt v)
per-metric scale c fitted on P (EUI fitted on EUI-scale residuals).
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_trunc2.py
"""
import json

import numpy as np
from scipy.stats import norm

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
pP, rP = z["gas_pP"], z["gas_rP"]
ok = pP > 1000
coef = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), 1.0 / M[P][ok]]), (rP[ok] / pP[ok]) ** 2, rcond=None)[0]
# log-scale variance model on P
lr = np.log(np.maximum(pP + rP, 1.0)) - np.log(np.maximum(pP, 1.0))
coefl = np.linalg.lstsq(np.column_stack([np.ones(len(lr)), 1.0 / M[P]]), (lr - lr.mean()) ** 2, rcond=None)[0]


def adj_normal(p, m):
    s = np.maximum(np.abs(p) * np.sqrt(np.maximum(coef[0] + coef[1] / m, 1e-6)), 1.0)
    a = (T - p) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def adj_lognormal(p, m):
    v = np.maximum(coefl[0] + coefl[1] / m, 1e-6)
    sv = np.sqrt(v)
    lmu = np.log(np.maximum(p, 1.0)) - v / 2
    ratio = np.exp(norm.logcdf((np.log(T) - lmu - v) / sv) - norm.logcdf((np.log(T) - lmu) / sv))
    return p * ratio - p


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


out = {"var_normal": coef.tolist(), "var_log": coefl.tolist()}
for kind, f in (("normal", adj_normal), ("lognormal", adj_lognormal)):
    res = {}
    for m in ("gas", "eui"):
        sc_P = A[P] if m == "eui" else np.ones(len(P))
        sc_H = A[H] if m == "eui" else np.ones(len(H))
        # fit c on P in the metric's own units
        pPm, rPm = z[f"{m}_pP"], z[f"{m}_rP"]
        # rP for gas is in kWh (gas scale); for eui in EUI units (pipeline: r = A*y - pf for gas, y - pf for eui)
        aPm = f(pPm * sc_P, M[P]) / sc_P
        c = float(np.sum(aPm * rPm) / np.sum(aPm ** 2))
        y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
        a = f(p * sc_H, M[H]) / sc_H
        pa = p + c * a
        base = nr(y, p, m)
        d = (nr(y, pa, m) - base) / 5
        halves = [(nr(y[i::2], pa[i::2], m) - nr(y[i::2], p[i::2], m)) / 5 for i in range(2)]
        rng = np.random.default_rng(1)
        e0, e1 = (y - p) ** 2, (y - pa) ** 2
        bs = []
        for _ in range(300):
            b = rng.integers(0, len(p), len(p))
            bs.append((np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] / 5)
        res[m] = dict(c_P=c, delta_total=d, halves=halves, boot_sd=float(np.std(bs)),
                      boot_p_worse=float(np.mean(np.array(bs) > 0)), max_pred_after=float((pa * sc_H).max()))
    res["sum_delta_total"] = res["gas"]["delta_total"] + res["eui"]["delta_total"]
    out[kind] = res
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_trunc2.json", "w"), indent=1)
