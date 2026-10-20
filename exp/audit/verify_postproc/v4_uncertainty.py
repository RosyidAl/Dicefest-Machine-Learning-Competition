"""verify_postproc v4: size and sampling spread of the S3 gain with the claimed constants (k from all H corrected).

(a) full-train OOF teacher: gain on P rows and on H rows separately (H rows were in the k fit, P rows were not).
(b) bootstrap of the gain over a private-LB-sized sample (128.8k rows) drawn from (i) the H chain with nested
    cross-fitted k, (ii) full-OOF P rows with k_H: mean, sd, P(gain >= 0).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/v4_uncertainty.py
"""
import sys

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
f = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
MP, MH = Mall[P], Mall[H]
T_P = float((z["gas_pP"] + z["gas_rP"]).max())
T_all = float(f["gas_y"].max())
SD = {}
for m in SIG:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


def xf(m, p, M, Ar, T):
    p = np.clip(p, 0, None)
    s = np.maximum(p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / M, 1e-8, None)), 1e-12)
    return tn_hi(p, s, T if m == "gas" else T / Ar)


kH, hchain = {}, {}
for m in SIG:
    y, p = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    x = xf(m, p, MH, A[H], T_P)
    kH[m] = float(x @ (y - p) / (x @ x))
    pc = p.copy()
    for a, b in KFold(5, shuffle=True, random_state=0).split(p):
        pc[b] = p[b] + float(x[a] @ (y - p)[a] / (x[a] @ x[a])) * x[b]
    hchain[m] = (y, p, pc)
inP = np.zeros(len(A), bool)
inP[P] = True
full = {}
print("== (a) full-train OOF teacher, S3 with k_H")
for m in SIG:
    y, p = f[f"{m}_y"], np.clip(f[f"{m}_p"], 0, None)
    pc = p + kH[m] * xf(m, p, Mall, A, T_all)
    full[m] = (y[inP], p[inP], pc[inP])
    for name, sel in (("P rows", inP), ("H rows", ~inP), ("all", np.ones(len(A), bool))):
        d = (np.sqrt(np.mean((y - pc)[sel] ** 2)) - np.sqrt(np.mean((y - p)[sel] ** 2))) / SIG[m]
        print(f"   {m:4s} {name:7s} d {d:+.7f} total {d / 5:+.8f}")


def boot(sets, n, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(next(iter(sets.values()))[0])
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, N, n)
        t = 0.0
        for m, (y, p, pc) in sets.items():
            t += (np.sqrt(np.mean((y[idx] - pc[idx]) ** 2)) - np.sqrt(np.mean((y[idx] - p[idx]) ** 2))) / SIG[m] / 5
        out[i] = t
    return out


print("== (b) bootstrap of the total gain on a 128.8k-row sample (private-LB size) and a 55.2k-row sample (public)")
for name, sets in (("H chain cross-fitted", hchain), ("full OOF P rows, k_H", full)):
    for n in (128791, 55196):
        b = boot(sets, n)
        print(f"   {name:22s} n {n:6d}: mean {b.mean():+.8f} sd {b.std():.8f} P(gain>=0) {np.mean(b >= 0):.3f}"
              f" q05 {np.quantile(b, 0.05):+.8f} q95 {np.quantile(b, 0.95):+.8f}")
