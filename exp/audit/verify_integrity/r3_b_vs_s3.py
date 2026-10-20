"""Verifier round 3 (leakage/optimism lens), step B: is the integrity truncation shrink (I) additive to what is ALREADY
in BEST_CURRENT? D015 (sub_v19ncapp, sub_v19nca2pp) adopted the postproc audit chain S1 (small-M shift) + S3 (soft
truncated-normal cap, spread multiplier 1, scale fitted on H) + S4 (hard clips), the same mechanism as I.

Honest replication on H (exp/audit/chain_H.npz only; all H-fitted scales 5-fold cross-fitted on H, everything else
from P):
  V0  S0                        base (v19vce analog, = the claim's baseline)
  V1  S0 + I                    the claim (I: a0, b0, k, c fitted on P, own code as r3_a)
  V2  S1 + S3 + S4              adopted analog (kS3 cross-fitted on H)
  V3  S1 + I + S4               replace S3 by I
  V4  S1 + S3 + S4 + I          stack I on top of the adopted chain (double shrink)
  V5  S1 + S3(in-sample k) + S4 (+ I)  the deployed kS3 fitted on all of H (in-sample on H, reference only)
Increments in TOTAL units (per-metric NRMSE change / 5, gas + eui only; elec untouched by I). Paired bootstrap and
20 permuted split-halves (S3 scale fitted on one half, scored on the other) for V3-V2 and V4-V2.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/r3_b_vs_s3.py
"""
import json

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
MET = ("gas", "eui")
OUT = "exp/audit/verify_integrity/r3_b_vs_s3.json"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AH, MP, MH = A[H], M[P], M[H]
y = {m: z[f"{m}_y"] for m in MET}
base = {m: np.clip(z[f"{m}_corrected"], 0, None) for m in MET}
yPg = z["gas_pP"] + z["gas_rP"]
GMAX_P = float(yPg.max())
EUIMIN_P = float((z["eui_pP"] + z["eui_rP"]).min())

# ---- I (integrity) parameters, own fit on P (same procedure as r3_a)
r3a = json.load(open("exp/audit/verify_integrity/r3_a_rederive.json"))
a0, b0 = r3a["var_model_P"]["a0"], r3a["var_model_P"]["b0"]
KI = {m: r3a["fit_P"][m]["k"] for m in MET}
CI = {m: r3a["fit_P"][m]["c"] for m in MET}


def adj_I(pg, m, k):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / m, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def apply_I(m, p):
    sc = AH if m == "eui" else 1.0
    return np.clip(p + CI[m] * adj_I(p * sc, MH, KI[m]) / sc, 0, None)


# ---- adopted S1 / S3 / S4 (exp/audit/postproc/11_combo.py and postproc_nca/13_candidate_nca.py)
SD = {}
for m in MET:
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(p)), 1 / MP]), (r / p) ** 2, rcond=None)[0]
S1D = {}
for m in MET:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    S1D[m] = [(lo, hi, float(pP[g] @ rP[g] / (pP[g] @ pP[g]))) for lo, hi in ((2, 3), (4, 4))
              for g in [(MP >= lo) & (MP <= hi)]]


def S1(m, p):
    p = p.copy()
    for lo, hi, d in S1D[m]:
        p[(MH >= lo) & (MH <= hi)] *= 1 + d
    return p


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


def S3x(m, p):
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / MH, 1e-8, None))
    return tn_hi(p, s, GMAX_P if m == "gas" else GMAX_P / AH)


def S4(m, p):
    lo, hi = {"gas": (EUIMIN_P * AH, np.full(len(AH), GMAX_P)), "eui": (np.full(len(AH), EUIMIN_P), GMAX_P / AH)}[m]
    return np.clip(p, lo, hi)


def kfit(x, r, idx):
    return float(x[idx] @ r[idx] / (x[idx] @ x[idx]))


def nr(m, p, idx=None):
    if idx is None:
        idx = np.arange(len(H))
    return float(np.sqrt(np.mean((y[m][idx] - p[idx]) ** 2)) / SIG[m])


V = {k: {} for k in ("V0", "V1", "V2", "V3", "V4", "V4b", "V5", "V5I")}
kS3_folds = {}
kS3_all = {}
for m in MET:
    p0 = base[m]
    p1 = S1(m, p0)
    x = S3x(m, p1)
    r1 = y[m] - p1
    p3 = np.empty(len(H))
    ks = []
    for a, b in KFold(5, shuffle=True, random_state=0).split(p1):
        k = kfit(x, r1, a)
        ks.append(k)
        p3[b] = p1[b] + k * x[b]
    kS3_folds[m] = ks
    kS3_all[m] = kfit(x, r1, np.arange(len(H)))
    p3_ins = p1 + kS3_all[m] * x
    V["V0"][m] = p0
    V["V1"][m] = apply_I(m, p0)
    V["V2"][m] = S4(m, p3)
    V["V3"][m] = S4(m, apply_I(m, p1))
    V["V4"][m] = apply_I(m, S4(m, p3))
    V["V4b"][m] = S4(m, apply_I(m, p3))
    V["V5"][m] = S4(m, p3_ins)
    V["V5I"][m] = apply_I(m, S4(m, p3_ins))

sc = {v: {m: nr(m, V[v][m]) for m in MET} for v in V}
tot = {v: sum(sc[v][m] for m in MET) / 5 for v in V}
out = dict(GMAX_P=GMAX_P, EUIMIN_P=EUIMIN_P, I_params=dict(a0=a0, b0=b0, k=KI, c=CI), S1_deltas=S1D,
           S3_var_model=SD and {m: SD[m].tolist() for m in MET}, kS3_crossfit=kS3_folds, kS3_all_H=kS3_all,
           nrmse=sc, total_vs_V0={v: tot[v] - tot["V0"] for v in V},
           inc_replace_S3_by_I=tot["V3"] - tot["V2"], inc_stack_I_on_adopted=tot["V4"] - tot["V2"],
           inc_stack_I_before_clip=tot["V4b"] - tot["V2"], inc_stack_I_on_deployed_inSample=tot["V5I"] - tot["V5"],
           per_metric_inc_stack={m: (sc["V4"][m] - sc["V2"][m]) / 5 for m in MET},
           per_metric_inc_replace={m: (sc["V3"][m] - sc["V2"][m]) / 5 for m in MET})

# paired bootstrap of the increments
rng = np.random.default_rng(7)
E = {v: {m: (y[m] - V[v][m]) ** 2 for m in MET} for v in V}
bs = {"V3-V2": [], "V4-V2": [], "V1-V0": [], "V2-V0": []}
for _ in range(1000):
    b = rng.integers(0, len(H), len(H))
    t = {v: sum(np.sqrt(E[v][m][b].mean()) / SIG[m] for m in MET) / 5 for v in ("V0", "V1", "V2", "V3", "V4")}
    for key in bs:
        u, w = key.split("-")
        bs[key].append(t[u] - t[w])
out["boot"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), p_worse=float(np.mean(np.array(v) > 0)))
               for k, v in bs.items()}

# permuted split-halves: S3 scale fitted on one half, everything scored on the other half
sh = {"V3-V2": [], "V4-V2": [], "V1-V0": []}
for rep in range(20):
    perm = np.random.default_rng(900 + rep).permutation(len(H))
    h1, h2 = np.sort(perm[: len(H) // 2]), np.sort(perm[len(H) // 2:])
    for fa, sb in ((h1, h2), (h2, h1)):
        t = {}
        for v in ("V0", "V1", "V2", "V3", "V4"):
            t[v] = 0.0
        for m in MET:
            p0 = base[m]
            p1 = S1(m, p0)
            x = S3x(m, p1)
            k = kfit(x, y[m] - p1, fa)
            p3 = S4(m, p1 + k * x)
            preds = {"V0": p0, "V1": apply_I(m, p0), "V2": p3, "V3": S4(m, apply_I(m, p1)), "V4": apply_I(m, p3)}
            for v, pv in preds.items():
                t[v] += nr(m, pv, sb) / 5
        sh["V3-V2"].append(t["V3"] - t["V2"])
        sh["V4-V2"].append(t["V4"] - t["V2"])
        sh["V1-V0"].append(t["V1"] - t["V0"])
out["split_halves"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), frac_neg=float(np.mean(np.array(v) < 0)),
                               min=float(np.min(v)), max=float(np.max(v))) for k, v in sh.items()}
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
