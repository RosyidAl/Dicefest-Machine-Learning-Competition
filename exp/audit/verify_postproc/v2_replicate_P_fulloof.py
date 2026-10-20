"""verify_postproc v2: independent replication of the near-cap over-prediction on rows that never touched H labels.

(a) P-OOF teacher (pP, rP from chain_H; P rows only, models trained inside P): residual bias by standardized distance
    to the cap, and S3 gain on P with k fitted on H corrected (k from H labels, scored on disjoint P labels).
(b) k fitted on P-OOF teacher ONLY (no H label anywhere), applied to H corrected: gain on H.
(c) full-train OOF of the v19vce VC blend (exp/audit/postproc/14_fullvc_oof.npz, the analogue of the test teacher:
    models trained on 90% of train): bias by distance to cap on P rows / H rows separately, S3 with k from H corrected
    scored on the P rows (disjoint from the k-fit rows), and its own 5-fold cross-fitted k.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/v2_replicate_P_fulloof.py
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
GMAX_P = float((z["gas_pP"] + z["gas_rP"]).max())
GMAX_ALL = float(f["gas_y"].max())
print(f"GMAX over P {GMAX_P:.1f}, over all train {GMAX_ALL:.1f}, over H {z['gas_y'].max():.1f}")
SD = {}
for m in SIG:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


def feats(m, p, M, Arows, T):
    p = np.clip(p, 0, None)
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / M, 1e-8, None))
    hi = T if m == "gas" else T / Arows
    return s, (hi - p) / np.maximum(s, 1e-12), tn_hi(p, np.maximum(s, 1e-12), hi)


def nr(m, yv, p):
    return float(np.sqrt(np.mean((yv - p) ** 2)) / SIG[m])


def bins(m, yv, p, dist, label):
    row = []
    for lo, hi in ((-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3)):
        sel = (dist >= lo) & (dist < hi)
        r = (yv - p)[sel]
        if sel.sum() > 1:
            row.append(f"[{lo},{hi}) n{sel.sum():4d} {r.mean() / SIG[m]:+.3f}s(t{r.mean() / (r.std() / np.sqrt(sel.sum())):+.1f})")
    print(f"   {m:4s} {label:22s} " + " ".join(row))


k_H = {}
for m in SIG:
    yH, pH = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    _, _, xH = feats(m, pH, MH, A[H], GMAX_P)
    k_H[m] = float(xH @ (yH - pH) / (xH @ xH))
print("k fitted on all H corrected (as claimed):", {m: round(v, 4) for m, v in k_H.items()})

print("== (a) P-OOF teacher (P rows only)")
k_P = {}
for m in SIG:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    yP = pP + rP
    pP = np.clip(pP, 0, None)
    s, dist, x = feats(m, pP, MP, A[P], GMAX_P)
    bins(m, yP, pP, dist, "P-OOF teacher")
    k_P[m] = float(x @ (yP - pP) / (x @ x))
    base = nr(m, yP, pP)
    d = nr(m, yP, pP + k_H[m] * x) - base
    dk1 = nr(m, yP, pP + x) - base
    print(f"   {m}: P base {base:.6f}; S3 with k_H {k_H[m]:.3f}: d {d:+.7f} (total {d / 5:+.8f});"
          f" k=1 d {dk1:+.7f}; k fitted on P itself {k_P[m]:.3f}")

print("== (b) k fitted on P-OOF teacher only, applied to H corrected (zero H labels in the fit)")
tot = 0.0
for m in SIG:
    yH, pH = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    _, _, xH = feats(m, pH, MH, A[H], GMAX_P)
    base = nr(m, yH, pH)
    d = nr(m, yH, pH + k_P[m] * xH) - base
    tot += d / 5
    print(f"   {m}: H base {base:.6f}; k_P {k_P[m]:.3f}: d {d:+.7f} (total {d / 5:+.8f})")
print(f"   TOTAL (gas+eui) {tot:+.8f}")

print("== (c) full-train OOF of the v19vce VC blend (teacher, no correction)")
inP = np.zeros(len(A), bool)
inP[P] = True
for m in SIG:
    yv, p = f[f"{m}_y"], np.clip(f[f"{m}_p"], 0, None)
    s, dist, x = feats(m, p, Mall, A, GMAX_ALL)
    bins(m, yv[inP], p[inP], dist[inP], "full OOF, P rows")
    bins(m, yv[~inP], p[~inP], dist[~inP], "full OOF, H rows")
    bins(m, yv, p, dist, "full OOF, all rows")
    # k from H corrected (H labels) scored on P rows only (disjoint labels)
    bP = nr(m, yv[inP], p[inP])
    dP = nr(m, yv[inP], p[inP] + k_H[m] * x[inP]) - bP
    # own cross-fitted k on all rows
    pc = p.copy()
    ks = []
    for a, b in KFold(5, shuffle=True, random_state=0).split(p):
        k = float(x[a] @ (yv - p)[a] / (x[a] @ x[a]))
        ks.append(round(k, 3))
        pc[b] = p[b] + k * x[b]
    dall = nr(m, yv, pc) - nr(m, yv, p)
    print(f"   {m}: P rows S3 with k_H: d {dP:+.7f} (total {dP / 5:+.8f}); all rows cross-fitted k {ks}: d {dall:+.7f}"
          f" (total {dall / 5:+.8f}); full-OOF k all rows {float(x @ (yv - p) / (x @ x)):.3f}")
