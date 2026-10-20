"""verify2_postproc v3: robustness of the S3 soft cap.

1. Gain as a function of k (flatness) on H corrected and on the full-train OOF teacher.
2. Where the H gain comes from (per standardized-distance band).
3. Test projection with per-band SSE reductions measured on the full-train OOF teacher (test-strength models).
4. Noise of the gain on a public-LB-sized (55,196 rows) and private-sized (128,791 rows) sample of H.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/softcap_s3/v3_robust.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.special import log_ndtr

mem = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable")))
assert mem >= 9_000_000, f"MemAvailable {mem} kB < 9000000, abort"
sys.path.insert(0, "final/src")
sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402
from dup_fix import duplicate_targets  # noqa: E402
from blend import nnls_blend, disagreement, vc_basis, predict_vc  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
COL = {"gas": GAS, "eui": EUI}
OUTD = "exp/audit/verify2_postproc/softcap_s3"
v1 = json.load(open(f"{OUTD}/v1_reproduce.json"))
VM = {m: v1["var_model"][m]["ols_rel"] for m in SIG}
KP = {m: v1["D1_D2"]["T=claimed"][m]["kP"] for m in SIG}
KCL = {"gas": 0.5046, "eui": 0.6574}
T = 999764.8988


def mills_adj(p, s, Tm):
    b = (Tm - p) / s
    return -s * np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))


def sd(m, p, Mv):
    a0, b0 = VM[m]
    return p * np.sqrt(np.clip(a0 + b0 / Mv, 1e-8, None))


def blend_light(Pf, Pa, yf, metric, A_fit, A_app, spec, M_fit, M_app, chunk=40000):
    s = A_fit if metric == "gas" else np.ones(len(yf))
    w = nnls_blend(Pf, yf, s if metric == "gas" else None)
    vf = dict(A=A_fit, M=M_fit, p=Pf @ w, d=disagreement(Pf))
    va = dict(A=A_app, M=M_app, p=Pa @ w, d=disagreement(Pa))
    Bf, Ba = vc_basis(spec, vf, vf), vc_basis(spec, vf, va)
    nb, npm = Bf.shape[1], Pf.shape[1]
    G, h = np.zeros((nb * npm, nb * npm)), np.zeros(nb * npm)
    for i in range(0, len(yf), chunk):
        sl = slice(i, i + chunk)
        Xc = np.concatenate([Pf[sl] * Bf[sl, [j]] for j in range(nb)], axis=1) * s[sl, None]
        G += Xc.T @ Xc
        h += Xc.T @ (yf[sl] * s[sl])
    L = np.linalg.cholesky(G)
    wv, _ = nnls(L.T, np.linalg.solve(L, h), maxiter=50 * len(G))
    W = wv.reshape(nb, npm).T
    pf, pa = predict_vc(Pf, W, Bf), predict_vc(Pa, W, Ba)
    return (pf * A_fit, pa * A_app) if metric == "gas" else (pf, pa)


out = {}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, AH, MH = z["H"], z["A"][z["H"]], z["M"][z["H"]]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
gm = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
spec = json.load(open("exp/v19vce/resid_knn.json"))["vc"]
Po = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in gm])
Pa = np.column_stack([np.load(f"exp/pred/{r}.npy") for r in gm])
yF = {"gas": tr[GAS].values, "eui": tr[EUI].values}
fo = {m: np.clip(blend_light(Po, Pa, yF["eui"], m, A, At, spec[m], M, Mt)[0], 0, None) for m in SIG}
del Po, Pa
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
live = np.isnan(duplicate_targets(tr, te)[:, 0])

ks = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
bands = [(-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)]
tot_curve_H, tot_curve_F = np.zeros(len(ks)), np.zeros(len(ks))
proj = {}
for m in SIG:
    # H corrected
    y, p = z[f"{m}_y"], z[f"{m}_corrected"]
    TmH = T if m == "gas" else T / AH
    sH = sd(m, p, MH)
    adjH, dH = mills_adj(p, sH, TmH), (TmH - p) / sH
    base = np.sqrt(np.mean((y - p) ** 2))
    cH = [(np.sqrt(np.mean((y - p - k * adjH) ** 2)) - base) / SIG[m] for k in ks]
    # full OOF teacher
    yf, pf = yF[m], fo[m]
    TmF = T if m == "gas" else T / A
    sF = sd(m, pf, M)
    adjF, dF = mills_adj(pf, sF, TmF), (TmF - pf) / sF
    baseF = np.sqrt(np.mean((yf - pf) ** 2))
    cF = [(np.sqrt(np.mean((yf - pf - k * adjF) ** 2)) - baseF) / SIG[m] for k in ks]
    tot_curve_H += np.array(cH) / 5
    tot_curve_F += np.array(cF) / 5
    print(f"{m}: dNRMSE vs k on H corrected  " + " ".join(f"k{k}:{v:+.7f}" for k, v in zip(ks, cH)))
    print(f"{m}: dNRMSE vs k on full-OOF tch " + " ".join(f"k{k}:{v:+.7f}" for k, v in zip(ks, cF)))
    # band contributions on H (k claimed): share of the SSE reduction
    dsse = (y - p - KCL[m] * adjH) ** 2 - (y - p) ** 2
    contrib = {f"[{lo},{hi})": float(dsse[(dH >= lo) & (dH < hi)].sum() / dsse.sum()) for lo, hi in bands}
    print(f"{m}: share of H SSE reduction by band (k claimed): " + " ".join(f"{b} {v:.2f}" for b, v in contrib.items()))
    # test projection from the full-OOF per-band SSE reduction
    pt = sub[COL[m]].values.astype(float)
    Tmt = T if m == "gas" else T / At
    st = sd(m, pt, Mt)
    dt = (Tmt - pt) / st
    res = {}
    for kname, kk in (("k_claimed", KCL[m]), ("kP", KP[m])):
        dsseF = (yf - pf - kk * adjF) ** 2 - (yf - pf) ** 2
        tot = 0.0
        for lo, hi in bands:
            selF = (dF >= lo) & (dF < hi)
            nt = int(((dt >= lo) & (dt < hi) & live).sum())
            tot += nt * dsseF[selF].mean()
        r0 = base  # H corrected rmse as the test base rmse
        res[kname] = float((np.sqrt(r0 ** 2 + tot / len(pt)) - r0) / SIG[m])
    proj[m] = res
    print(f"{m}: test projection from full-OOF band effects: " + " ".join(f"{k} {v:+.7f}" for k, v in res.items()))
    out[m] = dict(curve_H=dict(zip(map(str, ks), cH)), curve_fullOOF=dict(zip(map(str, ks), cF)), band_share_H=contrib,
                  test_projection_fullOOF=res)
print("TOTAL vs k, H corrected   " + " ".join(f"k{k}:{v:+.8f}" for k, v in zip(ks, tot_curve_H)))
print("TOTAL vs k, full-OOF tch  " + " ".join(f"k{k}:{v:+.8f}" for k, v in zip(ks, tot_curve_F)))
pt_total = {k: sum(proj[m][k] for m in SIG) / 5 for k in ("k_claimed", "kP")}
print("TOTAL test projection (full-OOF band effects):", {k: round(v, 9) for k, v in pt_total.items()})
out["total_curve_H"] = dict(zip(map(str, ks), tot_curve_H.tolist()))
out["total_curve_fullOOF"] = dict(zip(map(str, ks), tot_curve_F.tolist()))
out["test_projection_total"] = pt_total

# public / private sized samples of H (k claimed)
rng = np.random.default_rng(77)
e = {}
for m in SIG:
    y, p = z[f"{m}_y"], z[f"{m}_corrected"]
    TmH = T if m == "gas" else T / AH
    adjH = mills_adj(p, sd(m, p, MH), TmH)
    e[m] = ((y - p) ** 2, (y - p - KCL[m] * adjH) ** 2)
for name, n, repl in (("public_55196", 55196, False), ("private_128791_boot", 128791, True)):
    g = []
    for _ in range(2000):
        idx = rng.choice(len(H), n, replace=repl)
        g.append(sum((np.sqrt(e[m][1][idx].mean()) - np.sqrt(e[m][0][idx].mean())) / SIG[m] / 5 for m in SIG))
    g = np.array(g)
    out[name] = dict(mean=float(g.mean()), sd=float(g.std()), frac_neg=float((g < 0).mean()),
                     q05=float(np.quantile(g, 0.05)), q95=float(np.quantile(g, 0.95)))
    print(f"{name}: total gain mean {g.mean():+.8f} sd {g.std():.8f} P(<0) {(g < 0).mean():.3f} "
          f"90% [{np.quantile(g, 0.05):+.8f}, {np.quantile(g, 0.95):+.8f}]")
json.dump(out, open(f"{OUTD}/v3_robust.json", "w"), indent=1)
print(f"wrote {OUTD}/v3_robust.json")
