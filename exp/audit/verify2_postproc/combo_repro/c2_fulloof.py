"""verify2_postproc/combo_repro c2: second, larger honest split for the combined post-processing S1+S3+S4.

Rebuilds (own code) the v19vce teacher on the full-train member OOF: varying-coefficient NNLS A2xM2xp2 exactly as
final/src/pipeline._assemble -> postprocess.blend_preds -> blend.blend_vc (basis from src/blend.vc_basis, the
pipeline's own function), with the NNLS solved by an own Lawson-Hanson active-set on the Gram matrix accumulated in
row chunks (memory light). Then, on these 429,303 OOF rows:
  1. small-M multiplicative bias per group (compare with exp/audit/postproc/12_fullvc_smallM.json)
  2. S1 cross-fitted over rows (repeated 2- and 5-fold: deltas fitted on the fit rows, applied to the held-out rows),
     plus P->H / H->P with the project holdout split
  3. S3 (noise model, k) and S4 (bounds) cross-fitted the same way, then the whole chain S1+S3+S4
  4. concentration / stability of the S1 gain (A or p terciles inside the small-M groups)
Saves the teacher OOF predictions to c2_teacher_oof.npz (for c3).
usage (from /root/Dicefest): OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
    exp/audit/verify2_postproc/combo_repro/c2_fulloof.py
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.special import log_ndtr
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402  (the pipeline's own basis function)
from common import SIGMAS, TARGET_COLS, holdout_split  # noqa: E402

D = "exp/audit/verify2_postproc/combo_repro"
GAS, EL, EUI = TARGET_COLS[:3]
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
MET = ("gas", "eui", "elec")
GROUPS = ((2, 3), (4, 4))

tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EL, EUI])
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
n = len(tr)
print("rows", n, "M counts", {k: int((M == k).sum()) for k in (1, 2, 3, 4, 5)}, "max|A*EUI-gas|/gas",
      float(np.max(np.abs(A * tr[EUI].values - tr[GAS].values) / tr[GAS].values)))
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]


def nnls_normal(G, b, tol_rel=1e-11, maxit=5000):
    """Lawson-Hanson active set for min w'Gw - 2 b'w s.t. w >= 0 (G symmetric PSD), own implementation"""
    k = len(b)
    w = np.zeros(k)
    act = np.zeros(k, bool)  # passive (free) set
    tol = tol_rel * max(np.abs(b).max(), 1e-300)
    for _ in range(maxit):
        g = b - G @ w
        cand = np.where(~act, g, -np.inf)
        j = int(np.argmax(cand))
        if act.all() or cand[j] <= tol:
            break
        act[j] = True
        while True:
            idx = np.where(act)[0]
            zz = np.zeros(k)
            zz[idx] = np.linalg.lstsq(G[np.ix_(idx, idx)], b[idx], rcond=None)[0]
            if (zz[idx] > 0).all():
                w = zz
                break
            neg = act & (zz <= 0)
            alpha = np.min(w[neg] / (w[neg] - zz[neg]))
            w = w + alpha * (zz - w)
            act &= w > 1e-15
            w[~act] = 0.0
    return w


def gram(design, y, s, chunk=20000):
    G = b = None
    for i in range(0, len(y), chunk):
        sl = slice(i, min(i + chunk, len(y)))
        Xc = design(sl) * s[sl, None]
        yc = y[sl] * s[sl]
        G = Xc.T @ Xc if G is None else G + Xc.T @ Xc
        b = Xc.T @ yc if b is None else b + Xc.T @ yc
    return G, b


teacher, kkt = {}, {}
for fam, names in (("gas", mem["gas"]), ("elec", mem["elec"])):
    X = np.column_stack([np.load(f"exp/oof/{nm}.npy") for nm in names]).astype(float)
    mets = ("gas", "eui") if fam == "gas" else ("elec",)
    for met in mets:
        yv = tr[EL].values if met == "elec" else tr[EUI].values
        s = A if met == "gas" else np.ones(n)
        w0 = nnls_normal(*gram(lambda sl: X[sl], yv, s))
        p0 = X @ w0
        vf = dict(A=A, M=M, p=p0)
        Bs = vc_basis("A2xM2xp2", vf, vf)
        nb = Bs.shape[1]
        design = lambda sl: np.concatenate([X[sl] * Bs[sl, [j]] for j in range(nb)], axis=1)  # noqa: E731
        G, bb = gram(design, yv, s)
        Wf = nnls_normal(G, bb)
        grad = bb - G @ Wf
        kkt[met] = dict(n_pos=int((Wf > 0).sum()), max_grad_free=float(np.abs(grad[Wf > 0]).max() / np.abs(bb).max()),
                        max_grad_bound=float(grad[Wf == 0].max() / np.abs(bb).max()) if (Wf == 0).any() else None)
        W = Wf.reshape(nb, X.shape[1]).T
        p = ((X @ W) * Bs).sum(axis=1)
        teacher[met] = p * A if met == "gas" else p
    del X
yt = {"gas": tr[GAS].values.astype(float), "eui": tr[EUI].values.astype(float), "elec": tr[EL].values.astype(float)}
print("NNLS KKT (relative gradient on free / max on bound vars)", kkt)
base = {m: np.clip(teacher[m], 0, None) for m in MET}
r0 = {m: yt[m] - base[m] for m in MET}
print("teacher full-OOF NRMSE", {m: round(float(np.sqrt(np.mean(r0[m] ** 2)) / SIG[m]), 6) for m in MET})
np.savez_compressed(f"{D}/c2_teacher_oof.npz", **{m: base[m] for m in MET})
out = {"kkt": kkt}

# ---- 1. small-M bias on the full OOF ---------------------------------------------------------------------------
ref = json.load(open("exp/audit/postproc/12_fullvc_smallM.json"))
bias = {}
for m in MET:
    bias[m] = {}
    for gname, g in (("M<=3", M <= 3), ("M=4", M == 4), ("M=5", M == 5)):
        p, r = base[m][g], r0[m][g]
        rel = r / p
        bias[m][gname] = dict(n=int(g.sum()), ls=float(p @ r / (p @ p)), ros=float(r.sum() / p.sum()),
                              meanrel=float(rel.mean()), t=float(rel.mean() / rel.std() * np.sqrt(g.sum())),
                              claim_ls=ref[m][gname]["ls_delta"])
    print(m, {g: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for g, d in bias[m].items()})
out["bias"] = bias

# ---- helpers for the chain ---------------------------------------------------------------------------------------


def fit_params(fit):
    """all post-processing parameters from the rows `fit` (S1 deltas, S3 noise model, S4 bounds); k needs S1 first"""
    prm = {"d": {}, "nm": {}}
    for m in MET:
        prm["d"][m] = []
        for lo, hi in GROUPS:
            g = fit[(M[fit] >= lo) & (M[fit] <= hi)]
            prm["d"][m].append(float(base[m][g] @ r0[m][g] / (base[m][g] @ base[m][g])))
    for m in ("gas", "eui"):
        p, r = base[m][fit], r0[m][fit]
        Xn = np.column_stack([np.ones(len(fit)), 1.0 / M[fit]])
        prm["nm"][m] = np.linalg.lstsq(Xn, (r / p) ** 2, rcond=None)[0]
    yg = yt["gas"][fit]
    prm["gmax"] = float(yg.max())
    prm["euimin"] = float(yt["eui"][fit].min())
    ea = yt["elec"][fit] / A[fit]
    prm["eamin"], prm["eamax"] = float(ea.min()), float(ea.max())
    return prm


def s1(idx, prm, m, p):
    q = p.copy()
    for (lo, hi), dd in zip(GROUPS, prm["d"][m]):
        g = (M[idx] >= lo) & (M[idx] <= hi)
        q[g] *= 1 + dd
    return q


def tail(idx, prm, m, p):
    s = p * np.sqrt(np.clip(prm["nm"][m][0] + prm["nm"][m][1] / M[idx], 1e-8, None))
    T = np.full(len(idx), prm["gmax"]) if m == "gas" else prm["gmax"] / A[idx]
    b = (T - p) / s
    return -s * np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))


def clip4(idx, prm, m, p):
    a = A[idx]
    lo = {"gas": prm["euimin"] * a, "eui": np.full(len(a), prm["euimin"]), "elec": prm["eamin"] * a}[m]
    hi = {"gas": np.full(len(a), prm["gmax"]), "eui": prm["gmax"] / a, "elec": prm["eamax"] * a}[m]
    return np.clip(p, lo, hi)


def chain_split(fit, app, steps=("S1", "S3", "S4")):
    """parameters from `fit` rows only; returns {metric: prediction on app rows} per step"""
    prm = fit_params(fit)
    res = {"S0": {m: base[m][app].copy() for m in MET}}
    cf = {m: base[m][fit].copy() for m in MET}
    ca = {m: base[m][app].copy() for m in MET}
    if "S1" in steps:
        cf = {m: s1(fit, prm, m, cf[m]) for m in MET}
        ca = {m: s1(app, prm, m, ca[m]) for m in MET}
    res["S1"] = {m: ca[m].copy() for m in MET}
    if "S3" in steps:
        for m in ("gas", "eui"):
            xf = tail(fit, prm, m, cf[m])
            k = float(xf @ (yt[m][fit] - cf[m]) / (xf @ xf))
            ca[m] = ca[m] + k * tail(app, prm, m, ca[m])
    res["S3"] = {m: ca[m].copy() for m in MET}
    if "S4" in steps:
        ca = {m: clip4(app, prm, m, ca[m]) for m in MET}
    res["S4"] = {m: ca[m].copy() for m in MET}
    return res


def score_full(pred, m):
    return float(np.sqrt(np.mean((yt[m] - pred) ** 2)) / SIG[m])


def crossfit(nsp, seed, steps=("S1", "S3", "S4")):
    full = {s: {m: base[m].copy() for m in MET} for s in ("S0", "S1", "S3", "S4")}
    for fit, app in KFold(nsp, shuffle=True, random_state=seed).split(np.arange(n)):
        r = chain_split(fit, app, steps)
        for s in full:
            for m in MET:
                full[s][m][app] = r[s][m]
    sc = {s: {m: score_full(full[s][m], m) for m in MET} for s in full}
    return sc, full


# ---- 2/3. cross-fitted chain on the full OOF ---------------------------------------------------------------------
summary = {}
for nsp, seeds in ((5, range(0, 20)), (2, range(0, 20))):
    rows = []
    for sd in seeds:
        sc, full = crossfit(nsp, sd)
        st = {f"{a}->{b}": sum(sc[b][m] - sc[a][m] for m in MET) / 5
              for a, b in (("S0", "S1"), ("S1", "S3"), ("S3", "S4"), ("S0", "S4"))}
        per = {m: sc["S4"][m] - sc["S0"][m] for m in MET}
        rows.append(dict(st, **{f"{m}_S0S4": v for m, v in per.items()}))
    df = pd.DataFrame(rows)
    summary[f"{nsp}fold"] = {c: dict(mean=float(df[c].mean()), sd=float(df[c].std())) for c in df.columns}
    print(f"full-OOF {nsp}-fold x{len(rows)} seeds: " + " | ".join(f"{c} {df[c].mean():+.8f} (sd {df[c].std():.8f})"
                                                                   for c in df.columns))
out["crossfit"] = summary

# delta-method SE (seed 0, 5-fold) + concentration of the S1 gain on the full OOF
sc, full = crossfit(5, 0)
dtot = np.zeros(n)
d_s1 = {}
for m in MET:
    e0 = (yt[m] - full["S0"][m]) ** 2
    rm0 = np.sqrt(e0.mean())
    dtot += ((yt[m] - full["S4"][m]) ** 2 - e0) / (2 * rm0 * SIG[m]) / n / 5
    d_s1[m] = ((yt[m] - full["S1"][m]) ** 2 - e0) / (2 * rm0 * SIG[m]) / n
se = float(np.std(dtot) * np.sqrt(n))
o = np.argsort(dtot)
print(f"5-fold seed0 first-order total {dtot.sum():+.8f} SE {se:.8f} z {dtot.sum() / se:+.2f}; without top 3/5/10 improved "
      f"{dtot[o[3:]].sum():+.8f} {dtot[o[5:]].sum():+.8f} {dtot[o[10:]].sum():+.8f}; without top+worst 5 "
      f"{dtot[o[5:-5]].sum():+.8f}")
conc = {}
for m in MET:
    di = d_s1[m]
    oo = np.argsort(di)
    conc[m] = dict(all=float(di.sum()), wo_top3=float(di[oo[3:]].sum()), wo_top5=float(di[oo[5:]].sum()),
                   wo_top5_worst5=float(di[oo[5:-5]].sum()), n_imp=int((di < 0).sum()), n_wor=int((di > 0).sum()))
print("S1-only per metric (NRMSE units) on full OOF:", {m: {k: (round(v, 7) if isinstance(v, float) else v)
                                                          for k, v in d.items()} for m, d in conc.items()})
out["seed0_5fold"] = dict(first_order_total=float(dtot.sum()), se=se, s1_conc=conc)

# P->H and H->P with the project holdout split (rows of the full OOF)
Pi, Hi = holdout_split(n)
ph = {}
for nm_, (fit, app) in (("P->H", (Pi, Hi)), ("H->P", (Hi, Pi))):
    r = chain_split(fit, app)
    tot = {}
    for a, b in (("S0", "S1"), ("S1", "S3"), ("S3", "S4"), ("S0", "S4")):
        tot[f"{a}->{b}"] = sum(float(np.sqrt(np.mean((yt[m][app] - r[b][m]) ** 2)) - np.sqrt(np.mean((yt[m][app] - r[a][m]) ** 2)))
                               / SIG[m] for m in MET) / 5
    ph[nm_] = tot
    print(f"full-OOF {nm_}: " + " ".join(f"{k} {v:+.8f}" for k, v in tot.items()))
out["P_H"] = ph

# ---- 4. stability of the small-M bias inside the groups (terciles of A and of the prediction) ----------------------
stab = {}
for m in MET:
    stab[m] = {}
    for lo, hi in GROUPS:
        g = np.where((M >= lo) & (M <= hi))[0]
        for vname, v in (("A", A[g]), ("p", base[m][g])):
            q = np.quantile(v, [1 / 3, 2 / 3])
            tt = np.digitize(v, q)
            for j in range(3):
                gg = g[tt == j]
                p, r = base[m][gg], r0[m][gg]
                stab[m][f"M{lo}-{hi}_{vname}t{j}"] = (len(gg), round(float(p @ r / (p @ p)), 4),
                                                       round(float(np.mean(r / p)), 4),
                                                       round(float(np.std(r / p) / np.sqrt(len(gg))), 4))
    print(m, "LS delta / mean rel (se) by tercile:", stab[m])
out["stability"] = stab
json.dump(out, open(f"{D}/c2_fulloof.json", "w"), indent=1, default=float)
print("wrote", f"{D}/c2_fulloof.json")
