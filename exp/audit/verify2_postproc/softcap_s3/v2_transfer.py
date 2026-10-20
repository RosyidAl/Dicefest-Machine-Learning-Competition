"""verify2_postproc v2: does the S3 soft cap transfer from the holdout chain to the test predictions?

1. Rebuild the full-train VC teacher (exp/oof members of exp/v19vce, spec from exp/v19vce/resid_knn.json) with the
   pipeline's own blend_preds: OOF prediction on all 429k train rows from models trained on 90% of the FULL train
   (the test models' strength). Measure the near-cap bias there, fit k, and score it honestly (split-half, k fitted on
   one half and scored on the other; also k from P (v1) applied to the full OOF).
2. Apply the rule (k from P, v1; and the claimed k) to submissions/sub_v19vce.csv (non-duplicate rows) and compare the
   standardized-distance profile and the size of the changes with the H corrected chain.
3. Projected test gain: per-band mean squared-error reduction measured on H (k from P) times the test band counts.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/softcap_s3/v2_transfer.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.special import log_ndtr

mem = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable")))
assert mem >= 9_000_000, f"MemAvailable {mem} kB < 9000000, abort"
sys.path.insert(0, "final/src")
sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402
from dup_fix import duplicate_targets  # noqa: E402
from blend import nnls_blend, disagreement, vc_basis, predict_vc  # noqa: E402  (final/src/blend.py)
from scipy.optimize import nnls  # noqa: E402


def blend_preds_light(Pf, Pa, yf, metric, A_fit, A_app, spec, M_fit, M_app, chunk=40000):
    """Same optimisation problem as postprocess.blend_preds (VC NNLS), solved through the Gram matrix in row chunks so
    the 429k x 216 design is never materialised: ||s(Xw - y)||^2 = ||L^T w - L^-1 h||^2 + const with G = X'S^2X = LL^T."""
    s = A_fit if metric == "gas" else np.ones(len(yf))
    w = nnls_blend(Pf, yf, s if metric == "gas" else None)
    vf = dict(A=A_fit, M=M_fit, p=Pf @ w, d=disagreement(Pf))
    va = dict(A=A_app, M=M_app, p=Pa @ w, d=disagreement(Pa))
    Bf, Ba = vc_basis(spec, vf, vf), vc_basis(spec, vf, va)
    nb, npm = Bf.shape[1], Pf.shape[1]
    G = np.zeros((nb * npm, nb * npm))
    h = np.zeros(nb * npm)
    for i in range(0, len(yf), chunk):
        sl = slice(i, i + chunk)
        Xc = np.concatenate([Pf[sl] * Bf[sl, [j]] for j in range(nb)], axis=1) * s[sl, None]
        G += Xc.T @ Xc
        h += Xc.T @ (yf[sl] * s[sl])
    jit = 0.0
    while True:
        try:
            L = np.linalg.cholesky(G + jit * np.eye(len(G)))
            break
        except np.linalg.LinAlgError:
            jit = max(jit * 10, 1e-12 * np.trace(G) / len(G))
    wv, _ = nnls(L.T, np.linalg.solve(L, h), maxiter=50 * len(G))
    W = wv.reshape(nb, npm).T
    pf, pa = predict_vc(Pf, W, Bf), predict_vc(Pa, W, Ba)
    return ((pf * A_fit, pa * A_app) if metric == "gas" else (pf, pa)), jit

GAS, EL, EUI, CL, AD = TARGET_COLS
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI]}
COL = {"gas": GAS, "eui": EUI}
OUTD = "exp/audit/verify2_postproc/softcap_s3"
v1 = json.load(open(f"{OUTD}/v1_reproduce.json"))
VM = {m: v1["var_model"][m]["ols_rel"] for m in SIG}
KP = {m: v1["D1_D2"]["T=claimed"][m]["kP"] for m in SIG}  # k fitted on P only (T = claimed cap)
KCL = {"gas": 0.5046, "eui": 0.6574}  # claimed k (fitted on all H after S1)
T = 999764.8988  # claimed cap = full-train max gas (re-checked below)


def mills_adj(p, s, Tm):
    b = (Tm - p) / s
    return -s * np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))


def sd(m, p, Mv):
    a0, b0 = VM[m]
    return p * np.sqrt(np.clip(a0 + b0 / Mv, 1e-8, None))


def nrmse(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
print("train max gas", tr[GAS].max(), " (claimed T", T, ")  rows with gas >= 1e6:", int((tr[GAS] >= 1e6).sum()))
out = {"kP": KP, "k_claimed": KCL, "var_model": VM}

# ---------- 1. full-train VC teacher ----------
gas_members = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
spec = json.load(open("exp/v19vce/resid_knn.json"))["vc"]
Po = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in gas_members])
Pa = np.column_stack([np.load(f"exp/pred/{r}.npy") for r in gas_members])
yE = tr[EUI].values
full = {}
td = np.load("exp/pl/teacher_deploy.npz", allow_pickle=False)  # project's deployed teacher (test side), cross-check only
for m in ("gas", "eui"):
    (pf, pa), jit = blend_preds_light(Po, Pa, yE, m, A, At, spec[m], M, Mt)
    full[m] = (np.clip(pf, 0, None), np.clip(pa, 0, None))
    rel = np.abs(full[m][1] - td[m]) / td[m]
    print(f"{m}: rebuilt test teacher vs exp/pl/teacher_deploy.npz: max rel diff {rel.max():.2e}, "
          f"rms diff/sigma {np.sqrt(np.mean((full[m][1] - td[m]) ** 2)) / SIG[m]:.2e} (cholesky jitter {jit:.1e})", flush=True)
    out[f"rebuild_check_{m}"] = dict(max_rel_diff=float(rel.max()), jitter=jit)
del Po, Pa
yF = {"gas": tr[GAS].values, "eui": yE}
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
dup = duplicate_targets(tr, te)
isdup = ~np.isnan(dup[:, 0])
live = ~isdup
print("duplicate rows", int(isdup.sum()))
for m in ("gas", "eui"):
    pf, pa = full[m]
    corr = sub[COL[m]].values - pa
    print(f"{m}: full-OOF teacher NRMSE {nrmse(yF[m], pf, m):.6f}; test correction rms/sigma (non-dup) "
          f"{np.sqrt(np.mean(corr[live] ** 2)) / SIG[m]:.5f}")

bands = [(-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)]
out["full_oof"] = {}
for m in ("gas", "eui"):
    pf = full[m][0]
    y = yF[m]
    Tm = T if m == "gas" else T / A
    s = sd(m, pf, M)
    dist, r, adj = (Tm - pf) / s, y - pf, mills_adj(pf, s, Tm)
    tab = []
    print(f"  [full-train OOF teacher] {m}")
    for lo, hi in bands:
        sel = (dist >= lo) & (dist < hi)
        n = int(sel.sum())
        if n < 2:
            continue
        mr, se = float(r[sel].mean()), float(r[sel].std(ddof=1) / np.sqrt(n))
        tab.append(dict(band=[lo, hi], n=n, mean_resid=mr, se=se, resid_over_s=float(np.mean(r[sel] / s[sel])),
                        adj_over_s=float(np.mean(adj[sel] / s[sel]))))
        print(f"    dist [{lo:>4},{hi:>5}) n {n:6d} mean resid {mr:+12.3f} se {se:10.3f} t {mr / se:+5.1f} "
              f" resid/s {tab[-1]['resid_over_s']:+.3f} tn-adj/s {tab[-1]['adj_over_s']:+.3f}")
    k_all = float(adj @ r / (adj @ adj))
    base = nrmse(y, pf, m)
    d_kP = nrmse(y, pf + KP[m] * adj, m) - base
    sh = []
    for seed in range(501, 701):
        h = np.random.default_rng(seed).permutation(len(y)) % 2
        a, b = h == 0, h == 1
        k = float(adj[a] @ r[a] / (adj[a] @ adj[a]))
        sh.append(nrmse(y[b], pf[b] + k * adj[b], m) - nrmse(y[b], pf[b], m))
    sh = np.array(sh)
    out["full_oof"][m] = dict(table=tab, k_insample=k_all, base=base, d_kP=d_kP, total_kP=d_kP / 5,
                              halfscore_mean=float(sh.mean()), halfscore_sd=float(sh.std()),
                              halfscore_frac_neg=float((sh < 0).mean()))
    print(f"    k in-sample (full OOF) {k_all:.4f} vs kP {KP[m]:.4f}; dNRMSE with kP {d_kP:+.7f} (total {d_kP / 5:+.8f}); "
          f"split-half fit/score {sh.mean():+.7f} sd {sh.std():.7f} frac<0 {(sh < 0).mean():.2f}")

# ---------- 2. test application ----------
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, AH, MH = z["H"], z["A"][z["H"]], z["M"][z["H"]]
out["test"] = {}
rng_bands = [(-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3)]
for m in ("gas", "eui"):
    p = sub[COL[m]].values.astype(float)
    Tm = T if m == "gas" else T / At
    s = sd(m, p, Mt)
    dist, adj = (Tm - p) / s, mills_adj(p, s, Tm)
    adj = np.where(live, adj, 0.0)
    ph = z[f"{m}_corrected"]
    TmH = T if m == "gas" else T / AH
    sH = sd(m, ph, MH)
    distH, adjH = (TmH - ph) / sH, mills_adj(ph, sH, TmH)
    prof = {}
    for lo, hi in rng_bands:
        nt = int(((dist >= lo) & (dist < hi) & live).sum())
        nh = int(((distH >= lo) & (distH < hi)).sum())
        prof[f"[{lo},{hi})"] = dict(test=nt, test_per10k=1e4 * nt / live.sum(), H=nh, H_per10k=1e4 * nh / len(ph))
    st = {}
    for kname, kk in (("kP", KP[m]), ("k_claimed", KCL[m])):
        ch, chH = kk * adj, kk * adjH
        st[kname] = dict(k=kk,
                         test_per10k_gt_0p01sig=1e4 * float((np.abs(ch) > 0.01 * SIG[m]).sum()) / live.sum(),
                         H_per10k_gt_0p01sig=1e4 * float((np.abs(chH) > 0.01 * SIG[m]).sum()) / len(ph),
                         test_per10k_gt_0p1sig=1e4 * float((np.abs(ch) > 0.1 * SIG[m]).sum()) / live.sum(),
                         H_per10k_gt_0p1sig=1e4 * float((np.abs(chH) > 0.1 * SIG[m]).sum()) / len(ph),
                         test_rms_over_sig=float(np.sqrt(np.mean(ch[live] ** 2)) / SIG[m]),
                         H_rms_over_sig=float(np.sqrt(np.mean(chH ** 2)) / SIG[m]),
                         test_above_cap_after=int(((p + ch) > Tm)[live].sum()))
    o = np.argsort(adj)[:5]
    big = [dict(row=int(i), p=float(p[i]), M=float(Mt[i]), A=float(At[i]), new_kP=float(p[i] + KP[m] * adj[i]),
                new_claimed=float(p[i] + KCL[m] * adj[i])) for i in o]
    out["test"][m] = dict(profile=prof, change=st, largest=big, n_dup_near_cap=int(((dist < 3) & isdup).sum()))
    print(f"  test {m}: profile (rows per 10k) " + " ".join(f"{b}: test {v['test_per10k']:.2f} / H {v['H_per10k']:.2f}"
                                                         for b, v in prof.items()))
    for kname, v in st.items():
        print(f"    {kname} (k={v['k']:.4f}): >0.01sig test {v['test_per10k_gt_0p01sig']:.1f}/10k vs H {v['H_per10k_gt_0p01sig']:.1f};"
              f" >0.1sig test {v['test_per10k_gt_0p1sig']:.1f} vs H {v['H_per10k_gt_0p1sig']:.1f}; rms/sig test "
              f"{v['test_rms_over_sig']:.5f} vs H {v['H_rms_over_sig']:.5f}; test rows above cap after {v['test_above_cap_after']}")
    print("    largest test changes:", [(round(b['p']), round(b['new_kP']), round(b['new_claimed']), int(b['M'])) for b in big[:3]],
          " dup rows near cap:", out["test"][m]["n_dup_near_cap"])

# ---------- 3. projected test gain from the H per-band SSE reduction ----------
proj_total = 0.0
for m in ("gas", "eui"):
    y, ph = z[f"{m}_y"], z[f"{m}_corrected"]
    TmH = T if m == "gas" else T / AH
    sH = sd(m, ph, MH)
    distH, adjH = (TmH - ph) / sH, mills_adj(ph, sH, TmH)
    dsse_H = (y - ph - KP[m] * adjH) ** 2 - (y - ph) ** 2
    p = sub[COL[m]].values.astype(float)
    Tm = T if m == "gas" else T / At
    s = sd(m, p, Mt)
    dist = (Tm - p) / s
    tot_dsse = 0.0
    for lo, hi in [(-9, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9)]:
        selH = (distH >= lo) & (distH < hi)
        nt = int(((dist >= lo) & (dist < hi) & live).sum())
        tot_dsse += nt * (dsse_H[selH].mean() if selH.any() else 0.0)
    rmse0 = np.sqrt(np.mean((y - ph) ** 2))
    d_nrmse = (np.sqrt(rmse0 ** 2 + tot_dsse / len(p)) - rmse0) / SIG[m]
    dH = (np.sqrt(np.mean((y - ph - KP[m] * adjH) ** 2)) - rmse0) / SIG[m]
    proj_total += d_nrmse / 5
    out.setdefault("projection", {})[m] = dict(d_nrmse_test_proj=float(d_nrmse), d_nrmse_H=float(dH))
    print(f"  projected test dNRMSE {m} {d_nrmse:+.7f} (H measured {dH:+.7f})")
out["projection"]["total"] = proj_total
print(f"  projected TOTAL on test {proj_total:+.8f}")
json.dump(out, open(f"{OUTD}/v2_transfer.json", "w"), indent=1)
print(f"wrote {OUTD}/v2_transfer.json")
