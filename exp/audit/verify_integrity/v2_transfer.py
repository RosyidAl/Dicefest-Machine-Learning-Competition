"""Verifier (leakage/optimism lens) 2: does the truncation shrink transfer from the holdout chain to the test setting?

 a. Full-data OOF: rebuild the v19vce VC blend (A2xM2xp2) of the 27 gas-family members on ALL train rows from
    exp/oof (models trained on 90% of all train, i.e. closer to the test-time models than the 70%-data P models).
    NNLS solved through the normal equations (low memory; not bit-identical to the pipeline, close enough for a check).
    - bias of the full-OOF blend near the bound (all rows, P rows, H rows)
    - claimed shrink (params fitted on P-teacher) applied to the full-OOF blend, scored on H rows (labels of H never
      used for any parameter), and c refit P->H / H->P on the full OOF (k, a0, b0 fixed at the claimed values)
 b. Test side: tail counts of floor area A and meters M (train vs test; an untruncated test would carry an excess of
    large-A rows), and the share of rows whose prediction falls in bands near 1e6 (test vs H chain vs full OOF).
 c. Independent rebuild of the candidate file from submissions/sub_v19vce.csv: max |diff| vs sub_v19vce_trunc2.csv,
    which rows differ, other columns identical.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/v2_transfer.py
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.stats import norm

sys.path.insert(0, "final/src")
from blend import vc_basis  # noqa: E402

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
CL = dict(a0=0.008064793880232542, b0=0.3929692979589114, k=0.7, c_gas=1.0126184473794142, c_eui=1.141696475118523)
OUT = "exp/audit/verify_integrity/v2_transfer.json"
out = {}


def adj(pg, m, k=CL["k"]):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(CL["a0"] + CL["b0"] / m, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


def vc_lowmem(Pf, Pa, yf, spec, vf, va, s=None):
    """blend.blend_vc via normal equations: global NNLS -> 'p' variable -> product hat basis -> NNLS on L^T"""
    s = np.ones(len(yf)) if s is None else s
    w = nnls(Pf * s[:, None], yf * s, maxiter=5000)[0]
    vf = dict(vf, p=Pf @ w)
    va = dict(va, p=Pa @ w)
    Bf, Ba = vc_basis(spec, vf, vf), vc_basis(spec, vf, va)
    nb, nm = Bf.shape[1], Pf.shape[1]
    G = np.zeros((nb * nm, nb * nm))
    g = np.zeros(nb * nm)
    for j in range(nb):
        Xj = Pf * (Bf[:, j] * s)[:, None]
        g[j * nm:(j + 1) * nm] = Xj.T @ (yf * s)
        for l in range(j, nb):
            blk = Xj.T @ (Pf * (Bf[:, l] * s)[:, None])
            G[j * nm:(j + 1) * nm, l * nm:(l + 1) * nm] = blk
            G[l * nm:(l + 1) * nm, j * nm:(j + 1) * nm] = blk.T
    G += np.eye(len(G)) * 1e-12 * np.trace(G) / len(G)
    L = np.linalg.cholesky(G)
    wv = nnls(L.T, np.linalg.solve(L, g), maxiter=20000)[0]
    W = wv.reshape(nb, nm).T
    return ((Pf @ W) * Bf).sum(1), ((Pa @ W) * Ba).sum(1)


tr = pd.read_parquet("data/cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS, EUI])
te = pd.read_parquet("data/cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
yg, ye = tr[GAS].values, tr[EUI].values
n = len(tr)
perm = np.random.default_rng(7).permutation(n)
P, H = np.sort(perm[: int(0.7 * n)]), np.sort(perm[int(0.7 * n):])
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
assert (z["P"] == P).all() and (z["H"] == H).all()
out["max_abs_gas_y_vs_chain"] = float(np.abs(z["gas_y"] - yg[H]).max())

# a. full-data OOF VC blend
names = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
Pf = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in names])
Pa = np.column_stack([np.load(f"exp/pred/{r}.npy") for r in names])
pf_g, pa_g = vc_lowmem(Pf, Pa, ye, "A2xM2xp2", dict(A=A, M=M), dict(A=At, M=Mt), A)
pf_g, pa_g = np.clip(pf_g * A, 0, None), np.clip(pa_g * At, 0, None)
pf_e, pa_e = vc_lowmem(Pf, Pa, ye, "A2xM2xp2", dict(A=A, M=M), dict(A=At, M=Mt), None)
pf_e, pa_e = np.clip(pf_e, 0, None), np.clip(pa_e, 0, None)
del Pf, Pa
out["fullOOF_nrmse"] = dict(gas=nr(yg, pf_g, "gas"), eui=nr(ye, pf_e, "eui"), gas_H=nr(yg[H], pf_g[H], "gas"),
                            eui_H=nr(ye[H], pf_e[H], "eui"))
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
out["test_base_vs_submitted_rms_over_sigma"] = dict(
    gas=float(np.sqrt(np.mean((pa_g - sub[GAS].values) ** 2)) / SIG["gas"]),
    eui=float(np.sqrt(np.mean((pa_e - sub[EUI].values) ** 2)) / SIG["eui"]))

bands = ((7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e6), (1e6, 3e6))
bias = {}
for part, idx in (("all", np.arange(n)), ("P", P), ("H", H)):
    p, y = pf_g[idx], yg[idx]
    pa = p + CL["c_gas"] * adj(p, M[idx])
    bias[part] = {f"{lo:.0f}-{hi:.0f}": dict(n=int(((p >= lo) & (p < hi)).sum()),
                                             resid=float((y - p)[(p >= lo) & (p < hi)].mean()),
                                             resid_after=float((y - pa)[(p >= lo) & (p < hi)].mean()),
                                             se=float((y - p)[(p >= lo) & (p < hi)].std()
                                                      / np.sqrt(max(((p >= lo) & (p < hi)).sum(), 1))))
                  for lo, hi in bands}
out["fullOOF_bias_bands_gas"] = bias

# claimed shrink on the full OOF, scored on H rows (and P rows, all rows for reference)
res = {}
for part, idx in (("H", H), ("P", P), ("all", np.arange(n))):
    r = {}
    for m, p, y, sc in (("gas", pf_g, yg, np.ones(n)), ("eui", pf_e, ye, A)):
        pp, yy, ss = p[idx], y[idx], sc[idx]
        pa = (pp * ss + CL[f"c_{m}"] * adj(pp * ss, M[idx])) / ss
        r[m] = (nr(yy, pa, m) - nr(yy, pp, m)) / 5
    r["total"] = r["gas"] + r["eui"]
    res[part] = r
out["fullOOF_claimed_shrink_delta_total"] = res

# c refit on one part of the full OOF, applied to the other (k, a0, b0 fixed)
cf = {}
for fit, app, lab in ((P, H, "P->H"), (H, P, "H->P")):
    r = {}
    for m, p, y, sc in (("gas", pf_g, yg, np.ones(n)), ("eui", pf_e, ye, A)):
        a_f = adj(p[fit] * sc[fit], M[fit]) / sc[fit]
        c = float(np.sum(a_f * (y[fit] - p[fit])) / np.sum(a_f ** 2))
        a_a = adj(p[app] * sc[app], M[app]) / sc[app]
        r[m] = dict(c=c, delta_total=(nr(y[app], p[app] + c * a_a, m) - nr(y[app], p[app], m)) / 5)
    r["total"] = r["gas"]["delta_total"] + r["eui"]["delta_total"]
    cf[lab] = r
out["fullOOF_c_refit"] = cf
# k on the full OOF: P-fitted c per k, scored on H (does k=0.7 still look right with more training data?)
kg = {}
for k in (0.5, 0.7, 1.0):
    t = 0.0
    for m, p, y, sc in (("gas", pf_g, yg, np.ones(n)), ("eui", pf_e, ye, A)):
        a_f = adj(p[P] * sc[P], M[P], k) / sc[P]
        c = float(np.sum(a_f * (y[P] - p[P])) / np.sum(a_f ** 2))
        a_a = adj(p[H] * sc[H], M[H], k) / sc[H]
        t += (nr(y[H], p[H] + c * a_a, m) - nr(y[H], p[H], m)) / 5
    kg[str(k)] = t
out["fullOOF_k_grid_P_to_H_total"] = kg

# b. test side: tail counts and prediction bands
tails = {}
for lab, a_tr, a_te, cuts in (("A", A, At, (5e3, 7e3, 8e3, 1e4, 1.2e4)), ("M", M, Mt, (60, 80, 100, 120))):
    tails[lab] = {f">={c:g}": dict(train=int((a_tr >= c).sum()), test=int((a_te >= c).sum()),
                                  ratio=float((a_te >= c).sum() / max((a_tr >= c).sum(), 1)))
                  for c in cuts}
tails["ratio_expected_random_split"] = len(te) / n
out["tails"] = tails
pb = {}
for lab, p in (("test_base", pa_g), ("test_submitted", sub[GAS].values), ("H_teacher", z["gas_teacher"]),
               ("H_corrected", z["gas_corrected"]), ("fullOOF_all", pf_g)):
    pb[lab] = {f">={c:.0f}": float((p >= c).mean() * 1e4) for c in (8e5, 9e5, 9.5e5, 1e6)}
out["pred_band_rate_per_10k_rows"] = pb
# train label rate near the bound (per 10k) for reference
out["train_label_rate_per_10k"] = {f">={c:.0f}": float((yg >= c).mean() * 1e4) for c in (8e5, 9e5, 9.5e5, 1e6)}

# c. independent rebuild of the candidate
g, e = sub[GAS].values, sub[EUI].values
g2 = np.clip(g + CL["c_gas"] * adj(g, Mt), 0, None)
e2 = np.clip((e * At + CL["c_eui"] * adj(e * At, Mt)) / At, 0, None)
cand = pd.read_csv("exp/audit/integrity/sub_v19vce_trunc2.csv", float_precision="round_trip")
assert (cand["id"].values == sub["id"].values).all()
dg, de = np.abs(cand[GAS].values - g2), np.abs(cand[EUI].values - e2)
diff_rows = np.where((dg > 1e-6 * np.maximum(g2, 1)) | (de > 1e-6 * np.maximum(e2, 1)))[0]
gas_vals = set(np.round(yg, 6).tolist())
out["candidate"] = dict(
    sha256=hashlib.sha256(open("exp/audit/integrity/sub_v19vce_trunc2.csv", "rb").read()).hexdigest(),
    n_rows_differ_from_my_rebuild=int(len(diff_rows)),
    differing_rows_have_original_values=bool(np.all(cand[GAS].values[diff_rows] == g[diff_rows])),
    differing_rows_gas_is_a_train_value=int(sum(round(float(v), 6) in gas_vals for v in g[diff_rows])),
    max_rel_diff_other_rows=float(np.max(np.delete(dg / np.maximum(g2, 1), diff_rows))),
    other_cols_identical={c: bool((cand[c].values == sub[c].values).all()) for c in sub.columns
                          if c not in ("id", GAS, EUI)},
    gas_changed_rows=int((cand[GAS].values != g).sum()), eui_changed_rows=int((cand[EUI].values != e).sum()),
    gas_max=float(cand[GAS].max()), eui_min=float(cand[EUI].min()))
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
