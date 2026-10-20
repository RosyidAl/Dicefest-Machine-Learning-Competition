"""Independent full-train check of the small-meter-count scaling claim (deployment teacher = VC blend A2xM2xp2 of the
v19vce members on the full-train OOF arrays, recomputed here with a chunked normal-equation NNLS so the process stays
small). Teacher-level (no residual correction).

  1. reproduce the VC teacher: compare my test-side VC blend with exp/pl/teacher_deploy.npz (VC blend of the same
     members without correction)
  2. bias per M group on the full-train OOF teacher; deployment factors f = (y.pf)/(pf.pf)
  3. honest repeated K-fold cross-fitting of the rule on full-train rows (fit f on K-1 folds, apply to the held fold)
  4. test exposure: counts of M<=3 / M==4 test rows, overlap with the 77 duplicate rows, relative change of the
     submitted columns under the rule; check of the auditor's candidate file against sub_v19vce for those rows

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_tried/sm2_fulloof.py
"""
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

sys.path.insert(0, "final/src")
from blend import disagreement, vc_basis  # noqa: E402  (basis only; the NNLS solve is my own)

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
T = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
MET = ["gas", "eui", "elec"]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=sorted(set(list(T.values()) + KEY)))
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
A, At = tr["bld_dim_06"].values.astype(float), te["bld_dim_06"].values.astype(float)
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
FAM = {"gas": "gas", "eui": "gas", "elec": "elec"}
PO = {f: np.column_stack([np.load(f"exp/oof/{r}.npy") for r in mem[f]]) for f in ("gas", "elec")}
PT = {f: np.column_stack([np.load(f"exp/pred/{r}.npy") for r in mem[f]]) for f in ("gas", "elec")}
print({f: PO[f].shape for f in PO}, flush=True)


def my_nnls(X_fn, nrow, ncol, y, s, chunk=40000):
    """argmin_{w>=0} ||s*(y - X w)|| via the Gram matrix (chunked) and NNLS on its Cholesky factor"""
    G = np.zeros((ncol, ncol))
    b = np.zeros(ncol)
    for c0 in range(0, nrow, chunk):
        sl = slice(c0, min(nrow, c0 + chunk))
        Xc = X_fn(sl) * s[sl, None]
        G += Xc.T @ Xc
        b += Xc.T @ (y[sl] * s[sl])
    jit = 0.0
    for _ in range(8):
        try:
            L = np.linalg.cholesky(G + jit * np.eye(ncol))
            break
        except np.linalg.LinAlgError:
            jit = max(jit * 10, 1e-12 * np.trace(G) / ncol)
    z = np.linalg.solve(L, b)
    w, _ = nnls(L.T, z, maxiter=50 * ncol)
    return w


def vc_teacher(metric):
    Pf, Pa = PO[FAM[metric]], PT[FAM[metric]]
    yf = tr[T["elec" if metric == "elec" else "eui"]].values.astype(float)
    s = A if metric == "gas" else np.ones(len(A))
    w = my_nnls(lambda sl: Pf[sl], len(yf), Pf.shape[1], yf, s)
    vf = dict(A=A, M=M, p=Pf @ w, d=disagreement(Pf))
    va = dict(A=At, M=Mt, p=Pa @ w, d=disagreement(Pa))
    Bf, Ba = vc_basis("A2xM2xp2", vf, vf), vc_basis("A2xM2xp2", vf, va)
    k = Bf.shape[1]
    X_fn = lambda sl: np.concatenate([Pf[sl] * Bf[sl, [j]] for j in range(k)], axis=1)  # noqa: E731
    W = my_nnls(X_fn, len(yf), Pf.shape[1] * k, yf, s).reshape(k, Pf.shape[1]).T
    pf, pa = ((Pf @ W) * Bf).sum(1), ((Pa @ W) * Ba).sum(1)
    if metric == "gas":
        pf, pa = pf * A, pa * At
    return pf, pa


Y = {m: tr[T[m]].values.astype(float) for m in MET}
PF, PA = {}, {}
td = np.load("exp/pl/teacher_deploy.npz", allow_pickle=False)
print("== 1. VC teacher reproduction (test side vs exp/pl/teacher_deploy.npz; train OOF NRMSE)")
for m in MET:
    PF[m], PA[m] = vc_teacher(m)
    ref = td[m]
    print(f"  {m}: OOF NRMSE {np.sqrt(np.mean((Y[m] - PF[m]) ** 2)) / SIG[m]:.6f} | test rms diff vs teacher_deploy "
          f"{np.sqrt(np.mean((PA[m] - ref) ** 2)) / SIG[m]:.2e} sigma, max rel diff "
          f"{np.max(np.abs(PA[m] - ref) / np.maximum(np.abs(ref), 1e-9)):.2e}", flush=True)

GROUPS = {"le3": lambda v: v <= 3, "eq4": lambda v: v == 4}
CTRL = dict(GROUPS, eq5=lambda v: v == 5, eq6=lambda v: v == 6)
print("\n== 2. bias per M group on the full-train OOF VC teacher (deployment teacher)")
for m in MET:
    for g, fn in CTRL.items():
        sl = fn(M)
        r = Y[m][sl] - PF[m][sl]
        f = Y[m][sl] @ PF[m][sl] / (PF[m][sl] @ PF[m][sl])
        print(f"  {m:4s} {g}: n={sl.sum():6d} rel {r.sum() / PF[m][sl].sum():+.4f} f {f:.4f} "
              f"t {r.mean() / (r.std(ddof=1) / np.sqrt(sl.sum())):+.1f} frac y>p {np.mean(r > 0):.2f}")
FAC = {m: {g: float(Y[m][fn(M)] @ PF[m][fn(M)] / (PF[m][fn(M)] @ PF[m][fn(M)])) for g, fn in GROUPS.items()} for m in MET}
print("  deployment factors (full train):", {m: {g: round(v, 4) for g, v in FAC[m].items()} for m in MET})


def apply_rule(pred, Mv, fac, s):
    out = pred.copy()
    for g, fn in GROUPS.items():
        sel = fn(Mv)
        out[sel] = pred[sel] * (1 + s * (fac[g] - 1))
    return out


print("\n== 3. honest repeated K-fold cross-fitting on full-train rows (teacher level)")
n = len(A)
for K, reps in ((2, 60), (5, 30)):
    for s in (0.5, 1.0):
        tots, per = [], {m: [] for m in MET}
        for rep in range(reps):
            fold = np.random.default_rng(20000 + 97 * K + rep).integers(0, K, n)
            tot = 0.0
            for m in MET:
                q = PF[m].copy()
                for k in range(K):
                    fm = fold != k
                    fac = {g: Y[m][fm & fn(M)] @ PF[m][fm & fn(M)] / (PF[m][fm & fn(M)] @ PF[m][fm & fn(M)])
                           for g, fn in GROUPS.items()}
                    ap = ~fm
                    q[ap] = apply_rule(PF[m][ap], M[ap], fac, s)
                d = (np.sqrt(np.mean((Y[m] - q) ** 2)) - np.sqrt(np.mean((Y[m] - PF[m]) ** 2))) / SIG[m]
                per[m].append(d)
                tot += d / 5
            tots.append(tot)
        tots = np.array(tots)
        print(f"  K={K} shrink {s:.2f}: mean total {tots.mean():+.8f} sd {tots.std():.8f} frac<0 {np.mean(tots < 0):.3f} | "
              + " ".join(f"{m} {np.mean(per[m]):+.7f}" for m in MET), flush=True)

print("\n== 4. test exposure")
cnt = {g: int(fn(Mt).sum()) for g, fn in CTRL.items()}
print("  test rows per group:", cnt, "| train:", {g: int(fn(M).sum()) for g, fn in CTRL.items()},
      "| test M<=2:", int((Mt <= 2).sum()), "min test M", Mt.min())
ref = tr.groupby(KEY, sort=False).size().reset_index()[KEY]
ref["_dup"] = 1
hit = te[KEY].merge(ref, on=KEY, how="left")["_dup"].fillna(0).values.astype(bool)
print(f"  duplicate test rows {hit.sum()}; of them M<=4: {int((hit & (Mt <= 4)).sum())}")
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
COL = {"gas": T["gas"], "eui": T["eui"], "elec": T["elec"]}
for m in MET:
    mult = apply_rule(np.ones(len(Mt)), Mt, FAC[m], 0.5)
    v = sub[COL[m]].values
    dv = v * (mult - 1)
    print(f"  {m}: rule (shrink 0.5, full-train factors) changes {int((mult != 1).sum())} rows; rms change over all test "
          f"rows {np.sqrt(np.mean(dv ** 2)) / SIG[m]:.2e} sigma; multipliers le3 {1 + 0.5 * (FAC[m]['le3'] - 1):.4f} "
          f"eq4 {1 + 0.5 * (FAC[m]['eq4'] - 1):.4f}")
cand = pd.read_csv("exp/audit/tried/sub_v19vce_GS_audit.csv", float_precision="round_trip")
print("  candidate ids identical:", bool((cand["id"].values == sub["id"].values).all()),
      "| columns identical:", list(cand.columns) == list(sub.columns))
small = Mt <= 4
for c in sub.columns[1:]:
    a, b = sub[c].values, cand[c].values
    ch = a != b
    msg = f"  {c}: changed rows {int(ch.sum())} (M<=4: {int((ch & small).sum())}, M>4: {int((ch & ~small).sum())})"
    for g, fn in GROUPS.items():
        sel = fn(Mt) & (a > 0)
        if sel.any():
            ratio = b[sel] / a[sel]
            msg += f" | {g} ratio min {ratio.min():.4f} max {ratio.max():.4f}"
    print(msg)
