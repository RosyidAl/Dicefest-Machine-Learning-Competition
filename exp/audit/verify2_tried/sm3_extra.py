"""Extra robustness checks for the small-M scaling claim.
  a. holdout chain: P -> H at teacher level (no correction) vs corrected, shrink 0.5
  b. holdout chain + full-train K-fold: add M==5 as a third group (does it help?)
  c. full-train VC teacher: factor stability across spatial thirds (sub_n) and across the original 10 OOF folds
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_tried/sm3_extra.py
"""
import glob
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

sys.path.insert(0, "final/src")
from blend import disagreement, vc_basis  # noqa: E402

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
MET = ["gas", "eui", "elec"]
Z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M = Z["P"], Z["H"], Z["M"]
MP, MH = M[P], M[H]
G2 = {"le3": lambda v: v <= 3, "eq4": lambda v: v == 4}
G3 = dict(G2, eq5=lambda v: v == 5)


def lsf(y, p):
    return float(y @ p / (p @ p))


def apply_rule(pred, Mv, fac, s, groups):
    out = pred.copy()
    for g, fn in groups.items():
        sel = fn(Mv)
        out[sel] = pred[sel] * (1 + s * (fac[g] - 1))
    return out


def nr(y, p, m):
    return np.sqrt(np.mean((y - p) ** 2)) / SIG[m]


print("== a/b. holdout chain P -> H, shrink 0.5")
for groups, gname in ((G2, "{<=3,4}"), (G3, "{<=3,4,5}")):
    for base in ("teacher", "corrected"):
        tot, per = 0.0, {}
        for m in MET:
            yH, bH = Z[f"{m}_y"], Z[f"{m}_{base}"]
            pP = Z[f"{m}_pP"]
            yP = pP + Z[f"{m}_rP"]
            fac = {g: lsf(yP[fn(MP)], pP[fn(MP)]) for g, fn in groups.items()}
            per[m] = nr(yH, apply_rule(bH, MH, fac, 0.5, groups), m) - nr(yH, bH, m)
            tot += per[m] / 5
        print(f"  groups {gname:10s} base {base:9s}: total {tot:+.8f} | " + " ".join(f"{m} {per[m]:+.7f}" for m in MET))

# full-train VC teacher (same construction as sm2_fulloof.py)
T = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
tr = pd.read_parquet("cache/train.parquet", columns=sorted(list(T.values()) + ["bld_dim_06", "bld_dim_12", "subzone_id"]))
A, Mf = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
subn = tr["subzone_id"].str[4:].astype(int).values
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]
PO = {f: np.column_stack([np.load(f"exp/oof/{r}.npy") for r in mem[f]]) for f in ("gas", "elec")}


def my_nnls(X_fn, nrow, ncol, y, s, chunk=40000):
    Gm = np.zeros((ncol, ncol))
    b = np.zeros(ncol)
    for c0 in range(0, nrow, chunk):
        sl = slice(c0, min(nrow, c0 + chunk))
        Xc = X_fn(sl) * s[sl, None]
        Gm += Xc.T @ Xc
        b += Xc.T @ (y[sl] * s[sl])
    L = np.linalg.cholesky(Gm)
    w, _ = nnls(L.T, np.linalg.solve(L, b), maxiter=50 * ncol)
    return w


PF, Y = {}, {}
for m in MET:
    Pf = PO["elec" if m == "elec" else "gas"]
    yf = tr[T["elec" if m == "elec" else "eui"]].values.astype(float)
    s = A if m == "gas" else np.ones(len(A))
    w = my_nnls(lambda sl: Pf[sl], len(yf), Pf.shape[1], yf, s)
    vf = dict(A=A, M=Mf, p=Pf @ w, d=disagreement(Pf))
    Bf = vc_basis("A2xM2xp2", vf, vf)
    k = Bf.shape[1]
    W = my_nnls(lambda sl: np.concatenate([Pf[sl] * Bf[sl, [j]] for j in range(k)], axis=1), len(yf), Pf.shape[1] * k,
                yf, s).reshape(k, Pf.shape[1]).T
    PF[m] = ((Pf @ W) * Bf).sum(1) * (A if m == "gas" else 1.0)
    Y[m] = tr[T[m]].values.astype(float)
    print(f"  full-train VC teacher {m} NRMSE {nr(Y[m], PF[m], m):.6f}", flush=True)

print("\n== b2. full-train repeated 5-fold cross-fit: {<=3,4} vs {<=3,4,5}, shrink 0.5")
n = len(A)
for groups, gname in ((G2, "{<=3,4}"), (G3, "{<=3,4,5}")):
    tots = []
    for rep in range(20):
        fold = np.random.default_rng(31000 + rep).integers(0, 5, n)
        tot = 0.0
        for m in MET:
            q = PF[m].copy()
            for kf in range(5):
                fm = fold != kf
                fac = {g: lsf(Y[m][fm & fn(Mf)], PF[m][fm & fn(Mf)]) for g, fn in groups.items()}
                q[~fm] = apply_rule(PF[m][~fm], Mf[~fm], fac, 0.5, groups)
            tot += (nr(Y[m], q, m) - nr(Y[m], PF[m], m)) / 5
        tots.append(tot)
    print(f"  groups {gname:10s}: mean total {np.mean(tots):+.8f} sd {np.std(tots):.8f}")

print("\n== c. factor stability on the full-train VC teacher")
terc = np.digitize(subn, np.quantile(subn, [1 / 3, 2 / 3]))
for m in MET:
    msg = f"  {m:4s} sub_n thirds:"
    for g, fn in G2.items():
        msg += f" {g} " + "/".join(f"{lsf(Y[m][fn(Mf) & (terc == t)], PF[m][fn(Mf) & (terc == t)]):.3f}(n{int((fn(Mf) & (terc == t)).sum())})"
                                   for t in range(3))
    print(msg)
fold_files = sorted(glob.glob("data/folds/*.csv")) + sorted(glob.glob("data/folds/*.parquet")) + sorted(glob.glob("data/folds/*.npy"))
print("  fold files:", fold_files[:6])
fold = None
for f in fold_files:
    try:
        if f.endswith(".npy"):
            arr = np.load(f, allow_pickle=False)
        elif f.endswith(".parquet"):
            arr = pd.read_parquet(f).iloc[:, -1].values
        else:
            arr = pd.read_csv(f).iloc[:, -1].values
        if len(arr) == n and len(np.unique(arr)) == 10:
            fold = arr
            print("  using fold file", f)
            break
    except Exception as e:  # noqa: BLE001
        print("  skip", f, type(e).__name__)
if fold is not None:
    for m in MET:
        fs = []
        for g, fn in G2.items():
            vals = [lsf(Y[m][fn(Mf) & (fold == k)], PF[m][fn(Mf) & (fold == k)]) for k in np.unique(fold)]
            fs.append(f"{g} min {min(vals):.3f} max {max(vals):.3f} n>1 {sum(v > 1 for v in vals)}/10")
        print(f"  {m:4s} per OOF fold: " + " | ".join(fs))
