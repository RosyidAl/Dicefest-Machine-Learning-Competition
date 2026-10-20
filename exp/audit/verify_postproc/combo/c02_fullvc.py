"""verify_postproc/combo c02: recompute the actual v19vce varying-coefficient blend (A2xM2xp2, members of
exp/v19vce/members.yaml) on the FULL-train member OOF (in-sample fit, as final/src/pipeline._assemble) and on the test
member predictions (the pre-correction teacher behind submissions/sub_v19vce.csv). Independent numerical path: the VC
NNLS is solved through the Gram matrix with scipy.optimize.lsq_linear (BVLS) instead of scipy.optimize.nnls.

Outputs (exp/audit/verify_postproc/combo/c02_fullvc.npz): {m}_oof (train, metric scale), {m}_test (test teacher,
metric scale) for m in gas / eui / elec. Checks:
  - agreement with the postproc auditor's exp/audit/postproc/14_fullvc_oof.npz (train side);
  - test teacher vs submitted sub_v19vce (the difference is the residual correction + clip + duplicate copy):
    relative correction by M group on test, compared with the same ratio on H (corrected / teacher) - does the denser
    test-time neighbour pool absorb more of the small-M bias than on H?

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/combo/c02_fullvc.py
"""
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import lsq_linear, nnls

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402  (basis definition shared with the pipeline on purpose: it is what is applied)
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
COL = {"gas": GAS, "eui": EUI, "elec": EL}
OUT = "exp/audit/verify_postproc/combo"
SPEC = "A2xM2xp2"
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
A, M = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]


def gram_vc(X, B, y, s, chunk=25000):
    nm, nb = X.shape[1], B.shape[1]
    G, b = np.zeros((nm * nb, nm * nb)), np.zeros(nm * nb)
    for i in range(0, len(y), chunk):
        sl = slice(i, min(i + chunk, len(y)))
        Z = (B[sl, :, None] * X[sl, None, :]).reshape(-1, nb * nm) * s[sl, None]  # column j * nm + i
        G += Z.T @ Z
        b += Z.T @ (y[sl] * s[sl])
    return G, b


def solve_gram(G, b):
    L = np.linalg.cholesky(G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G)))
    res = lsq_linear(L.T, np.linalg.solve(L, b), bounds=(0, np.inf), method="bvls", tol=1e-14, max_iter=20000)
    return res.x


out = {}
for fam, names in (("gas", mem["gas"]), ("elec", mem["elec"])):
    X = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
    Xt = np.column_stack([np.load(f"exp/pred/{n}.npy") for n in names])
    mets = ("eui", "gas") if fam == "gas" else ("elec",)
    for m in mets:
        y = tr[EL].values if m == "elec" else tr[EUI].values
        s = A if m == "gas" else np.ones(len(y))
        w0 = nnls(X * s[:, None], y * s)[0]  # global blend (pipeline nnls_blend) -> the 'p' conditioning variable
        vf = dict(A=A, M=M, p=X @ w0)
        va = dict(A=At, M=Mt, p=Xt @ w0)
        B, Bt = vc_basis(SPEC, vf, vf), vc_basis(SPEC, vf, va)
        w = solve_gram(*gram_vc(X, B, y, s))
        W = w.reshape(B.shape[1], X.shape[1]).T
        pf, pa = ((X @ W) * B).sum(1), ((Xt @ W) * Bt).sum(1)
        if m == "gas":
            pf, pa = pf * A, pa * At
        yy = y * A if m == "gas" else y
        out[f"{m}_oof"], out[f"{m}_test"] = pf, pa
        print(f"{m}: in-sample VC OOF NRMSE {np.sqrt(np.mean((yy - pf) ** 2)) / SIGMAS[COL[m]]:.6f}", flush=True)
np.savez_compressed(f"{OUT}/c02_fullvc.npz", **out)

aud = np.load("exp/audit/postproc/14_fullvc_oof.npz")
for m in ("gas", "eui", "elec"):
    d = np.abs(aud[f"{m}_p"] - out[f"{m}_oof"]) / SIGMAS[COL[m]]
    print(f"{m}: |mine - auditor 14_fullvc_oof| / sigma: max {d.max():.2e} mean {d.mean():.2e}")

# test-time correction by M group vs the same on H
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
trk = pd.read_parquet("cache/train.parquet", columns=KEY)
tek = pd.read_parquet("cache/test.parquet", columns=KEY)
ref = trk.drop_duplicates(KEY).assign(_hit=1)
isdup = tek.merge(ref, on=KEY, how="left")["_hit"].notna().values
print("test duplicate rows", int(isdup.sum()))
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
MH = z["M"][z["H"]]
for m in ("gas", "eui", "elec"):
    ps, pt = sub[COL[m]].values, out[f"{m}_test"]
    line = []
    for name, g, gh in (("M2-3", (Mt >= 2) & (Mt <= 3), (MH >= 2) & (MH <= 3)), ("M4", Mt == 4, MH == 4),
                        ("M5", Mt == 5, MH == 5), ("M>=6", Mt >= 6, MH >= 6)):
        g = g & ~isdup
        rel_t = (ps[g] - pt[g]).sum() / pt[g].sum()
        cH, tH = z[f"{m}_corrected"][gh], z[f"{m}_teacher"][gh]
        rel_h = (cH - tH).sum() / tH.sum()
        line.append(f"{name}: test n {int(g.sum())} corr {rel_t:+.4f} | H n {int(gh.sum())} corr {rel_h:+.4f}")
    print(f"{m:4s} relative correction sum(corrected - teacher) / sum(teacher): " + " ; ".join(line))
