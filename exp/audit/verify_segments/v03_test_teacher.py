"""Transfer check: does the residual correction on TEST (all-train neighbour pools) already pull near-cap gas
predictions down, so that the cap calibration would double-correct there?

Rebuilds the v19vce test VC teacher (A2xM2xp2 NNLS on the full-train member OOF, gas metric with row scale A, as
final/src/postprocess.blend_preds) with a memory-light Gram/Cholesky NNLS, then compares near the 1e6 cap:
  test teacher vs test corrected (= sub_v19vce gas, minus the 77 duplicate rows)
  H teacher vs H corrected (chain_H)
and the mean correction (corrected - teacher) for teacher predictions in each near-cap bin.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/v03_test_teacher.py \
       > exp/audit/verify_segments/v03_test_teacher.log
"""
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.special import log_ndtr

sys.path.insert(0, "src")
from blend import vc_basis  # noqa: E402

U, K, LO = 1e6, 0.08, 6e5
LOG2PI = 0.5 * np.log(2 * np.pi)
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]


def cal(g, k=K, cap=U):
    s = k * g
    a = (cap - g) / s
    return g - s * np.exp(-0.5 * a * a - LOG2PI - log_ndtr(a))


def nnls_gram(G, b):
    L = np.linalg.cholesky(G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G)))
    return nnls(L.T, np.linalg.solve(L, b), maxiter=50 * len(G))[0]


tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
ye = tr[EUI].values
ref = tr.groupby(KEY, sort=False)[[GAS]].first().reset_index()
hit = ~np.isnan(te[KEY].merge(ref, on=KEY, how="left")[GAS].values)
names = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
Po = np.column_stack([np.load(f"exp/oof/{n}.npy") for n in names])
Pa = np.column_stack([np.load(f"exp/pred/{n}.npy") for n in names])
print(f"members {len(names)}; OOF {Po.shape}; test {Pa.shape}; dup rows {int(hit.sum())}", flush=True)

sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
teach = {}
for met, s in (("gas", A), ("eui", np.ones(len(A)))):
    w0, _ = nnls(Po * s[:, None], ye * s)
    vf = dict(A=A, M=M, p=Po @ w0)
    va = dict(A=At, M=Mt, p=Pa @ w0)
    Bf, Ba = vc_basis("A2xM2xp2", vf, vf), vc_basis("A2xM2xp2", vf, va)
    G = b = None
    for i in range(0, len(ye), 20000):
        sl = slice(i, min(i + 20000, len(ye)))
        X = np.concatenate([Po[sl] * Bf[sl, [j]] for j in range(Bf.shape[1])], axis=1) * s[sl, None]
        yc = ye[sl] * s[sl]
        G = X.T @ X if G is None else G + X.T @ X
        b = X.T @ yc if b is None else b + X.T @ yc
    W = nnls_gram(G, b).reshape(Bf.shape[1], Po.shape[1]).T
    pa = np.clip(((Pa @ W) * Ba).sum(axis=1), 0, None)
    teach[met] = pa * At if met == "gas" else pa
    col = GAS if met == "gas" else EUI
    corr = sub[col].values - teach[met]
    sc = 151750.04 if met == "gas" else 40.65
    print(f"{met}: rebuilt test teacher; rms(sub - teacher)/sigma over non-dup rows "
          f"{np.sqrt(np.mean(corr[~hit] ** 2)) / sc:.5f}", flush=True)

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
tH, cH = z["gas_teacher"], z["gas_corrected"]
gT, cT = teach["gas"][~hit], sub[GAS].values[~hit]
bins = [0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 9]
print("\nrows per 100k by bin of gas / 1e6:   test teacher | test corrected || H teacher | H corrected")
for lo, hi in zip(bins[:-1], bins[1:]):
    f = lambda p: 1e5 * np.mean((p / U >= lo) & (p / U < hi))  # noqa: E731
    print(f"   [{lo:.2f},{hi:.2f})  {f(gT):7.1f}  {f(cT):7.1f}  ||  {f(tH):7.1f}  {f(cH):7.1f}")
print("\nmean correction (corrected - teacher) by bin of the TEACHER prediction: test | H   (n test, n H)")
for lo, hi in zip(bins[:-1], bins[1:]):
    mt = (gT / U >= lo) & (gT / U < hi)
    mh = (tH / U >= lo) & (tH / U < hi)
    ct = (cT - gT)[mt].mean() if mt.any() else np.nan
    ch = (cH - tH)[mh].mean() if mh.any() else np.nan
    print(f"   [{lo:.2f},{hi:.2f})  {ct:+9.0f}  {ch:+9.0f}   ({int(mt.sum())}, {int(mh.sum())})")
# relative correction (corrected / teacher - 1) for teacher > 0.9e6
mt, mh = gT > 9e5, tH > 9e5
print(f"\nteacher > 0.9e6: mean relative correction test {np.mean(cT[mt] / gT[mt] - 1):+.4f} (n {int(mt.sum())}) | "
      f"H {np.mean(cH[mh] / tH[mh] - 1):+.4f} (n {int(mh.sum())})")
mt, mh = (gT > 6e5) & (gT <= 9e5), (tH > 6e5) & (tH <= 9e5)
print(f"teacher in (0.6e6, 0.9e6]: mean relative correction test {np.mean(cT[mt] / gT[mt] - 1):+.4f} (n {int(mt.sum())}) | "
      f"H {np.mean(cH[mh] / tH[mh] - 1):+.4f} (n {int(mh.sum())})")
# what the calibration does on test if applied to teacher vs corrected (mean shift for rows teacher > 0.9e6)
mt = gT > 9e5
print(f"test rows teacher > 0.9e6: mean (cal(corrected) - corrected) {np.mean(cal(cT[mt]) - cT[mt]):+.0f}; "
      f"mean (cal(teacher) - teacher) {np.mean(cal(gT[mt]) - gT[mt]):+.0f}")
mh = tH > 9e5
print(f"H rows teacher > 0.9e6:    mean (cal(corrected) - corrected) {np.mean(cal(cH[mh]) - cH[mh]):+.0f}; "
      f"mean (cal(teacher) - teacher) {np.mean(cal(tH[mh]) - tH[mh]):+.0f}")
np.save("exp/audit/verify_segments/test_teacher_gas.npy", teach["gas"])
np.save("exp/audit/verify_segments/test_teacher_eui.npy", teach["eui"])
