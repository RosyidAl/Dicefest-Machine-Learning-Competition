"""Range / truncation calibration for gas, eui, elec on the honest holdout chain and on the submission.

Bounds are learned from P rows only when scored on H (honest), from all train rows when counted on test.
Checks: absolute caps (gas < 1e6 data filter), per-meter caps/floors (y / bld_dim_12), per-area (EUI) caps, and the
implied EUI cap from the gas cap (EUI <= gas_cap / A).

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_bounds.py > exp/audit/segments/seg_bounds.log
"""
import numpy as np
import pandas as pd

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
COL = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
ytr = pd.read_parquet("cache/train.parquet", columns=list(COL.values()))
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=list(COL.values()))
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)


def nr(e, m):
    return np.sqrt(np.mean(e ** 2)) / SIG[m]


print("train M min", M.min(), "test M min", Mt.min(), " train A min", A.min(), "test A min", At.min())
for m in ("gas", "eui", "elec"):
    y = ytr[COL[m]].values
    print(f"\n=== {m}: train y min {y.min():.6g} max {y.max():.6g}")
    for name, den in [("per meter", M), ("per area", A)]:
        r = y / den
        qs = np.quantile(r, [0, 1e-4, 1e-3, 0.5, 0.999, 0.9999, 1])
        print(f"  y {name}: " + " ".join(f"{v:.5g}" for v in qs))
    # top of the distribution: gap analysis
    ys = np.sort(y)[::-1]
    print("  top 10 y:", " ".join(f"{v:.6g}" for v in ys[:10]))
    print("  bottom 5 y:", " ".join(f"{v:.6g}" for v in ys[-5:][::-1]))

# honest bound application on H: bounds from P rows only
res = {}
for m in ("gas", "eui", "elec"):
    yH, pc = z[f"{m}_y"], z[f"{m}_corrected"]
    yP = ytr[COL[m]].values[P]
    base = nr(yH - pc, m)
    AH, MH = A[H], M[H]
    gasP = ytr[COL["gas"]].values[P]
    cands = {
        "abs max_P": (None, yP.max()),
        "per-meter range_P": ((yP / M[P]).min() * MH, (yP / M[P]).max() * MH),
        "per-area range_P": ((yP / A[P]).min() * AH, (yP / A[P]).max() * AH),
    }
    if m == "gas":
        cands["abs 1e6"] = (None, 1e6)
    if m == "eui":
        cands["gas max_P / A"] = (None, gasP.max() / AH)
        cands["1e6 / A"] = (None, 1e6 / AH)
    for name, (lo, hi) in cands.items():
        pn = np.clip(pc, lo, hi)
        k = int(np.sum(pn != pc))
        print(f"  H {m:4s} clip {name:18s}: rows changed {k:4d}  dNRMSE {nr(yH - pn, m) - base:+.7f}  dTotal {(nr(yH - pn, m) - base) / 5:+.8f}")
    # test: how many rows would change with bounds from all train rows
    y = ytr[COL[m]].values
    ps = sub[COL[m]].values
    tc = {"abs max_train": (None, y.max()), "per-meter range": ((y / M).min() * Mt, (y / M).max() * Mt),
          "per-area range": ((y / A).min() * At, (y / A).max() * At)}
    if m == "gas":
        tc["abs 1e6"] = (None, 1e6)
    if m == "eui":
        tc["gas max / A"] = (None, ytr[COL["gas"]].max() / At)
        tc["1e6 / A"] = (None, 1e6 / At)
    for name, (lo, hi) in tc.items():
        pn = np.clip(ps, lo, hi)
        ch = pn != ps
        print(f"  test {m:4s} clip {name:18s}: rows changed {int(ch.sum()):4d}  mean |change|/sig {np.abs(pn - ps)[ch].mean() / SIG[m] if ch.any() else 0:.4f}"
              f"  max |change| {np.abs(pn - ps).max():.6g}")
    if m == "gas":
        idx = np.argsort(-ps)[:12]
        print("  test top gas preds:", " ".join(f"{v:.6g}" for v in ps[idx]))
        idxH = np.argsort(-pc)[:15]
        print("  H top gas preds / truth:", " ".join(f"{p_:.6g}/{y_:.6g}" for p_, y_ in zip(pc[idxH], yH[idxH])))
