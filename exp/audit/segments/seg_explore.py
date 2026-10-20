"""Segment-level bias scan of the honest holdout-chain predictions (exp/audit/chain_H.npz, m_corrected).

For gas / eui / elec: mean residual (y - corrected) by segment, its t-stat, the in-sample 'oracle' gain of removing it
(upper bound) and the cross-fitted gain (shift fitted on 4/5 of H, applied to the other 1/5; 10 repeats with different
fold seeds). Effects expressed as change of the TOTAL score (= per-target NRMSE change / 5).

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_explore.py > exp/audit/segments/seg_explore.log
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M, subn = z["P"], z["H"], z["A"], z["M"], z["subn"]
tr = pd.read_parquet("cache/train.parquet", columns=["zone_region_id", "subzone_id"])
te = pd.read_parquet("cache/test.parquet", columns=["zone_region_id", "subzone_id", "bld_dim_06", "bld_dim_12"])
zone = tr["zone_region_id"].str[4:].astype(int).values
sub_tr = tr["subzone_id"].str[4:].astype(int).values
assert (sub_tr == subn).all(), "subn mismatch"
sub_te = te["subzone_id"].str[4:].astype(int).values

# subzone support counts
cntP = pd.Series(subn[P]).value_counts()
nP_H = pd.Series(subn[H]).map(cntP).fillna(0).astype(int).values
cntT = pd.Series(subn).value_counts()
nT_te = pd.Series(sub_te).map(cntT).fillna(0).astype(int).values
print(f"H rows {len(H)}, unseen-subzone (no P row) share {np.mean(nP_H == 0):.4f}")
print(f"test rows {len(sub_te)}, unseen-subzone (no train row) share {np.mean(nT_te == 0):.4f}")
for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 10 ** 9)]:
    print(f"  support {lo}-{hi}: H {np.mean((nP_H >= lo) & (nP_H <= hi)):.4f}  test {np.mean((nT_te >= lo) & (nT_te <= hi)):.4f}")

AH, MH = A[H], M[H]
segs = {}


def qbin(x, q=10):
    r = pd.Series(x).rank(method="first").values
    return np.minimum((r - 1) * q // len(x), q - 1).astype(int)


segs["A_dec"] = qbin(AH)
segs["M_dec"] = qbin(MH)
segs["AM_dec"] = qbin(AH / MH)
segs["unseen"] = (nP_H == 0).astype(int)
segs["support"] = np.digitize(nP_H, [1, 2, 3, 6, 11])
segs["zone"] = zone[H]
segs["subn_win200"] = subn[H] // 200


def nrmse(e, m):
    return np.sqrt(np.mean(e ** 2)) / SIG[m]


def cf_shift(e, g, shrink=0.0, reps=10):
    """cross-fitted per-segment additive shift; returns mean total-score change over reps and its sd"""
    base = nrmse(e, MET)
    out = []
    for rep in range(reps):
        en = e.copy()
        for a, b in KFold(5, shuffle=True, random_state=rep).split(e):
            s = pd.DataFrame({"g": g[a], "e": e[a]}).groupby("g")["e"].agg(["sum", "count"])
            c = s["sum"] / (s["count"] + shrink)
            en[b] = e[b] - pd.Series(g[b]).map(c).fillna(0).values
        out.append((nrmse(en, MET) - base) / 5)
    return np.mean(out), np.std(out)


for MET in ("gas", "eui", "elec"):
    y, pc = z[f"{MET}_y"], z[f"{MET}_corrected"]
    e = y - pc
    base = nrmse(e, MET)
    print(f"\n=== {MET}: NRMSE {base:.5f}  mean resid/sig {e.mean() / SIG[MET]:+.5f}  t {e.mean() / (e.std() / np.sqrt(len(e))):+.2f}"
          f"  min pred {pc.min():.3g}  n<0 {(pc < 0).sum()}")
    # global linear recalibration y ~ a + b p, cross-fitted
    for name, X in [("shift", np.ones((len(e), 1))), ("slope0", pc[:, None]), ("a+bp", np.column_stack([np.ones(len(e)), pc]))]:
        o = []
        for rep in range(10):
            en = e.copy()
            for a, b in KFold(5, shuffle=True, random_state=rep).split(e):
                beta = np.linalg.lstsq(X[a], e[a], rcond=None)[0]
                en[b] = e[b] - X[b] @ beta
            o.append((nrmse(en, MET) - base) / 5)
        beta = np.linalg.lstsq(X, e, rcond=None)[0]
        print(f"  global {name:6s} beta {np.round(beta, 6)}  cf dTotal {np.mean(o):+.7f} (sd {np.std(o):.7f})")
    for sname, g in segs.items():
        df = pd.DataFrame({"g": g, "e": e, "e2": e ** 2})
        s = df.groupby("g").agg(n=("e", "size"), mean=("e", "mean"), sd=("e", "std"), mse=("e2", "mean"))
        s["t"] = s["mean"] / (s["sd"] / np.sqrt(s["n"]))
        oracle = (np.sqrt((df["e2"].sum() - (s["n"] * s["mean"] ** 2).sum()) / len(e)) / SIG[MET] - base) / 5
        cf0, sd0 = cf_shift(e, g, 0.0)
        cf30, sd30 = cf_shift(e, g, 30.0)
        chi = float((s["t"] ** 2).sum())
        print(f"  [{sname}] groups {len(s)}  sum t^2 {chi:.1f} (df {len(s)})  oracle dTotal {oracle:+.7f}  "
              f"cf dTotal {cf0:+.7f} (sd {sd0:.7f})  cf shrink30 {cf30:+.7f} (sd {sd30:.7f})")
        if len(s) <= 12:
            for gi, r in s.iterrows():
                print(f"     g={gi:>3}  n {int(r['n']):6d}  mean/sig {r['mean'] / SIG[MET]:+.5f}  t {r['t']:+6.2f}  "
                      f"rmse/sig {np.sqrt(r['mse']) / SIG[MET]:.4f}")
