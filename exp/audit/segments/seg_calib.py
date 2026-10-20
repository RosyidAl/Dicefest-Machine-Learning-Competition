"""Segment calibration checks on the honest holdout chain (exp/audit/chain_H.npz), part 2.

1. bias by prediction level (deciles of teacher / corrected, tails), by correction size, by support
2. per-segment slope calibration e ~ a_s + b_s (p - mean_s p), cross-fitted 5-fold x 10 reps
3. per-support-bin correction weights (lambda per bin) and test-mix importance-weighted lambda, cross-fitted
4. range/truncation check: train target range vs H and test predictions (submissions/sub_v19vce.csv)
Total-score change = per-target NRMSE change / 5.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_calib.py > exp/audit/segments/seg_calib.log
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
COL = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M, subn = z["P"], z["H"], z["A"], z["M"], z["subn"]
te = pd.read_parquet("cache/test.parquet", columns=["subzone_id"])
sub_te = te["subzone_id"].str[4:].astype(int).values
ytr = pd.read_parquet("cache/train.parquet", columns=list(COL.values()))
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["id"] + list(COL.values()))
cntP = pd.Series(subn[P]).value_counts()
nP_H = pd.Series(subn[H]).map(cntP).fillna(0).astype(int).values
cntT = pd.Series(subn).value_counts()
nT_te = pd.Series(sub_te).map(cntT).fillna(0).astype(int).values
EDGES = [1, 2, 3, 6]
supH, supT = np.digitize(nP_H, EDGES), np.digitize(nT_te, EDGES)
shareH = np.bincount(supH, minlength=5) / len(supH)
shareT = np.bincount(supT, minlength=5) / len(supT)
wmix = (shareT / shareH)[supH]  # importance weights: H support mix -> test support mix
print("support bins (0,1,2,3-5,6+) H share", np.round(shareH, 4), "test share", np.round(shareT, 4))
REPS = 10


def qbin(x, q=10):
    r = pd.Series(x).rank(method="first").values
    return np.minimum((r - 1) * q // len(x), q - 1).astype(int)


def nr(e, m, w=None):
    w = np.ones(len(e)) if w is None else w
    return np.sqrt(np.sum(w * e ** 2) / np.sum(w)) / SIG[m]


def cf(e, design, m, w_eval=None, w_fit=None, reps=REPS):
    """cross-fitted linear residual model e ~ design; returns mean/sd total change (weighted eval if w_eval)"""
    base = nr(e, m, w_eval)
    out = []
    for rep in range(reps):
        en = e.copy()
        for a, b in KFold(5, shuffle=True, random_state=rep).split(e):
            Xa = design[a] if w_fit is None else design[a] * np.sqrt(w_fit[a])[:, None]
            ea = e[a] if w_fit is None else e[a] * np.sqrt(w_fit[a])
            beta = np.linalg.lstsq(Xa, ea, rcond=None)[0]
            en[b] = e[b] - design[b] @ beta
        out.append((nr(en, m, w_eval) - base) / 5)
    return np.mean(out), np.std(out)


def onehot(g):
    u = np.unique(g)
    return (g[:, None] == u[None, :]).astype(float)


for m in ("gas", "eui", "elec"):
    y, pt, pc = z[f"{m}_y"], z[f"{m}_teacher"], z[f"{m}_corrected"]
    e = y - pc
    print(f"\n=== {m}: NRMSE H {nr(e, m):.5f}  test-support-weighted {nr(e, m, wmix):.5f}")
    # 1. bias tables
    corr = pc - pt
    for name, g in [("teacher_dec", qbin(pt)), ("corrected_dec", qbin(pc)), ("corr_dec", qbin(corr)),
                    ("corrected_pct", qbin(pc, 100))]:
        df = pd.DataFrame({"g": g, "e": e})
        s = df.groupby("g")["e"].agg(["size", "mean", "std"])
        s["t"] = s["mean"] / (s["std"] / np.sqrt(s["size"]))
        c0, sd0 = cf(e, onehot(g), m)
        print(f"  [{name}] sum t^2 {float((s['t'] ** 2).sum()):.1f} (df {len(s)})  cf shift dTotal {c0:+.7f} (sd {sd0:.7f})")
        if name != "corrected_pct":
            print("     mean/sig:", " ".join(f"{v / SIG[m]:+.4f}" for v in s["mean"]), " | t:",
                  " ".join(f"{v:+.1f}" for v in s["t"]))
        else:
            print("     top 3 pct mean/sig:", " ".join(f"{v / SIG[m]:+.4f}" for v in s["mean"].values[-3:]),
                  "t:", " ".join(f"{v:+.1f}" for v in s["t"].values[-3:]))
            print("     bottom 3 pct mean/sig:", " ".join(f"{v / SIG[m]:+.4f}" for v in s["mean"].values[:3]),
                  "t:", " ".join(f"{v:+.1f}" for v in s["t"].values[:3]))
    # slope of residual on the correction itself (is the correction over/under-shooting?) and on teacher
    for name, X in [("e~corr", corr[:, None]), ("e~corr|support", onehot(supH) * corr[:, None]),
                    ("e~corr|Adec", onehot(qbin(A[H], 5)) * corr[:, None])]:
        c0, sd0 = cf(e, X, m)
        print(f"  {name:16s} cf dTotal {c0:+.7f} (sd {sd0:.7f})")
    # 2. per-segment slope calibration
    for sname, g in [("support", supH), ("A_q5", qbin(A[H], 5)), ("M_q5", qbin(M[H], 5)), ("pc_q5", qbin(pc, 5)),
                     ("unseen", (supH == 0).astype(int))]:
        oh = onehot(g)
        pm = pd.Series(pc).groupby(g).transform("mean").values
        X = np.column_stack([oh, oh * (pc - pm)[:, None]])
        c0, sd0 = cf(e, X, m)
        c1, sd1 = cf(e, oh * (pc - pm)[:, None], m)
        c2, sd2 = cf(e, X, m, w_eval=wmix)
        print(f"  slope calib [{sname}] shift+slope cf dTotal {c0:+.7f} (sd {sd0:.7f})  slope-only {c1:+.7f} (sd {sd1:.7f})"
              f"  shift+slope test-mix-weighted eval {c2:+.7f} (sd {sd2:.7f})")
        b = np.linalg.lstsq(X, e, rcond=None)[0]
        k = oh.shape[1]
        print("     in-sample slopes:", " ".join(f"{v:+.4f}" for v in b[k:]))
    # 3. correction weights per support bin, and importance-weighted lambda (relative to the chain's corrected)
    F = z[f"{m}_F"].astype(float)
    rH = y - pt  # teacher residual
    base_cf = []
    rows = {}
    for label, design, wfit in [("global lam", F, None), ("global lam, test-mix weighted fit", F, wmix),
                                ("lam per support bin", np.column_stack([F * oh_[:, None] for oh_ in onehot(supH).T]), None),
                                ("lam per seen/unseen", np.column_stack([F * oh_[:, None] for oh_ in onehot((supH == 0).astype(int)).T]), None)]:
        res_unw, res_w = [], []
        for rep in range(REPS):
            en = np.empty(len(rH))
            for a, b in KFold(5, shuffle=True, random_state=rep).split(rH):
                Xa = design[a] if wfit is None else design[a] * np.sqrt(wfit[a])[:, None]
                ra = rH[a] if wfit is None else rH[a] * np.sqrt(wfit[a])
                beta = np.linalg.lstsq(Xa, ra, rcond=None)[0]
                en[b] = rH[b] - design[b] @ beta
            res_unw.append(nr(en, m))
            res_w.append(nr(en, m, wmix))
        rows[label] = (np.array(res_unw), np.array(res_w))
    g_unw, g_w = rows["global lam"]
    for label, (u, w) in rows.items():
        print(f"  {label:36s} H NRMSE {u.mean():.6f}  test-mix NRMSE {w.mean():.6f}  "
              f"dTotal vs global: H {(u - g_unw).mean() / 5:+.7f} (sd {(u - g_unw).std() / 5:.7f})  "
              f"test-mix {(w - g_w).mean() / 5:+.7f} (sd {(w - g_w).std() / 5:.7f})")
    # 4. range check
    yt = ytr[COL[m]].values
    ps = sub[COL[m]].values
    print(f"  train y range [{yt.min():.6g}, {yt.max():.6g}]  H pred range [{pc.min():.6g}, {pc.max():.6g}]  "
          f"test pred range [{ps.min():.6g}, {ps.max():.6g}]  H pred>max {np.sum(pc > yt.max())}  "
          f"test pred>max {np.sum(ps > yt.max())}  test pred<min {np.sum(ps < yt.min())}  H pred<min {np.sum(pc < yt.min())}")
    for q in (0.999, 0.9999):
        print(f"    train y q{q}: {np.quantile(yt, q):.6g}")
