"""Independent replication of the small-meter fix S on rows whose labels the forward check never scored.

 1. reverse chain: factor fitted on H (corrected-level least squares per group), applied to the P teacher-level
    predictions pP, scored on the P labels (bootstrap over P rows)
 2. inside P only (no H rows at all): 2-fold cross-fitting over 50 random splits, s in {0.5, 1.0}
 3. symmetric trimming on H (forward check): drop the k most helped AND the k most hurt rows of the group
usage: python exp/audit/verify_tried/S_reverse.py
"""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M = z["P"], z["H"], z["M"]
MP, MH = M[P], M[H]
GROUPS = (("<=3", lambda v: v <= 3), ("4", lambda v: v == 4))
D = {m: dict(y=z[f"{m}_y"], c=z[f"{m}_corrected"], pP=z[f"{m}_pP"], yP=z[f"{m}_rP"] + z[f"{m}_pP"]) for m in SIG}


def dn(y, a, b, m, idx=None):
    if idx is not None:
        y, a, b = y[idx], a[idx], b[idx]
    return (np.sqrt(np.mean((y - b) ** 2)) - np.sqrt(np.mean((y - a) ** 2))) / SIG[m]


# 1. reverse chain
for s in (0.5, 1.0):
    tot, tb = 0.0, np.zeros(300)
    msg = []
    g = np.random.default_rng(5)
    BI = [g.integers(0, len(P), len(P)) for _ in range(300)]
    for m in SIG:
        y, c, pP, yP = D[m]["y"], D[m]["c"], D[m]["pP"], D[m]["yP"]
        p2 = pP.copy()
        fs = {}
        for k, grp in GROUPS:
            sh, sp = grp(MH), grp(MP)
            f = y[sh] @ c[sh] / (c[sh] @ c[sh])
            fs[k] = round(float(f), 4)
            p2[sp] = pP[sp] * (1 + s * (f - 1))
        d = dn(yP, pP, p2, m)
        e0, e1 = (yP - pP) ** 2, (yP - p2) ** 2
        b = np.array([(np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())) / SIG[m] for i in BI])
        tot += d / 5
        tb += b / 5
        msg.append(f"{m} f_H {fs} {d:+.7f}")
    print(f"[1] reverse (fit H corrected -> score P teacher) s={s}: {'; '.join(msg)} | TOTAL {tot:+.8f} "
          f"P-boot sd {tb.std():.8f} P(<0) {(tb < 0).mean():.3f}", flush=True)

# 2. inside P only, 2-fold cross-fitting over random splits
for s in (0.5, 1.0):
    res = []
    per = {m: [] for m in SIG}
    for seed in range(50):
        half = np.random.default_rng(100 + seed).permutation(len(P)) % 2
        tot = 0.0
        for m in SIG:
            pP, yP = D[m]["pP"], D[m]["yP"]
            p2 = pP.copy()
            for h in (0, 1):
                a, b = half != h, half == h
                for k, grp in GROUPS:
                    sa, sb = grp(MP) & a, grp(MP) & b
                    f = yP[sa] @ pP[sa] / (pP[sa] @ pP[sa])
                    p2[sb] = pP[sb] * (1 + s * (f - 1))
            d = dn(yP, pP, p2, m) / 5
            per[m].append(d)
            tot += d
        res.append(tot)
    res = np.array(res)
    print(f"[2] inside-P 2-fold (50 splits) s={s}: mean TOTAL {res.mean():+.8f} sd over splits {res.std():.8f} "
          f"min {res.min():+.8f} max {res.max():+.8f} share<0 {(res < 0).mean():.2f} | per metric "
          + ", ".join(f"{m} {np.mean(v):+.8f}" for m, v in per.items()), flush=True)

# 3. symmetric trimming on the forward H check (P factors, s = 0.5)
F0 = {m: {k: float(D[m]["yP"][g(MP)] @ D[m]["pP"][g(MP)] / (D[m]["pP"][g(MP)] @ D[m]["pP"][g(MP)])) for k, g in GROUPS}
      for m in SIG}
for kcut in (0, 3, 5, 10, 20):
    tot = 0.0
    for m in SIG:
        y, c = D[m]["y"], D[m]["c"]
        c2 = c.copy()
        for k, g in GROUPS:
            sel = g(MH)
            c2[sel] = c[sel] * (1 + 0.5 * (F0[m][k] - 1))
        di = (y - c2) ** 2 - (y - c) ** 2
        ch = np.flatnonzero(di != 0)
        o = ch[np.argsort(di[ch])]
        keep = np.ones(len(H), bool)
        if kcut:
            keep[o[:kcut]] = False
            keep[o[-kcut:]] = False
        tot += dn(y, c, c2, m, np.flatnonzero(keep)) / 5
    print(f"[3] H forward, s=0.5, dropping the {kcut} most helped and {kcut} most hurt rows per metric: TOTAL {tot:+.8f}",
          flush=True)
