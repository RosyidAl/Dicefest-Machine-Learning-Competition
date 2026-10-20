"""Adversarial re-derivation of the small-meter fix S (bld_dim_12 <= 4) on the honest holdout chain.

Claim under test: factor per metric and group {M<=3, M==4} fitted on the P side (teacher-level fitted prediction pP vs
y_P), applied as 1 + s * (f - 1) to the corrected H predictions (the honest analog of the submission) gives a total
change of -0.0000415 at s = 0.5.

Checks (all parameters from P only unless stated; total = per-metric NRMSE change / 5):
 1. reproduce the per-metric / total change for s in {0.5, 0.75, 1.0}
 2. H-row bootstrap (private-LB size) and public-size subsets (55,197 rows)
 3. joint bootstrap: P rows resampled (factor uncertainty) together with H rows (scoring noise)
 4. how much of the small-M bias the residual correction already absorbs on H (teacher vs corrected relative bias)
 5. concentration: share of the gain carried by the few most helpful rows; gain after dropping them
 6. same factor applied at teacher level on H (consistency)
 7. split-half on H with the factor re-fitted on the other half of H at corrected level (no P information)
usage: python exp/audit/verify_tried/S_verify.py
"""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
MP, MH, AH = M[P], M[H], A[H]
GROUPS = (("<=3", lambda v: v <= 3), ("4", lambda v: v == 4))
D = {m: dict(y=z[f"{m}_y"], c=z[f"{m}_corrected"], t=z[f"{m}_teacher"], pP=z[f"{m}_pP"], yP=z[f"{m}_rP"] + z[f"{m}_pP"])
     for m in SIG}
rng = np.random.default_rng(12345)  # different seed from the auditor's
NB = 1000


def boot_means(e, nb=NB, seed=777, chunk=50):
    """means of e over nb H-row bootstrap resamples; the same resamples for every call (fixed seed)"""
    g = np.random.default_rng(seed)
    out = []
    for i in range(0, nb, chunk):
        idx = g.integers(0, len(H), (min(chunk, nb - i), len(H)), dtype=np.int32)
        out.append(e[idx].mean(1))
    return np.concatenate(out)


def boot_idx(i, seed=999):
    return np.random.default_rng([seed, i]).integers(0, len(H), len(H))


def factors(m, idxP=None):
    pP, yP, Mv = D[m]["pP"], D[m]["yP"], MP
    if idxP is not None:
        pP, yP, Mv = pP[idxP], yP[idxP], Mv[idxP]
    return {k: float(yP[g(Mv)] @ pP[g(Mv)] / (pP[g(Mv)] @ pP[g(Mv)])) for k, g in GROUPS}


def apply(c, f, s):
    out = c.copy()
    for k, g in GROUPS:
        sel = g(MH)
        out[sel] = c[sel] * (1 + s * (f[k] - 1))
    return out


def dn(y, a, b, m, idx=None):
    if idx is None:
        return (np.sqrt(np.mean((y - b) ** 2)) - np.sqrt(np.mean((y - a) ** 2))) / SIG[m]
    return (np.sqrt(np.mean((y[idx] - b[idx]) ** 2)) - np.sqrt(np.mean((y[idx] - a[idx]) ** 2))) / SIG[m]


print("counts: H M<=3", int((MH <= 3).sum()), "H M==4", int((MH == 4).sum()), "| P M<=3", int((MP <= 3).sum()),
      "P M==4", int((MP == 4).sum()), flush=True)
F0 = {m: factors(m) for m in SIG}
print("P factors:", {m: {k: round(v, 4) for k, v in f.items()} for m, f in F0.items()}, flush=True)

# 1 + 2: reproduce and bootstrap over H rows (vectorised: per-row squared errors, resampled means)
for s in (0.5, 0.75, 1.0):
    tot, tb, msg = 0.0, np.zeros(NB), []
    for m in SIG:
        y, c = D[m]["y"], D[m]["c"]
        c2 = apply(c, F0[m], s)
        e0, e1 = (y - c) ** 2, (y - c2) ** 2
        d = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m]
        b = (np.sqrt(boot_means(e1)) - np.sqrt(boot_means(e0))) / SIG[m]
        tot += d / 5
        tb += b / 5
        msg.append(f"{m} {d:+.7f}")
    print(f"[1/2] s={s}: {'; '.join(msg)} | TOTAL {tot:+.8f}  H-boot sd {tb.std():.8f} P(<0) {(tb < 0).mean():.3f} "
          f"5-95% [{np.quantile(tb, .05):+.8f}, {np.quantile(tb, .95):+.8f}]", flush=True)

# public-size subsets (30% of test = 55,197 rows): how visible is the change on the public LB
PUB = np.stack([rng.choice(len(H), 55197, replace=False) for _ in range(400)]).astype(np.int32)
for s in (0.5, 1.0):
    tp = np.zeros(len(PUB))
    for m in SIG:
        y, c = D[m]["y"], D[m]["c"]
        c2 = apply(c, F0[m], s)
        e0, e1 = (y - c) ** 2, (y - c2) ** 2
        tp += (np.sqrt(e1[PUB].mean(1)) - np.sqrt(e0[PUB].mean(1))) / SIG[m] / 5
    print(f"[2] public-size subsets s={s}: mean {tp.mean():+.8f} sd {tp.std():.8f} P(<0) {(tp < 0).mean():.3f} "
          f"P(<= -0.00005) {(tp <= -0.00005).mean():.3f}", flush=True)

# 3: joint bootstrap (P rows for the factor, H rows for the score)
for s in (0.5, 1.0):
    tj = np.zeros(400)
    for i in range(400):
        ip = rng.integers(0, len(P), len(P))
        ih = boot_idx(i)
        for m in SIG:
            y, c = D[m]["y"], D[m]["c"]
            c2 = apply(c, factors(m, ip), s)
            tj[i] += dn(y, c, c2, m, ih) / 5
    print(f"[3] joint P+H bootstrap s={s}: mean {tj.mean():+.8f} sd {tj.std():.8f} P(<0) {(tj < 0).mean():.3f} "
          f"5% {np.quantile(tj, .05):+.8f} 95% {np.quantile(tj, .95):+.8f}", flush=True)

# 4: bias absorbed by the correction on H
for m in SIG:
    y, c, t = D[m]["y"], D[m]["c"], D[m]["t"]
    parts = []
    for k, g in GROUPS:
        sel = g(MH)
        parts.append(f"{k}: teacher rel bias {(y[sel] - t[sel]).sum() / t[sel].sum():+.4f}, corrected "
                     f"{(y[sel] - c[sel]).sum() / c[sel].sum():+.4f}, mean corr/teacher {(c[sel] - t[sel]).sum() / t[sel].sum():+.4f}"
                     f" | LS factor on H corrected {y[sel] @ c[sel] / (c[sel] @ c[sel]):.4f}")
    print(f"[4] {m}: " + " || ".join(parts), flush=True)

# 5: concentration of the gain (s = 0.5)
for m in SIG:
    y, c = D[m]["y"], D[m]["c"]
    c2 = apply(c, F0[m], 0.5)
    di = (y - c2) ** 2 - (y - c) ** 2  # negative = helps
    sel = np.flatnonzero(di != 0)
    order = sel[np.argsort(di[sel])]
    tot_sse = di.sum()
    top5, top10 = di[order[:5]].sum() / tot_sse, di[order[:10]].sum() / tot_sse
    keep = np.ones(len(H), bool)
    keep[order[:5]] = False
    d_drop5 = dn(y, c, c2, m, np.flatnonzero(keep))
    keep[order[:10]] = False
    d_drop10 = dn(y, c, c2, m, np.flatnonzero(keep))
    nh, nhurt = int((di[sel] < 0).sum()), int((di[sel] > 0).sum())
    print(f"[5] {m}: rows changed {len(sel)} helped {nh} hurt {nhurt}; share of SSE gain in top-5 rows {top5:.2f}, "
          f"top-10 {top10:.2f}; total change {dn(y, c, c2, m) / 5:+.8f}, without top-5 {d_drop5 / 5:+.8f}, "
          f"without top-10 {d_drop10 / 5:+.8f}", flush=True)

# 6: factor applied at teacher level on H (no correction)
tot = 0.0
for m in SIG:
    y, t = D[m]["y"], D[m]["t"]
    d = dn(y, t, apply(t, F0[m], 0.5), m)
    tot += d / 5
print(f"[6] teacher-level H, s=0.5: TOTAL {tot:+.8f}", flush=True)

# 7: split-half inside H (factor fitted on the corrected predictions of one half, scored on the other)
res = []
for seed in range(20):
    half = np.random.default_rng(seed).permutation(len(H)) % 2
    tot = 0.0
    for m in SIG:
        y, c = D[m]["y"], D[m]["c"]
        c2 = c.copy()
        for h in (0, 1):
            a, b = half != h, half == h
            for k, g in GROUPS:
                sa, sb = g(MH) & a, g(MH) & b
                f = y[sa] @ c[sa] / (c[sa] @ c[sa])
                c2[sb] = c[sb] * (1 + 0.5 * (f - 1))
        tot += dn(y, c, c2, m) / 5
    res.append(tot)
res = np.array(res)
print(f"[7] H split-half (20 splits), s=0.5: mean {res.mean():+.8f} min {res.min():+.8f} max {res.max():+.8f}", flush=True)
