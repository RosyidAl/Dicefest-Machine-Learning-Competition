"""Variants of the small-meter fix S on the holdout chain (parameters from P teacher level, scored on corrected H)."""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M = z["P"], z["H"], z["M"]
MP, MH = M[P], M[H]
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(H), len(H)) for _ in range(300)]
PB = [rng.integers(0, len(P), len(P)) for _ in range(100)]  # P-row bootstrap for factor uncertainty
d = {m: dict(y=z[f"{m}_y"], c=z[f"{m}_corrected"], pP=z[f"{m}_pP"], yP=z[f"{m}_rP"] + z[f"{m}_pP"]) for m in SIG}


def fit(m, groups, idx=None):
    pP, yP, Mv = d[m]["pP"], d[m]["yP"], MP
    if idx is not None:
        pP, yP, Mv = pP[idx], yP[idx], Mv[idx]
    return [float((yP[g(Mv)] @ pP[g(Mv)]) / (pP[g(Mv)] @ pP[g(Mv)])) for g in groups]


def score(groups, shrink=1.0, shared=None, facs=None):
    tot = 0.0
    tb = 0.0
    for m in SIG:
        y, c = d[m]["y"], d[m]["c"]
        f = facs[m] if facs is not None else fit(shared or m, groups)
        c2 = c.copy()
        for g, fv in zip(groups, f):
            s = g(MH)
            c2[s] = c[s] * (1 + shrink * (fv - 1))
        e0, e1 = (y - c) ** 2, (y - c2) ** 2
        tot += (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m] / 5
        tb = tb + np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] / 5 for b in BOOT])
    return tot, np.std(tb), (tb < 0).mean()


G2 = [lambda v: v <= 3, lambda v: v == 4]
G3 = [lambda v: v <= 3, lambda v: v == 4, lambda v: v == 5]
for tag, kw in (("S {<=3,4}", dict(groups=G2)), ("S {<=3,4,5}", dict(groups=G3)),
                ("S {<=3,4} shrink 0.5", dict(groups=G2, shrink=0.5)), ("S {<=3,4} shrink 0.75", dict(groups=G2, shrink=0.75)),
                ("S {<=3,4} shrink 1.25", dict(groups=G2, shrink=1.25)),
                ("S {<=3,4} gas factors for all metrics", dict(groups=G2, shared="gas")),
                ("S {<=4} pooled", dict(groups=[lambda v: v <= 4]))):
    t, sd, p = score(**kw)
    print(f"{tag:42s} TOTAL {t:+.8f} (sd {sd:.8f}, P<0 {p:.3f})", flush=True)
for m in SIG:
    F = np.array([fit(m, G3, b) for b in PB])
    print(f"{m}: P-bootstrap of factors (<=3, 4, 5): mean {np.round(F.mean(0), 4)} sd {np.round(F.std(0), 4)}", flush=True)
