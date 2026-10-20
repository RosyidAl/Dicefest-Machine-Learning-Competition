"""Honest check of a multiplicative fix for rows with very few meters (M = bld_dim_12 <= 4): factor per M value fitted
on P (teacher-level fitted predictions, least squares y ~ f * pred within the M value), applied to the corrected H
predictions; also cross-fitted 5-fold on H. Bootstrap sd over H rows."""
import numpy as np
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, M = z["P"], z["H"], z["M"]
MH, MP = M[H], M[P]
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(H), len(H)) for _ in range(300)]
KF = list(KFold(5, shuffle=True, random_state=0).split(np.zeros(len(H))))
tot = {}
for variant in ("P-fit per M (2-4 pooled as <=3 / 4)", "P-fit pooled M<=4", "H cross-fit per M"):
    out = []
    for m in ("gas", "eui", "elec"):
        y, c = z[f"{m}_y"], z[f"{m}_corrected"]
        yP = z[f"{m}_rP"] + z[f"{m}_pP"]
        pP = z[f"{m}_pP"]
        c2 = c.copy()
        if variant.startswith("P-fit per M"):
            for grp in ((2, 3), (4,)):
                sp, sh = np.isin(MP, grp), np.isin(MH, grp)
                f = (yP[sp] @ pP[sp]) / (pP[sp] @ pP[sp])
                c2[sh] = c[sh] * f
                if m == "gas" or True:
                    pass
            fs = {g: round(float((yP[np.isin(MP, g)] @ pP[np.isin(MP, g)]) / (pP[np.isin(MP, g)] @ pP[np.isin(MP, g)])), 4)
                  for g in ((2, 3), (4,))}
        elif variant == "P-fit pooled M<=4":
            sp, sh = MP <= 4, MH <= 4
            f = (yP[sp] @ pP[sp]) / (pP[sp] @ pP[sp])
            c2[sh] = c[sh] * f
            fs = round(float(f), 4)
        else:
            fs = []
            for a, b in KF:
                for grp in ((2, 3), (4,)):
                    sa = np.isin(MH[a], grp)
                    sb = np.isin(MH[b], grp)
                    f = (y[a][sa] @ c[a][sa]) / (c[a][sa] @ c[a][sa])
                    idx = b[sb]
                    c2[idx] = c[idx] * f
                    fs.append(round(float(f), 3))
        e0, e1 = (y - c) ** 2, (y - c2) ** 2
        d = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m]
        bs = np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] for b in BOOT])
        print(f"{variant:38s} {m:4s} factors {fs} delta {d:+.7f} (sd {bs.std():.7f}, P<0 {(bs < 0).mean():.2f}) "
              f"total {d / 5:+.8f} rows {int((c2 != c).sum())}", flush=True)
        out.append(d / 5)
    print(f"{variant:38s} TOTAL {sum(out):+.8f}", flush=True)
