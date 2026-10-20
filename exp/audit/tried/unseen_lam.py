"""Separate correction weights for rows whose subzone has no fit-side row (unseen subzone; 16.8% of H, 8.1% of test):
F augmented with F * unseen (and an unseen intercept scaled by the teacher), lambda cross-fitted 5-fold on H.
Tested in v9 for the group correction only; not re-tested with the kNN features + VC blend."""
import numpy as np
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, subn = z["P"], z["H"], z["subn"]
un = (~np.isin(subn[H], np.unique(subn[P]))).astype(float)
KF = list(KFold(5, shuffle=True, random_state=0).split(np.zeros(len(H))))
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(H), len(H)) for _ in range(300)]
tot = {}
for m in SIG:
    y, t = z[f"{m}_y"], z[f"{m}_teacher"]
    F = z[f"{m}_F"].astype(np.float64)
    r = y - t

    def cf(X):
        pc = np.zeros(len(r))
        for a, b in KF:
            pc[b] = X[b] @ np.linalg.lstsq(X[a], r[a], rcond=None)[0]
        return t + pc

    c0 = cf(F)
    e0 = (y - c0) ** 2
    print(f"{m}: unseen rows mean resid after correction {np.mean((y - c0)[un == 1]):+.4g}, seen {np.mean((y - c0)[un == 0]):+.4g}")
    for tag, X in (("F, F*unseen", np.column_stack([F, F * un[:, None]])),
                   ("F, unseen*teacher", np.column_stack([F, un * t])),
                   ("F, F*unseen, unseen*teacher", np.column_stack([F, F * un[:, None], un * t]))):
        e1 = (y - cf(X)) ** 2
        dd = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m]
        bs = np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] for b in BOOT])
        print(f"  {tag:30s} delta {dd:+.7f} (sd {bs.std():.7f}) total {dd / 5:+.8f}")
