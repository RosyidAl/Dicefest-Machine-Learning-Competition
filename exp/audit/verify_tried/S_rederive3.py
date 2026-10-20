"""Independent re-derivation of the small-meter factor S (lens tried claim): factors fit on P only
(y_P = pP + rP), applied to H corrected predictions; also leave-subzone-out check on H and P-fit vs H-fit sign."""
import numpy as np
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
print("keys", sorted(z.files))
P, H, M, subn = z["P"], z["H"], z["M"], z["subn"]
print("len P H", len(P), len(H), "overlap", len(np.intersect1d(P, H)))
MP, MH = M[P], M[H]
grp = {"<=3": (MP <= 3, MH <= 3), "4": (MP == 4, MH == 4)}
rng = np.random.default_rng(123)
B = [rng.integers(0, len(H), len(H)) for _ in range(500)]
for s in (0.5, 1.0):
    tot = 0.0; tb = np.zeros(len(B))
    for m in SIG:
        pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
        yP = pP + rP
        y, c = z[f"{m}_y"], z[f"{m}_corrected"]
        v = c.copy(); fs = {}
        for k, (sp, sh) in grp.items():
            f = (yP[sp] @ pP[sp]) / (pP[sp] @ pP[sp]); fs[k] = round(float(f), 4)
            v[sh] *= 1 + s * (f - 1)
        v = np.maximum(v, 0)
        e0, e1 = (y - c) ** 2, (y - v) ** 2
        d = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m] / 5
        tot += d
        tb += np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / SIG[m] / 5 for b in B])
        print(f"s={s} {m} f_P {fs} dTOTAL {d:+.8f}")
    print(f"s={s} TOTAL {tot:+.8f} boot sd {tb.std():.8f} P(<0) {(tb<0).mean():.3f}")
# sanity: no shrink-to-H labels used; check y on P really y (pP+rP) nonneg
for m in SIG:
    print(m, "min yP", float((z[f'{m}_pP'] + z[f'{m}_rP']).min()), "H rel bias M<=3 corrected",
          float(z[f'{m}_y'][MH <= 3].sum() / z[f'{m}_corrected'][MH <= 3].sum()))
