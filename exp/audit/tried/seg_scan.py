"""Scan for systematic bias of the honest corrected H predictions in rare segments (the gas-cap bias was missed by the
LightGBM residual model PP0008 because it lives in 0.06% of rows). For each segment: n, mean residual, t, and the
in-sample upper bound of removing the segment mean (total-score units). Segments with |t| > 3 are re-checked honestly
(segment mean shift fitted on P teacher-level residuals, applied to H corrected)."""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M, subn = z["P"], z["H"], z["A"], z["M"], z["subn"]
unseenH = ~np.isin(subn[H], np.unique(subn[P]))
print("H rows whose subzone has no P row:", unseenH.mean().round(4), flush=True)


def segs(pred, Ah, Mh, sub, unseen):
    out = {}
    qs = np.quantile(pred, [0, .001, .01, .05, .1, .5, .9, .95, .99, .999, 1])
    for i in range(len(qs) - 1):
        out[f"pred q[{i}]"] = (pred >= qs[i]) & (pred <= qs[i + 1])
    for name, v in (("A", Ah), ("M", Mh), ("A/M", Ah / Mh)):
        q = np.quantile(v, [0, .001, .01, .05, .5, .95, .99, .999, 1])
        for i in range(len(q) - 1):
            out[f"{name} q[{i}]"] = (v >= q[i]) & (v <= q[i + 1])
    out["unseen subzone"] = unseen
    out["seen subzone"] = ~unseen
    for i, (lo, hi) in enumerate(zip(np.linspace(0, sub.max() + 1, 21)[:-1], np.linspace(0, sub.max() + 1, 21)[1:])):
        out[f"subn bin {i}"] = (sub >= lo) & (sub < hi)
    return out


for m in ("gas", "eui", "elec"):
    y, c, t = z[f"{m}_y"], z[f"{m}_corrected"], z[f"{m}_teacher"]
    r = y - c
    mse = np.mean(r ** 2)
    print(f"== {m}", flush=True)
    S = segs(c, A[H], M[H], subn[H], unseenH)
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    SP = segs(pP, A[P], M[P], subn[P], np.zeros(len(P), bool))
    for k, s in S.items():
        n = s.sum()
        if n < 5:
            continue
        mu, sd = r[s].mean(), r[s].std()
        tt = mu / (sd / np.sqrt(n))
        gain = (np.sqrt(mse - n * mu ** 2 / len(r)) - np.sqrt(mse)) / SIG[m] / 5
        if abs(tt) > 3:
            # honest: shift by the P-side mean residual of the same segment definition (thresholds from H quantiles
            # are recomputed on P for prediction / size segments; unseen has no P analog)
            msg = ""
            if k in SP and SP[k].sum() > 0:
                shift = rP[SP[k]].mean()
                c2 = c + s * shift
                d = (np.sqrt(np.mean((y - c2) ** 2)) - np.sqrt(mse)) / SIG[m] / 5
                msg = f" | P-fitted shift {shift:+.4g} -> honest total {d:+.8f}"
            print(f"  {k:16s} n={n:6d} mean resid {mu:+.4g} t={tt:+.1f} in-sample bound total {gain:+.8f}{msg}",
                  flush=True)
