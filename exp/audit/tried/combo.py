"""Joint honest estimate of the two rare-segment post-processing fixes found by the 'tried' audit:
  G: gas-cap fix. Gas is hard-truncated at 1e6 kWh in the data; predictions above 9e5 are shrunk:
     g' = min(g + beta * max(g - 9e5, 0), 1e6), beta fitted on P (teacher-level); EUI via the implied gas: e' = f(e*A)/A.
  S: small-meter fix. Rows with bld_dim_12 (M) <= 4 are under-predicted by ~20% (M<=3) / ~6% (M=4) in all three
     metrics; multiplicative factor per group fitted on P (teacher-level), applied to gas, EUI, elec.
All parameters from P only; scored on the corrected H predictions (the honest analog of the submission).
Bootstrap: 300 resamples of all H rows (private-LB size = 128,791 rows = |H|) and of 55,197-row subsets (public size)."""
import json

import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
AP, AH, MP, MH = A[P], A[H], M[P], M[H]
CAP, KNOT = 1e6, 9e5
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(H), len(H)) for _ in range(300)]
PUB = [rng.choice(len(H), 55197, replace=False) for _ in range(300)]
d = {m: dict(y=z[f"{m}_y"], c=z[f"{m}_corrected"], pP=z[f"{m}_pP"], yP=z[f"{m}_rP"] + z[f"{m}_pP"]) for m in SIG}
# G: beta on P (gas teacher level)
hP = np.maximum(d["gas"]["pP"] - KNOT, 0)
beta = float(hP @ (d["gas"]["yP"] - d["gas"]["pP"]) / (hP @ hP))
G = lambda g: np.minimum(g + beta * np.maximum(g - KNOT, 0), CAP)  # noqa: E731
# S: factors on P
GROUPS = {"<=3": lambda v: v <= 3, "4": lambda v: v == 4}
fac = {m: {k: float((d[m]["yP"][f(MP)] @ d[m]["pP"][f(MP)]) / (d[m]["pP"][f(MP)] @ d[m]["pP"][f(MP)])) for k, f in GROUPS.items()}
       for m in SIG}
print("beta", round(beta, 4), "factors", json.dumps({m: {k: round(v, 4) for k, v in f.items()} for m, f in fac.items()}))


def apply(m, c, use_g, use_s):
    out = c.copy()
    if use_s:
        for k, f in GROUPS.items():
            s = f(MH)
            out[s] = out[s] * fac[m][k]
    if use_g:
        if m == "gas":
            out = G(out)
        elif m == "eui":
            out = G(out * AH) / AH
    return out


def boot(e0, e1, sig, idxs):
    return np.array([(np.sqrt(e1[b].mean()) - np.sqrt(e0[b].mean())) / sig for b in idxs])


for tag, ug, us in (("G only", True, False), ("S only", False, True), ("G + S", True, True)):
    tot, tb, tpub = 0.0, 0.0, 0.0
    for m in SIG:
        y, c = d[m]["y"], d[m]["c"]
        c2 = apply(m, c, ug, us)
        e0, e1 = (y - c) ** 2, (y - c2) ** 2
        dd = (np.sqrt(e1.mean()) - np.sqrt(e0.mean())) / SIG[m]
        bs, bp = boot(e0, e1, SIG[m], BOOT), boot(e0, e1, SIG[m], PUB)
        tot += dd / 5
        tb = tb + bs / 5
        tpub = tpub + bp / 5
        print(f"  {tag:6s} {m:4s} H {np.sqrt(e0.mean()) / SIG[m]:.6f} -> {np.sqrt(e1.mean()) / SIG[m]:.6f} ({dd:+.7f}) rows {(c2 != c).sum()}")
    print(f"{tag:6s} TOTAL {tot:+.8f}; private-size bootstrap sd {np.std(tb):.8f} P(<0) {(tb < 0).mean():.3f}; "
          f"public-size subset mean {np.mean(tpub):+.8f} sd {np.std(tpub):.8f} P(<0) {(tpub < 0).mean():.3f}", flush=True)
json.dump(dict(beta=beta, knot=KNOT, cap=CAP, factors=fac), open("exp/audit/tried/combo_params_P.json", "w"), indent=1)
