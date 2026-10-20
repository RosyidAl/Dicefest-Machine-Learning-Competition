"""Robustness of the small-M shift (claim S1): heavy tails, influence and test-side concentration.

1. relative residual r/p in the groups (nested VC fit): mean, median, quartiles, share positive
2. bootstrap of the group delta (rows resampled inside the group): percentile CI, P(d <= 0)
3. delta after dropping the k most influential rows (largest p*r)
4. test side: concentration of sum(p^2) (which drives the expected gain) in the 60 / 298 test rows vs train
5. simulated realized gain for the public (30%) and private (70%) parts of the test: rows of the nested train
   predictions resampled to the test group sizes, deltas re-estimated on a bootstrap of the train rows

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/r2_smallM/s04_robust.py
"""
import json

import numpy as np
import pandas as pd

OUT = "exp/audit/verify2_postproc/r2_smallM"
GAS, EL, EUI = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"
COL = dict(gas=GAS, eui=EUI, elec=EL)
SIG = dict(gas=151750.04, eui=40.65, elec=41725.37)
MET = ("gas", "eui", "elec")
z = np.load(f"{OUT}/s02_fullvc.npz")
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_12", GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_12"])
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
M, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
sel = {"g23": (M == 2) | (M == 3), "g4": M == 4}
selt = {"g23": (Mt == 2) | (Mt == 3), "g4": Mt == 4}
rng = np.random.default_rng(11)
res = {}


def lsd(p, r):
    return float(np.sum(p * r) / np.sum(p * p))


for m in MET:
    y, pn = tr[COL[m]].values, z[f"pn_{m}"]
    r = y - pn
    rmse0 = float(np.sqrt(np.mean(r ** 2)))
    out = {}
    for k in ("g23", "g4"):
        s = sel[k]
        p_, r_ = pn[s], r[s]
        q = r_ / p_
        d = lsd(p_, r_)
        bs = np.array([lsd(p_[i], r_[i]) for i in (rng.integers(0, len(p_), len(p_)) for _ in range(4000))])
        infl = np.argsort(-(p_ * r_))
        drop = {kk: lsd(np.delete(p_, infl[:kk]), np.delete(r_, infl[:kk])) for kk in (1, 3, 5, 10)}
        # test-side concentration of p^2 (sub_v19vce predictions) vs train (nested predictions)
        pt = sub[COL[m]].values[selt[k]]
        share_te = float(np.max(pt ** 2) / np.sum(pt ** 2))
        share_tr = float(np.max(p_ ** 2) / np.sum(p_ ** 2))
        msq_te, msq_tr = float(np.mean(pt ** 2)), float(np.mean(p_ ** 2))
        out[k] = dict(n=int(s.sum()), d=d, rel_mean=float(q.mean()), rel_median=float(np.median(q)),
                      rel_q25=float(np.quantile(q, 0.25)), rel_q75=float(np.quantile(q, 0.75)),
                      share_pos=float(np.mean(q > 0)), boot_ci=(float(np.quantile(bs, 0.025)),
                                                                float(np.quantile(bs, 0.975))),
                      boot_p_le0=float(np.mean(bs <= 0)), drop_top=drop, test_max_share_p2=share_te,
                      train_max_share_p2=share_tr, test_mean_p2_over_train=msq_te / msq_tr)
        print(f"{m:4s} {k:3s} n {s.sum():4d} d {d:+.3f} boot95 [{out[k]['boot_ci'][0]:+.3f},{out[k]['boot_ci'][1]:+.3f}]"
              f" P(d<=0) {out[k]['boot_p_le0']:.4f} | r/p mean {q.mean():+.3f} median {np.median(q):+.3f} "
              f"IQR [{out[k]['rel_q25']:+.3f},{out[k]['rel_q75']:+.3f}] share>0 {np.mean(q > 0):.2f} | d without top "
              f"1/3/5/10 infl rows " + " ".join(f"{drop[kk]:+.3f}" for kk in (1, 3, 5, 10))
              + f" | max row share of sum p^2: train {share_tr:.3f} test {share_te:.3f}; test/train mean p^2 "
              f"{msq_te / msq_tr:.2f}", flush=True)
    # 5. simulated realized gain on test parts: resample (p, y) pairs of the train group rows to the test group size
    sims = {}
    for part, frac in (("public", 0.3), ("private", 0.7)):
        n_tot = int(round(len(Mt) * frac))
        g = np.empty(3000)
        for b in range(3000):
            dmse = 0.0
            for k in ("g23", "g4"):
                s = np.where(sel[k])[0]
                ib = rng.integers(0, len(s), len(s))  # train sample the deltas are estimated from
                dk = lsd(pn[s][ib], r[s][ib])
                nk = rng.binomial(int(selt[k].sum()), frac)  # test rows of the group in this part
                it = rng.integers(0, len(s), nk)  # exchangeable stand-ins for the test rows
                pp, rr = pn[s][it], r[s][it]
                dmse += float(np.sum((rr - dk * pp) ** 2 - rr ** 2))
            g[b] = (np.sqrt(rmse0 ** 2 + dmse / n_tot) - rmse0) / SIG[m]
        sims[part] = g
    out["sim"] = {part: dict(mean=float(v.mean()), sd=float(v.std()), p_pos=float(np.mean(v > 0)))
                  for part, v in sims.items()}
    res[m] = out
    res.setdefault("_sims", {})[m] = sims
print()
for part in ("public", "private"):
    tot = sum(res["_sims"][m][part] for m in MET) / 5
    res[f"total_sim_{part}"] = dict(mean=float(tot.mean()), sd=float(tot.std()), p_pos=float(np.mean(tot > 0)),
                                    q05=float(np.quantile(tot, 0.05)), q95=float(np.quantile(tot, 0.95)))
    print(f"simulated TOTAL change on the {part} part: mean {tot.mean():+.7f} sd {tot.std():.7f} "
          f"P(>0) {np.mean(tot > 0):.3f} 90% [{np.quantile(tot, 0.05):+.7f}, {np.quantile(tot, 0.95):+.7f}]  "
          + " ".join(f"{m} {res[m]['sim'][part]['mean']:+.7f}" for m in MET))
del res["_sims"]
json.dump(res, open(f"{OUT}/s04_robust.json", "w"), indent=1, default=float)
print("wrote", f"{OUT}/s04_robust.json")
