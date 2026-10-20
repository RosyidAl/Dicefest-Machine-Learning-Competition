"""verify_postproc/adv v05: expected S1 gain on the actual test rows, allowing the true small-M bias to differ between
rows whose subzone has other train rows (seen) and rows alone in / absent from the train subzones (unseen), with a
bootstrap over the train small-M rows for the uncertainty of the true bias. The candidate's deltas are fixed (full-train
OOF least squares); E[dSE_i] = p_i^2 d (d - 2 b_i).
Train OOF analog of an unseen test row = a train row that is the only train row of its subzone (its OOF prediction has
no subzone encoding). Full-train VC OOF from the previous verifier (verify_postproc/combo/c02_fullvc.npz).

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/adv/v05_s1_test_expectation.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[:3]
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
OUT = "exp/audit/verify_postproc/adv"
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS, EL, EUI])
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
f2 = np.load("exp/audit/verify_postproc/combo/c02_fullvc.npz", allow_pickle=False)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
rmseH = {m: float(np.sqrt(np.mean((z[f"{m}_y"] - np.maximum(z[f"{m}_corrected"], 0)) ** 2))) for m in MET}
Mtr, Mt = tr["bld_dim_12"].values.astype(float), te["bld_dim_12"].values.astype(float)
cnt = tr["subzone_id"].map(tr["subzone_id"].value_counts()).values
alone_tr = cnt == 1
unseen_te = ~te["subzone_id"].isin(set(tr["subzone_id"])).values
ref = tr.groupby(KEY, sort=False)[[GAS]].first().reset_index()
live = np.isnan(te[KEY].merge(ref, on=KEY, how="left")[GAS].values)
nT = len(te)
GROUPS = ((2, 3), (4, 4))
rng = np.random.default_rng(7)
res = {}
tot_pooled = tot_split = 0.0
boot_tot = np.zeros(2000)
for m in MET:
    y, p = tr[COL[m]].values, f2[f"{m}_oof"]
    r = y - p
    pt = sub[COL[m]].values.astype(float)
    for j, (a, b) in enumerate(GROUPS):
        g = (Mtr >= a) & (Mtr <= b)
        d = float(p[g] @ r[g] / (p[g] @ p[g]))                          # the candidate's delta
        ga, gs = g & alone_tr, g & ~alone_tr
        b_a = float(p[ga] @ r[ga] / (p[ga] @ p[ga]))
        b_s = float(p[gs] @ r[gs] / (p[gs] @ p[gs]))
        gt = (Mt >= a) & (Mt <= b) & live
        u = unseen_te[gt]
        p2 = pt[gt] ** 2
        conv = lambda v: v / nT / (2 * rmseH[m]) / SIGMAS[COL[m]] / 5  # noqa: E731  (sum dSE -> TOTAL change)
        e_pool = conv(float(np.sum(p2 * d * (d - 2 * d))))
        e_split = conv(float(np.sum(p2 * d * (d - 2 * np.where(u, b_a, b_s)))))
        # bootstrap the train rows of the group (seen and alone separately) -> distribution of the expected gain
        ia, is_ = np.flatnonzero(ga), np.flatnonzero(gs)
        bt = np.empty(2000)
        for t in range(2000):
            sa, ss = rng.choice(ia, len(ia)), rng.choice(is_, len(is_))
            ba = p[sa] @ r[sa] / (p[sa] @ p[sa])
            bs = p[ss] @ r[ss] / (p[ss] @ p[ss])
            bt[t] = conv(float(np.sum(p2 * d * (d - 2 * np.where(u, ba, bs)))))
        boot_tot += bt
        print(f"{m:4s} M{a}-{b}: d {d:+.4f} | bias train-alone {b_a:+.3f} (n {ga.sum()}) seen {b_s:+.3f} (n {gs.sum()}) | "
              f"test n {gt.sum()} unseen {u.sum()} | expected TOTAL pooled-b {e_pool:+.8f} split-b {e_split:+.8f} "
              f"(boot 5-95% {np.percentile(bt, 5):+.8f} {np.percentile(bt, 95):+.8f})")
        res[f"{m}_M{a}{b}"] = dict(d=d, b_alone=b_a, b_seen=b_s, n_alone=int(ga.sum()), n_seen=int(gs.sum()),
                                  n_test=int(gt.sum()), n_test_unseen=int(u.sum()), exp_pooled=e_pool, exp_split=e_split)
        tot_pooled += e_pool
        tot_split += e_split
print(f"S1 expected TOTAL on test: pooled-b {tot_pooled:+.8f} | seen/unseen-b {tot_split:+.8f} | bootstrap of b: "
      f"mean {boot_tot.mean():+.8f}, 5-95% [{np.percentile(boot_tot, 5):+.8f}, {np.percentile(boot_tot, 95):+.8f}], "
      f"P(expected gain >= 0) {np.mean(boot_tot >= 0):.3f}")
v2 = json.load(open(f"{OUT}/v02_expected_vs_realized.json"))
s3 = v2["T_S3_gas"]["exp_test"] + v2["T_S3_eui"]["exp_test"]
print(f"S3 expected TOTAL on test (v02): {s3:+.8f}; S4 on test (3 elec rows) ~0")
print(f"=> expected TOTAL change on test S1+S3+S4: {tot_split + s3:+.8f} (pooled-b {tot_pooled + s3:+.8f}); "
      f"claim -0.0000652")
res.update(S1_pooled=tot_pooled, S1_split=tot_split, S1_boot=[float(np.percentile(boot_tot, q)) for q in (5, 50, 95)],
           S3=s3, total_split=tot_split + s3, total_pooled=tot_pooled + s3)
json.dump(res, open(f"{OUT}/v05_s1_test_expectation.json", "w"), indent=1, default=float)
