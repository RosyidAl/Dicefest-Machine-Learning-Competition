"""Verify postproc S1, step 2: does the shift transfer to the test rows?

- the residual correction in sub_v19vce uses the FULL train pool for test rows (H used a P-only pool): compare the
  correction's relative shift sub / teacher on small-M test rows with the same ratio on H (corrected / teacher);
  if the denser pool already lifted small-M test rows, S1 would double-correct;
- small-M test rows vs H rows: level of the teacher prediction (p, p / A), so the H gain is representative;
- analytic expected test change with the submission deltas (full-train VC OOF, 12_fullvc_smallM.json) under the
  multiplicative-bias model: dMSE = -(1/N) sum d^2 p^2 on the touched rows, first-order dNRMSE = dMSE / (2 sigma^2 nrmse);
  plus the same with the P-OOF deltas, and the public 30% share;
- check of the candidate file exp/audit/postproc/sub_v19vce_pp.csv: S1 rows scaled by exactly 1 + delta, duplicates
  untouched, other changes (S3/S4) counted.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_postproc/02_test_side.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
COL = {"gas": GAS, "eui": EUI, "elec": EL}
SIG = {m: SIGMAS[c] for m, c in COL.items()}
MET = ("gas", "eui", "elec")
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]

tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
ref = tr.groupby(KEY, sort=False)[TARGET_COLS].first().reset_index()
dup = te[KEY].merge(ref, on=KEY, how="left")[TARGET_COLS].values
isdup = ~np.isnan(dup[:, 0])
Mt, At = te["bld_dim_12"].values.astype(float), te["bld_dim_06"].values
print("test rows: M=1", int((Mt == 1).sum()), "M2-3", int(((Mt >= 2) & (Mt <= 3)).sum()), "M=4", int((Mt == 4).sum()),
      "| duplicates", int(isdup.sum()), "of which M<=4", int((isdup & (Mt <= 4)).sum()))
td = np.load("exp/pl/teacher_deploy.npz", allow_pickle=False)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, A, Mall = z["H"], z["A"], z["M"]
MH, AH = Mall[H], A[H]
groups = {"M2-3": (2, 3), "M4": (4, 4), "M5": (5, 5), "M6-8": (6, 8), "M>=9": (9, 1e9)}

print("\n(1) correction's relative shift on small-M rows: test (full train pool) vs H (P-only pool)")
for m in MET:
    s, t = sub[COL[m]].values, td[m]
    live = ~isdup & (s > 0)
    hc, ht = np.clip(z[f"{m}_corrected"], 0, None), z[f"{m}_teacher"]
    line = []
    for g, (lo, hi) in groups.items():
        gt = live & (Mt >= lo) & (Mt <= hi)
        gh = (MH >= lo) & (MH <= hi) & (ht > 0)
        rt = s[gt].sum() / t[gt].sum() - 1
        rh = hc[gh].sum() / ht[gh].sum() - 1
        line.append(f"{g}: test {rt:+.4f} (n {gt.sum()}) H {rh:+.4f} (n {gh.sum()})")
    print(f"  {m:4s} sum(corr)/sum(teacher) | " + " | ".join(line))

print("\n(2) teacher level on small-M rows: test vs H (mean p, mean p/A, rms p)")
for m in MET:
    t, ht = td[m], z[f"{m}_teacher"]
    for g, (lo, hi) in list(groups.items())[:2]:
        gt = (Mt >= lo) & (Mt <= hi) & ~isdup
        gh = (MH >= lo) & (MH <= hi)
        print(f"  {m:4s} {g}: test n {gt.sum():3d} mean p {t[gt].mean():10.2f} rms p {np.sqrt(np.mean(t[gt] ** 2)):10.2f} "
              f"median A {np.median(At[gt]):7.1f} | H n {gh.sum():3d} mean p {ht[gh].mean():10.2f} "
              f"rms p {np.sqrt(np.mean(ht[gh] ** 2)):10.2f} median A {np.median(AH[gh]):7.1f}")

print("\n(3) analytic expected change (multiplicative-bias model, d_true = d)")
fo = json.load(open("exp/audit/postproc/12_fullvc_smallM.json"))
dfull = {m: (fo[m]["M<=3"]["ls_delta"], fo[m]["M=4"]["ls_delta"]) for m in MET}
nrm = {m: float(np.sqrt(np.mean((z[f"{m}_y"] - np.clip(z[f"{m}_corrected"], 0, None)) ** 2)) / SIG[m]) for m in MET}
print("  submission deltas", {m: [round(v, 5) for v in d] for m, d in dfull.items()})
tot_test, tot_H = 0.0, 0.0
for m in MET:
    s = sub[COL[m]].values
    dm_t, dm_h = 0.0, 0.0
    for (lo, hi), d in zip(((2, 3), (4, 4)), dfull[m]):
        gt = (Mt >= lo) & (Mt <= hi) & ~isdup
        gh = (MH >= lo) & (MH <= hi)
        dm_t -= d * d * np.sum(s[gt] ** 2) / len(s)
        hc = np.clip(z[f"{m}_corrected"], 0, None)
        dm_h -= d * d * np.sum(hc[gh] ** 2) / len(H)
    ct, ch = dm_t / (2 * SIG[m] ** 2 * nrm[m]), dm_h / (2 * SIG[m] ** 2 * nrm[m])
    tot_test += ct / 5
    tot_H += ch / 5
    print(f"  {m:4s}: expected dNRMSE test {ct:+.7f} | same formula on H rows {ch:+.7f}")
print(f"  TOTAL expected: test {tot_test:+.8f} | H (formula) {tot_H:+.8f}")

print("\n(4) candidate file exp/audit/postproc/sub_v19vce_pp.csv vs sub_v19vce")
for name in ("sub_v19vce_pp", "sub_v19vce_pp_s4", "sub_v19vce_pp_s34"):
    cand = pd.read_csv(f"exp/audit/postproc/{name}.csv", float_precision="round_trip")
    assert (cand["id"].values == sub["id"].values).all()
    info = {}
    for m in MET:
        a, b = sub[COL[m]].values, cand[COL[m]].values
        ch = a != b
        g23 = (Mt >= 2) & (Mt <= 3) & ~isdup
        g4 = (Mt == 4) & ~isdup
        r23 = b[g23] / a[g23] - 1
        r4 = b[g4] / a[g4] - 1
        info[m] = dict(changed=int(ch.sum()), changed_not_smallM=int((ch & ~(g23 | g4)).sum()),
                       dup_changed=int((ch & isdup).sum()), M1_changed=int((ch & (Mt == 1)).sum()),
                       r23=(round(float(r23.min()), 5), round(float(r23.max()), 5)),
                       r4=(round(float(r4.min()), 5), round(float(r4.max()), 5)))
    same_other = all((cand[c].values == sub[c].values).all() for c in (CL, AD))
    print(f"  {name}: clean/adopt identical {same_other}; " + json.dumps(info))
