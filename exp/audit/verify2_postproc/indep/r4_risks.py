"""verify2_postproc/indep r4: risk checks for the postproc candidate.
  (1) Is the S1 gain on H carried by a handful of rows? per-row squared-error change, share of the top rows, and the
      gain with the top-k most favourable rows removed (jackknife style)
  (2) Do the test rows look truncated like train (gas < 1e6)? compare the top tail of the test prediction
      (sub_v19vce) with the full-train OOF VC blend (r2) and the size features A, M, A*M of train vs test
  (3) Is the S1 bias stable across sub-samples of the full OOF (zones split in halves, A terciles within small M)?
  (4) Expected public / private change of the candidate: the S1 part on the 359 test rows is fixed by the deltas;
      simulate a random 30/70 public/private split of H-like rows

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/indep/r4_risks.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402

GAS, EL, EUI = TARGET_COLS[0], TARGET_COLS[1], TARGET_COLS[2]
SIG = {"gas": SIGMAS[GAS], "eui": SIGMAS[EUI], "elec": SIGMAS[EL]}
MET = ["gas", "eui", "elec"]
OUT = "exp/audit/verify2_postproc/indep"
GROUPS = [(2, 3), (4, 4)]
rep = {}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
MP, MH = M[P], M[H]
nH = len(H)
y = {m: z[f"{m}_y"] for m in MET}
base = {m: np.maximum(z[f"{m}_corrected"], 0.0) for m in MET}
DP = {}
for m in MET:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    DP[m] = [float(pP[(MP >= lo) & (MP <= hi)] @ rP[(MP >= lo) & (MP <= hi)] / (pP[(MP >= lo) & (MP <= hi)] @ pP[(MP >= lo) & (MP <= hi)]))
             for lo, hi in GROUPS]

# (1) concentration of the S1 gain on H
conc = {}
for m in MET:
    q = base[m].copy()
    for (lo, hi), d in zip(GROUPS, DP[m]):
        q[(MH >= lo) & (MH <= hi)] *= 1 + d
    dse = (y[m] - q) ** 2 - (y[m] - base[m]) ** 2
    g = dse != 0
    srt = np.sort(dse[g])
    mse0 = np.mean((y[m] - base[m]) ** 2)
    rm = np.sqrt(mse0)

    def dnr(total_dse):
        return (np.sqrt(mse0 + total_dse / nH) - rm) / SIG[m]

    conc[m] = dict(rows=int(g.sum()), improved=int((dse < 0).sum()), worsened=int((dse > 0).sum()),
                   dnrmse=float(dnr(dse.sum())), share_top5_improving=float(srt[:5].sum() / dse.sum()),
                   share_top10_improving=float(srt[:10].sum() / dse.sum()),
                   dnrmse_without_top3=float(dnr(dse.sum() - srt[:3].sum())),
                   dnrmse_without_top5=float(dnr(dse.sum() - srt[:5].sum())),
                   dnrmse_without_top10=float(dnr(dse.sum() - srt[:10].sum())),
                   worst5_rows_dse_sum=float(srt[-5:].sum()))
rep["S1_concentration_H"] = conc
print("(1) S1 concentration on H", json.dumps(conc, indent=0), flush=True)

# (2) truncation of test vs train at the top end
fo = np.load(f"{OUT}/r2_fulloof_pred.npz")
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", GAS])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=[GAS, EUI], float_precision="round_trip")
trunc = {}
for lab, v in (("train_oof_gas_pred", fo["gas_p"]), ("test_gas_pred", sub[GAS].values), ("train_gas_true", tr[GAS].values)):
    trunc[lab] = {f"frac_gt_{t}": float(np.mean(v > t)) for t in (5e5, 8e5, 9e5, 9.5e5, 1e6)}
    trunc[lab]["max"] = float(v.max())
    trunc[lab]["q9999"] = float(np.quantile(v, 0.9999))
for lab, df in (("train", tr), ("test", te)):
    a, mm = df["bld_dim_06"].values, df["bld_dim_12"].values.astype(float)
    trunc[f"{lab}_size"] = dict(A_q999=float(np.quantile(a, .999)), A_max=float(a.max()), M_q999=float(np.quantile(mm, .999)),
                                M_max=float(mm.max()), AxM_q999=float(np.quantile(a * mm, .999)))
# gas-per-row expected under "test untruncated": share of train rows near the cap vs test predictions near the cap
rep["truncation"] = trunc
print("(2) truncation", json.dumps(trunc, indent=0), flush=True)

# (3) stability of the full-OOF small-M bias across halves of the data (sub_n parity -> spatial halves) and A terciles
subn = z["subn"]
stab = {}
for m in MET:
    yy, pp = fo[f"{m}_y"], fo[f"{m}_p"]
    r = yy - pp
    out = {}
    for lo, hi in GROUPS:
        g = (M >= lo) & (M <= hi)
        for lab, h in (("subn_even", subn % 2 == 0), ("subn_odd", subn % 2 == 1), ("subn_low", subn < np.median(subn)),
                       ("subn_high", subn >= np.median(subn))):
            gg = g & h
            out[f"M{lo}-{hi}_{lab}"] = [int(gg.sum()), round(float(pp[gg] @ r[gg] / (pp[gg] @ pp[gg])), 4)]
        aq = np.quantile(A[g], [1 / 3, 2 / 3])
        for j, h in enumerate((A <= aq[0], (A > aq[0]) & (A <= aq[1]), A > aq[1])):
            gg = g & h
            out[f"M{lo}-{hi}_Atercile{j}"] = [int(gg.sum()), round(float(pp[gg] @ r[gg] / (pp[gg] @ pp[gg])), 4)]
    stab[m] = out
rep["S1_bias_stability_full_oof"] = stab
print("(3) S1 bias stability (n, LS delta)", json.dumps(stab, indent=0), flush=True)
json.dump(rep, open(f"{OUT}/r4_risks.json", "w"), indent=1)
print("wrote", f"{OUT}/r4_risks.json")
