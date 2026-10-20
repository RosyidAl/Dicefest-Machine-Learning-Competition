"""Audit postproc 13 (copy for v19nca): build the candidate submission sub_v19nca + post-processing (written ONLY under
exp/audit/postproc/, never submitted) and summarise how it differs from sub_v19vce.

Rules (validated honestly on H in 11_combo.py):
  S1 small-M relative shift: p *= 1 + delta for M in {2, 3} and for M == 4 (gas, eui, elec), delta = least-squares
     multiplicative bias of the v19vce VC blend on the full-train OOF (12_fullvc_smallM.json). M == 1 (one test row,
     outside the train support) is left unchanged.
  S3 soft upper cap (gas, eui): p += k * tn_hi(p, s), s = p sqrt(a0 + b0 / M) (a0, b0 from P-OOF residuals),
     upper bound gas_max (gas) / gas_max / A (eui); k fitted on all of H after S1 (as in the validated chain).
  S4 hard clip to the train-implied bounds: gas <= max train gas; eui in [min eui, max gas / A];
     elec in [min elec/A, max elec/A] * A; gas >= min eui * A.
The 77 duplicate rows (exact copies of train targets, src/dup_fix.py) and the clean-ratio / adoption columns are not
touched.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/13_candidate.py [--no-s3] [--out name]
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
COL = {"gas": GAS, "eui": EUI, "elec": EL}
MET = ("gas", "eui", "elec")
ap = argparse.ArgumentParser()
ap.add_argument("--no-s1", action="store_true")
ap.add_argument("--no-s3", action="store_true")
ap.add_argument("--out", default="sub_v19nca2_pp")
args = ap.parse_args()

KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19nca2.csv", float_precision="round_trip")
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
assert (sub["id"].values == te["id"].values).all() and (sub["id"].values == ss["id"].values).all()
# duplicate rows (same rule as src/dup_fix.duplicate_targets)
ref = tr.groupby(KEY, sort=False)[TARGET_COLS].first().reset_index()
dup = te[KEY].merge(ref, on=KEY, how="left")[TARGET_COLS].values
isdup = ~np.isnan(dup[:, 0])
assert isdup.sum() == 77 and np.allclose(sub.loc[isdup, TARGET_COLS].values, dup[isdup], rtol=0, atol=0)
At, Mt = te["bld_dim_06"].values, te["bld_dim_12"].values.astype(float)
Atr = tr["bld_dim_06"].values
B = dict(gas_max=tr[GAS].max(), eui_min=tr[EUI].min(), elecA_min=(tr[EL] / Atr).min(), elecA_max=(tr[EL] / Atr).max())

# parameters
fo = json.load(open("exp/audit/postproc_nca2/12_fullvc_smallM_nca2.json"))
delta = {m: {"M23": fo[m]["M<=3"]["ls_delta"], "M4": fo[m]["M=4"]["ls_delta"]} for m in MET}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, Mall = z["P"], z["H"], z["A"], z["M"]
MP, MH, AH = Mall[P], Mall[H], A[H]
SD = {}
for m in MET:
    p, r = z[f"{m}_pP"], z[f"{m}_rP"]
    SD[m] = np.linalg.lstsq(np.column_stack([np.ones(len(p)), 1 / MP]), (r / p) ** 2, rcond=None)[0]
GMAX_P = (z["gas_pP"] + z["gas_rP"]).max()


def tn_hi(p, s, hi):
    b = (hi - p) / s
    return -s * norm.pdf(b) / np.clip(norm.cdf(b), 1e-300, None)


# k of S3 on all of H, after S1 with P-OOF deltas (exactly the validated chain, 11_combo.py)
kS3 = {}
for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    p = np.clip(z[f"{m}_corrected"], 0, None).copy()
    for lo, hi in ((2, 3), (4, 4)):
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p[gH] *= 1 + float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / MH, 1e-8, None))
    x = tn_hi(p, s, GMAX_P if m == "gas" else GMAX_P / AH)
    kS3[m] = float(x @ (z[f"{m}_y"] - p) / (x @ x))
print("deltas (full-train VC OOF)", {m: {k: round(v, 4) for k, v in d.items()} for m, d in delta.items()})
print("sd model", {m: np.round(v, 5).tolist() for m, v in SD.items()}, "k S3", {m: round(v, 4) for m, v in kS3.items()})
print("bounds", {k: round(float(v), 4) for k, v in B.items()})

new = sub.copy()
live = ~isdup
log = {}
for m in MET:
    p = sub[COL[m]].values.astype(float).copy()
    p0 = p.copy()
    if not args.no_s1:
        for key, g in (("M23", (Mt >= 2) & (Mt <= 3)), ("M4", Mt == 4)):
            g = g & live
            p[g] *= 1 + delta[m][key]
    p1 = p.copy()
    if m in kS3 and not args.no_s3:
        s = p * np.sqrt(np.clip(SD[m][0] + SD[m][1] / Mt, 1e-8, None))
        x = tn_hi(p, s, B["gas_max"] if m == "gas" else B["gas_max"] / At)
        p = np.where(live, p + kS3[m] * x, p)
    p3 = p.copy()
    lo, hi = {"gas": (B["eui_min"] * At, np.full(len(At), B["gas_max"])),
              "eui": (np.full(len(At), B["eui_min"]), B["gas_max"] / At),
              "elec": (B["elecA_min"] * At, B["elecA_max"] * At)}[m]
    p = np.where(live, np.clip(p, lo, hi), p)
    sig = SIGMAS[COL[m]]
    log[m] = dict(rows_S1=int((p1 != p0).sum()), rows_S3=int((np.abs(p3 - p1) > 1e-9 * np.abs(p1)).sum()),
                  rows_S3_over_0p001sigma=int((np.abs(p3 - p1) > 1e-3 * sig).sum()), rows_S4=int((p != p3).sum()),
                  rms_change_over_sigma=float(np.sqrt(np.mean((p - p0) ** 2)) / sig),
                  max_abs_change_over_sigma=float(np.max(np.abs(p - p0)) / sig))
    new[COL[m]] = p
print(json.dumps(log, indent=1))
# checks
assert (new["id"].values == ss["id"].values).all() and list(new.columns) == list(sub.columns)
assert np.isfinite(new[TARGET_COLS].values).all()
assert (new[[GAS, EL, EUI]].values >= 0).all() and new[AD].between(0, 100).all()
assert (new[[CL, AD]].values == sub[[CL, AD]].values).all()
assert np.array_equal(new.loc[isdup, TARGET_COLS].values, sub.loc[isdup, TARGET_COLS].values)
out = f"exp/audit/postproc_nca2/{args.out}.csv"
new.to_csv(out, index=False)
print("wrote", out, "sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())
