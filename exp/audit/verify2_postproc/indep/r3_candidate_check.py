"""verify2_postproc/indep r3: rebuild the postproc candidate (S1 + S3 + S4 on sub_v19vce) with own code and own
parameters (full-OOF deltas from r2, k / variance model from chain_H), compare it with the auditor's files
exp/audit/postproc/sub_v19vce_pp*.csv, and check the files for blunders (ids, order, untouched columns, duplicates,
bounds, which rows changed and by how much). Writes nothing outside exp/audit/verify2_postproc/indep/.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify2_postproc/indep/r3_candidate_check.py
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import norm

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS  # noqa: E402
from dup_fix import KEY, duplicate_targets  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
COL = {"gas": GAS, "eui": EUI, "elec": EL}
SIG = {m: SIGMAS[c] for m, c in COL.items()}
MET = ["gas", "eui", "elec"]
OUT = "exp/audit/verify2_postproc/indep"
r2 = json.load(open(f"{OUT}/r2_fulloof.json"))
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGET_COLS)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
base = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
files = {k: f"exp/audit/postproc/{k}.csv" for k in ("sub_v19vce_pp", "sub_v19vce_pp_s34", "sub_v19vce_pp_s4")}
cand = {k: pd.read_csv(f, float_precision="round_trip") for k, f in files.items()}
sha = {k: hashlib.sha256(open(f, "rb").read()).hexdigest() for k, f in files.items()}
sha["sub_v19vce"] = hashlib.sha256(open("submissions/sub_v19vce.csv", "rb").read()).hexdigest()
print("sha256", sha, flush=True)
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
dup = duplicate_targets(tr, te)
isdup = ~np.isnan(dup[:, 0])
live = ~isdup
rep = {"n_dup": int(isdup.sum()), "test_M_counts": {"M=1": int((Mt == 1).sum()), "M2-3": int(((Mt >= 2) & (Mt <= 3)).sum()),
                                                    "M=4": int((Mt == 4).sum())},
       "dup_in_smallM": int((isdup & (Mt >= 2) & (Mt <= 4)).sum())}
assert (base["id"].values == te["id"].values).all() and (base["id"].values == ss["id"].values).all()
print(rep, flush=True)

# parameters (own estimates)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
MP, MH, AH = M[P], M[H], A[H]
GROUPS = [(2, 3), (4, 4)]
D_full = {m: r2["deltas_full_oof"][m] for m in MET}
VM, kH = {}, {}


def tn_shift(p, s, U):
    b = (U - p) / s
    return -s * np.exp(norm.logpdf(b) - log_ndtr(b))


for m in MET:
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    VM[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / MP]), (rP / pP) ** 2, rcond=None)[0]
gmaxP = float((z["gas_pP"] + z["gas_rP"]).max())
for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"], z[f"{m}_rP"]
    p = np.maximum(z[f"{m}_corrected"], 0.0)
    for lo, hi in GROUPS:
        gP, gH = (MP >= lo) & (MP <= hi), (MH >= lo) & (MH <= hi)
        p = p.copy()
        p[gH] *= 1 + float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    s = p * np.sqrt(np.maximum(VM[m][0] + VM[m][1] / MH, 1e-8))
    x = tn_shift(p, s, gmaxP if m == "gas" else gmaxP / AH)
    kH[m] = float(x @ (z[f"{m}_y"] - p) / (x @ x))
bt = r2["bounds_train"]
print("deltas", D_full, "k", kH, "var model", {m: VM[m].tolist() for m in MET}, "bounds", bt, flush=True)


def build(s1=True, s3=True, s4=True):
    new = base.copy()
    for m in MET:
        p = base[COL[m]].values.astype(float).copy()
        if s1:
            for (lo, hi), d in zip(GROUPS, D_full[m]):
                g = (Mt >= lo) & (Mt <= hi) & live
                p[g] *= 1 + d
        if s3 and m in kH:
            s = p * np.sqrt(np.maximum(VM[m][0] + VM[m][1] / Mt, 1e-8))
            x = tn_shift(p, s, bt["gmax"] if m == "gas" else bt["gmax"] / At)
            p = np.where(live, p + kH[m] * x, p)
        if s4:
            lo_, hi_ = {"gas": (bt["euimin"] * At, np.full(len(At), bt["gmax"])), "eui": (np.full(len(At), bt["euimin"]), bt["gmax"] / At),
                        "elec": (bt["eamin"] * At, bt["eamax"] * At)}[m]
            p = np.where(live, np.clip(p, lo_, hi_), p)
        new[COL[m]] = p
    return new


mine = {"sub_v19vce_pp": build(), "sub_v19vce_pp_s34": build(s1=False), "sub_v19vce_pp_s4": build(s1=False, s3=False)}
for k in files:
    c, o = cand[k], mine[k]
    chk = {"ids_equal_sample": bool((c["id"].values == ss["id"].values).all()), "columns": list(c.columns) == list(base.columns),
           "rows": len(c), "finite": bool(np.isfinite(c[TARGET_COLS].values).all()),
           "clean_adopt_identical_to_base": bool((c[[CL, AD]].values == base[[CL, AD]].values).all()),
           "dup_rows_identical_to_base": bool(np.array_equal(c.loc[isdup, TARGET_COLS].values, base.loc[isdup, TARGET_COLS].values)),
           "dup_rows_equal_train_targets": bool(np.allclose(c.loc[isdup, TARGET_COLS].values, dup[isdup], rtol=0, atol=1e-9)),
           "nonneg": bool((c[[GAS, EL, EUI]].values >= 0).all()), "adopt_0_100": bool(c[AD].between(0, 100).all())}
    for m in MET:
        cv, ov, bv = c[COL[m]].values, o[COL[m]].values, base[COL[m]].values
        chk[f"{m}_max_absdiff_vs_my_rebuild_over_sigma"] = float(np.max(np.abs(cv - ov)) / SIG[m])
        chk[f"{m}_rows_changed_vs_base"] = int((cv != bv).sum())
        chk[f"{m}_rows_changed_gt_1e-4sigma"] = int((np.abs(cv - bv) > 1e-4 * SIG[m]).sum())
        chk[f"{m}_rms_change_over_sigma"] = float(np.sqrt(np.mean((cv - bv) ** 2)) / SIG[m])
        chk[f"{m}_max_change_over_sigma"] = float(np.max(np.abs(cv - bv)) / SIG[m])
        chk[f"{m}_changed_rows_outside_rules"] = int(((np.abs(cv - bv) > 1e-12 * np.abs(bv)) & isdup).sum())
    # where the S1-type changes are: small-M rows, and their relative factor
    for m in MET:
        rel = c[COL[m]].values / base[COL[m]].values - 1
        for lo, hi in GROUPS:
            g = (Mt >= lo) & (Mt <= hi) & live
            if g.any():
                chk[f"{m}_rel_change_M{lo}-{hi}_median"] = float(np.median(rel[g]))
    print(k, json.dumps(chk), flush=True)
    rep[k] = chk
# eui vs gas consistency of the change: ratio gas / (eui * A) before and after for the changed rows
for k in ("sub_v19vce_pp",):
    c = cand[k]
    r_before = base[GAS].values / (base[EUI].values * At)
    r_after = c[GAS].values / (c[EUI].values * At)
    rep[f"{k}_gas_over_euiA"] = dict(before_median=float(np.median(r_before)), after_median=float(np.median(r_after)),
                                     before_p01_p99=np.quantile(r_before, [.01, .99]).tolist(),
                                     after_p01_p99=np.quantile(r_after, [.01, .99]).tolist())
json.dump(rep, open(f"{OUT}/r3_candidate_check.json", "w"), indent=1)
print("wrote", f"{OUT}/r3_candidate_check.json")
