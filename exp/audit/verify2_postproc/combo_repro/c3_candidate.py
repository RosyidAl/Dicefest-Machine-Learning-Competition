"""verify2_postproc/combo_repro c3: rebuild the claimed candidate(s) from submissions/sub_v19vce.csv with own code and
compare with exp/audit/postproc/sub_v19vce_pp.csv (S1+S3+S4), _pp_s34.csv (S3+S4) and _pp_s4.csv (S4 only); check
ids/order, untouched columns and duplicate rows; model-based expected S1 gain on the test rows and its noise on the
public 30%; transfer check of the small-M under-prediction (test prediction level vs full-OOF level).

Parameters (as the claim specifies): S1 deltas = LS multiplicative bias of the v19vce VC teacher on the full-train OOF
(own rebuild, c2_teacher_oof.npz); S3 noise model (a0, b0) from the P-side residuals and k fitted on all of H after S1
with P deltas (chain_H.npz); S4 bounds from the full train targets. Duplicate rows (own key match) are not touched.
usage (from /root/Dicefest): OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
    exp/audit/verify2_postproc/combo_repro/c3_candidate.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

D = "exp/audit/verify2_postproc/combo_repro"
TC = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm", "target_clean_building_ratio",
      "target_electric_heating_adoption"]
GAS, EL, EUI, CL, AD = TC
COL = {"gas": GAS, "eui": EUI, "elec": EL}
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
MET = ("gas", "eui", "elec")
GROUPS = ((2, 3), (4, 4))
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]

tr = pd.read_parquet("cache/train.parquet", columns=KEY + TC)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
cands = {nm: pd.read_csv(f"exp/audit/postproc/{nm}.csv", float_precision="round_trip")
         for nm in ("sub_v19vce_pp", "sub_v19vce_pp_s34", "sub_v19vce_pp_s4")}
out = {}
assert (sub["id"].values == ss["id"].values).all() and (te["id"].values == ss["id"].values).all()

# duplicates: own key match (left merge keeps the test order)
mg = te[KEY].merge(tr[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
isdup = (mg["_merge"] == "both").values
print("duplicate test rows (own key match):", int(isdup.sum()))
live = ~isdup

A, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
Atr, Mtr = tr["bld_dim_06"].values.astype(float), tr["bld_dim_12"].values.astype(float)
print("test M counts", {k: int((Mt == k).sum()) for k in (1, 2, 3, 4, 5)}, "live M<=4", int(((Mt <= 4) & live).sum()))
Bd = dict(gmax=float(tr[GAS].max()), euimin=float(tr[EUI].min()), eamin=float((tr[EL] / Atr).min()),
          eamax=float((tr[EL] / Atr).max()))
print("bounds (full train)", Bd)

# S1 deltas from the own full-OOF teacher rebuild
t = np.load(f"{D}/c2_teacher_oof.npz")
ytr = {"gas": tr[GAS].values, "eui": tr[EUI].values, "elec": tr[EL].values}
dfull = {}
for m in MET:
    dfull[m] = []
    for lo, hi in GROUPS:
        g = (Mtr >= lo) & (Mtr <= hi)
        p, r = t[m][g], ytr[m][g] - t[m][g]
        dfull[m].append(float(p @ r / (p @ p)))
print("S1 deltas full-OOF", {m: [round(v, 5) for v in d] for m, d in dfull.items()})

# S3 parameters from the chain (P noise model, k on all H after S1 with P deltas, bound = P max)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, Ac, Mc = z["P"], z["H"], z["A"].astype(float), z["M"].astype(float)
nmP, kS3 = {}, {}
gmaxP = float((z["gas_pP"] + z["gas_rP"]).max())


def tail(p, s, T):
    b = (T - p) / s
    return -s * np.exp(-0.5 * b * b - 0.5 * np.log(2 * np.pi) - log_ndtr(b))


for m in ("gas", "eui"):
    pP, rP = z[f"{m}_pP"].astype(float), z[f"{m}_rP"].astype(float)
    nmP[m] = np.linalg.lstsq(np.column_stack([np.ones(len(pP)), 1 / Mc[P]]), (rP / pP) ** 2, rcond=None)[0]
    p = np.clip(z[f"{m}_corrected"].astype(float), 0, None)
    for lo, hi in GROUPS:
        gP, gH = (Mc[P] >= lo) & (Mc[P] <= hi), (Mc[H] >= lo) & (Mc[H] <= hi)
        p[gH] *= 1 + float(pP[gP] @ rP[gP] / (pP[gP] @ pP[gP]))
    s = p * np.sqrt(np.clip(nmP[m][0] + nmP[m][1] / Mc[H], 1e-8, None))
    x = tail(p, s, gmaxP if m == "gas" else gmaxP / Ac[H])
    kS3[m] = float(x @ (z[f"{m}_y"] - p) / (x @ x))
print("S3 noise model", {m: np.round(v, 5).tolist() for m, v in nmP.items()}, "k", {m: round(v, 4) for m, v in kS3.items()})
out["params"] = dict(deltas=dfull, k=kS3, nm={m: v.tolist() for m, v in nmP.items()}, bounds=Bd)


def build(s1=True, s3=True, s4=True):
    new = sub.copy()
    for m in MET:
        p = sub[COL[m]].values.astype(float).copy()
        if s1:
            for (lo, hi), dd in zip(GROUPS, dfull[m]):
                g = (Mt >= lo) & (Mt <= hi) & live
                p[g] *= 1 + dd
        if s3 and m in kS3:
            sd = p * np.sqrt(np.clip(nmP[m][0] + nmP[m][1] / Mt, 1e-8, None))
            T = np.full(len(p), Bd["gmax"]) if m == "gas" else Bd["gmax"] / A
            p = np.where(live, p + kS3[m] * tail(p, sd, T), p)
        if s4:
            lo = {"gas": Bd["euimin"] * A, "eui": np.full(len(A), Bd["euimin"]), "elec": Bd["eamin"] * A}[m]
            hi = {"gas": np.full(len(A), Bd["gmax"]), "eui": Bd["gmax"] / A, "elec": Bd["eamax"] * A}[m]
            p = np.where(live, np.clip(p, lo, hi), p)
        new[COL[m]] = p
    return new


cmp = {}
for nm, kw in (("sub_v19vce_pp", {}), ("sub_v19vce_pp_s34", dict(s1=False)), ("sub_v19vce_pp_s4", dict(s1=False, s3=False))):
    mine, theirs = build(**kw), cands[nm]
    c = {"ids_equal_sample": bool((theirs["id"].values == ss["id"].values).all()),
         "columns_equal": list(theirs.columns) == ["id"] + TC,
         "sha256": hashlib.sha256(open(f"exp/audit/postproc/{nm}.csv", "rb").read()).hexdigest(),
         "clean_adopt_identical_to_base": bool((theirs[[CL, AD]].values == sub[[CL, AD]].values).all()),
         "dup_rows_identical_to_base": bool((theirs.loc[isdup, TC].values == sub.loc[isdup, TC].values).all())}
    for m in MET:
        a, b, b0 = mine[COL[m]].values, theirs[COL[m]].values, sub[COL[m]].values
        c[m] = dict(max_abs_diff_mine_vs_theirs_sigma=float(np.max(np.abs(a - b)) / SIG[m]),
                    rows_changed_vs_base=int((b != b0).sum()),
                    rows_changed_over_1em3sigma=int((np.abs(b - b0) > 1e-3 * SIG[m]).sum()),
                    rms_change_sigma=float(np.sqrt(np.mean((b - b0) ** 2)) / SIG[m]),
                    max_abs_change_sigma=float(np.max(np.abs(b - b0)) / SIG[m]),
                    mean_change_sigma=float(np.mean(b - b0) / SIG[m]))
    cmp[nm] = c
    print(nm, json.dumps(c))
out["compare"] = cmp

# where the candidate differs from base outside S1 rows: are all those rows near the truncation or bounds?
th = cands["sub_v19vce_pp"]
s1rows = (Mt <= 4) & live
for m in ("gas", "eui"):
    dlt = th[COL[m]].values - sub[COL[m]].values
    other = (~s1rows) & (np.abs(dlt) > 1e-3 * SIG[m])
    gas_pred = sub[GAS].values
    print(f"{m}: non-S1 rows changed > 1e-3 sigma: {int(other.sum())}, their base gas pred min "
          f"{gas_pred[other].min() if other.any() else np.nan:.0f}, all changes negative: {bool((dlt[other] < 0).all())}")

# ---- model-based expected S1 gain on the test and its noise on a 30% public subset ------------------------------
nrm_H = {"gas": 0.24514, "eui": 0.64859, "elec": 0.31462}  # corrected chain NRMSE on H (scale of RMSE0)
nt = len(sub)
exp_tot, sd_pub = 0.0, 0.0
detail = {}
for m in MET:
    rm0 = nrm_H[m] * SIG[m]
    e_sse, v_sse = 0.0, 0.0
    for (lo, hi), dd in zip(GROUPS, dfull[m]):
        g = (Mt >= lo) & (Mt <= hi) & live
        p = sub[COL[m]].values[g]
        gt = (Mtr >= lo) & (Mtr <= hi)
        pt, rt = t[m][gt], ytr[m][gt] - t[m][gt]
        vrel = float(np.mean((rt / pt - dd) ** 2))  # residual relative variance after the shift (train)
        e_sse += -dd ** 2 * float(p @ p)
        v_sse += 4 * dd ** 2 * vrel * float(np.sum(p ** 4))
        detail[f"{m}_M{lo}-{hi}"] = dict(n=int(g.sum()), delta=dd, vrel=vrel, sum_p2=float(p @ p))
    e_nr = e_sse / (nt * 2 * rm0 * SIG[m])
    sd_nr_pub = np.sqrt(v_sse * 0.3) / (0.3 * nt * 2 * rm0 * SIG[m])
    detail[m] = dict(expected_nrmse_change=e_nr, public_sd=float(sd_nr_pub))
    exp_tot += e_nr / 5
    sd_pub += (sd_nr_pub / 5) ** 2
print(f"model-based expected S1 total change on test {exp_tot:+.8f}; sd on a 30% public subset {np.sqrt(sd_pub):.8f}")
print("   ", {k: {kk: (round(vv, 7) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in detail.items()})
out["expected_S1_test"] = dict(total=exp_tot, public_sd=float(np.sqrt(sd_pub)), detail=detail)

# ---- transfer: is the test prediction level for small-M rows like the OOF level? ---------------------------------
tf = {}
for lo, hi in GROUPS:
    gt = (Mtr >= lo) & (Mtr <= hi)
    ge = (Mt >= lo) & (Mt <= hi) & live
    for m in MET:
        sc = Atr if m == "gas" else 1.0
        sce = A if m == "gas" else 1.0
        tf[f"{m}_M{lo}-{hi}"] = dict(
            n_tr=int(gt.sum()), n_te=int(ge.sum()),
            train_mean_y_per_A=float(np.mean(ytr[m][gt] / (Atr[gt] if m != "eui" else 1))),
            train_mean_oof_per_A=float(np.mean(t[m][gt] / (Atr[gt] if m != "eui" else 1))),
            test_mean_pred_per_A=float(np.mean(sub[COL[m]].values[ge] / (A[ge] if m != "eui" else 1))),
            train_ratio_mean_y_over_oof=float(np.mean(ytr[m][gt]) / np.mean(t[m][gt])))
        del sc, sce
print("transfer (per-area level of truth / OOF on train vs test prediction):")
for k, v in tf.items():
    print("   ", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
out["transfer"] = tf
json.dump(out, open(f"{D}/c3_candidate.json", "w"), indent=1, default=float)
print("wrote", f"{D}/c3_candidate.json")
