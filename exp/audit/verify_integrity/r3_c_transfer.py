"""Verifier round 3 (leakage/optimism lens), step C: transfer of the integrity truncation shrink (I) to the test setting,
plus an independent rebuild of the candidate file.

 1. Full-train OOF of the v19vce VC blend (exp/audit/postproc/14_fullvc_oof.npz: members trained on 90% of ALL train,
    no residual correction). I (P-fitted parameters, r3_a) scored on H rows (labels unused by the fit): delta and
    calibration gamma = sum(d e)/sum(d^2). Same on P rows (NOT honest: labels used by the fit, reference only).
 2. Label-free transfer: under the bias model e = gamma * d + noise, the expected SSE change of adding d is
    (1 - 2 gamma) sum(d^2). The amount of adjustment mean(d^2) on the TEST predictions (sub_v19vce, duplicates
    excluded) vs the H chain fixes the expected test gain for a given gamma (H chain gamma, full-OOF H gamma, 1).
 3. Prediction-band rates per 10k rows: test vs H chain vs full-OOF.
 4. Independent rebuild of exp/audit/integrity/sub_v19vce_trunc2.csv from submissions/sub_v19vce.csv.
usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_integrity/r3_c_transfer.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

T = 1e6
SIG = dict(gas=151750.04, eui=40.65)
MET = ("gas", "eui")
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
OUT = "exp/audit/verify_integrity/r3_c_transfer.json"
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
r3a = json.load(open("exp/audit/verify_integrity/r3_a_rederive.json"))
a0, b0 = r3a["var_model_P"]["a0"], r3a["var_model_P"]["b0"]
KI = {m: r3a["fit_P"][m]["k"] for m in MET}
CI = {m: r3a["fit_P"][m]["c"] for m in MET}


def adj_I(pg, m, k):
    s = np.maximum(k * np.abs(pg) * np.sqrt(np.maximum(a0 + b0 / m, 1e-6)), 1.0)
    a = (T - pg) / s
    return -s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def dI(m, p, a, mm):
    sc = a if m == "eui" else 1.0
    return np.clip(p + CI[m] * adj_I(p * sc, mm, KI[m]) / sc, 0, None) - p


def nr(y, p, m):
    return float(np.sqrt(np.mean((y - p) ** 2)) / SIG[m])


out = {}
# ---- 1. full-train OOF (no correction)
f = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
amount = {}
gam = {}
for m in MET:
    yy, pp = f[f"{m}_y"], np.clip(f[f"{m}_p"], 0, None)
    assert np.allclose(yy[H], z[f"{m}_y"], rtol=1e-12, atol=1e-6)
    d = dI(m, pp, A, M)
    res = {}
    for name, idx in (("H", H), ("P_not_honest", P)):
        e = yy[idx] - pp[idx]
        res[name] = dict(base=nr(yy[idx], pp[idx], m), delta_total=(nr(yy[idx], pp[idx] + d[idx], m)
                                                                      - nr(yy[idx], pp[idx], m)) / 5,
                         gamma=float(d[idx] @ e / (d[idx] @ d[idx])), mean_d2=float(np.mean(d[idx] ** 2)))
    out[f"fullOOF_{m}"] = res
    amount[f"fullOOF_H_{m}"] = res["H"]["mean_d2"]
    gam[f"fullOOF_H_{m}"] = res["H"]["gamma"]
out["fullOOF_H_total"] = out["fullOOF_gas"]["H"]["delta_total"] + out["fullOOF_eui"]["H"]["delta_total"]

# ---- H chain amounts and gammas
for m in MET:
    yy, pp = z[f"{m}_y"], np.clip(z[f"{m}_corrected"], 0, None)
    d = dI(m, pp, A[H], M[H])
    amount[f"chainH_{m}"] = float(np.mean(d ** 2))
    gam[f"chainH_{m}"] = float(d @ (yy - pp) / (d @ d))
    out[f"chainH_{m}_delta_total"] = (nr(yy, pp + d, m) - nr(yy, pp, m)) / 5

# ---- 2. test side
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("data/cache/train.parquet", columns=KEY + [GAS])
te = pd.read_parquet("data/cache/test.parquet", columns=["id"] + KEY)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
assert (sub["id"].values == te["id"].values).all()
ref = tr.groupby(KEY, sort=False)[GAS].first().reset_index()
hit = ~te[KEY].merge(ref, on=KEY, how="left")[GAS].isna().values
out["test_dup_rows"] = int(hit.sum())
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
dt = {}
for m, col in (("gas", GAS), ("eui", EUI)):
    p = sub[col].values
    d = np.where(hit, 0.0, dI(m, p, At, Mt))
    dt[m] = d
    amount[f"test_{m}"] = float(np.mean(d ** 2))
est = {}
for m in MET:
    nr0 = r3a["H"][m]["base"]
    mse0 = (nr0 * SIG[m]) ** 2
    e = {}
    for gname in (f"chainH_{m}", f"fullOOF_H_{m}", "one"):
        g = 1.0 if gname == "one" else gam[gname]
        dmse = (1 - 2 * g) * amount[f"test_{m}"]
        e[f"gamma_{gname}"] = dict(gamma=g, delta_total=(np.sqrt(mse0 + dmse) - np.sqrt(mse0)) / SIG[m] / 5)
    e["ratio_amount_test_over_chainH"] = amount[f"test_{m}"] / amount[f"chainH_{m}"]
    e["ratio_amount_fullOOF_H_over_chainH"] = amount[f"fullOOF_H_{m}"] / amount[f"chainH_{m}"]
    est[m] = e
for gname in ("chainH", "fullOOF_H", "one"):
    est[f"total_gamma_{gname}"] = sum(est[m][f"gamma_{gname}_{m}" if gname != "one" else "gamma_one"]["delta_total"]
                                      for m in MET)
out["amount_mean_d2"] = amount
out["gamma"] = gam
out["test_label_free_estimate"] = est

# ---- 3. band rates per 10k
def rates(p):
    return {f">={t:.0f}": float(np.mean(p >= t) * 1e4) for t in (8e5, 9e5, 9.5e5, 1e6)}


out["band_rate_per_10k"] = dict(test_gas=rates(sub[GAS].values), chainH_gas=rates(z["gas_corrected"]),
                                chainH_teacher_gas=rates(z["gas_teacher"]), fullOOF_H_gas=rates(f["gas_p"][H]),
                                fullOOF_all_gas=rates(f["gas_p"]), train_label_gas=rates(f["gas_y"]))

# ---- 4. candidate rebuild
new = sub.copy()
new[GAS] = sub[GAS].values + dt["gas"]
new[EUI] = sub[EUI].values + dt["eui"]
cand = pd.read_csv("exp/audit/integrity/sub_v19vce_trunc2.csv", float_precision="round_trip")
sha = hashlib.sha256(open("exp/audit/integrity/sub_v19vce_trunc2.csv", "rb").read()).hexdigest()
out["candidate"] = dict(sha256=sha, sha_matches_claim=sha.startswith("aca75294"),
                        ids_equal=bool((cand["id"].values == sub["id"].values).all()),
                        other_cols_identical={c: bool(np.array_equal(cand[c].values, sub[c].values))
                                              for c in sub.columns if c not in ("id", GAS, EUI)},
                        gas_max_rel_diff=float(np.max(np.abs(cand[GAS].values - new[GAS].values)
                                                      / np.maximum(new[GAS].values, 1))),
                        eui_max_rel_diff=float(np.max(np.abs(cand[EUI].values - new[EUI].values)
                                                      / np.maximum(new[EUI].values, 1e-9))),
                        rows_gas_moved_gt_1pct=int((np.abs(dt["gas"]) > 0.01 * sub[GAS].values).sum()),
                        rows_eui_moved_gt_1pct=int((np.abs(dt["eui"]) > 0.01 * sub[EUI].values).sum()),
                        gas_max_after=float(new[GAS].max()), gas_min_change=float(dt["gas"].min()))
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
