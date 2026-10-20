"""verify2_postproc/combo_repro c4: does the residual correction already move the small-M rows on the test side more
than on H (denser neighbour pool), which would make the teacher-based S1 delta overshoot on the test?
Compares corrected / teacher ratios for M 2-3 and M 4 rows: test (submissions/sub_v19vce.csv vs exp/pl/teacher_deploy.npz,
the deployed VC teacher without correction) and H (chain_H.npz corrected vs teacher). Also the implied residual
under-prediction left after the correction, and the S1 gain on H when the delta is reduced by the test-side correction.
usage (from /root/Dicefest): OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
    exp/audit/verify2_postproc/combo_repro/c4_transfer.py
"""
import json

import numpy as np
import pandas as pd

D = "exp/audit/verify2_postproc/combo_repro"
COL = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm", "elec": "target_macro_elec_kwh"}
SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
MET = ("gas", "eui", "elec")
GROUPS = ((2, 3), (4, 4))
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY)
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
mg = te[KEY].merge(tr[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
live = (mg["_merge"] != "both").values
Mt = te["bld_dim_12"].values.astype(float)
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
td = np.load("exp/pl/teacher_deploy.npz", allow_pickle=False)
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
MH = z["M"][z["H"]].astype(float)
prm = json.load(open(f"{D}/c3_candidate.json"))["params"]["deltas"]
out = {}
for m in MET:
    c_te, t_te = sub[COL[m]].values, td[m]
    # all-rows sanity: correction size on test vs H
    allrat_te = np.sqrt(np.mean((c_te[live] - t_te[live]) ** 2)) / SIG[m]
    c_h, t_h = np.clip(z[f"{m}_corrected"], 0, None), z[f"{m}_teacher"]
    allrat_h = np.sqrt(np.mean((c_h - t_h) ** 2)) / SIG[m]
    out[m] = {"rms_correction_sigma_test": float(allrat_te), "rms_correction_sigma_H": float(allrat_h)}
    for (lo, hi), dd in zip(GROUPS, prm[m]):
        ge = (Mt >= lo) & (Mt <= hi) & live
        gh = (MH >= lo) & (MH <= hi)
        lr_te = np.log(c_te[ge] / t_te[ge])
        lr_h = np.log(c_h[gh] / t_h[gh])
        ls_te = float(c_te[ge] @ t_te[ge] / (t_te[ge] @ t_te[ge]) - 1)
        ls_h = float(c_h[gh] @ t_h[gh] / (t_h[gh] @ t_h[gh]) - 1)
        # remaining bias relative to the corrected prediction if the teacher bias is dd: (1 + dd) / (1 + c) - 1
        out[m][f"M{lo}-{hi}"] = dict(n_test=int(ge.sum()), n_H=int(gh.sum()), corr_ls_shift_test=ls_te,
                                     corr_ls_shift_H=ls_h, corr_mean_logratio_test=float(lr_te.mean()),
                                     corr_mean_logratio_H=float(lr_h.mean()), teacher_delta=dd,
                                     implied_delta_after_test_correction=float((1 + dd) / (1 + ls_te) - 1))
    print(m, json.dumps({k: ({kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                             if isinstance(v, dict) else round(v, 6)) for k, v in out[m].items()}))
json.dump(out, open(f"{D}/c4_transfer.json", "w"), indent=1)
print("wrote", f"{D}/c4_transfer.json")
