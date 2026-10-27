"""Honest (nested) score of the varying-coefficient blend A2xM2xp2 used by the final submission.

The notebook (cell 35) prints only the honest score of the global NNLS blend (blend.cv_blend_score: weights fitted on
4/5 of the train rows, scored on the remaining 1/5, KFold(5, shuffle, seed 0)). This script repeats that procedure on
the same out-of-fold arrays of the final run, first for the global blend (must reproduce cell 35) and then for the
A2xM2xp2 blend (blend.blend_vc refitted inside every outer fold, ranks taken within the fit rows only).

usage: .venv/bin/python deck/ppt01/bukti/skor_jujur_bobot_adaptif.py   (writes skor_jujur_bobot_adaptif.json)
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = "/root/Dicefest"
os.environ.setdefault("DICEFEST_DATA_DIR", os.path.join(ROOT, "data/raw"))
sys.path.insert(0, os.path.join(ROOT, "final/nbrun_final/dicefest_nb/src"))
from blend import blend_vc, cv_blend_score  # noqa: E402
from common import TARGET_COLS, load, nrmse  # noqa: E402
from pipeline import load_config  # noqa: E402

WORK = os.path.join(ROOT, "final/nbrun_final/work")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skor_jujur_bobot_adaptif.json")
NOTEBOOK_CELL35 = {"gas": 0.24494174802833815, "eui": 0.6502285547443805, "elec": 0.315874614001809}
GAS, EL, EUI = TARGET_COLS[:3]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


cfg = load_config()
spec = json.load(open(os.path.join(ROOT, "final/nbrun_final/dicefest_nb/configs/resid_knn.json")))["vc"]
tr, _ = load()
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
P = {f: np.column_stack([np.load(f"{WORK}/oof/{r}.npy") for r in cfg["blend"][f]]) for f in ("gas", "elec")}
log(f"members gas/EUI {P['gas'].shape[1]}, elec {P['elec'].shape[1]}, rows {len(A)}; vc {spec}")

# (family, y used for fitting, row scale, scored column, multiplier back to the scored column)
METRICS = {"gas": ("gas", tr[EUI].values, A, GAS, A), "eui": ("gas", tr[EUI].values, None, EUI, 1.0),
           "elec": ("elec", tr[EL].values, None, EL, 1.0)}
res = {"global": {}, "adaptif": {}}
with threadpool_limits(4):
    for m, (fam, yfit, scale, col, mult) in METRICS.items():
        g = cv_blend_score(P[fam], yfit, scale) * mult
        res["global"][m] = nrmse(tr[col].values, g, col)
        log(f"{m}: global honest {res['global'][m]:.6f} (cell 35: {NOTEBOOK_CELL35[m]:.6f})")
        pred = np.zeros(len(yfit))
        for a, b in KFold(5, shuffle=True, random_state=0).split(P[fam]):
            _, pb, _ = blend_vc(P[fam][a], P[fam][b], yfit[a], spec[m], dict(A=A[a], M=M[a]), dict(A=A[b], M=M[b]),
                                None if scale is None else scale[a])
            pred[b] = pb
        res["adaptif"][m] = nrmse(tr[col].values, pred * mult, col)
        log(f"{m}: A2xM2xp2 honest {res['adaptif'][m]:.6f} (delta {res['adaptif'][m] - res['global'][m]:+.6f})")

check = {m: abs(res["global"][m] - NOTEBOOK_CELL35[m]) < 1e-6 for m in METRICS}
res["selisih"] = {m: res["adaptif"][m] - res["global"][m] for m in METRICS}
res["selisih_rata2_5_target"] = sum(res["selisih"].values()) / 5  # clean ratio and adoption unchanged
res["cocok_dengan_sel_35"] = check
res["detik"] = round(time.time() - t0)
json.dump(res, open(OUT, "w"), indent=1)
log(f"global matches cell 35: {check}; wrote {OUT}")
