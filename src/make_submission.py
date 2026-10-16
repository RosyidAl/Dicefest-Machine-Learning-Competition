"""Assemble a submission from saved OOF/test predictions.

usage: python src/make_submission.py <tag> <adopt_pred.npy> --gas run1,run2,... --elec runA,runB,... [--hold]
Gas-family runs store EUI predictions; elec-family runs store elec predictions.
--hold fits the blend weights on the holdout predictions of exp_hold.py instead of on OOF predictions.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blend import cv_blend_score, nnls_blend  # noqa: E402
from common import SIGMAS, TARGET_COLS, holdout_split, load, nrmse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("adopt")
    ap.add_argument("--gas", required=True)
    ap.add_argument("--elec", required=True)
    ap.add_argument("--dir", default="exp")
    ap.add_argument("--adopt_rmse", type=float, default=0.897)
    ap.add_argument("--hold", action="store_true",
                    help="fit blend weights on fold-averaged predictions for the holdout rows H (exp/hold/pred, "
                         "models trained on the other 70%%), which behave like test predictions; the printed CV "
                         "is then the cross-fitted holdout score")
    a = ap.parse_args()

    tr, te = load()
    A, At = tr["bld_dim_06"].values, te["bld_dim_06"].values
    y = {c: tr[c].values for c in TARGET_COLS}
    gas_runs, elec_runs = a.gas.split(","), a.elec.split(",")
    # rows and predictions the blend weights are fitted on
    rows, src = (holdout_split(len(tr))[1], "exp/hold/pred") if a.hold else (np.arange(len(tr)), f"{a.dir}/oof")
    Pg = np.column_stack([np.load(f"{src}/{r}.npy") for r in gas_runs])
    Pgt = np.column_stack([np.load(f"{a.dir}/pred/{r}.npy") for r in gas_runs])
    Pe = np.column_stack([np.load(f"{src}/{r}.npy") for r in elec_runs])
    Pet = np.column_stack([np.load(f"{a.dir}/pred/{r}.npy") for r in elec_runs])
    Ab, yb = A[rows], {c: y[c][rows] for c in TARGET_COLS}

    w_eui = nnls_blend(Pg, yb[TARGET_COLS[2]])
    w_gas = nnls_blend(Pg, yb[TARGET_COLS[2]], row_scale=Ab)  # minimises gas-scale error
    w_el = nnls_blend(Pe, yb[TARGET_COLS[1]])
    for nm, w, runs in [("eui", w_eui, gas_runs), ("gas", w_gas, gas_runs), ("elec", w_el, elec_runs)]:
        print(nm, dict(zip(runs, np.round(w, 4))), "sum", round(w.sum(), 4))

    # honest (cross-fitted) blend scores
    s_gas = nrmse(yb[TARGET_COLS[0]], cv_blend_score(Pg, yb[TARGET_COLS[2]], Ab) * Ab, TARGET_COLS[0])
    s_eui = nrmse(yb[TARGET_COLS[2]], cv_blend_score(Pg, yb[TARGET_COLS[2]]), TARGET_COLS[2])
    s_el = nrmse(yb[TARGET_COLS[1]], cv_blend_score(Pe, yb[TARGET_COLS[1]]), TARGET_COLS[1])
    clean_tr = np.round(100 * tr["bld_dim_02"].values / tr["bld_dim_01"].values, 2)
    s_cl = nrmse(y[TARGET_COLS[3]], clean_tr, TARGET_COLS[3])
    frac_unseen = 1 - te["subzone_id"].isin(set(tr["subzone_id"])).mean()
    s_ad = np.sqrt(frac_unseen * a.adopt_rmse ** 2) / SIGMAS[TARGET_COLS[4]]
    parts = [s_gas, s_el, s_eui, s_cl, s_ad]
    print("CV per target:", dict(zip(TARGET_COLS, np.round(parts, 5))), " mean:", round(float(np.mean(parts)), 5))

    sub = pd.DataFrame({"id": te["id"].values})
    sub[TARGET_COLS[0]] = np.clip(Pgt @ w_gas, 0, None) * At
    sub[TARGET_COLS[1]] = np.clip(Pet @ w_el, 0, None)
    sub[TARGET_COLS[2]] = np.clip(Pgt @ w_eui, 0, None)
    sub[TARGET_COLS[3]] = np.round(100 * te["bld_dim_02"].values / te["bld_dim_01"].values, 2)
    sub[TARGET_COLS[4]] = np.clip(np.load(a.adopt), 0, 100)

    ss = pd.read_parquet("cache/sample.parquet")
    assert list(sub.columns) == list(ss.columns), "column order mismatch"
    assert len(sub) == len(ss) and (sub["id"].values == ss["id"].values).all(), "id order mismatch"
    assert np.isfinite(sub[TARGET_COLS].values).all()
    os.makedirs("submissions", exist_ok=True)
    out = f"submissions/sub_{a.tag}.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape)
    print(sub[TARGET_COLS].describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
