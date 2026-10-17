"""CV_SECONDARY robustness check (MASTER_PROMPT V02): one LightGBM screening model (lr 0.1, early stopping on the
validation fold) with and without the spatial target encodings, scored on CV_PRIMARY (KFold per row, mirrors the
test split, D001) and on CV_SECONDARY (GroupKFold per zone_region_id, the organisers' suggestion).

It shows how much of the score depends on regions seen in training: under the random row split 92% of test rows
have their subzone in train, under the zone split none do.
Outputs: exp/cv2/{oof,pred}/*.npy, exp/cv2/results.jsonl, registry rows E0271-E0278.
usage: python src/exp_cv_secondary.py [--smoke]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import TARGET_COLS, load, nrmse  # noqa: E402
from tracking import data_hash, git_commit, now_wib, upsert  # noqa: E402
from train_lib import Data, run  # noqa: E402

SMOKE = "--smoke" in sys.argv
OUT = "exp/cv2_smoke" if SMOKE else "exp/cv2"
P = dict(objective="regression", learning_rate=0.1, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=12)
if SMOKE:
    P.update(learning_rate=0.3, num_leaves=31)

tr, _ = load()
d = Data(n_folds=5)
ex, ext = d.agg_features()


def folds_from(f):
    x = pd.read_parquet(f)
    assert (x["id"].values == tr["id"].values).all()
    return [(np.where(x["fold"].values != k)[0], np.where(x["fold"].values == k)[0]) for k in sorted(x["fold"].unique())]


SCHEMES = {"primary": folds_from("data/folds/folds_primary_k5_s42.parquet"),
           "secondary": folds_from("data/folds/folds_secondary_k5_s42.parquet")}
JOBS = [("gpm", 1 / d.A), ("epm", None)]  # gas per meter (weight M^2/A), elec per meter (weight M^2)
eid = 271
for scheme, folds in SCHEMES.items():
    d.folds = folds[:2] if SMOKE else folds
    for te in (False, True):
        for kind, wmul in JOBS:
            name = f"lgb_{kind}_{scheme}_{'te' if te else 'note'}"
            t0 = time.time()
            oof, _ = run(d, name, "lgb", kind, P, use_te=te, windows=(2, 10) if te else (), cells=te, out_dir=OUT,
                         wmul=wmul, extra=ex, extra_t=ext, verbose=False)
            idx = np.concatenate([b for _, b in d.folds])
            if kind == "gpm":
                sc = {"t1": nrmse(tr[TARGET_COLS[0]].values[idx], (oof * d.A)[idx], TARGET_COLS[0]),
                      "t3": nrmse(tr[TARGET_COLS[2]].values[idx], oof[idx], TARGET_COLS[2])}
            else:
                sc = {"t2": nrmse(tr[TARGET_COLS[1]].values[idx], oof[idx], TARGET_COLS[1])}
            rec = dict(name=name, scheme=scheme, te=te, kind=kind, minutes=round((time.time() - t0) / 60, 1), **sc)
            print(json.dumps(rec), flush=True)
            if SMOKE:
                continue
            with open(f"{OUT}/results.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            upsert(dict(exp_id=f"E{eid:04d}", status="done", phase=2, finished_wib=now_wib(),
                        hypothesis=f"CV_SECONDARY check: LGB {kind} {'dengan' if te else 'tanpa'} TE spasial, skema {scheme}",
                        model_family="lgb", device="cpu", targets="t1,t3" if kind == "gpm" else "t2",
                        target_transform=f"{kind} (ratio to bld_dim_12)", feature_set_id="legacy+agg" + ("+te" if te else ""),
                        cv_scheme=scheme, fold_file=f"data/folds/folds_{scheme}_k5_s42.parquet", seeds=42,
                        config_path="src/exp_cv_secondary.py", git_commit=git_commit(), data_hash=data_hash(),
                        nrmse_t1=sc.get("t1", ""), nrmse_t2=sc.get("t2", ""), nrmse_t3=sc.get("t3", ""),
                        runtime_min=rec["minutes"], oof_path=f"{OUT}/oof/{name}.npy", test_pred_path=f"{OUT}/pred/{name}.npy",
                        notes="screening lr 0.1, early stopping on the validation fold (relative comparison only)"))
            eid += 1
print("done")
