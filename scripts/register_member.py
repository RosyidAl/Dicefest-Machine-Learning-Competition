"""Register exp/oof|pred members trained by the src/exp_*.py scripts (outside train_cv) in the registry.
usage: python scripts/register_member.py name1,name2 --phase 4 --hyp "..." [--parent E0191]"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from backfill_registry import FAM, FORM, family  # noqa: E402
from common import SIGMAS, TARGET_COLS, load  # noqa: E402
from prune import ELEC  # noqa: E402
from tracking import data_hash, next_exp_id, now_wib, read_registry, upsert  # noqa: E402

sys.path.insert(0, "scripts")
ap = argparse.ArgumentParser()
ap.add_argument("names")
ap.add_argument("--phase", default="5")
ap.add_argument("--hyp", default="")
ap.add_argument("--parent", default="")
ap.add_argument("--log", default="")
a = ap.parse_args()
tr, _ = load()
A = tr["bld_dim_06"].values
reg = read_registry()
for n in a.names.split(","):
    if (reg["oof_path"] == f"exp/oof/{n}.npy").any():
        print(n, "already registered"); continue
    o = np.load(f"exp/oof/{n}.npy").astype(float)
    el = any(t in n.split("_") for t in ELEC)
    form = [k for k in n.split("_") if k in FORM]
    row = dict(exp_id=next_exp_id("EM"), parent_id=a.parent, status="done", phase=a.phase, model_family=family(n),
               device="cpu" if family(n) == "lightgbm" else "gpu", finished_wib=now_wib(), cv_scheme="primary",
               fold_file="data/folds/folds_primary_k10_s42.parquet", seeds="42", data_hash=data_hash(),
               oof_path=f"exp/oof/{n}.npy", test_pred_path=f"exp/pred/{n}.npy", hypothesis=a.hyp or f"member {n}",
               target_transform=FORM[form[0]] if form else "", feature_set_id="FS_legacy+agg+te+knn" if "k_" in n[:3] else "",
               targets="t2" if el else "t1,t3", config_path=a.log, notes=f"member {n}")
    if a.log and os.path.exists(a.log):
        m = re.findall(rf"^\[{n}\] .*time=(\d+)s", open(a.log).read(), re.M)
        if m:
            row["runtime_min"] = f"{int(m[-1]) / 60:.1f}"
    if el:
        row["nrmse_t2"] = f"{np.sqrt(np.mean((o - tr[TARGET_COLS[1]].values) ** 2)) / SIGMAS[TARGET_COLS[1]]:.5f}"
    else:
        row["nrmse_t3"] = f"{np.sqrt(np.mean((o - tr[TARGET_COLS[2]].values) ** 2)) / SIGMAS[TARGET_COLS[2]]:.5f}"
        row["nrmse_t1"] = f"{np.sqrt(np.mean((o * A - tr[TARGET_COLS[0]].values) ** 2)) / SIGMAS[TARGET_COLS[0]]:.5f}"
    upsert(row)
    print(row["exp_id"], n, row.get("nrmse_t3", row.get("nrmse_t2")))
