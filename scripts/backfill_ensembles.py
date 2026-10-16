"""One-off: register the legacy blend versions (ENS) and post-processing steps (PP) built before the registry."""
import os
import re
import sys

sys.path.insert(0, "src")
from tracking import data_hash, upsert  # noqa: E402

VERS = ["v6", "v8", "v8_nocat", "v9", "v9nc", "v10", "v10nc", "v11", "v11nc", "v12", "v12nc", "v13", "v13nc", "v14",
        "v14nc", "v15", "v15nc"]
for i, v in enumerate(VERS, 1):
    log = open(f"exp/prune_{v}.log").read()
    m = re.findall(r"^top-(\d+): honest gas (\S+) elec (\S+) eui (\S+) total (\S+)", log, re.M)[-1]
    g = re.search(r"^gas runs: (\S+)", log, re.M).group(1)
    e = re.search(r"^elec runs: (\S+)", log, re.M).group(1)
    upsert(dict(exp_id=f"ENS{i:04d}", status="done", phase=8, model_family="ensemble", device="cpu",
                targets="t1,t2,t3,t4,t5", hypothesis=f"NNLS blend {v} (greedy top-{m[0]} per family, prune.py)",
                cv_scheme="primary", config_path=f"exp/prune_{v}.log", data_hash=data_hash(),
                nrmse_mean=m[4], nrmse_t1=m[1], nrmse_t2=m[2], nrmse_t3=m[3], nrmse_t4="0.00000",
                notes=f"honest = nested CV of the blend weights; t4 exact identity; gas members {g}; elec members {e}"))
PP = [("PP0001", "group residual correction (zone, climate cell, +-30 sub_n window), lambda fitted on holdout",
       "exp/resid_corr fit; holdout gain gas -0.00070 EUI -0.00100 elec -0.00023 (total ~-0.0004)", "src/resid_corr.py"),
      ("PP0002", "adoption v2: binomial q model with exact household count N", "total ~-0.0001", "src/adopt_v2.py"),
      ("PP0003", "train-duplicate copy (77 test rows identical to a train row)", "total ~-0.00003", "src/dup_fix.py"),
      ("PP0004", "similarity-kNN residual correction on top of PP0001",
       "holdout gain over PP0001: gas -0.0008 EUI -0.0013 elec -0.0003 (total ~-0.0005); exp/resid_knn_fit.log",
       "src/resid_knn.py")]
for eid, hyp, res, cfg in PP:
    upsert(dict(exp_id=eid, status="done", phase=9, model_family="postprocess", device="cpu", targets="t1,t2,t3",
                hypothesis=hyp, cv_scheme="holdout P70/H30 seed 7", config_path=cfg, data_hash=data_hash(), notes=res))
print("ok")
