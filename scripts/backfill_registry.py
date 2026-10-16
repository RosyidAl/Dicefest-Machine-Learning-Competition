"""One-off: register the experiments run before the registry existed (2026-10-12 .. 10-16).

Single models: one row per exp/oof/<name>.npy. NRMSE recomputed from the saved OOF arrays (gas family: EUI and
gas = EUI x A; elec family: elec kWh). Fold count, runtime and source log are read from the run's summary line.
Rows are ordered by the mtime of the test prediction file (chronological)."""
import datetime as dt
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, "src")
from common import SIGMAS, TARGET_COLS, load  # noqa: E402
from prune import ELEC  # noqa: E402
from tracking import WIB, data_hash, read_registry, upsert  # noqa: E402

FAM = [(("xgbl",), "xgboost_linear"), (("xgb", "x4", "x8", "xa", "xs"), "xgboost"),
       (("lgb", "l4", "l7", "la"), "lightgbm"), (("cat", "c4"), "catboost"),
       (("nn", "nn3", "nn4", "nn5", "n4", "n4f", "n5", "n6", "n6b"), "mlp"), (("p1", "p2", "p3"), "mlp_plr"),
       (("t1",), "tabm_mini_plr"), (("m1",), "mlp_multitask")]
FORM = {"gpm": "gas per meter (bld_dim_12)", "lgpm": "log gas per meter", "gpgea": "gas per bld_dim_03",
        "gpn": "gas per EPC count (bld_dim_01)", "gas": "gas kWh", "eui": "EUI", "leui": "log EUI",
        "epm": "elec per meter", "lepm": "log elec per meter", "epa": "elec per area", "lepa": "log elec per area",
        "elec": "elec kWh"}


def family(name):
    p = name.split("_")[0]
    for pre, fam in FAM:
        if p in pre:
            return fam
    return "other"


def log_info():
    info = {}
    for f in sorted(glob.glob("exp/*.log"), key=os.path.getmtime):
        for line in open(f, errors="ignore"):
            m = re.match(r"^\[(\S+)\] (.*?)(iters|epochs)=(\[[^\]]*\]).*time=(\d+)s", line)
            if m:
                n_it = len(m.group(4).strip("[]").split(","))
                folds = n_it if m.group(3) == "iters" else n_it // 3
                info[m.group(1)] = (folds, int(m.group(5)), f)
    return info


if __name__ == "__main__":
    tr, _ = load()
    A = tr["bld_dim_06"].values
    y = {c: tr[c].values for c in TARGET_COLS}
    info = log_info()
    done = set(read_registry()["exp_id"])
    names = sorted((f[:-4] for f in os.listdir("exp/oof") if os.path.exists(f"exp/pred/{f}")),
                   key=lambda n: os.path.getmtime(f"exp/pred/{n}.npy"))
    blend = {}
    for tag in ("v15nc", "v15"):
        for line in open(f"exp/prune_{tag}.log"):
            if line.startswith(("gas runs:", "elec runs:")):
                for r in line.split()[2].split(","):
                    blend.setdefault(r, tag)
    for i, n in enumerate(names, 1):
        eid = f"E{i:04d}"
        o = np.load(f"exp/oof/{n}.npy").astype(float)
        el = any(t in n.split("_") for t in ELEC)
        row = dict(exp_id=eid, status="done", phase=5, model_family=family(n),
                   device="cpu" if family(n) == "lightgbm" else "gpu",
                   finished_wib=dt.datetime.fromtimestamp(os.path.getmtime(f"exp/pred/{n}.npy"), WIB).strftime(
                       "%Y-%m-%d %H:%M"), cv_scheme="primary", seeds="42", data_hash=data_hash(),
                   oof_path=f"exp/oof/{n}.npy", test_pred_path=f"exp/pred/{n}.npy", feature_set_id="FS_legacy")
        form = [k for k in n.split("_") if k in FORM]
        row["target_transform"] = FORM[form[0]] if form else ""
        if el:
            row["targets"] = "t2"
            row["nrmse_t2"] = f"{np.sqrt(np.mean((o - y[TARGET_COLS[1]]) ** 2)) / SIGMAS[TARGET_COLS[1]]:.5f}"
        else:
            row["targets"] = "t1,t3"
            row["nrmse_t3"] = f"{np.sqrt(np.mean((o - y[TARGET_COLS[2]]) ** 2)) / SIGMAS[TARGET_COLS[2]]:.5f}"
            row["nrmse_t1"] = f"{np.sqrt(np.mean((o * A - y[TARGET_COLS[0]]) ** 2)) / SIGMAS[TARGET_COLS[0]]:.5f}"
        if n in info:
            k, t, f = info[n]
            row.update(fold_file=f"data/folds/folds_primary_k{k}_s42.parquet", runtime_min=f"{t / 60:.1f}",
                       config_path=f, notes=f"legacy run {n}; log {f}")
        else:
            row["notes"] = f"legacy run {n}; no summary line found"
        if n in blend:
            row["notes"] += f"; member of blend {blend[n]}"
        row["hypothesis"] = f"legacy model zoo member {n}"
        upsert(row)
    print(len(names), "rows")
