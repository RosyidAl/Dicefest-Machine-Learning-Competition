"""Experiment tracking: sequential ids, registry rows, environment and hashes (see MASTER_PROMPT Lampiran A)."""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess

import pandas as pd

REGISTRY = "experiments/registry.csv"
COLUMNS = ["exp_id", "parent_id", "status", "phase", "created_wib", "finished_wib", "hypothesis", "model_family",
           "device", "targets", "target_transform", "feature_set_id", "n_features", "cv_scheme", "fold_file", "seeds",
           "config_path", "params_hash", "git_commit", "data_hash", "nrmse_mean", "nrmse_t1", "nrmse_t2", "nrmse_t3",
           "nrmse_t4", "nrmse_t5", "fold_nrmse_json", "delta_vs_best_json", "folds_improved_json", "runtime_min",
           "peak_ram_gb", "peak_vram_gb", "oof_path", "test_pred_path", "notes"]
WIB = dt.timezone(dt.timedelta(hours=7))
DATA_MD5 = {"train": "fa14e9aa16ab8fa65a9a55e2e7188447", "test": "9511b3f7f8fcd912eb25cee8a109900d"}


def now_wib():
    return dt.datetime.now(WIB).strftime("%Y-%m-%d %H:%M")


def read_registry():
    if not os.path.exists(REGISTRY):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)


def next_exp_id(prefix="E"):
    reg = read_registry()
    nums = [int(e[len(prefix):]) for e in reg["exp_id"] if e.startswith(prefix) and e[len(prefix):].isdigit()]
    return f"{prefix}{max(nums, default=0) + 1:04d}"


def upsert(row):
    """insert a registry row or update the fields of an existing exp_id"""
    reg = read_registry()
    row = {k: ("" if v is None else str(v)) for k, v in row.items()}
    unknown = set(row) - set(COLUMNS)
    assert not unknown, unknown
    if row["exp_id"] in set(reg["exp_id"]):
        for k, v in row.items():
            reg.loc[reg["exp_id"] == row["exp_id"], k] = v
    else:
        reg = pd.concat([reg, pd.DataFrame([{c: row.get(c, "") for c in COLUMNS}])], ignore_index=True)
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    reg[COLUMNS].to_csv(REGISTRY, index=False)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def params_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def data_hash():
    return hashlib.sha256(json.dumps(DATA_MD5, sort_keys=True).encode()).hexdigest()[:12]


def env_info():
    info = {"python": platform.python_version(), "host": platform.node()}
    for m in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "torch"]:
        try:
            info[m] = __import__(m).__version__
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
