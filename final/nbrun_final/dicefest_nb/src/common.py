"""Shared data loading, feature preparation and metric for DICE FEST 2026."""
import os

import numpy as np
import pandas as pd

SEED = 42

TARGET_COLS = [
    "target_macro_gas_kwh",
    "target_macro_elec_kwh",
    "target_eui_gas_kwh_per_sqm",
    "target_clean_building_ratio",
    "target_electric_heating_adoption",
]
SIGMAS = {
    "target_macro_gas_kwh": 151750.04,
    "target_macro_elec_kwh": 41725.37,
    "target_eui_gas_kwh_per_sqm": 40.65,
    "target_clean_building_ratio": 12.49,
    "target_electric_heating_adoption": 7.18,
}
HVAC = [f"hvac_dist_{i:02d}" for i in range(1, 11)]
ERA_MAP = {"High (<0%)": 0, "Medium (0-6%)": 1, "Low (6-18%)": 2, "Very Low (>18%)": 3}


def nrmse(y_true, y_pred, col):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)) / SIGMAS[col])


def holdout_split(n, frac=0.7, seed=7):
    """Fixed split of the train rows into P (models trained here) and H (holdout for fitting blend weights)."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[: int(frac * n)]), np.sort(perm[int(frac * n):])


DATA_DIR = os.environ.get("DICEFEST_DATA_DIR", "data/raw")  # folder with train.csv, test.csv, sample_submission.csv
_CACHE = {}


def load(cache=None):
    """train / test frames read from DATA_DIR with the pyarrow CSV parser (bit-identical to the parquet cache used
    during development, see env/data_check.json). The argument is kept for API compatibility and ignored."""
    if "tr" not in _CACHE:
        _CACHE["tr"] = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), engine="pyarrow")
        _CACHE["te"] = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), engine="pyarrow")
    return _CACHE["tr"].copy(), _CACHE["te"].copy()


def load_sample():
    return pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), engine="pyarrow")


def base_features(df, const_cols):
    """Numeric feature frame: encoded strings, id codes, and a few physics-motivated ratios."""
    X = df.drop(columns=[c for c in df.columns if c.startswith("target_")] + ["id"] + const_cols).copy()
    X["bld_dim_18"] = X["bld_dim_18"].map(ERA_MAP).astype("float32")
    X["socio_demog_016"] = (X["socio_demog_016"] == "Urban").astype("float32")
    X["zone_n"] = X.pop("zone_region_id").str[4:].astype("int32")
    X["sub_n"] = X.pop("subzone_id").str[4:].astype("int32")
    A = df["bld_dim_06"]
    X["hvac_rest"] = 100 - df[HVAC].sum(axis=1)
    X["area_per_meter"] = A / df["bld_dim_12"]
    X["area_per_epc"] = A / df["bld_dim_01"]
    X["gea_ratio"] = A / df["bld_dim_03"]
    X["a07_ratio"] = df["bld_dim_07"] / A
    X["a08_ratio"] = df["bld_dim_08"] / A
    X["a05_ratio"] = df["bld_dim_05"] / df["bld_dim_03"]
    X["a04_ratio"] = df["bld_dim_04"] / df["bld_dim_03"]
    X["s002_ratio"] = df["socio_demog_002"] / A
    X["meters_per_epc"] = df["bld_dim_12"] / df["bld_dim_01"]
    X["clean_ratio"] = df["bld_dim_02"] / df["bld_dim_01"]
    return X.astype("float32")


def constant_cols(tr):
    num = tr.select_dtypes("number")
    return [c for c in num.columns if num[c].nunique() <= 1]
