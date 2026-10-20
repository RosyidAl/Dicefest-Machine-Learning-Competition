"""Audit postproc 02: are there hard truncation bounds in the train targets (gas < 1e6, elec/A < 150, EUI/elec-per-area
>= 5, ...)? Prints counts in bins right below/above candidate bounds, to tell a hard cut from a smooth tail.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/postproc/02_tails.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import TARGET_COLS  # noqa: E402

GAS, EL, EUI, CL, AD = TARGET_COLS
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", "bld_dim_01", GAS, EL, EUI])
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
g, el, eui = tr[GAS].values, tr[EL].values, tr[EUI].values
series = {"gas": g, "elec": el, "eui": eui, "elec_per_area": el / A, "gas_per_meter": g / M, "elec_per_meter": el / M,
          "gas_per_epc": g / tr["bld_dim_01"].values, "elec_per_epc": el / tr["bld_dim_01"].values}


def bins(name, x, lo, hi, nb):
    e = np.linspace(lo, hi, nb + 1)
    h = np.histogram(x, e)[0]
    print(f"{name:16s} [{lo:g},{hi:g}] n/bin:", " ".join(str(int(v)) for v in h))


print("n rows", len(g))
bins("gas hi", g, 800000, 1000000, 20)
bins("gas hi fine", g, 990000, 1002000, 12)
bins("elec hi", el, 200000, 400000, 20)
bins("eui hi", eui, 400000 / 1000, 700, 15)
bins("eui lo", eui, 0, 60, 12)
bins("elec/A hi", el / A, 100, 160, 12)
bins("elec/A hi fine", el / A, 145, 151, 12)
bins("elec/A lo", el / A, 0, 15, 15)
bins("gas/M lo", g / M, 0, 3000, 15)
bins("gas/M hi", g / M, 30000, 70000, 20)
bins("elec/M lo", el / M, 0, 1500, 15)
bins("elec/M hi", el / M, 10000, 26000, 16)
# EUI relative to its implied upper cap 1e6/A: fraction of rows with eui > c * 1e6/A
cap = 1e6 / A
for f in (0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
    print(f"eui > {f} * 1e6/A: {int((eui > f * cap).sum())}")
# how much of the A distribution makes 1e6/A binding: A quantiles and 1e6/A there
for qq in (0.5, 0.9, 0.99, 0.999):
    a = np.quantile(A, qq)
    print(f"A q{qq} = {a:.0f} -> eui cap {1e6 / a:.1f}; rows with A>={a:.0f}: eui q99 {np.quantile(eui[A >= a], 0.99):.1f} max {eui[A >= a].max():.1f}")
# elec per area cap 150: rows by A
for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 20000)):
    s = (A >= lo) & (A < hi)
    x = el[s] / A[s]
    print(f"A in [{lo},{hi}) n={s.sum()} elec/A q99 {np.quantile(x, .99):.1f} q999 {np.quantile(x, .999):.1f} max {x.max():.2f}"
          f" n>140 {(x > 140).sum()}")
