"""Independent check of the data facts behind the gas-cap calibration claim (verify2_segments lens).

Reads only a few columns from the parquet cache (light). Prints: edge counts of the train gas target near 1e6,
train share of all rows, identity eui == gas / bld_dim_06, and the same edge view for gas per meter and for EUI*A.
"""
import numpy as np
import pandas as pd

GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
tr = pd.read_parquet("cache/train.parquet", columns=[GAS, EUI, "bld_dim_06", "bld_dim_12"])
te = pd.read_parquet("cache/test.parquet", columns=["bld_dim_06"])
g = tr[GAS].values.astype(float)
A = tr["bld_dim_06"].values.astype(float)
e = tr[EUI].values.astype(float)
print(f"train rows {len(tr)}  test rows {len(te)}  train share {len(tr) / (len(tr) + len(te)):.10f}")
print(f"gas max {g.max():.4f}  rows >= 1e6: {(g >= 1e6).sum()}  min {g.min():.4f}")
edges = [8e5, 8.5e5, 9e5, 9.25e5, 9.5e5, 9.75e5, 9.9e5, 1e6, 1.01e6, 1.05e6]
cnt, _ = np.histogram(g, bins=edges)
for lo, hi, c in zip(edges[:-1], edges[1:], cnt):
    print(f"  gas in [{lo:9.0f},{hi:9.0f}): {c:5d}  per 10k width {c / ((hi - lo) / 1e4):7.2f}")
# smooth extrapolation: compare density just below the edge with the density further down
for lo, hi in ((7e5, 8e5), (8e5, 9e5), (9e5, 1e6)):
    print(f"  density per 10k in [{lo:.0f},{hi:.0f}): {((g >= lo) & (g < hi)).sum() / 10:.2f}")
rel = np.abs(e * A - g) / g
print(f"eui*A vs gas: max rel diff {rel.max():.3e}")
print(f"test bld_dim_06 min {te['bld_dim_06'].min():.2f}; train A min {A.min():.2f}")
