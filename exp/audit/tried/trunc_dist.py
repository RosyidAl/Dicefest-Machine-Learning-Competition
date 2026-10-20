"""Density of the scored targets near their extremes (is there a hard truncation?)."""
import numpy as np
import pandas as pd

T = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm"]
tr = pd.read_parquet("cache/train.parquet", columns=T + ["bld_dim_06", "bld_dim_12"])
g, e, u = (tr[c].values for c in T)
A, M = tr.bld_dim_06.values, tr.bld_dim_12.values.astype(float)
for name, v, edges in (("gas", g, np.arange(8.0e5, 1.0201e6, 2e4)), ("elec", e, np.arange(2.6e5, 4.01e5, 1e4)),
                       ("eui", u, np.arange(450, 701, 25))):
    h, _ = np.histogram(v, edges)
    print(name, "counts per bin from", edges[0], "step", edges[1] - edges[0], ":", h.tolist())
for name, v, edges in (("gas low", g, np.arange(0, 6.01e4, 5e3)), ("elec low", e, np.arange(0, 3.01e4, 2.5e3)),
                       ("eui low", u, np.arange(0, 41, 2.5))):
    h, _ = np.histogram(v, edges)
    print(name, "counts per bin from", edges[0], "step", edges[1] - edges[0], ":", h.tolist())
# per-meter gas and elec extremes
for name, v in (("gas/M", g / M), ("elec/M", e / M)):
    q = np.quantile(v, [0, 1e-4, 1e-3, 0.01, 0.99, 0.999, 0.9999, 1])
    print(name, np.round(q, 1).tolist())
# rows with big M: share of gas near cap, shows truncation selection: mean gas/M by M bin
for lo, hi in ((2, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 160)):
    s = (M >= lo) & (M < hi)
    print(f"M in [{lo},{hi}): n={s.sum()}, mean gas/M {np.mean(g[s] / M[s]):.0f}, share gas>9e5 {np.mean(g[s] > 9e5):.4f}, "
          f"max gas {g[s].max():.0f}")
