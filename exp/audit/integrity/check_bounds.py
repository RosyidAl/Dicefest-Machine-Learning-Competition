"""Integrity audit 8: hard bounds in the targets (sharp truncation?) for gas, elec, EUI and per-meter ratios.
usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/integrity/check_bounds.py
"""
import json
import numpy as np
import pandas as pd
cols = ["bld_dim_12", "bld_dim_06", "bld_dim_01", "target_macro_gas_kwh", "target_macro_elec_kwh",
        "target_eui_gas_kwh_per_sqm"]
tr = pd.read_parquet("data/cache/train.parquet", columns=cols)
g, e, u = tr["target_macro_gas_kwh"].values, tr["target_macro_elec_kwh"].values, tr["target_eui_gas_kwh_per_sqm"].values
M = tr["bld_dim_12"].values.astype(float)
out = {}
def hist(x, edges):
    return {f"{edges[i]:.4g}-{edges[i+1]:.4g}": int(((x >= edges[i]) & (x < edges[i + 1])).sum()) for i in range(len(edges) - 1)}
out["gas_upper"] = hist(g, [7e5, 7.5e5, 8e5, 8.5e5, 9e5, 9.5e5, 9.75e5, 1e6, 1.05e6, 1.1e6])
out["gas_lower"] = hist(g, [0, 2e3, 4e3, 6e3, 8e3, 1e4, 1.5e4, 2e4])
out["elec_upper"] = hist(e, [2e5, 2.5e5, 3e5, 3.5e5, 4e5, 4.5e5])
out["elec_lower"] = hist(e, [0, 2e3, 3e3, 4e3, 5e3, 6e3, 8e3])
gpm, epm = g / M, e / M
out["gas_per_meter"] = dict(min=float(gpm.min()), max=float(gpm.max()), q=np.quantile(gpm, [0.001, 0.5, 0.999]).round(1).tolist())
out["elec_per_meter"] = dict(min=float(epm.min()), max=float(epm.max()), q=np.quantile(epm, [0.001, 0.5, 0.999]).round(1).tolist())
out["gpm_upper"] = hist(gpm, [3e4, 4e4, 5e4, 6e4, 7e4, 7.32e4, 8e4, 1e5])
out["gpm_lower"] = hist(gpm, [0, 500, 1000, 1500, 2000, 3000])
out["epm_upper"] = hist(epm, [1e4, 1.5e4, 2e4, 2.5e4, 3e4, 4e4, 1e5])
out["epm_lower"] = hist(epm, [0, 500, 1000, 1500, 2000])
out["eui_upper"] = hist(u, [300, 400, 500, 600, 650, 700])
out["eui_lower"] = hist(u, [0, 5, 10, 20, 30, 40])
out["M"] = dict(min=float(M.min()), max=float(M.max()), n_le_4=int((M <= 4).sum()))
print(json.dumps(out, indent=1))
json.dump(out, open("exp/audit/integrity/check_bounds.json", "w"), indent=1)
