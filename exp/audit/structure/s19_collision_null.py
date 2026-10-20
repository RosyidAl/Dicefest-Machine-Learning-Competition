"""Null check for exact ratio collisions: compare with the same count after shuffling the denominator across rows
(chance level). Also overlap of gas/E collisions with gas/M collisions."""
import numpy as np, pandas as pd
tr = pd.read_parquet("cache/train.parquet", columns=["bld_dim_12", "bld_dim_01", "target_macro_gas_kwh", "target_macro_elec_kwh"])
g = tr.target_macro_gas_kwh.values; e = tr.target_macro_elec_kwh.values; M = tr.bld_dim_12.values; E = tr.bld_dim_01.values
def ncoll(x):
    s = pd.Series(np.round(x, 9)); return int((s.map(s.value_counts()).values > 1).sum())
rng = np.random.default_rng(0)
for n, num, den in [("gas/M", g, M), ("gas/E", g, E), ("elec/M", e, M), ("elec/E", e, E)]:
    obs = ncoll(num / den); null = [ncoll(num / rng.permutation(den)) for _ in range(2)]
    print(f"{n}: observed collision rows {obs}, shuffled-denominator {null}")
gm = pd.Series(np.round(g / M, 9)); ge = pd.Series(np.round(g / E, 9)); gg = pd.Series(g)
cm = gm.map(gm.value_counts()).values > 1; ce = ge.map(ge.value_counts()).values > 1; cg = gg.map(gg.value_counts()).values > 1
print("rows with any of gas / gas/M / gas/E collision:", int((cm | ce | cg).sum()), "share", round((cm | ce | cg).mean(), 4),
      "| gas/E-only (not gas, not gas/M):", int((ce & ~cm & ~cg).sum()))
