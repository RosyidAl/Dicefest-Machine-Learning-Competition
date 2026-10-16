"""Baseline table (Phase 2): target x model x CV scheme from the registry rows of the baseline experiments."""
import sys
import pandas as pd

sys.path.insert(0, "src")
from tracking import read_registry  # noqa: E402

reg = read_registry().set_index("exp_id")
B = {"B0 mean": ("E0169", "E0179"), "B1 Ridge": ("E0170", "E0180"), "B2 LGB tanpa ID": ("E0171", "E0181"),
     "B3 LGB + ID": ("E0172", "E0182"), "B4 XGB + ID": ("E0173", "E0183"), "B5 CatBoost + ID cat": ("E0174", "E0184"),
     "B6 MLP multi-output": ("E0175", "E0185")}
T = ["nrmse_t1", "nrmse_t2", "nrmse_t3", "nrmse_t4", "nrmse_t5", "nrmse_mean"]
lines = ["| Model | Skema | " + " | ".join(["t1 gas", "t2 elec", "t3 EUI", "t4 clean", "t5 adopsi", "mean"]) + " |",
         "|---|---|" + "---|" * 6]
for name, (p, s) in B.items():
    for scheme, e in (("primary", p), ("secondary", s)):
        if e in reg.index and reg.loc[e, "status"] == "done":
            lines.append(f"| {name} ({e}) | {scheme} | " + " | ".join(reg.loc[e, t] for t in T) + " |")
        else:
            lines.append(f"| {name} ({e}) | {scheme} | " + " | ".join(["-"] * 6) + f" | ({reg.loc[e, 'status'] if e in reg.index else 'belum'})")
print("\n".join(lines))
gap = []
for name, (p, s) in B.items():
    if all(e in reg.index and reg.loc[e, "status"] == "done" for e in (p, s)):
        gap.append(f"| {name} | " + " | ".join(f"{float(reg.loc[s, t]) - float(reg.loc[p, t]):+.4f}" for t in T) + " |")
if gap:
    print("\nGap secondary - primary:\n| Model | t1 | t2 | t3 | t4 | t5 | mean |\n|---|---|---|---|---|---|---|")
    print("\n".join(gap))
