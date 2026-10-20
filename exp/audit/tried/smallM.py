"""Rows with very few meters (bld_dim_12): residual bias per M value on H (corrected) and P (teacher-level)."""
import numpy as np

SIG = {"gas": 151750.04, "eui": 40.65, "elec": 41725.37}
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
P, H, A, M = z["P"], z["H"], z["A"], z["M"]
MH, MP = M[H], M[P]
print("M value counts (train):", {int(v): int((M == v).sum()) for v in range(2, 12)}, flush=True)
for m in ("gas", "eui", "elec"):
    y, c = z[f"{m}_y"], z[f"{m}_corrected"]
    rP = z[f"{m}_rP"]
    pP = z[f"{m}_pP"]
    print(f"== {m}")
    for v in range(2, 12):
        s, sp = MH == v, MP == v
        r = y[s] - c[s]
        print(f"  M={v:2d} H n={s.sum():5d} mean resid {r.mean():+9.4g} (t {r.mean() / (r.std() / np.sqrt(s.sum())):+.1f}) "
              f"rel {np.mean(r) / np.mean(c[s]):+.4f} | P n={sp.sum():5d} mean resid {rP[sp].mean():+9.4g} "
              f"(t {rP[sp].mean() / (rP[sp].std() / np.sqrt(sp.sum())):+.1f}) rel {rP[sp].mean() / pP[sp].mean():+.4f}", flush=True)
