"""Does the M2-3 relative bias depend on area per meter (A/M)? Test M2-3 rows have larger A/M than train (MW p 0.034)."""
import numpy as np, pandas as pd
z = np.load("/root/Dicefest/exp/audit/verify_postproc/adv_leak/r02_vc_pred.npz")
tr = pd.read_parquet("/root/Dicefest/cache/train.parquet", columns=["bld_dim_06", "bld_dim_12", "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm", "target_macro_elec_kwh"])
te = pd.read_parquet("/root/Dicefest/cache/test.parquet", columns=["bld_dim_06", "bld_dim_12"])
A, M = tr.bld_dim_06.values, tr.bld_dim_12.values.astype(float)
am, amt = A / M, te.bld_dim_06.values / te.bld_dim_12.values
Y = {"gas": tr.target_macro_gas_kwh.values, "eui": tr.target_eui_gas_kwh_per_sqm.values, "elec": tr.target_macro_elec_kwh.values}
for g, s, st in (("M2-3", (M >= 2) & (M <= 3), (te.bld_dim_12 >= 2).values & (te.bld_dim_12 <= 3).values), ("M4", M == 4, (te.bld_dim_12 == 4).values)):
    cut = np.median(am[s])
    print(f"{g}: train A/M median {cut:.1f}; share of test rows above it {np.mean(amt[st] > cut):.2f}")
    for m in Y:
        p, r = z["pf_" + m], Y[m] - z["pf_" + m]
        out = []
        for nm, ss in (("A/M low", s & (am <= cut)), ("A/M high", s & (am > cut))):
            d = p[ss] @ r[ss] / (p[ss] @ p[ss]); se = np.sqrt(np.sum((p[ss] * r[ss] - d * p[ss] ** 2) ** 2)) / (p[ss] @ p[ss])
            out.append(f"{nm} n {ss.sum()} d {d:+.3f} (se {se:.3f})")
        print(f"   {m:4s} " + " | ".join(out))
