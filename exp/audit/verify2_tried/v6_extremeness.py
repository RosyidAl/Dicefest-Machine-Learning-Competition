"""How extreme are high gas predictions on H, the full train OOF and the test (teacher and corrected)?
mean of (g - 9e5)_+^2 per row; the fix's gain scales with it (verify2_tried)."""
import numpy as np
import pandas as pd
K = 9e5
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
pf = np.load("exp/audit/verify2_tried/full_pf_gas.npy")
pa = np.load("exp/audit/verify2_tried/full_pa_gas.npy")
sub = pd.read_csv("submissions/sub_v19vce.csv", usecols=["target_macro_gas_kwh"])["target_macro_gas_kwh"].values
s = lambda g: np.sum(np.maximum(g - K, 0) ** 2) / len(g)
c = lambda g, t: np.sum(g > t) / len(g) * 183987
ref = s(z["gas_corrected"])
for name, g in [("H corrected", z["gas_corrected"]), ("H teacher", z["gas_teacher"]), ("P teacher (OOF)", z["gas_pP"]),
                ("full-train OOF teacher", pf), ("test teacher", pa), ("test corrected (sub)", sub)]:
    print(f"{name:24s} mean (g-9e5)+^2 {s(g):.3e} (ratio to H corrected {s(g) / ref:.3f}); per-183,987-rows count "
          f">9e5 {c(g, 9e5):6.1f} >9.5e5 {c(g, 9.5e5):5.1f} >1e6 {c(g, 1e6):5.1f}")
