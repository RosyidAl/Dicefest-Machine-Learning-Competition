"""verify2_postproc v0: inspect chain_H.npz keys/shapes and basic facts (independent reproduction)."""
import sys
import numpy as np
sys.path.insert(0, "src")
from common import TARGET_COLS, SIGMAS  # noqa: E402

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
for k in z.files:
    a = z[k]
    print(f"{k:16s} shape {a.shape} dtype {a.dtype} min {np.nanmin(a):.6g} max {np.nanmax(a):.6g}")
print("SIGMAS", {k: SIGMAS[k] for k in TARGET_COLS})
P, H = z["P"], z["H"]
print("len P", len(P), "len H", len(H), "overlap", len(np.intersect1d(P, H)), "union", len(np.union1d(P, H)))
