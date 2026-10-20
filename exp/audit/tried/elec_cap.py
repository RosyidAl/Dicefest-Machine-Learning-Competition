"""Does the gas cap (rows with gas >= 1e6 were removed) bias elec / leave signal for rows near the cap?
mean residual of the corrected elec on H by the corrected gas prediction; honest hinge fit on P, applied on H."""
import numpy as np
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
cg, ye, ce = z["gas_corrected"], z["elec_y"], z["elec_corrected"]
pPg, rPe, pPe = z["gas_pP"], z["elec_rP"], z["elec_pP"]
for lo, hi in ((6e5, 7e5), (7e5, 8e5), (8e5, 9e5), (9e5, 9.5e5), (9.5e5, 1e7)):
    s = (cg >= lo) & (cg < hi)
    sP = (pPg >= lo) & (pPg < hi)
    r = ye[s] - ce[s]
    print(f"gas pred in [{lo:.3g},{hi:.3g}) H n={s.sum()} elec resid {r.mean():+.0f} (se {r.std() / np.sqrt(s.sum()):.0f})"
          f" rel {np.mean(r / ce[s]):+.4f} | P n={sP.sum()} elec resid {rPe[sP].mean():+.0f} rel {np.mean(rPe[sP] / pPe[sP]):+.4f}")
