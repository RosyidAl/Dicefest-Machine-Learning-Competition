"""Gas-cap calibration (k = 0.08, applied to gas and to EUI via eui * A) on H corrected: total-score gain, its spread
over random 30% subsets (public-LB-sized, scaled to the test row count) and 70% subsets (private-like), and the gain
in each half of H.

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_cap_noise.py > exp/audit/segments/seg_cap_noise.log
"""
import numpy as np
from scipy.stats import norm

z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
H, A = z["H"], z["A"]
AH = A[H]
SG, SE = 151750.04, 40.65
K, U = 0.08, 1e6


def cap(g):
    s = K * g
    a = (U - g) / s
    return g - s * np.exp(norm.logpdf(a) - norm.logcdf(a))


yg, pg = z["gas_y"], z["gas_corrected"]
ye, pe = z["eui_y"], z["eui_corrected"]
pg1 = np.where(pg > 6e5, cap(np.maximum(pg, 1.0)), pg)
gE = pe * AH
pe1 = np.where(gE > 6e5, pe * cap(np.maximum(gE, 1.0)) / gE, pe)
eg0, eg1, ee0, ee1 = (yg - pg) ** 2, (yg - pg1) ** 2, (ye - pe) ** 2, (ye - pe1) ** 2


def gain(idx):
    dg = (np.sqrt(eg1[idx].mean()) - np.sqrt(eg0[idx].mean())) / SG
    de = (np.sqrt(ee1[idx].mean()) - np.sqrt(ee0[idx].mean())) / SE
    return dg / 5, de / 5


n = len(H)
g_all = gain(np.arange(n))
print(f"H all rows: gas dTotal {g_all[0]:+.8f}  eui dTotal {g_all[1]:+.8f}  sum {sum(g_all):+.8f}")
rng = np.random.default_rng(1)
for name, size in [("public-like (55,196 rows)", 55196), ("private-like (128,791 rows)", 128791)]:
    tot = []
    for _ in range(3000):
        idx = rng.integers(0, n, size)  # bootstrap at the test-subset size
        tot.append(sum(gain(idx)))
    tot = np.array(tot)
    print(f"{name}: mean {tot.mean():+.8f}  sd {tot.std():.8f}  P(change < 0) {np.mean(tot < 0):.4f}  "
          f"P(change <= -0.00001) {np.mean(tot <= -1e-5):.4f}")
perm = rng.permutation(n)
for h, idx in enumerate((perm[: n // 2], perm[n // 2:])):
    g = gain(idx)
    print(f"random half {h}: gas {g[0]:+.8f}  eui {g[1]:+.8f}")
for h in (0, 1):
    idx = np.arange(n)[np.arange(n) % 2 == h]
    g = gain(idx)
    print(f"row-parity half {h}: gas {g[0]:+.8f}  eui {g[1]:+.8f}")
