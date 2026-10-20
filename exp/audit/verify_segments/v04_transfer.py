"""Bin-transfer estimate of the TEST gain of the cap calibration (k = 0.08): the per-row SSE change is estimated per bin
of the predicted gas (or eui * A) in three honest views (H corrected; P-side teacher; full-train VC OOF), then weighted
by the test row counts per bin (sub_v19vce, duplicate rows excluded). Also: public-size (30%) / private-size (70%)
resampling of the test bins, and a check of regression-to-the-mean bias just below the cap region.

usage: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python exp/audit/verify_segments/v04_transfer.py \
       > exp/audit/verify_segments/v04_transfer.log
"""
import json

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

SG, SE = 151750.04, 40.65
U, K, LO = 1e6, 0.08, 6e5
LOG2PI = 0.5 * np.log(2 * np.pi)
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
EDGES = [LO, 8.5e5, 9e5, 9.3e5, 9.6e5, 9.8e5, 1e6, 1.02e6, 1e9]


def cal(g, k=K, cap=U):
    g = np.asarray(g, float)
    out = g.copy()
    m = g > LO
    s = k * g[m]
    a = (cap - g[m]) / s
    out[m] = g[m] - s * np.exp(-0.5 * a * a - LOG2PI - log_ndtr(a))
    return out


z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
full = np.load("exp/audit/postproc/14_fullvc_oof.npz", allow_pickle=False)
P, H, A = z["P"], z["H"], z["A"]
tr = pd.read_parquet("cache/train.parquet", columns=KEY + [GAS])
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
ref = tr.groupby(KEY, sort=False)[[GAS]].first().reset_index()
hit = ~np.isnan(te[KEY].merge(ref, on=KEY, how="left")[GAS].values)
del tr
sub = pd.read_csv("submissions/sub_v19vce.csv", float_precision="round_trip")
At = te["bld_dim_06"].values
gT = sub[GAS].values[~hit]
eT = (sub[EUI].values * At)[~hit]
NT = int((~hit).sum()) + int(hit.sum())  # all test rows count in the metric (dup rows have ~0 change)

# views: (name, y on metric scale, p on metric scale, binning variable on gas scale, scale A for eui)
views = {
    "gas": [("H corrected", z["gas_y"], z["gas_corrected"], z["gas_corrected"]),
            ("P teacher", z["gas_pP"] + z["gas_rP"], z["gas_pP"], z["gas_pP"]),
            ("full VC OOF", full["gas_y"], full["gas_p"], full["gas_p"])],
    "eui": [("H corrected", z["eui_y"], z["eui_corrected"], z["eui_corrected"] * A[H]),
            ("P teacher", z["eui_pP"] + z["eui_rP"], z["eui_pP"], z["eui_pP"] * A[P]),
            ("full VC OOF", full["eui_y"], full["eui_p"], full["eui_p"] * A)],
}
test_bin = {"gas": gT, "eui": eT}
sig = {"gas": SG, "eui": SE}
base_rmse = {"gas": float(np.sqrt(np.mean((z["gas_y"] - z["gas_corrected"]) ** 2))),
             "eui": float(np.sqrt(np.mean((z["eui_y"] - z["eui_corrected"]) ** 2)))}
res = {}
rng = np.random.default_rng(5)
for met in ("gas", "eui"):
    nTb = np.histogram(test_bin[met], bins=EDGES)[0]
    print(f"\n== {met}: test rows per bin {dict(zip([f'{e / 1e6:.2f}' for e in EDGES[:-1]], nTb.tolist()))}")
    for name, y, p, g in views[met]:
        pn = p * cal(g) / np.where(g > 0, g, 1.0) if met == "eui" else cal(p)
        pn = np.where(g > LO, pn, p)
        d = (y - pn) ** 2 - (y - p) ** 2
        b = np.digitize(g, EDGES) - 1
        nVb = np.array([np.sum(b == j) for j in range(len(EDGES) - 1)])
        mVb = np.array([d[b == j].mean() if nVb[j] else 0.0 for j in range(len(EDGES) - 1)])
        own = (np.sqrt(base_rmse[met] ** 2 + d.sum() / len(y)) - base_rmse[met]) / sig[met] / 5
        dT = (nTb * mVb).sum() / NT
        est = (np.sqrt(base_rmse[met] ** 2 + dT) - base_rmse[met]) / sig[met] / 5
        # uncertainty of the transfer estimate: bootstrap the view rows within bins
        bs = []
        idx_b = [np.where(b == j)[0] for j in range(len(EDGES) - 1)]
        for _ in range(1000):
            m_ = np.array([d[rng.choice(ix, len(ix))].mean() if len(ix) else 0.0 for ix in idx_b])
            bs.append((np.sqrt(base_rmse[met] ** 2 + (nTb * m_).sum() / NT) - base_rmse[met]) / sig[met] / 5)
        bs = np.array(bs)
        # public (30%) and private (70%) subsets of the test: binomial thinning + resampled rows
        pubs, privs = [], []
        for _ in range(1000):
            for frac, store in ((0.3, pubs), (0.7, privs)):
                nsub = rng.binomial(nTb, frac)
                tot = sum(d[rng.choice(ix, n_)].sum() if (len(ix) and n_) else 0.0 for ix, n_ in zip(idx_b, nsub))
                store.append((np.sqrt(base_rmse[met] ** 2 + tot / (NT * frac)) - base_rmse[met]) / sig[met] / 5)
        pubs, privs = np.array(pubs), np.array(privs)
        print(f"  {name:12s} own-rows dTotal {own:+.8f} | test bin-transfer {est:+.8f} (sd {bs.std():.8f}) | "
              f"public-size mean {pubs.mean():+.8f} sd {pubs.std():.8f} P(<0) {np.mean(pubs < 0):.3f} | "
              f"private-size mean {privs.mean():+.8f} sd {privs.std():.8f} P(<0) {np.mean(privs < 0):.3f}")
        print(f"      per-bin mean dSSE (view): {np.round(mVb, 1).tolist()}  n view {nVb.tolist()}")
        res[f"{met}_{name}"] = dict(own=own, transfer=est, transfer_sd=float(bs.std()), pub_mean=float(pubs.mean()),
                                    pub_sd=float(pubs.std()), priv_mean=float(privs.mean()), priv_sd=float(privs.std()))

# regression-to-the-mean check: mean residual just below the cap region, where truncation is negligible
print("\n== mean residual (y - p) for p / 1e6 in [0.6, 0.85): negative would mean tail over-shoot not due to the cap")
for name, y, p, _ in views["gas"]:
    m = (p >= 6e5) & (p < 8.5e5)
    r = y[m] - p[m]
    print(f"  {name:12s} n {int(m.sum())} mean {r.mean():+.0f} (se {r.std(ddof=1) / np.sqrt(m.sum()):.0f})")
json.dump(res, open("exp/audit/verify_segments/v04_transfer.json", "w"), indent=1)
