"""NCA0001 members (src/nca.py): retrieval regressor on the inputs of the kNN NN members (base + aggregates + spatial
encodings + FB07/FB07b kNN-target block). full: 10-fold on all train rows -> exp/oof|pred; hold: 10-fold inside the
70% part P, predictions for H -> exp/hold/oof|pred; smoke: 2 folds of full mode on a few epochs.
usage: python src/exp_nca.py full|hold|smoke name1,name2,...
"""
import copy
import os
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, TARGET_COLS, load  # noqa: E402
from fb07b import DataX  # noqa: E402
from nca import run_nca  # noqa: E402
from resid_knn import spaces  # noqa: E402

JOBS = {  # name: kind, weight code, settings
    "nc_lgpm": ("lgas_per:bld_dim_12", "A", dict(epochs=12)),
    "nc_lepm": ("lepm", "M", dict(epochs=12)),
    "ncp_lgpm": ("lgas_per:bld_dim_12", "A", dict(epochs=20, emb=dict(k=16, sigma=0.05, d=8))),
}

mode, names = sys.argv[1], sys.argv[2].split(",")
tr, te = load()
d = DataX(n_folds=10)
S = spaces(tr, te)
n = len(tr)
d.sp = {k: (v[:n], v[n:]) for k, v in S.items()}
d.knn_cache_dir = "cache/knn"
d.knn_x_cache_dir = "cache/knn_x"
ex, ext = d.agg_features()
if mode == "hold":
    perm = np.random.default_rng(7).permutation(len(d.X))
    P, H = np.sort(perm[: int(0.7 * len(perm))]), np.sort(perm[int(0.7 * len(perm)):])
    s = copy.copy(d)
    s.X, s.Xt = d.X.iloc[P].reset_index(drop=True), d.X.iloc[H].reset_index(drop=True)
    s.A, s.At, s.M, s.Mt = d.A[P], d.A[H], d.M[P], d.M[H]
    s.y = {c: d.y[c][P] for c in TARGET_COLS}
    s.sub, s.subt, s.zone, s.zonet = d.sub[P], d.sub[H], d.zone[P], d.zone[H]
    s.subn, s.subnt = d.subn[P], d.subn[H]
    s.V, s.W = d.V[P], d.W[P]
    s.cells, s.cellst = [c[P] for c in d.cells], [c[H] for c in d.cells]
    s.folds = list(KFold(10, shuffle=True, random_state=SEED).split(s.X))
    s._kz = None
    s.sp = {k: (v[0][P], v[0][H]) for k, v in d.sp.items()}
    D, out, agg = s, "exp/hold", dict(extra=ex.iloc[P].reset_index(drop=True), extra_t=ex.iloc[H].reset_index(drop=True))
else:
    D, out, agg = d, ("exp/nca_smoke" if mode == "smoke" else "exp"), dict(extra=ex, extra_t=ext)
W = {"A": D.A, "iA": 1.0 / D.A, "M": D.M.astype(float), "iM": 1.0 / D.M, None: None}
for name in names:
    kind, wk, kw = JOBS[name]
    if mode != "smoke" and os.path.exists(f"{out}/pred/{name}.npy"):
        continue
    kw = dict(kw)
    folds = None
    if mode == "smoke":
        kw["epochs"], folds = 2, [0, 1]
    run_nca(D, name, kind, out_dir=out, wmul=W[wk], folds=folds, **agg, **kw)
