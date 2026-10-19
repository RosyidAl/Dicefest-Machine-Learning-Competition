"""Test adoption vector from saved adoption-model outputs (exp/adopt/<name>_unseen.npy: r for the unseen subzones,
averaged over the 5 fold models) combined with fixed weights; seen subzones copy the train value.

usage: python src/adopt_make.py name1:w1,name2:w2,... <out.npy>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from adopt_v2 import subzone_table  # noqa: E402
from common import load  # noqa: E402

if __name__ == "__main__":
    spec = [(p.split(":")[0], float(p.split(":")[1])) for p in sys.argv[1].split(",")]
    out = sys.argv[2]
    tr, te = load()
    S = subzone_table(tr, te)
    lab = S.ad.notna().values
    L, U = S[lab], S[~lab]
    r_u = sum(w * np.load(f"exp/adopt/{n}_unseen.npy") for n, w in spec)
    known = dict(zip(L.index, L.ad.values))
    unseen = dict(zip(U.index, np.clip(U.rest.values - r_u, 0, 100)))
    pred = np.array([known[s] if s in known else unseen[s] for s in te["subzone_id"].values])
    np.save(out, pred)
    print(f"wrote {out}: unseen subzones {len(U)}, mean r {r_u.mean():.4f}")
