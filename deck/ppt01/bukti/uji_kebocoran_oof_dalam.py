"""Label-permutation test for the training rows (inner out-of-fold step), on the modules embedded in the notebook.

uji_kebocoran_notebook.py shuffles the labels of validation rows. This test covers the other half of the cross-fit:
the target-based features of the training rows of an outer fold are computed from four inner parts (KFold(5, shuffle,
seed 42) over the fit rows) and must never use the labels of the row's own inner part, its own label included.

Procedure: outer fold 0 of KFold(10, shuffle, 42); compute the inner-OOF kNN-target block (FB07 + FB07b) and the area
target encodings for the fit rows; shuffle the labels of the rows in inner part 0 among themselves; recompute. Rows in
inner part 0 must be bit-identical; rows of the other inner parts are expected to change (they use part 0 as
neighbours), which shows that the shuffle reached the computation.

usage: .venv/bin/python deck/ppt01/bukti/uji_kebocoran_oof_dalam.py   (writes uji_kebocoran_oof_dalam.json)
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import KFold

ROOT = "/root/Dicefest"
os.environ.setdefault("DICEFEST_DATA_DIR", os.path.join(ROOT, "data/raw"))
sys.path.insert(0, os.path.join(ROOT, "final/nbrun_final/dicefest_nb/src"))
from pipeline import make_data  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uji_kebocoran_oof_dalam.json")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


d = make_data("/tmp/unused_work")
d.knn_cache_dir = None
d.knn_x_cache_dir = None
d.use_x = True
a, _ = list(KFold(10, shuffle=True, random_state=42).split(d.X))[0]
ia, ib = list(KFold(5, shuffle=True, random_state=42).split(a))[0]  # inner part 0 = positions ib within a
log(f"fit rows {len(a)}, inner part 0: {len(ib)} rows, other inner parts: {len(ia)} rows")


def blocks():
    k = d.knn_features(a, "oof").to_numpy()
    log("knn inner OOF done")
    t = d.te_features(a, "oof", windows=(2, 10), cells=True).to_numpy()
    log("target encodings inner OOF done")
    return {"knn_latih_oof": k, "te_latih_oof": t}


F1 = blocks()
V0 = d.V.copy()
rows = a[ib]
d.V[rows] = V0[rows[np.random.default_rng(3).permutation(len(rows))]]
assert not np.array_equal(d.V[rows], V0[rows])
log("labels of inner part 0 shuffled")
F2 = blocks()


def changed(x1, x2):
    return int((~np.all((x1 == x2) | (np.isnan(x1) & np.isnan(x2)), axis=1)).sum())


res = {}
for name in F1:
    res[name] = {"kolom": int(F1[name].shape[1]),
                 "baris_bagian_0": int(len(ib)), "berubah_bagian_0": changed(F1[name][ib], F2[name][ib]),
                 "baris_bagian_lain": int(len(ia)), "berubah_bagian_lain": changed(F1[name][ia], F2[name][ia])}
    log(f"{name}: {res[name]}")
ok = all(v["berubah_bagian_0"] == 0 and v["berubah_bagian_lain"] > 0 for v in res.values())
json.dump({"lulus": ok, "fold_luar": "0 dari KFold(10, shuffle=True, random_state=42)",
           "bagian_dalam": "0 dari KFold(5, shuffle=True, random_state=42) atas baris latih", "blok": res,
           "detik": round(time.time() - t0)}, open(OUT, "w"), indent=1, ensure_ascii=False)
log(f"PASS={ok}; wrote {OUT}")
