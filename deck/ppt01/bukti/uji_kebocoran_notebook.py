"""Label-permutation leakage test on the exact modules embedded in the final notebook (dicefest_nb/src).

For one outer fold (fold 0 of the 10-fold KFold(shuffle, seed 42) used by the final members), every target-based
feature block is computed twice: once with the real targets and once after the targets of the validation rows are
shuffled among themselves. A block that never reads validation labels gives bit-identical validation, test and
fit-row features. A deliberately leaky variant (neighbours taken from all rows, validation rows included) is run as a
positive control and must change, which shows that the test can detect leakage.

Blocks: FB07 + FB07b kNN-target (fb07b.DataX.knn_features with use_x=True, as built by pipeline.make_data) and the
area target encodings (train_lib.Data.te_features with sub_n windows 2 and 10 and climate cells).

usage: .venv/bin/python deck/ppt01/bukti/uji_kebocoran_notebook.py   (writes uji_kebocoran_notebook.json next to it)
"""
import json
import os
import sys
import time

import numpy as np

ROOT = "/root/Dicefest"
os.environ.setdefault("DICEFEST_DATA_DIR", os.path.join(ROOT, "data/raw"))
sys.path.insert(0, os.path.join(ROOT, "final/nbrun_final/dicefest_nb/src"))
from pipeline import make_data  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uji_kebocoran_notebook.json")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


d = make_data("/tmp/unused_work")
d.knn_cache_dir = None  # no cache: every block is recomputed from the (possibly shuffled) targets
d.knn_x_cache_dir = None
d.use_x = True  # FB07 + FB07b, the widest kNN-target block in the ensemble
folds = list(__import__("sklearn.model_selection", fromlist=["KFold"]).KFold(10, shuffle=True, random_state=42)
             .split(d.X))
a, b = folds[0]
log(f"train rows {len(d.X)}, fit rows {len(a)}, validation rows {len(b)}, test rows {len(d.Xt)}")


def blocks(d):
    out = {}
    out["knn_validasi"] = d.knn_features(a, ("train", b))
    log("knn validation done")
    out["knn_uji"] = d.knn_features(a, "test")
    log("knn test done")
    out["knn_latih_oof"] = d.knn_features(a, "oof")
    log("knn fit-row (inner OOF) done")
    out["te_validasi"] = d.te_features(a, d.te_apply(b), windows=(2, 10), cells=True)
    out["te_uji"] = d.te_features(a, d.te_apply(), windows=(2, 10), cells=True)
    out["te_latih_oof"] = d.te_features(a, "oof", windows=(2, 10), cells=True)
    log("target encodings done")
    # positive control: a leaky kNN block whose neighbour pool contains the validation rows themselves
    d.use_x = False
    out["kontrol_bocor_knn_validasi"] = d.knn_features(np.arange(len(d.X)), ("train", b))
    d.use_x = True
    log("leaky control done")
    return out


F1 = blocks(d)
rng = np.random.default_rng(1)
V_orig = d.V.copy()
d.V[b] = V_orig[b[rng.permutation(len(b))]]  # shuffle every target-derived column among the validation rows
assert not np.array_equal(d.V[b], V_orig[b]) and np.array_equal(d.V[a], V_orig[a])
log("validation labels shuffled")
F2 = blocks(d)

res = {}
for k in F1:
    x1, x2 = F1[k].to_numpy(), F2[k].to_numpy()
    same = bool(F1[k].equals(F2[k]))
    changed_rows = int((~np.all((x1 == x2) | (np.isnan(x1) & np.isnan(x2)), axis=1)).sum())
    res[k] = {"identik_bit": same, "kolom": int(x1.shape[1]), "baris": int(x1.shape[0]),
              "baris_berubah": changed_rows}
    log(f"{k}: identical={same} changed rows={changed_rows} of {x1.shape[0]} ({x1.shape[1]} columns)")

ok = all(v["identik_bit"] for k, v in res.items() if not k.startswith("kontrol")) and \
    not res["kontrol_bocor_knn_validasi"]["identik_bit"]
summary = {"lulus": ok, "fold": "0 dari KFold(10, shuffle=True, random_state=42)", "baris_latih": int(len(a)),
           "baris_validasi": int(len(b)), "baris_uji": int(len(d.Xt)), "blok": res,
           "kode": "final/nbrun_final/dicefest_nb/src (modul yang tertanam di notebook final)",
           "detik": round(time.time() - t0)}
json.dump(summary, open(OUT, "w"), indent=1, ensure_ascii=False)
log(f"PASS={ok}; wrote {OUT}")
