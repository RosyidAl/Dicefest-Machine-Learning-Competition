"""Determinism check (R05) for the NN variants used in the final blend: PLR MLP (p2/p3), TabM-mini + PLR (t1) and the
multi-task MLP (m1). Fits fold 0 of a 60k-row subsample for 2 epochs and prints the sha256 of the predictions.
Run twice in separate processes (scripts/check_repro.sh) and compare."""
import hashlib
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, "src")
import numpy as np  # noqa: E402

from nn import fit_fold, preprocess, set_seed  # noqa: E402
from train_lib import Data  # noqa: E402

d = Data(n_folds=10)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(d.X), 60000, replace=False))
Z, Zt = preprocess(d.X.iloc[idx], d.Xt.iloc[:20000])
y = np.log(d.y["target_macro_elec_kwh"][idx] / d.M[idx])
y2 = d.y["target_eui_gas_kwh_per_sqm"][idx]
a, b = np.arange(45000), np.arange(45000, 60000)
EMB = dict(k=24, sigma=0.05, d=24)
cases = {
    "plr": dict(emb=EMB, hidden=(1024, 512, 256), drop=0.3, lr=1e-3, wd=1e-4),
    "tabm_plr": dict(emb=EMB, hidden=(512, 512, 512), drop=0.1, lr=2e-3, wd=1e-4, ens=16),
    "plain": dict(hidden=(512, 256, 128)),
}
for name, kw in cases.items():
    set_seed(42)
    pb, pt, _ = fit_fold(Z[a], y[a], None, Z[b], y[b], None, Zt, epochs=2, fixed_ep=1, **kw)
    print(name, hashlib.sha256(np.ascontiguousarray(pb).tobytes() + np.ascontiguousarray(pt).tobytes()).hexdigest()[:16])
set_seed(42)
Y = np.column_stack([y, y2])
pb, pt, _ = fit_fold(Z[a], Y[a], np.ones_like(Y[a]), Z[b], Y[b], np.ones_like(Y[b]), Zt, epochs=2, fixed_ep=1,
                     hidden=(2048, 1024, 512))
print("multitask", hashlib.sha256(np.ascontiguousarray(pb).tobytes() + np.ascontiguousarray(pt).tobytes()).hexdigest()[:16])
