"""v9: MLPs with PLR numerical embeddings (nn.PLR), full-data 10-fold. On the holdout split (exp_hold_plr.py)
p1_lepm was a weaker single model than n6_lepm but improved the elec blend by ~0.0005 NRMSE (diversity);
p1_gpm improved the gas / EUI blends by ~0.0005 / ~0.0009. The other formulations mirror the n6_* MLPs.

usage: python src/exp_v9_nn.py name1,name2,...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from train_lib import Data  # noqa: E402
from nn import run_nn  # noqa: E402

d = Data(n_folds=10)
kw = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))
TE = dict(use_te=True, windows=(2, 10), cells=True)
EMB = dict(k=24, sigma=0.05, d=24)
iA, iM = 1.0 / d.A, 1.0 / d.M
# name: (kind, epochs, sample weight, embedding settings)
JOBS = {
    "p1_lepm": ("lepm", 24, d.M.astype(float), EMB),
    "p1_epm_wM": ("epm", 23, iM, EMB),
    "p1_gpm": ("gpm", 44, iA, EMB),
    "p1_lgpm": ("lgas_per:bld_dim_12", 40, d.A, EMB),
    "p1_gpgea": ("gas_per:bld_dim_03", 36, iA, EMB),
    "p1_leui_wA": ("leui", 33, d.A, EMB),
    "p1_epa": ("elec_per:bld_dim_06", 20, iM, EMB),
}
for name in sys.argv[1].split(","):
    kind, E, wmul, emb = JOBS[name]
    if not os.path.exists(f"exp/pred/{name}.npy"):
        run_nn(d, name, kind, wmul=wmul, epochs=E, fixed_ep=E - 1, emb=emb, **TE, **kw)
