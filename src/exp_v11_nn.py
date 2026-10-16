"""v11: MLPs with PLR embeddings on short schedules (p2_*) and TabM-mini + PLR (t1_*), all with the subzone / zone
aggregates over train+test rows (Data.agg_features). Full-data 10-fold.

Single-fold tests (fold 0, one seed) showed the v9 PLR runs were overtrained: PLR lepm 24 epochs 0.32946 vs
12 epochs 0.32204 (plain MLP 0.32371, TabM-mini + PLR 0.32080); PLR gpm EUI 24 epochs 0.67754 vs 12 epochs 0.67050.
The aggregates moved PLR lepm 0.32204 -> 0.32172.

usage: python src/exp_v11_nn.py name1,name2,...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from train_lib import Data  # noqa: E402
from nn import run_nn  # noqa: E402

d = Data(n_folds=10)
ex, ext = d.agg_features()
TE = dict(use_te=True, windows=(2, 10), cells=True, extra=ex, extra_t=ext)
EMB = dict(k=24, sigma=0.05, d=24)
PLR = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44), emb=EMB)
TABM = dict(lr=2e-3, wd=1e-4, drop=0.1, hidden=(512, 512, 512), seeds=(42,), emb=EMB, ens=16)
# fold-0 sweep (lepm, 1 seed): 10 epochs + sigma 0.02 + d 48 gave 0.32084 vs 0.32172 for the p2 settings; for gpm every
# change from the p2 settings was worse, so p3 is used for the elec family only
PLR3 = dict(PLR, emb=dict(k=24, sigma=0.02, d=48))
iA, iM = 1.0 / d.A, 1.0 / d.M
# name: (target formulation, epochs, sample weight, model settings)
JOBS = {
    "p2_lepm": ("lepm", 12, d.M.astype(float), PLR),
    "p2_gpm": ("gpm", 12, iA, PLR),
    "t1_lepm": ("lepm", 24, d.M.astype(float), TABM),
    "t1_gpm": ("gpm", 16, iA, TABM),
    "p2_lgpm": ("lgas_per:bld_dim_12", 12, d.A, PLR),
    "p2_gpgea": ("gas_per:bld_dim_03", 12, iA, PLR),
    "p2_epm_wM": ("epm", 12, iM, PLR),
    "p2_leui_wA": ("leui", 12, d.A, PLR),
    "p2_gpn": ("gas_per:bld_dim_01", 12, iA, PLR),
    "p2_gas": ("gas", 12, None, PLR),
    "p2_elec": ("elec", 12, None, PLR),
    "p2_epa": ("elec_per:bld_dim_06", 12, iM, PLR),
    "p2_eui_wA": ("eui", 12, d.A, PLR),
    "t1_epm_wM": ("epm", 24, iM, TABM),
    "t1_gpgea": ("gas_per:bld_dim_03", 16, iA, TABM),
    "p3_lepm": ("lepm", 10, d.M.astype(float), PLR3),
    "p3_epm_wM": ("epm", 10, iM, PLR3),
}
for name in sys.argv[1].split(","):
    kind, E, wmul, cfg = JOBS[name]
    if not os.path.exists(f"exp/pred/{name}.npy"):
        run_nn(d, name, kind, wmul=wmul, epochs=E, fixed_ep=E - 1, **TE, **cfg)
