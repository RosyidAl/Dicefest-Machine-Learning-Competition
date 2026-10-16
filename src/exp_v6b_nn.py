"""v6b MLPs: v6 setup with seeds 45-47 (more seed averaging). Each fold trains E epochs of one-cycle annealing and keeps the final model,
so the validation rows never choose the epoch (v4/v5 best-epoch OOF was optimistic)."""
import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data(n_folds=10)
part = sys.argv[1]
kw = dict(lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(45, 46, 47))
TE = dict(use_te=True, windows=(2, 10), cells=True)
iA, iM = 1.0 / d.A, 1.0 / d.M


def go(name, kind, E, wmul):
    run_nn(d, name, kind, wmul=wmul, epochs=E, fixed_ep=E - 1, **TE, **kw)


if part == "gas":
    go("n6b_gpm", "gpm", 44, iA)
    go("n6b_gpgea", "gas_per:bld_dim_03", 36, iA)
    go("n6b_lgpm", "lgas_per:bld_dim_12", 40, d.A)
    go("n6b_eui_wA", "eui", 36, d.A)
    go("n6b_leui_wA", "leui", 33, d.A)
else:
    go("n6b_lepm", "lepm", 24, d.M.astype(float))
    go("n6b_epm_wM", "epm", 23, iM)
    go("n6b_epa", "elec_per:bld_dim_06", 20, iM)
