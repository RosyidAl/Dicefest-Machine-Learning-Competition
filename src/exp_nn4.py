import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data()
kw = dict(epochs=60, lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))
TE = dict(use_te=True, windows=(2, 10))
run_nn(d, "nn3_leui_wA_te", "leui", wmul=d.A, **TE, **kw)
run_nn(d, "nn3_lepm_te", "lepm", wmul=d.M.astype(float), **TE, **kw)
run_nn(d, "nn3_gpm_te", "gpm", wmul=1.0 / d.A, **TE, **kw)
kw2 = dict(epochs=60, lr=7e-4, wd=3e-4, drop=0.2, hidden=(2048, 1024, 512, 256), seeds=(45, 46, 47))
run_nn(d, "nn4_eui_wA_te", "eui", wmul=d.A, **TE, **kw2)
run_nn(d, "nn4_epm_wM_te", "epm", wmul=1.0 / d.M, **TE, **kw2)
