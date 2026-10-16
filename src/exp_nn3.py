import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data()
kw = dict(epochs=60, lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))
run_nn(d, "nn3_eui_wA_te", "eui", wmul=d.A, use_te=True, windows=(2, 10), **kw)
run_nn(d, "nn3_epm_wM_te", "epm", wmul=1.0 / d.M, use_te=True, windows=(2, 10), **kw)
