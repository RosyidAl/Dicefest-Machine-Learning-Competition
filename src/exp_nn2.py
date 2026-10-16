import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data()
# regularised, weighted, with spatial encodings
kw = dict(epochs=30, lr=1e-3, wd=1e-4, drop=0.25, hidden=(512, 256, 128))
run_nn(d, "nn_eui_wA_te", "eui", wmul=d.A, use_te=True, windows=(2, 10), **kw)
run_nn(d, "nn_epm_wM_te", "epm", wmul=1.0 / d.M, use_te=True, windows=(2, 10), **kw)
run_nn(d, "nn_leui_wA_te", "leui", wmul=d.A, use_te=True, windows=(2, 10), **kw)
run_nn(d, "nn_lepm_te", "lepm", wmul=d.M.astype(float), use_te=True, windows=(2, 10), **kw)
