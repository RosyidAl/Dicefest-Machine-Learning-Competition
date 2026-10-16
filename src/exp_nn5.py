import sys; sys.path.insert(0, "src")
from train_lib import Data
from nn import run_nn
d = Data()
kw = dict(epochs=60, lr=1e-3, wd=1e-4, drop=0.3, hidden=(1024, 512, 256), seeds=(42, 43, 44))
TE = dict(use_te=True, windows=(2, 10))
iA, iM = 1.0 / d.A, 1.0 / d.M
run_nn(d, "nn5_gpgea_te", "gas_per:bld_dim_03", wmul=iA, **TE, **kw)
run_nn(d, "nn5_gpn_te", "gas_per:bld_dim_01", wmul=iA, **TE, **kw)
run_nn(d, "nn5_lgpm_te", "lgas_per:bld_dim_12", wmul=d.A, **TE, **kw)
run_nn(d, "nn5_epa_te", "elec_per:bld_dim_06", wmul=iM, **TE, **kw)
run_nn(d, "nn5_gas_te", "gas", wmul=iA, **TE, **kw)
run_nn(d, "nn5_elec_te", "elec", wmul=iM, **TE, **kw)
