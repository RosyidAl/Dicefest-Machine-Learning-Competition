"""Which rows explain the max |actual - predicted S3| gap in the deployed pp files (rows M>4)?"""
import numpy as np, pandas as pd
from scipy.stats import norm
SIG = {"gas": 151750.04, "eui": 40.65}
COL = {"gas": "target_macro_gas_kwh", "eui": "target_eui_gas_kwh_per_sqm"}
CLAIM = {"gas": dict(k=0.5046, a0=0.00806, b0=0.39297), "eui": dict(k=0.6574, a0=0.00839, b0=0.38715)}
T = 999764.8988
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
At, Mt = te["bld_dim_06"].values.astype(float), te["bld_dim_12"].values.astype(float)
a, b = pd.read_csv("submissions/sub_v19nca.csv"), pd.read_csv("submissions/sub_v19ncapp.csv")
for m in ("gas", "eui"):
    c = CLAIM[m]; p = a[COL[m]].values; Tm = T if m == "gas" else T / At
    s = p * np.sqrt(c["a0"] + c["b0"] / Mt); bb = (Tm - p) / s
    ch = -c["k"] * s * np.exp(norm.logpdf(bb) - norm.logcdf(bb))
    diff = b[COL[m]].values - p
    sel = Mt > 4
    err = np.where(sel, diff - ch, 0)
    o = np.argsort(-np.abs(err))[:6]
    for i in o:
        print(m, te["id"].values[i], f"M {Mt[i]:.0f} p {p[i]:.2f} pred_chg {ch[i]:+.3f} actual {diff[i]:+.3f} err/sig {err[i]/SIG[m]:+.2e}")
    live = sel & (np.abs(diff) > 0)
    print(m, "rows M>4 with nonzero actual change: max |err|/sig", f"{np.abs(diff[live]-ch[live]).max()/SIG[m]:.2e}",
          "max rel err", f"{np.max(np.abs(diff[live]-ch[live])/np.maximum(np.abs(ch[live]),1e-12)):.2e}")
