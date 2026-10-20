"""verify2 (segments lens): independent check of the 'gas > 1e6' blunder claim on the actual files.

Reads only column subsets (train targets + area/meters, test id/area/meters) and the submission CSVs.
Writes nothing except this script's log (stdout is redirected by the caller).
"""
import resource
import time

import numpy as np
import pandas as pd

t0 = time.time()
CAP = 1e6
tr = pd.read_parquet("cache/train.parquet",
                     columns=["id", "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm", "bld_dim_06", "bld_dim_12"])
te = pd.read_parquet("cache/test.parquet", columns=["id", "bld_dim_06", "bld_dim_12"])
ss = pd.read_csv("data/raw/sample_submission.csv", usecols=["id"])
g = tr["target_macro_gas_kwh"].to_numpy()
e = tr["target_eui_gas_kwh_per_sqm"].to_numpy()
A = tr["bld_dim_06"].to_numpy()
print(f"train n={len(tr)} test n={len(te)} total={len(tr)+len(te)} train share={len(tr)/(len(tr)+len(te)):.10f} "
      f"0.7*total={0.7*(len(tr)+len(te)):.3f}")
print(f"[train gas] max={g.max():.4f} n>=1e6={(g >= CAP).sum()} n>=999000={(g >= 999000).sum()} "
      f"n>=990000={(g >= 990000).sum()} nan={np.isnan(g).sum()}")
print(f"[train eui*A vs gas] max |eui*A-gas|/gas = {np.max(np.abs(e * A - g) / g):.3e}; max eui*A = {np.max(e * A):.4f}")
# tail density per 10k kWh band, to see whether the distribution is cut (not tapering) at 1e6
edges = np.arange(700_000, 1_100_001, 50_000)
cnt, _ = np.histogram(g, bins=edges)
print("[train gas] rows per 50k band 700k..1.1M:", " ".join(f"{int(a/1e3)}k:{c}" for a, c in zip(edges[:-1], cnt)))
# simple log-linear tail fit on 700k..1e6 (6 bands), extrapolated to 1e6..1.1e6 if there were no cap
x = (edges[:-1] + 25_000)[:6]
yv = np.log(cnt[:6])
b, a = np.polyfit(x, yv, 1)
exp_above = np.exp(a + b * np.array([1_025_000, 1_075_000])).sum()
print(f"[train gas] log-linear tail fit on 700k-1M extrapolates ~{exp_above:.0f} train rows in [1e6,1.1e6) "
      f"if uncapped; observed {((g >= 1e6) & (g < 1.1e6)).sum()}")

# test/sample alignment
te_ids = te["id"].to_numpy()
print(f"[ids] sample_submission order == cache/test order: {np.array_equal(ss['id'].to_numpy(), te_ids)}")
At = te.set_index("id")["bld_dim_06"]
for f in ["submissions/sub_v19vce.csv", "submissions/BEST_CURRENT.csv", "submissions/sub_v19nca2pp.csv",
          "submissions/sub_v19ncapp.csv", "submissions/sub_v19nca.csv"]:
    try:
        s = pd.read_csv(f)
    except FileNotFoundError:
        print(f"[{f}] missing")
        continue
    same_order = np.array_equal(s["id"].to_numpy(), ss["id"].to_numpy())
    gs = s["target_macro_gas_kwh"].to_numpy()
    es = s["target_eui_gas_kwh_per_sqm"].to_numpy()
    As = At.reindex(s["id"]).to_numpy()
    over = np.where(gs > CAP)[0]
    over_tm = np.where(gs > g.max())[0]
    over_e = np.where(es * As > CAP)[0]
    print(f"[{f}] rows={len(s)} id order ok={same_order} gas>1e6: {len(over)}  gas>train max: {len(over_tm)}  "
          f"eui*A>1e6: {len(over_e)}  gas max={gs.max():.1f}  max eui*A={np.max(es * As):.1f}")
    if f.endswith("sub_v19vce.csv"):
        o = over[np.argsort(-gs[over])]
        for i in o:
            print(f"    gas  {s['id'].iat[i]:>10s} pred={gs[i]:12.1f} excess={gs[i]-CAP:9.1f} excess/sigma={(gs[i]-CAP)/151750.04:.4f}"
                  f"  M={te.set_index('id')['bld_dim_12'].get(s['id'].iat[i])}  A={As[i]:.1f}")
        oe = over_e[np.argsort(-(es[over_e] * As[over_e]))]
        for i in oe:
            cap_e = CAP / As[i]
            print(f"    eui  {s['id'].iat[i]:>10s} pred={es[i]:9.3f} cap 1e6/A={cap_e:9.3f} excess/sigma={(es[i]-cap_e)/40.65:.4f}"
                  f"  gas pred={gs[i]:12.1f}")
        np.save("exp/audit/verify2_segments/hc_test_gas_over_idx.npy", over)
        np.save("exp/audit/verify2_segments/hc_test_eui_over_idx.npy", over_e)
print(f"elapsed {time.time()-t0:.1f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
