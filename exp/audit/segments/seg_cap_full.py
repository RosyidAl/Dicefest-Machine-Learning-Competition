"""Gas cap (1e6) truncation calibration: third check on the full-train OOF of the v19vce gas members (global NNLS
blend, as the light proxy of the VC blend), test-side impact, and a candidate CSV (never submitted).

1. full-train OOF (exp/oof/<member>.npy, gas list of exp/v19vce/members.yaml): global NNLS blend for gas (EUI target,
   row scale A, times A) and EUI; near-cap residual table; k fitted on all OOF rows and 5-fold cross-fitted gain.
2. test: apply the IMR calibration (k = 0.085 from the P-side teacher fit, U = 1e6) to sub_v19vce gas and to EUI via
   g = eui * A; the 77 duplicate rows keep their copied train targets. Writes exp/audit/segments/sub_v19vce_cap.csv
   and checks that every other cell is byte-identical (re-write round trip).

usage: OMP_NUM_THREADS=2 .venv/bin/python exp/audit/segments/seg_cap_full.py > exp/audit/segments/seg_cap_full.log
"""
import hashlib
import io

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.stats import norm
from sklearn.model_selection import KFold

SIG = {"gas": 151750.04, "eui": 40.65}
GAS, EUI = "target_macro_gas_kwh", "target_eui_gas_kwh_per_sqm"
TARGETS = ["target_macro_gas_kwh", "target_macro_elec_kwh", "target_eui_gas_kwh_per_sqm",
           "target_clean_building_ratio", "target_electric_heating_adoption"]
KEY = ["socio_demog_005", "subzone_id", "bld_dim_12", "bld_dim_01", "bld_dim_06"]
U, K_P = 1e6, 0.085
KGRID = np.round(np.arange(0.01, 0.301, 0.005), 4)


def imr_adj(p, k, cap=U):
    s = np.maximum(k * p, 1e-9)
    a = (cap - p) / s
    return p - s * np.exp(norm.logpdf(a) - norm.logcdf(a))


def fitk(y, p):
    sse = [np.sum((y - imr_adj(p, k)) ** 2) for k in KGRID]
    return KGRID[int(np.argmin(sse))]


def nr(e, m):
    return np.sqrt(np.mean(e ** 2)) / SIG[m]


# ---- 1. full-train OOF check
tr = pd.read_parquet("cache/train.parquet", columns=KEY + TARGETS)
A = tr["bld_dim_06"].values
yg, ye = tr[GAS].values, tr[EUI].values
mem = yaml.safe_load(open("exp/v19vce/members.yaml"))["blend"]["gas"]
Po = np.column_stack([np.load(f"exp/oof/{r}.npy") for r in mem])
wg, _ = nnls(Po * A[:, None], ye * A)
we, _ = nnls(Po, ye)
pg, pe = Po @ wg * A, Po @ we
del Po
print(f"full OOF global NNLS ({len(mem)} gas members): gas NRMSE {nr(yg - pg, 'gas'):.5f}  eui {nr(ye - pe, 'eui'):.5f}")
bins = [0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0, 1.02]
g = np.digitize(pg / U, bins)
s = pd.DataFrame({"g": g, "e": yg - pg}).groupby("g")["e"].agg(["size", "mean", "std"])
s["t"] = s["mean"] / (s["std"] / np.sqrt(s["size"]))
for gi, r in s.iterrows():
    lo = bins[gi - 1] if gi > 0 else -np.inf
    hi = bins[gi] if gi < len(bins) else np.inf
    print(f"   p/1e6 in [{lo:>5.3g},{hi:>5.3g})  n {int(r['size']):6d}  mean resid {r['mean']:+10.4g}  t {r['t']:+6.2f}")
kf = fitk(yg, pg)
print(f"  k fitted on all full-OOF rows: {kf}  (P-side teacher fit {K_P}, all-H fit 0.08)")
for k in (K_P, kf):
    d = (nr(yg - imr_adj(pg, k), "gas") - nr(yg - pg, "gas")) / 5
    gE = pe * A
    dE = (nr(ye - pe * imr_adj(gE, k) / gE, "eui") - nr(ye - pe, "eui")) / 5
    print(f"  k={k}: full-OOF gas dTotal {d:+.8f}  eui (via eui*A) dTotal {dE:+.8f}")
outs = []
for rep in range(10):
    pn = pg.copy()
    for a, b in KFold(5, shuffle=True, random_state=rep).split(yg):
        pn[b] = imr_adj(pg[b], fitk(yg[a], pg[a]))
    outs.append((nr(yg - pn, "gas") - nr(yg - pg, "gas")) / 5)
print(f"  5-fold cross-fitted k on full OOF: gas dTotal {np.mean(outs):+.8f} (sd {np.std(outs):.8f})")

# ---- 2. test impact + candidate CSV
te = pd.read_parquet("cache/test.parquet", columns=["id"] + KEY)
raw = open("submissions/sub_v19vce.csv", "rb").read()
sub = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
buf = io.StringIO()
sub.to_csv(buf, index=False)
assert buf.getvalue().encode() == raw, "round trip of sub_v19vce.csv is not byte-identical"
assert (sub["id"].values == te["id"].values).all()
grp = tr.groupby(KEY, sort=False)[TARGETS]
ref = grp.first().reset_index()
dup = te[KEY].merge(ref, on=KEY, how="left")[TARGETS].values
hit = ~np.isnan(dup[:, 0])
print(f"\ntest: duplicate rows {int(hit.sum())}; their gas equals the copied train value: "
      f"{bool(np.all(sub.loc[hit, GAS].values == dup[hit, 0]))}")
At = te["bld_dim_06"].values
p0, e0 = sub[GAS].values.copy(), sub[EUI].values.copy()
p1 = np.where(hit, p0, imr_adj(p0, K_P))
g0 = e0 * At
e1 = np.where(hit, e0, e0 * imr_adj(g0, K_P) / g0)
for name, a0, a1, sg in [("gas", p0, p1, SIG["gas"]), ("eui", e0, e1, SIG["eui"])]:
    d = np.abs(a1 - a0)
    print(f"  {name}: rows changed >1e-9 rel {int(np.sum(d > 1e-9 * np.abs(a0)))}, >0.1% sigma {int(np.sum(d > 1e-3 * sg))}, "
          f">1% sigma {int(np.sum(d > 1e-2 * sg))}, >10% sigma {int(np.sum(d > 0.1 * sg))}; rms change/sigma "
          f"{np.sqrt(np.mean(d ** 2)) / sg:.6f}; max change {d.max():.6g}; new max {a1.max():.6g}")
z = np.load("exp/audit/chain_H.npz", allow_pickle=False)
print(f"  share of rows with gas pred > 0.85e6: H corrected {np.mean(z['gas_corrected'] > 0.85e6):.5f}  "
      f"test {np.mean(p0 > 0.85e6):.5f};  > 0.9e6: H {np.mean(z['gas_corrected'] > 0.9e6):.5f}  test {np.mean(p0 > 0.9e6):.5f}")
out = sub.copy()
out[GAS] = p1
out[EUI] = e1
out.to_csv("exp/audit/segments/sub_v19vce_cap.csv", index=False)
chk = pd.read_csv("exp/audit/segments/sub_v19vce_cap.csv", float_precision="round_trip")
for c in ["id"] + TARGETS:
    same = (chk[c].values == sub[c].values)
    print(f"  column {c}: cells identical to sub_v19vce {int(same.sum())} / {len(same)}")
print("  sha256", hashlib.sha256(open("exp/audit/segments/sub_v19vce_cap.csv", "rb").read()).hexdigest())
