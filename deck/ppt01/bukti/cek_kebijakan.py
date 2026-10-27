"""Checks behind the policy slides, run after the independent review of deck v2 (read-only, deterministic).

1. Adoption by gas-network share (train rows, weighted by meters): adoption rate and share of electric-heating
   households per band of hvac_dist_02.
2. Zone gas-network share (meter-weighted, as explain.zone_indices): how many zones are above 80 %.
3. Robustness of the zone ranking of the electricity increase beyond uniform scaling:
   (a) heat-pump COP that varies with climate: COP 3.5 in the mildest zone down to 2.5 in the harshest, by the zone's
       mean env_climate_05 (higher values go with higher EUI in the explanation model, cell 47);
   (b) prediction error on the test rows: multiplicative log-normal noise on gas and electricity predictions with the
       relative error of the honest blend (gas 0.2449 x 151750 / mean gas, elec 0.3159 x 41725 / mean elec), 200 draws;
   (c) both. Reported: Spearman rank correlation with the baseline ranking, zones that change quadrant, overlap of the
       top 5 zones.
4. Composition of the "retrofit" potential (cell 48 definition: above-average SHAP of building era + building dimensions
   + derived physical ratios) on the SHAP sample of cell 45 (20,000 train rows): share left when only the building-era
   family is counted, and when the size and count features are excluded.
5. Size contrast behind +65 % / +75 % (cell 48): mean floor area per meter of the extreme 5 % bins, and the contrast
   between the bottom and top quartile.
6. Ranking of zones by absolute added electricity (gas x 0.8 / 3) versus by percentage.

usage: .venv/bin/python deck/ppt01/bukti/cek_kebijakan.py   (writes cek_kebijakan.json)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/root/Dicefest"
os.environ.setdefault("DICEFEST_DATA_DIR", os.path.join(ROOT, "data/raw"))
sys.path.insert(0, os.path.join(ROOT, "final/nbrun_final/dicefest_nb/src"))
from common import SIGMAS, TARGET_COLS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cek_kebijakan.json")
GAS, EL, EUI, CLEAN, ADOPT = TARGET_COLS
R = {}

tr = pd.read_csv(os.path.join(ROOT, "data/raw/train.csv"))
te = pd.read_csv(os.path.join(ROOT, "data/raw/test.csv"))
sub = pd.read_csv(os.path.join(ROOT, "submissions/BEST_CURRENT.csv"))
assert (sub["id"].values == te["id"].values).all()

# 1. adoption by gas-network share band (train rows)
band = pd.cut(tr["hvac_dist_02"], [-0.01, 40, 60, 80, 100.01], labels=["<=40", "40-60", "60-80", ">80"])
M = tr["bld_dim_12"].astype(float)
g = pd.DataFrame({"band": band, "M": M, "eh": tr[ADOPT] / 100 * M}).groupby("band", observed=True).sum()
R["adopsi_per_pita_gas_jaringan"] = {
    b: {"porsi_kawasan_pct": round(100 * float((band == b).mean()), 2),
        "tingkat_adopsi_pct": round(100 * float(g.loc[b, "eh"] / g.loc[b, "M"]), 2),
        "porsi_rumah_tangga_pemanas_listrik_pct": round(100 * float(g.loc[b, "eh"] / g["eh"].sum()), 2)}
    for b in g.index}

# zone aggregates: observed train targets, final predictions for test rows (as explain.zone_indices)
for c in (GAS, EL, ADOPT):
    te[c] = sub[c].values
allr = pd.concat([tr.assign(_t=0), te.assign(_t=1)], ignore_index=True)
Mall = allr["bld_dim_12"].astype(float).values
base = pd.DataFrame({"zone": allr["zone_region_id"].values, "gas": allr[GAS].values, "elec": allr[EL].values,
                     "area": allr["bld_dim_06"].values, "M": Mall, "gnw": allr["hvac_dist_02"].values * Mall,
                     "clim": allr["env_climate_05"].values, "test": allr["_t"].values})
Z = base.groupby("zone").agg(gas=("gas", "sum"), elec=("elec", "sum"), area=("area", "sum"), M=("M", "sum"),
                             gnw=("gnw", "sum"), clim=("clim", "mean"))
Z["gas_share"] = Z.gnw / Z.M
Z["eui"] = Z.gas / Z.area
Z["inc"] = 100 * Z.gas * 0.8 / 3.0 / Z.elec
assert round(Z.gas.sum() / 1e9, 2) == 141.35 and round(Z.inc.median(), 2) == 94.85


def quadrant(eui, inc):
    xm, ym = np.median(eui), np.median(inc)
    return np.where(eui >= xm, np.where(inc >= ym, "Q1", "Q2"), np.where(inc >= ym, "Q3", "Q4"))


q0 = quadrant(Z.eui.values, Z.inc.values)
top0 = set(Z.inc.sort_values(ascending=False).index[:5])

# 2. zone gas-network share
R["zona_gas_jaringan"] = {"zona_di_atas_80_pct": int((Z.gas_share > 80).sum()), "zona_total": int(len(Z)),
                         "median_zona_pct": round(float(Z.gas_share.median()), 2),
                         "kawasan_di_atas_80_pct": round(100 * float((allr["hvac_dist_02"] > 80).mean()), 2)}


def compare(inc_new, eui_new=None):
    eui_new = Z.eui.values if eui_new is None else eui_new
    rho = spearmanr(Z.inc.values, inc_new).statistic
    q = quadrant(eui_new, inc_new)
    top = set(Z.index[np.argsort(-inc_new)[:5]])
    return float(rho), int((q != q0).sum()), len(top & top0)


# 3a. climate-dependent COP
r = Z.clim.rank(pct=True).values  # 0 = mildest, 1 = harshest
cop = 3.5 - 1.0 * (r - r.min()) / (r.max() - r.min())
inc_cop = 100 * Z.gas.values * 0.8 / cop / Z.elec.values
rho, sw, top = compare(inc_cop)
R["cop_per_iklim"] = {"spearman": round(rho, 4), "zona_pindah_kuadran": sw, "irisan_5_teratas": top,
                      "cop_min": 2.5, "cop_max": 3.5}

# 3b. prediction noise on test rows, 3c. noise + climate COP
rel_gas = 0.2449 * SIGMAS[GAS] / sub[GAS].mean()
rel_el = 0.3159 * SIGMAS[EL] / sub[EL].mean()
rng = np.random.default_rng(0)
is_test = base["test"].values == 1
codes, zones = pd.factorize(base["zone"].values)
assert list(zones) == list(Z.index) or set(zones) == set(Z.index)
order = pd.Index(zones).get_indexer(Z.index)
gas_v, el_v = base["gas"].values, base["elec"].values
area_z = Z.area.values
stats = {"noise": [], "noise_cop": []}
for _ in range(200):
    gn = gas_v * np.where(is_test, np.exp(rng.normal(-rel_gas ** 2 / 2, rel_gas, len(gas_v))), 1.0)
    en = el_v * np.where(is_test, np.exp(rng.normal(-rel_el ** 2 / 2, rel_el, len(el_v))), 1.0)
    gz = np.bincount(codes, gn)[order]
    ez = np.bincount(codes, en)[order]
    stats["noise"].append(compare(100 * gz * 0.8 / 3.0 / ez, gz / area_z))
    stats["noise_cop"].append(compare(100 * gz * 0.8 / cop / ez, gz / area_z))
for k, v in stats.items():
    v = np.array(v)
    R[f"derau_prediksi{'_dan_cop' if k == 'noise_cop' else ''}"] = {
        "galat_relatif_gas": round(float(rel_gas), 3), "galat_relatif_listrik": round(float(rel_el), 3),
        "spearman_median": round(float(np.median(v[:, 0])), 4), "spearman_min": round(float(v[:, 0].min()), 4),
        "pindah_kuadran_median": float(np.median(v[:, 1])), "pindah_kuadran_maks": int(v[:, 1].max()),
        "irisan_5_teratas_median": float(np.median(v[:, 2]))}

# 6. absolute added electricity
add = Z.gas * 0.8 / 3.0
rank_abs = add.rank(ascending=False)
R["tambahan_absolut"] = {"5_teratas_TWh": {z: round(float(add[z] / 1e9), 3) for z in add.sort_values(
    ascending=False).index[:5]}, "peringkat_REG_290": int(rank_abs["REG_290"]),
    "spearman_persen_vs_absolut": round(float(spearmanr(Z.inc, add).statistic), 3)}

# 4 and 5: SHAP sample of the explanation models (same arrays as cells 45 and 47)
from train_lib import Data  # noqa: E402
import explain  # noqa: E402

d = Data()
ex, ext = d.agg_features()
X, _ = explain.feature_frame(d, ex, ext)
cols = list(X.columns)
fam = np.array([explain.family(c) for c in cols])
z = np.load(os.path.join(ROOT, "reports/policy/shap_eui.npz"))
idx, sv = z["idx"], z["sv"]
assert round(float(np.abs(sv[:, cols.index("area_per_meter")]).mean()), 4) == 0.0895
gas_s = tr[GAS].values[idx]
size_cols = {"area_per_meter", "area_per_epc", "meters_per_epc", "s002_ratio", "bld_dim_06", "bld_dim_12",
             "bld_dim_01"}
stock_mask = np.isin(fam, ["era bangunan", "dimensi bangunan", "rasio fisik turunan"])
era_mask = fam == "era bangunan"
nosize_mask = stock_mask & ~np.isin(cols, list(size_cols))


def saving(mask):
    s = sv[:, mask].sum(axis=1)
    return float((gas_s * (1 - np.exp(-np.clip(s, 0, None)))).sum())


full = saving(stock_mask)
R["komposisi_potensi_retrofit"] = {"porsi_jika_hanya_periode_bangunan": round(saving(era_mask) / full, 3),
                                   "porsi_tanpa_fitur_ukuran_dan_jumlah": round(saving(nosize_mask) / full, 3),
                                   "sampel": int(len(idx))}

apm = X["area_per_meter"].values
for t, key, sgn in (("eui", "gas_per_m2", 1), ("epm", "listrik_per_rumah", -1)):  # gas: small vs large; elec: large vs small
    z = np.load(os.path.join(ROOT, f"reports/policy/shap_{t}.npz"))
    idx, sv = z["idx"], z["sv"]
    x, s = apm[idx], sv[:, cols.index("area_per_meter")]
    bins = pd.qcut(x, 20, labels=False)
    lo, hi = bins == 0, bins == bins.max()
    q1, q3 = np.quantile(x, [0.25, 0.75])
    R[f"kontras_ukuran_{key}"] = {
        "pita_5pct_terkecil_rata2_m2": round(float(x[lo].mean()), 1),
        "pita_5pct_terbesar_rata2_m2": round(float(x[hi].mean()), 1),
        "selisih_pita_ekstrem_pct": round(100 * float(np.exp(sgn * (s[lo].mean() - s[hi].mean())) - 1), 1),
        "kuartil_bawah_atas_m2": [round(float(q1), 1), round(float(q3), 1)],
        "selisih_kuartil_pct": round(100 * float(np.exp(sgn * (s[x <= q1].mean() - s[x >= q3].mean())) - 1), 1)}

json.dump(R, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
