"""Where the zones named in the v3 deck sit under the two COP scenarios, and how subzone numbering follows zones.

Zone aggregates are built exactly as in deck/ppt01/bukti/cek_kebijakan.py and explain.zone_indices: observed targets
for train rows, final predictions (submissions/BEST_CURRENT.csv, sha aec745ab...d66e) for test rows. Scenarios:
  - base: 80% of gas for heating, heat-pump COP 3 for every zone (notebook cell 48);
  - climate COP: COP 3.5 in the mildest zone down to 2.5 in the harshest, by the rank of the zone's mean
    env_climate_05 (cek_kebijakan.py 3a; the script asserts its Spearman 0.8147 and 68 quadrant changes).
For each named zone: quadrant, rank of the relative increase, rank of the absolute addition, gas-network share and
adoption (both meter-weighted, as cell 50). Also the Q2 zones that stay Q2 under the climate COP.

Subzone numbering (notebook appendix A reports 99.07% of rows sorted by subzone number in the same zone as the row
before): the share of those row pairs that are inside one subzone, and the same check over pairs of consecutive
subzone numbers.

usage: .venv/bin/python deck/ppt02/bukti/cek_zona_contoh.py   (writes cek_zona_contoh.json, about 30 s)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cek_zona_contoh.json")
GAS, EL, ADOPT = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_electric_heating_adoption"
NAMED = ["REG_273", "REG_270", "REG_045", "REG_290", "REG_263", "REG_249", "REG_242", "REG_259", "REG_275"]

tr = pd.read_csv(os.path.join(ROOT, "data/raw/train.csv"))
te = pd.read_csv(os.path.join(ROOT, "data/raw/test.csv"))
sub = pd.read_csv(os.path.join(ROOT, "submissions/BEST_CURRENT.csv"))
assert (sub["id"].values == te["id"].values).all()
for c in (GAS, EL, ADOPT):
    te[c] = sub[c].values
allr = pd.concat([tr, te], ignore_index=True)
M = allr["bld_dim_12"].astype(float).values
base = pd.DataFrame({"zone": allr["zone_region_id"].values, "gas": allr[GAS].values, "elec": allr[EL].values,
                     "area": allr["bld_dim_06"].values, "M": M, "gnw": allr["hvac_dist_02"].values * M,
                     "adw": allr[ADOPT].values * M, "clim": allr["env_climate_05"].values})
Z = base.groupby("zone").agg(gas=("gas", "sum"), elec=("elec", "sum"), area=("area", "sum"), M=("M", "sum"),
                             gnw=("gnw", "sum"), adw=("adw", "sum"), clim=("clim", "mean"))
Z["gas_share"] = Z.gnw / Z.M
Z["adopt"] = Z.adw / Z.M
Z["eui"] = Z.gas / Z.area
Z["inc"] = 100 * Z.gas * 0.8 / 3.0 / Z.elec
assert round(Z.gas.sum() / 1e9, 2) == 141.35 and round(Z.inc.median(), 2) == 94.85
assert round(Z.loc["REG_249", "gas_share"], 2) == 85.54 and round(Z.loc["REG_249", "adopt"], 2) == 3.75  # cell 50


def quadrant(eui, inc):
    xm, ym = np.median(eui), np.median(inc)
    return np.where(eui >= xm, np.where(inc >= ym, "Q1", "Q2"), np.where(inc >= ym, "Q3", "Q4"))


r = Z.clim.rank(pct=True).values                      # 0 = mildest, 1 = harshest (as cek_kebijakan.py)
cop = 3.5 - 1.0 * (r - r.min()) / (r.max() - r.min())
Z["inc_c"] = 100 * Z.gas * 0.8 / cop / Z.elec
Z["q0"], Z["q1"] = quadrant(Z.eui.values, Z.inc.values), quadrant(Z.eui.values, Z.inc_c.values)
rho = spearmanr(Z.inc, Z.inc_c).statistic
assert round(rho, 4) == 0.8147 and int((Z.q0 != Z.q1).sum()) == 68   # cek_kebijakan.json "cop_per_iklim"
assert dict(Z.q0.value_counts()) == {"Q1": 118, "Q4": 118, "Q2": 47, "Q3": 47}  # cell 49
Z["rank_inc0"] = Z.inc.rank(ascending=False, method="min").astype(int)
Z["rank_inc1"] = Z.inc_c.rank(ascending=False, method="min").astype(int)
Z["rank_add0"] = (Z.gas * 0.8 / 3.0).rank(ascending=False, method="min").astype(int)
Z["rank_add1"] = (Z.gas * 0.8 / cop).rank(ascending=False, method="min").astype(int)

R = {"median_kenaikan_dasar_pct": round(float(Z.inc.median()), 2),
     "median_kenaikan_cop_iklim_pct": round(float(Z.inc_c.median()), 2),
     "spearman_cop_iklim": round(float(rho), 4), "zona_pindah_kuadran": int((Z.q0 != Z.q1).sum()),
     "lima_teratas_kenaikan_dasar": list(Z.inc.sort_values(ascending=False).index[:5]),
     "lima_teratas_kenaikan_cop_iklim": list(Z.inc_c.sort_values(ascending=False).index[:5]),
     "zona_contoh": {z: {"kuadran_dasar": Z.loc[z, "q0"], "kuadran_cop_iklim": Z.loc[z, "q1"],
                         "kenaikan_dasar_pct": round(float(Z.loc[z, "inc"]), 2),
                         "kenaikan_cop_iklim_pct": round(float(Z.loc[z, "inc_c"]), 2),
                         "peringkat_kenaikan_dasar": int(Z.loc[z, "rank_inc0"]),
                         "peringkat_kenaikan_cop_iklim": int(Z.loc[z, "rank_inc1"]),
                         "peringkat_tambahan_absolut_dasar": int(Z.loc[z, "rank_add0"]),
                         "peringkat_tambahan_absolut_cop_iklim": int(Z.loc[z, "rank_add1"]),
                         "porsi_gas_jaringan_pct": round(float(Z.loc[z, "gas_share"]), 2),
                         "adopsi_pct": round(float(Z.loc[z, "adopt"]), 2)} for z in NAMED}}

# Q2 zones that stay Q2 when the COP varies with climate, largest gas use first
q2 = Z[(Z.q0 == "Q2") & (Z.q1 == "Q2")].sort_values("gas", ascending=False)
R["q2_tetap_q2"] = {"jumlah": int(len(q2)), "dari": int((Z.q0 == "Q2").sum()),
                    "zona": [{"zona": z, "gas_TWh": round(float(q2.loc[z, "gas"]) / 1e9, 3),
                              "porsi_gas_jaringan_pct": round(float(q2.loc[z, "gas_share"]), 2),
                              "adopsi_pct": round(float(q2.loc[z, "adopt"]), 2),
                              "kenaikan_dasar_pct": round(float(q2.loc[z, "inc"]), 2),
                              "kenaikan_cop_iklim_pct": round(float(q2.loc[z, "inc_c"]), 2)} for z in q2.index[:10]]}
R["q1_tetap_q1"] = {"jumlah": int(((Z.q0 == "Q1") & (Z.q1 == "Q1")).sum()), "dari": int((Z.q0 == "Q1").sum())}

# subzone numbering: rows sorted by subzone number (as report.split_checks), then pairs of consecutive subzones
rows = pd.concat([tr[["zone_region_id", "subzone_id"]], te[["zone_region_id", "subzone_id"]]], ignore_index=True)
num = rows.subzone_id.str[4:].astype(int).values
order = np.argsort(num, kind="stable")
zone, sz = rows.zone_region_id.values[order], num[order]
subs = rows.assign(n=num).drop_duplicates("subzone_id").sort_values("n")
assert subs.groupby("subzone_id").zone_region_id.nunique().max() == 1   # a subzone lies in one zone
R["urutan_subzona"] = {
    "baris_berurutan_zona_sama_pct": round(100 * float(np.mean(zone[1:] == zone[:-1])), 2),
    "baris_berurutan_subzona_sama_pct": round(100 * float(np.mean(sz[1:] == sz[:-1])), 2),
    "pasangan_subzona_berurutan_zona_sama_pct": round(
        100 * float(np.mean(subs.zone_region_id.values[1:] == subs.zone_region_id.values[:-1])), 2),
    "jumlah_subzona": int(len(subs))}
assert R["urutan_subzona"]["baris_berurutan_zona_sama_pct"] == 99.07   # cell 54

json.dump(R, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
