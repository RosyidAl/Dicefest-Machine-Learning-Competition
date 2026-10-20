"""Policy / interpretability analysis (same code the final notebook runs, via final/src/explain.py).

Outputs: reports/policy/*.csv (tables) and reports/figures/pol_*.png (figures).
usage: OMP_NUM_THREADS=4 python exp/analysis/policy_explain.py [--quick]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "final/src")
import explain as E  # noqa: E402
from common import TARGET_COLS, load  # noqa: E402
from train_lib import Data  # noqa: E402

QUICK = "--quick" in sys.argv
OUT, FIG = "reports/policy", "reports/figures"
os.makedirs(OUT, exist_ok=True)
GAS, EL, EUI, CLEAN, ADOPT = TARGET_COLS
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

tr, te = load()
d = Data()
ex, ext = d.agg_features()
X, Xt = E.feature_frame(d, ex, ext)
cols = list(X.columns)
summary = {}
models = {}
for t, (label, fn) in E.TARGETS.items():
    y = fn(tr)
    m, r2 = E.fit_explain(X, y, rounds=200 if QUICK else 800)
    idx, sv, base = E.shap_values(m, X, n=5000 if QUICK else 20000)
    models[t] = m
    fs = E.family_shares(sv, cols)
    tf = E.top_features(sv, cols)
    fs.to_csv(f"{OUT}/family_{t}.csv")
    tf.to_csv(f"{OUT}/top_{t}.csv", index=False)
    np.savez_compressed(f"{OUT}/shap_{t}.npz", idx=idx, sv=sv.astype(np.float32), base=base)
    summary[t] = dict(label=label, r2_holdout=round(r2, 4), families=fs.round(4).to_dict())
    print(t, label, "R2", round(r2, 4), fs.round(3).to_dict(), flush=True)

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

# ---- figure 1: family shares per target
fam_tab = pd.DataFrame({E.TARGETS[t][0]: pd.Series(summary[t]["families"]) for t in E.TARGETS}).fillna(0)
fam_tab = fam_tab.loc[fam_tab.sum(axis=1).sort_values().index]
ax = fam_tab.T.plot.barh(stacked=True, figsize=(8, 3.6), colormap="tab20", width=0.75)
ax.set_xlabel("porsi mean |SHAP|")
ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
plt.title("Apa yang mendorong tiap target? (model penjelas, fitur terukur saja)")
plt.tight_layout()
plt.savefig(f"{FIG}/pol_families.png", bbox_inches="tight")
plt.close()


def dep(t, feat, ax, xlabel, xlim=None, logx=False):
    z = np.load(f"{OUT}/shap_{t}.npz")
    j = cols.index(feat)
    x = X[feat].values[z["idx"]]
    s = z["sv"][:, j]
    ax.scatter(x, s, s=2, alpha=0.25, color="#3b6ea5", rasterized=True)
    q = pd.Series(s).groupby(pd.qcut(x, 20, duplicates="drop")).mean()
    mids = [iv.mid for iv in q.index]
    ax.plot(mids, q.values, color="#c0392b", lw=2)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"SHAP ({E.TARGETS[t][0]})")
    if logx:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)


# ---- figure 2: key dependence curves
fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
dep("eui", "area_per_meter", axs[0, 0], "luas lantai per meter (m2)", logx=True)
dep("epm", "area_per_meter", axs[0, 1], "luas lantai per meter (m2)", logx=True)
dep("adopt", "hvac_dist_02", axs[0, 2], "pangsa pemanas gas jaringan (%)")
dep("eui", "env_climate_05", axs[1, 0], "env_climate_05 (variabel iklim, anonim)")
dep("eui", "bld_era_15", axs[1, 1], "porsi bangunan periode bld_era_15 (%)")
dep("eui", "bld_era_09", axs[1, 2], "porsi bangunan periode bld_era_09 (%)")
plt.tight_layout()
plt.savefig(f"{FIG}/pol_dependence.png", bbox_inches="tight")
plt.close()

# ---- area-level indices (zones): observed targets for train rows, final predictions for test rows
sub = pd.read_csv("submissions/BEST_CURRENT.csv")
allr = pd.concat([tr.assign(_src="train"), te.assign(_src="test")], ignore_index=True)
for c in (GAS, EL, EUI, ADOPT):
    allr.loc[allr._src == "test", c] = sub[c].values
Xall = pd.concat([X, Xt], ignore_index=True)
fc = E.family_contrib(models["eui"], Xall, cols)
stock = fc[[c for c in ("era bangunan", "dimensi bangunan", "rasio fisik turunan") if c in fc]].sum(axis=1).values
A = allr["bld_dim_06"].values
Mtr = allr["bld_dim_12"].values
g = pd.DataFrame({"zone": allr["zone_region_id"].values, "gas": allr[GAS].values, "elec": allr[EL].values,
                  "area": A, "meters": Mtr, "adopt_w": allr[ADOPT].values * Mtr,
                  "gas_net_w": allr["hvac_dist_02"].values * Mtr,
                  # gas saved if the building-stock part of log EUI above the national mean were removed
                  "gas_saving": allr[GAS].values * (1 - np.exp(-np.clip(stock, 0, None)))})
Z = g.groupby("zone").sum()
Z["eui"] = Z.gas / Z.area
Z["adopt"] = Z.adopt_w / Z.meters
Z["gas_network_share"] = Z.gas_net_w / Z.meters
HEAT_SHARE, SCOP = 0.8, 3.0  # scenario assumptions: 80% of domestic gas is space/water heating; heat pump SCOP 3
Z["elec_added"] = Z.gas * HEAT_SHARE / SCOP
Z["load_increase_pct"] = 100 * Z.elec_added / Z.elec
Z["saving_pct"] = 100 * Z.gas_saving / Z.gas
Z = Z.drop(columns=["adopt_w", "gas_net_w"])
Z.round(4).to_csv(f"{OUT}/zones.csv")
top_retrofit = Z.sort_values("gas_saving", ascending=False).head(10)
top_grid = Z.sort_values("load_increase_pct", ascending=False).head(10)
top_retrofit.round(3).to_csv(f"{OUT}/top_retrofit.csv")
top_grid.round(3).to_csv(f"{OUT}/top_grid.csv")
tot = dict(gas_twh=Z.gas.sum() / 1e9, saving_twh=Z.gas_saving.sum() / 1e9,
           saving_pct=100 * Z.gas_saving.sum() / Z.gas.sum(), top10_retrofit_share=top_retrofit.gas_saving.sum() / Z.gas_saving.sum(),
           load_increase_median=float(Z.load_increase_pct.median()), load_increase_p90=float(Z.load_increase_pct.quantile(0.9)))
json.dump(tot, open(f"{OUT}/totals.json", "w"), indent=1)
print("totals", tot, flush=True)

# ---- figure 3: trilemma map of the zones
fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(Z.eui, Z.load_increase_pct, s=8 + 300 * Z.gas / Z.gas.max(), c=Z.adopt, cmap="viridis", alpha=0.75,
                edgecolor="k", lw=0.2)
ax.axvline(Z.eui.median(), color="grey", ls="--", lw=0.8)
ax.axhline(Z.load_increase_pct.median(), color="grey", ls="--", lw=0.8)
ax.set_xlabel("intensitas gas zona (kWh/m2)")
ax.set_ylabel("kenaikan beban listrik bila pemanas gas dielektrifikasi (%)")
plt.colorbar(sc, label="adopsi pemanas listrik saat ini (%)")
ax.set_title("Peta trilema per zona (ukuran titik = konsumsi gas total)")
xm, ym = Z.eui.median(), Z.load_increase_pct.median()
for (x, y, t) in ((0.98, 0.98, "Q1 retrofit dulu,\nlalu elektrifikasi"), (0.98, 0.02, "Q2 heat pump\nsekarang"),
                  (0.02, 0.98, "Q3 perkuat jaringan\nsebelum elektrifikasi"), (0.02, 0.02, "Q4 program standar")):
    ax.text(x, y, t, transform=ax.transAxes, ha="right" if x > 0.5 else "left", va="top" if y > 0.5 else "bottom",
            fontsize=8, color="#555", bbox=dict(fc="white", ec="none", alpha=0.7))
Z["quadrant"] = np.where(Z.eui >= xm, np.where(Z.load_increase_pct >= ym, "Q1", "Q2"),
                         np.where(Z.load_increase_pct >= ym, "Q3", "Q4"))
qt = Z.groupby("quadrant").agg(zona=("gas", "size"), gas_twh=("gas", lambda v: v.sum() / 1e9),
                               hemat_retrofit_twh=("gas_saving", lambda v: v.sum() / 1e9),
                               adopsi_rata=("adopt", "mean"), kenaikan_beban_rata=("load_increase_pct", "mean"))
qt["porsi_gas"] = qt.gas_twh / qt.gas_twh.sum()
qt.round(3).to_csv(f"{OUT}/quadrants.csv")
Z.round(4).to_csv(f"{OUT}/zones.csv")
print(qt.round(3).to_string(), flush=True)
plt.tight_layout()
plt.savefig(f"{FIG}/pol_trilemma.png", bbox_inches="tight")
plt.close()
print("done", flush=True)
