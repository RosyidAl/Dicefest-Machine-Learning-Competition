"""Rebuild the writeup's trilemma map (Figure 7) without the explainer model.

The axes and colour of the map use only observed train targets, the final submission for test rows, floor area,
meter count and adoption, so this reproduces final/src/explain.py zone_indices/plot_trilemma exactly, with two
presentation changes: a decile colour scale for the skewed adoption values and an axis label that states annual
electricity consumption (not peak load). Checks the zone aggregates against the notebook before writing.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm

sys.path.insert(0, "final/src")
from common import TARGET_COLS  # noqa: E402
from explain import HEAT_SHARE, SCOP  # noqa: E402

GAS, EL, EUI, CLEAN, ADOPT = TARGET_COLS
COLS = ["zone_region_id", "bld_dim_06", "bld_dim_12", GAS, EL, ADOPT]

tr = pd.read_csv("data/raw/train.csv", usecols=["id"] + COLS)
te = pd.read_csv("data/raw/test.csv", usecols=["id", "zone_region_id", "bld_dim_06", "bld_dim_12"])
sub = pd.read_csv("submissions/BEST_CURRENT.csv")
assert (sub["id"].values == te["id"].values).all()
for c in (GAS, EL, ADOPT):
    te[c] = sub[c].values
allr = pd.concat([tr, te], ignore_index=True)
M = allr["bld_dim_12"].values
g = pd.DataFrame({"zone": allr["zone_region_id"].values, "gas": allr[GAS].values, "elec": allr[EL].values,
                  "area": allr["bld_dim_06"].values, "meters": M, "adopt_w": allr[ADOPT].values * M})
Z = g.groupby("zone").sum()
Z["eui"] = Z.gas / Z.area
Z["adopt"] = Z.adopt_w / Z.meters
Z["load_increase_pct"] = 100 * Z.gas * HEAT_SHARE / SCOP / Z.elec
xm, ym = Z.eui.median(), Z.load_increase_pct.median()
Z["kuadran"] = np.where(Z.eui >= xm, np.where(Z.load_increase_pct >= ym, "Q1", "Q2"),
                        np.where(Z.load_increase_pct >= ym, "Q3", "Q4"))

counts = Z.kuadran.value_counts().sort_index().to_dict()
check = {"counts": counts, "gas_TWh": round(Z.gas.sum() / 1e9, 2), "mean": round(Z.load_increase_pct.mean(), 1),
         "min": round(Z.load_increase_pct.min(), 1), "max": round(Z.load_increase_pct.max(), 1)}
print(check)
assert counts == {"Q1": 118, "Q2": 47, "Q3": 47, "Q4": 118} and check["gas_TWh"] == 141.35, check

bounds = np.unique(np.quantile(Z.adopt, np.linspace(0, 1, 11)))
norm = BoundaryNorm(bounds, plt.get_cmap("viridis").N)
fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(Z.eui, Z.load_increase_pct, s=8 + 300 * Z.gas / Z.gas.max(), c=Z.adopt, cmap="viridis", norm=norm,
                alpha=0.8, edgecolor="k", lw=0.2)
ax.axvline(xm, color="grey", ls="--", lw=0.8)
ax.axhline(ym, color="grey", ls="--", lw=0.8)
for (x, y, t) in ((0.98, 0.98, "Q1 retrofit dulu,\nlalu elektrifikasi"), (0.98, 0.02, "Q2 kandidat\nheat pump"),
                  (0.02, 0.98, "Q3 perkuat jaringan\nsebelum elektrifikasi"), (0.02, 0.02, "Q4 program standar")):
    ax.text(x, y, t, transform=ax.transAxes, ha="right" if x > 0.5 else "left", va="top" if y > 0.5 else "bottom",
            fontsize=8, color="#444", bbox=dict(fc="white", ec="none", alpha=0.7))
ax.set_xlabel("intensitas gas zona (kWh/m2)")
ax.set_ylabel("kenaikan konsumsi listrik tahunan bila\npemanas gas dielektrifikasi (%)")
cb = plt.colorbar(sc, label="adopsi pemanas listrik saat ini (%, skala desil)")
cb.set_ticks(bounds)
cb.set_ticklabels([f"{b:.1f}" for b in bounds])
ax.set_title("Peta trilema per zona (ukuran titik = konsumsi gas total)")
plt.tight_layout()
plt.savefig("reports/figures/pol_trilemma.png", dpi=150, bbox_inches="tight")
print("wrote reports/figures/pol_trilemma.png")
