"""Phase 12 figures (300 dpi, reports/figures/) for the writeup and the presentation.

fig1_targets       distributions of the five targets
fig2_noise_floor   variance of the v16 blend's OOF log residual against meter count M, fit a + b/M
fig3_cv_lb         nested CV (+ holdout-estimated post-processing) against public LB per submission
fig4_blend_weights final NNLS weights per metric, grouped by model family
fig5_gains         where the score came from: blend vs best single model, residual correction on the holdout
Inputs: cache/train.parquet, exp/oof/*.npy (dev OOF arrays), final/work/v16_dev/summary.json,
reports/cv_lb_history.csv, reports/ablation_summary.csv. SHAP figures come from src/interpret.py.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import TARGET_COLS, load  # noqa: E402

FIG = "reports/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
tr, _ = load()
A, M = tr["bld_dim_06"].values, tr["bld_dim_12"].values.astype(float)
summ = json.load(open("final/work/v16_dev/summary.json"))
W = summ["weights"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


# 1. target distributions
fig, ax = plt.subplots(1, 5, figsize=(15, 2.8))
spec = [("t1 gas makro (log10 kWh)", np.log10(tr[TARGET_COLS[0]])), ("t2 listrik makro (log10 kWh)", np.log10(tr[TARGET_COLS[1]])),
        ("t3 EUI gas (kWh/m2)", tr[TARGET_COLS[2]].clip(upper=tr[TARGET_COLS[2]].quantile(0.995))),
        ("t4 rasio bangunan bersih (%)", tr[TARGET_COLS[3]]), ("t5 adopsi pemanas listrik (%)", tr[TARGET_COLS[4]])]
for a, (t, v) in zip(ax, spec):
    a.hist(v, bins=80, color="#3b6ea5")
    a.set_title(t)
    a.set_yticks([])
save(fig, "fig1_targets")

# 2. label-noise floor: per-M-bin variance of the OOF log residual of the v16 blend
pred = {m: sum(w * np.load(f"exp/oof/{n}.npy") for n, w in W[m].items()) for m in ("gas", "elec")}
res = {"gas": np.log(tr[TARGET_COLS[0]].values / (pred["gas"] * A)), "elec": np.log(tr[TARGET_COLS[1]].values / pred["elec"])}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
rows = []
for a, (k, r) in zip(ax, res.items()):
    q = np.unique(np.quantile(M, np.linspace(0, 1, 41)))
    b = np.clip(np.searchsorted(q, M, side="right") - 1, 0, len(q) - 2)
    mm = np.array([M[b == i].mean() for i in range(len(q) - 1)])
    vv = np.array([r[b == i].var() for i in range(len(q) - 1)])
    nn = np.array([(b == i).sum() for i in range(len(q) - 1)])
    X = np.c_[np.ones_like(mm), 1 / mm]
    coef = np.linalg.lstsq(X * np.sqrt(nn)[:, None], vv * np.sqrt(nn), rcond=None)[0]
    noise_share = 1 - coef[0] * len(r) / ((X @ coef) * nn).sum()
    rows.append({"target": k, "a": coef[0], "b": coef[1], "noise_share": noise_share})
    g = np.geomspace(mm.min(), mm.max(), 200)
    a.scatter(mm, vv, s=14, color="#3b6ea5", label="varians residual per kelompok M")
    a.plot(g, coef[0] + coef[1] / g, color="#c0392b", label=f"a + b/M  (a={coef[0]:.4f}, b={coef[1]:.3f})")
    a.axhline(coef[0], color="gray", ls="--", lw=1, label="a: bagian yang masih bisa dijelaskan model")
    a.set_xscale("log")
    a.set_xticks([4, 6, 10, 20, 40, 60])
    a.set_xticklabels(["4", "6", "10", "20", "40", "60"])
    a.minorticks_off()
    a.set_xlabel("jumlah meter M (skala log)")
    a.set_ylabel("varians residual log")
    a.set_title(f"{k}: {noise_share:.0%} varians residual adalah noise\nrata-rata konsumsi rumah tangga")
    a.legend(fontsize=7, frameon=False)
pd.DataFrame(rows).to_csv("reports/noise_floor.csv", index=False)
save(fig, "fig2_noise_floor")

# 3. CV vs public LB
h = pd.read_csv("reports/cv_lb_history.csv")
h["cv_est"] = h.cv_blend_nested + h.post_est
off = (h.public_lb - h.cv_est)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(len(h))
ax[0].plot(x, h.cv_blend_nested, "o-", color="#7f8c8d", label="CV nested (blend saja)")
ax[0].plot(x, h.cv_est, "o-", color="#3b6ea5", label="CV + estimasi holdout post-processing")
ax[0].plot(x, h.public_lb, "s-", color="#c0392b", label="public LB")
ax[0].set_xticks(x)
ax[0].set_xticklabels(h.version, rotation=40, ha="right")
ax[0].set_ylabel("NRMSE kolom (lebih kecil lebih baik)")
ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title("Perkembangan skor per submisi")
ax[1].scatter(h.cv_est, h.public_lb, color="#3b6ea5")
for _, rr in h.iterrows():
    ax[1].annotate(rr.version, (rr.cv_est, rr.public_lb), fontsize=6, xytext=(3, 2), textcoords="offset points")
lo, hi = h.cv_est.min() - 0.0002, h.cv_est.max() + 0.0002
ax[1].plot([lo, hi], [lo + off.mean(), hi + off.mean()], color="#c0392b", lw=1,
           label=f"LB = CV + {off.mean():.5f} (sd {off.std():.5f})")
ax[1].set_xlabel("CV + estimasi post-processing")
ax[1].set_ylabel("public LB")
ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("CV memprediksi LB dengan selisih tetap")
save(fig, "fig3_cv_lb")


# 4. final blend weights by model family
def fam(n):
    p = n.split("_")[0]
    if p.startswith("mk") or p.startswith("m1"):
        return "MLP multi-task"
    if p.startswith(("pk", "p1", "p2", "p3")):
        return "MLP + embedding PLR"
    if p.startswith("t1"):
        return "TabM-mini"
    if p.startswith(("n4", "n6")):
        return "MLP"
    if p.startswith(("x", "xgb")):
        return "XGBoost"
    return "LightGBM"


fams = ["XGBoost", "LightGBM", "MLP", "MLP + embedding PLR", "TabM-mini", "MLP multi-task"]
cols = ["#2c3e50", "#16a085", "#f39c12", "#8e44ad", "#d35400", "#c0392b"]
share = pd.DataFrame({m: pd.Series({f: sum(w for n, w in W[m].items() if fam(n) == f) for f in fams}) for m in ("gas", "eui", "elec")}).T
share = share.div(share.sum(1), axis=0)
fig, ax = plt.subplots(figsize=(9, 2.2))
left = np.zeros(3)
for f, c in zip(fams, cols):
    ax.barh(["t1 gas", "t3 EUI", "t2 listrik"], share[f].values, left=left, color=c, label=f)
    left += share[f].values
ax.set_xlim(0, 1)
ax.set_xlabel("porsi bobot NNLS")
n_mem = {m: sum(1 for w in W[m].values() if w > 0) for m in W}
ax.set_title(f"Komposisi blend final (anggota berbobot: gas {n_mem['gas']}, EUI {n_mem['eui']}, listrik {n_mem['elec']})")
ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "fig4_blend_weights")
share.round(4).to_csv("reports/blend_family_share.csv")

# 5. gains: blend over best single member and post-processing on the holdout
ab = pd.read_csv("reports/ablation_summary.csv")
fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
s = ab[ab.level == "single"]
lab = [f"{c} [{m}]" for c, m in zip(s.change, s.metric)]
delta = np.where(s.before == 0, s.after, s.after - s.before)
ax[0].barh(lab[::-1], delta[::-1], color="#3b6ea5")
ax[0].set_xlabel("perubahan NRMSE model tunggal (negatif = lebih baik)")
ax[0].set_title("Ablation model tunggal")
ax[0].locator_params(axis="x", nbins=5)
p = ab[ab.level.isin(["blend", "post"])]
lab = [f"{c} [{m}]" for c, m in zip(p.change, p.metric)]
ax[1].barh(lab[::-1], (p.after - p.before).values[::-1], color=["#16a085" if lv == "blend" else "#c0392b" for lv in p.level[::-1]])
ax[1].set_xlabel("perubahan NRMSE (negatif = lebih baik)")
ax[1].set_title("Blend dan koreksi residual (holdout 70/30)")
save(fig, "fig5_gains")
print("done")
