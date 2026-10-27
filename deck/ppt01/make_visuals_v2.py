"""Charts for the final deck v2 (deck/ppt01). Every number comes from the executed notebook
final/deliverables/DICEFEST2026_BISMILLAH.ipynb (cell numbers in the comments), from data that reproduces a notebook
output exactly (checked by assertions below), or from the Kaggle private leaderboard (one value, marked).

- s08 / s12 dependence plots: SHAP values of the notebook's explanation models (reports/policy/shap_*.npz, the same
  deterministic run; mean |SHAP| is checked against the cell 45 output) on the notebook's own feature frame.
- s09 trilemma map: zone aggregates of observed train targets and the final submission (sha aec745ab), recomputed with
  the notebook formulas (explain.zone_indices); quadrant counts, gas totals and mean increases are checked against the
  cell 49 output.

usage: .venv/bin/python deck/ppt01/make_visuals_v2.py   (writes deck/ppt01/img/*.png)
"""
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = "/root/Dicefest"
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
NB_SRC = os.path.join(ROOT, "final/nbrun_final/dicefest_nb/src")
os.environ.setdefault("DICEFEST_DATA_DIR", os.path.join(ROOT, "data/raw"))
sys.path.insert(0, NB_SRC)
os.makedirs(IMG, exist_ok=True)

for f in sorted(os.listdir(os.path.join(HERE, "fonts"))):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))
NAVY, TEAL, AMBER, INK, MUTED = "#16324F", "#1F7A8C", "#D97B29", "#1F2933", "#5F6B7A"
LINE, SOFT, GREY = "#D5DBE3", "#EEF2F6", "#A7B1BD"
QCOL = {"Q1": AMBER, "Q2": TEAL, "Q3": NAVY, "Q4": GREY}
plt.rcParams.update({"font.family": "Carlito", "font.size": 14, "axes.edgecolor": LINE, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 16, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 100,
                     "axes.formatter.use_locale": False})
FACTS = {}  # derived numbers written to img/angka_visual.json for the fact sheet


def idn(x, nd=0, plus=False):
    """Indonesian number format: decimal comma, thousands dot"""
    s = f"{x:+,.{nd}f}" if plus else f"{x:,.{nd}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=250, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def comma_axis(ax, axis="y", nd=0, suffix=""):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: idn(v, nd) + suffix)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---- slide 4: top five mean |SHAP| (cell 45, In [32]) ------------------------------------------------------------
def s04_shap_top():
    panels = [("log intensitas gas (EUI)", [("area_per_meter", 0.0895), ("env_climate_05", 0.0321),
                                            ("socio_demog_082", 0.0276), ("area_per_epc", 0.0260),
                                            ("s002_ratio", 0.0250)]),
              ("log listrik per meter", [("area_per_meter", 0.0895), ("meters_per_epc", 0.0294),
                                         ("socio_demog_005", 0.0222), ("socio_demog_038", 0.0196),
                                         ("socio_demog_061", 0.0163)])]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for ax, (title, rows) in zip(axs, panels):
        names, vals = [r[0] for r in rows][::-1], [r[1] for r in rows][::-1]
        cols = [AMBER if n == "area_per_meter" else GREY for n in names]
        ax.barh(names, vals, height=0.62, color=cols)
        for y, v in enumerate(vals):
            ax.text(v + 0.002, y, idn(v, 4), va="center", fontsize=13, color=INK)
        ax.set_title(title, fontsize=15, pad=8)
        ax.set_xlim(0, 0.125)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=13, colors=INK)
        FACTS[f"rasio_fitur_kedua_{title}"] = round(rows[0][1] / rows[1][1], 2)
    fig.text(0.01, 0.01, "rata-rata |SHAP| pada sampel 20.000 baris latih; target dalam skala log", fontsize=12,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=2.0)
    save(fig, "s04_shap_top.png")


# ---- slide 7: public LB per step (cell 37, In [28]) and the private score (Kaggle) ---------------------------------
def s07_lb():
    steps = [("v12", "agregat\n+ PLR", 0.25025), ("v15", "koreksi kNN\ndan lainnya", 0.24954),
             ("v16", "kawasan\ntermirip", 0.24943), ("v19", "objektif\nPoisson", 0.24934),
             ("v19vc", "bobot\nadaptif", 0.24925), ("v19nca", "model\nNCA", 0.24920),
             ("final", "penyesuaian\nakhir", 0.24913)]
    private = 0.24844  # Kaggle private leaderboard, 24 Oct 2026 (reports/STATUS.md), rank 8
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(steps))
    y = [s[2] for s in steps]
    ax.plot(x, y, color=NAVY, lw=2.2, marker="o", ms=7, zorder=3)
    for xi, yi in zip(x, y):
        if xi == 0:  # first point: label above right, clear of the steep first segment
            ax.text(xi + 0.12, yi + 0.00005, idn(yi, 5), ha="left", va="bottom", fontsize=12, color=INK)
        else:
            ax.text(xi + 0.06, yi - 0.00008, idn(yi, 5), ha="center", va="top", fontsize=12, color=INK)
    ax.scatter([x[-1]], [private], s=150, color=AMBER, zorder=4)
    ax.annotate("private " + idn(private, 5) + "\nperingkat 8", (x[-1], private), xytext=(x[-1] - 1.35, private + 0.00002),
                fontsize=13, color=AMBER, fontweight="bold", va="center", ha="center",
                arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in steps], fontsize=12, color=INK)
    ax.set_ylim(0.2480, 0.2508)
    comma_axis(ax, "y", 4)
    ax.set_ylabel("skor (makin kecil makin baik)", fontsize=13, color=MUTED)
    ax.yaxis.grid(True, color=SOFT, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save(fig, "s07_lb.png")


# ---- slides 8 and 12: SHAP dependence of the explanation models (cells 45 and 47) -----------------------------------
def explain_frame():
    from train_lib import Data
    import explain
    d = Data()
    ex, ext = d.agg_features()
    X, _ = explain.feature_frame(d, ex, ext)
    return X


def load_shap(t, X, expected):
    z = np.load(os.path.join(ROOT, f"reports/policy/shap_{t}.npz"))
    idx, sv = z["idx"], z["sv"]
    assert sv.shape[1] == X.shape[1], (sv.shape, X.shape)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    for feat, val in expected.items():  # cell 45 output, rounded to 4 decimals
        assert round(float(imp[feat]), 4) == val, (t, feat, imp[feat], val)
    return idx, sv


def binned(x, s):
    q = pd.Series(s).groupby(pd.qcut(x, 20, duplicates="drop"), observed=True).mean()
    return np.array([iv.mid for iv in q.index]), q.values


def dep_panel(ax, x, s, color, xlabel, ylabel, logx):
    ax.scatter(x, s, s=3, alpha=0.18, color=color, rasterized=True, lw=0)
    bx, by = binned(x, s)
    ax.plot(bx, by, color=INK, lw=2.4)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if logx:
        ax.set_xscale("log")
    return bx, by


def s08_s12_dependence():
    X = explain_frame()
    cols = list(X.columns)
    res = {"eui": load_shap("eui", X, {"area_per_meter": 0.0895, "env_climate_05": 0.0321}),
           "epm": load_shap("epm", X, {"area_per_meter": 0.0895, "meters_per_epc": 0.0294}),
           "adopt": load_shap("adopt", X, {"hvac_rest": 3.5653, "hvac_dist_02": 0.9053})}
    apm = X["area_per_meter"].values
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.9))
    for ax, t, color, ylabel, title in ((axs[0], "eui", AMBER, "SHAP (log EUI)", "Intensitas gas per m²"),
                                        (axs[1], "epm", TEAL, "SHAP (log listrik per meter)",
                                         "Listrik per rumah")):
        idx, sv = res[t]
        bx, by = dep_panel(ax, apm[idx], sv[:, cols.index("area_per_meter")], color,
                           "luas lantai per meter (m², skala log)", ylabel, True)
        ax.set_title(title, fontsize=15, pad=6)
        ax.set_xlim(12, 600)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([20, 50, 100, 200, 500]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["20", "50", "100", "200", "500"]))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.55, 0.65)
        comma_axis(ax, "y", 1)
        FACTS[f"garis_rata2_{t}_ujung"] = [round(float(by[0]), 3), round(float(by[-1]), 3),
                                          round(float(bx[0]), 1), round(float(bx[-1]), 1)]
    fig.tight_layout(w_pad=2.5)
    save(fig, "s08_dep_size.png")

    idx, sv = res["adopt"]
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    bx, by = dep_panel(ax, X["hvac_dist_02"].values[idx], sv[:, cols.index("hvac_dist_02")], TEAL,
                       "porsi pemanas gas jaringan di kawasan (%)", "SHAP adopsi (poin persentase)", False)
    ax.axvspan(80, 100, color=AMBER, alpha=0.10, lw=0)
    ax.text(90, 9.3, "di atas 80%:\nsumbangan negatif", ha="center", fontsize=13, color=AMBER, fontweight="bold")
    ax.text(22, 9.3, "di bawah 40%:\n+4 sampai +12 poin", ha="center", fontsize=13, color=TEAL, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.5, 14.5)
    ax.set_yticks([-2, 0, 2, 4, 6, 8, 10, 12, 14])
    comma_axis(ax, "y", 0)
    fig.tight_layout()
    save(fig, "s12_dep_adopt.png")
    FACTS["garis_rata2_adopt"] = [[round(float(a), 1), round(float(b), 2)] for a, b in zip(bx, by)]


# ---- slide 9: trilemma map (cell 49) ------------------------------------------------------------------------------
def s09_trilemma():
    from common import TARGET_COLS
    from explain import HEAT_SHARE, SCOP
    gas_c, el_c, _, _, ad_c = TARGET_COLS
    sub_path = os.path.join(ROOT, "submissions/BEST_CURRENT.csv")
    assert hashlib.sha256(open(sub_path, "rb").read()).hexdigest().startswith("aec745ab")
    cols = ["zone_region_id", "bld_dim_06", "bld_dim_12", gas_c, el_c, ad_c]
    tr = pd.read_csv(os.path.join(ROOT, "data/raw/train.csv"), usecols=["id"] + cols)
    te = pd.read_csv(os.path.join(ROOT, "data/raw/test.csv"), usecols=["id", "zone_region_id", "bld_dim_06",
                                                                       "bld_dim_12"])
    sub = pd.read_csv(sub_path)
    assert (sub["id"].values == te["id"].values).all()
    for c in (gas_c, el_c, ad_c):
        te[c] = sub[c].values
    allr = pd.concat([tr, te], ignore_index=True)
    g = pd.DataFrame({"zone": allr["zone_region_id"].values, "gas": allr[gas_c].values, "elec": allr[el_c].values,
                      "area": allr["bld_dim_06"].values})
    Z = g.groupby("zone").sum()
    Z["eui"] = Z.gas / Z.area
    Z["inc"] = 100 * Z.gas * HEAT_SHARE / SCOP / Z.elec
    xm, ym = Z.eui.median(), Z.inc.median()
    Z["q"] = np.where(Z.eui >= xm, np.where(Z.inc >= ym, "Q1", "Q2"), np.where(Z.inc >= ym, "Q3", "Q4"))
    # cell 49 / 50 outputs
    qt = Z.groupby("q").agg(n=("gas", "size"), gas=("gas", lambda v: v.sum() / 1e9), inc=("inc", "mean"))
    expect = {"Q1": (118, 66.095, 102.494), "Q2": (47, 16.299, 90.245), "Q3": (47, 20.607, 99.102),
              "Q4": (118, 38.354, 87.042)}
    for q, (n, gas, inc) in expect.items():
        assert qt.loc[q, "n"] == n and round(qt.loc[q, "gas"], 3) == gas and round(qt.loc[q, "inc"], 3) == inc, qt
    assert round(Z.gas.sum() / 1e9, 2) == 141.35 and round(ym, 2) == 94.85
    FACTS["trilema"] = {"median_pct": round(float(ym), 4), "mean_pct": round(float(Z.inc.mean()), 4),
                        "min_pct": round(float(Z.inc.min()), 2), "max_pct": round(float(Z.inc.max()), 2),
                        "eui_median": round(float(xm), 2)}
    # sensitivity grid (cell 51 text: COP 2.5-3.5 x heating share 70-90 %; ranks unchanged because the factor is common)
    grid = {f"{int(s * 100)}%_COP{c}": round(float(ym * (s / HEAT_SHARE) * (SCOP / c)), 1)
            for s in (0.7, 0.8, 0.9) for c in (2.5, 3.0, 3.5)}
    FACTS["sensitivitas_median"] = grid
    FACTS["sensitivitas_rata2"] = {k: round(float(Z.inc.mean() * v / ym), 1) for k, v in grid.items()}
    r1 = Z.inc.rank()
    r2 = (Z.gas * 0.7 / 2.5 / Z.elec).rank()
    FACTS["peringkat_tetap"] = bool((r1 == r2).all())

    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    for q in ("Q4", "Q3", "Q2", "Q1"):
        m = Z.q == q
        ax.scatter(Z.eui[m], Z.inc[m], s=10 + 380 * Z.gas[m] / Z.gas.max(), color=QCOL[q], alpha=0.78,
                   edgecolor="white", lw=0.6, zorder=3)
    ax.axvline(xm, color=MUTED, ls="--", lw=1)
    ax.axhline(ym, color=MUTED, ls="--", lw=1)
    lab = {"Q1": ("Q1 · retrofit dulu", 0.985, 0.975, "right", "top"),
           "Q2": ("Q2 · kandidat pompa panas", 0.985, 0.03, "right", "bottom"),
           "Q3": ("Q3 · siapkan jaringan dulu", 0.015, 0.975, "left", "top"),
           "Q4": ("Q4 · program standar", 0.015, 0.03, "left", "bottom")}
    for q, (t, x, y, ha, va) in lab.items():
        ax.text(x, y, t, transform=ax.transAxes, ha=ha, va=va, fontsize=14, color=QCOL[q] if q != "Q4" else MUTED,
                fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
    for z, dx, dy in (("REG_290", -62, -4), ("REG_270", -8, 16)):
        ax.annotate(z, (Z.eui[z], Z.inc[z]), xytext=(dx, dy), textcoords="offset points", fontsize=11, color=INK,
                    zorder=6, arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax.set_xlabel("intensitas gas zona (kWh per m²)", fontsize=13)
    ax.set_ylabel("kenaikan konsumsi listrik tahunan (%)", fontsize=13)
    ax.text(xm, 64.2, " median", fontsize=11, color=MUTED, va="bottom")
    ax.set_ylim(64, 131)
    comma_axis(ax, "y", 0)
    ax.yaxis.grid(False)
    fig.text(0.01, 0.005, "ukuran titik = konsumsi gas zona", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "s09_trilema.png")


# ---- slides 10 and 11: top zones (two tables of cell 50, In [35]) ---------------------------------------------------
def s10_s11_zones():
    # zone, gas (kWh), saving_pct, quadrant: first table of cell 50 (sorted by absolute saving)
    sav = [("REG_273", 2.311622e9, 7.36, "Q1"), ("REG_270", 1.627015e9, 8.60, "Q1"), ("REG_045", 1.655768e9, 6.69, "Q1"),
           ("REG_242", 1.378593e9, 7.35, "Q2"), ("REG_263", 2.838367e9, 3.43, "Q1")]
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    names = [f"{z}  ({q})" for z, _, _, q in sav][::-1]
    gwh = [g * p / 100 / 1e6 for _, g, p, _ in sav][::-1]
    pct = [p for _, _, p, _ in sav][::-1]
    cols = [QCOL[q] for *_, q in sav][::-1]
    ax.barh(names, gwh, height=0.62, color=cols)
    for yv, (v, p) in enumerate(zip(gwh, pct)):
        ax.text(v + 3, yv, f"≈{idn(v)} GWh  ·  {idn(p, 2)}%", va="center", fontsize=13, color=INK)
    ax.set_xlim(0, 265)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=13, colors=INK)
    ax.set_title("Potensi hemat retrofit terbesar (GWh · % gas zona)", fontsize=15, pad=8)
    fig.tight_layout()
    save(fig, "s10_retrofit.png")
    FACTS["hemat_gwh"] = {z: round(g * p / 100 / 1e6, 1) for z, g, p, _ in sav}

    inc = [("REG_290", 121.41, "Q1"), ("REG_259", 115.10, "Q1"), ("REG_275", 114.29, "Q1"), ("REG_260", 113.91, "Q1"),
           ("REG_270", 113.46, "Q1")]  # second table of cell 50
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    names = [f"{z}  ({q})" for z, _, q in inc][::-1]
    vals = [v for _, v, _ in inc][::-1]
    ax.barh(names, vals, height=0.62, color=AMBER)
    for yv, v in enumerate(vals):
        ax.text(v + 1.5, yv, idn(v, 2, plus=True) + "%", va="center", fontsize=13, color=INK)
    ax.axvline(94.85, color=NAVY, ls="--", lw=1.4)
    ax.text(93.0, -0.78, "median 330 zona " + idn(94.85, 2, plus=True) + "%", ha="right", va="center",
            fontsize=12, color=NAVY)
    ax.set_ylim(-1.15, 4.6)
    ax.set_xlim(0, 150)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=13, colors=INK)
    ax.set_title("Kenaikan konsumsi listrik tahunan tertinggi", fontsize=15, pad=8)
    fig.tight_layout()
    save(fig, "s11_kenaikan.png")


# ---- backup C1: two-knot hat basis of blend.size_basis --------------------------------------------------------------
def c1_basis():
    t = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.plot(t, 1 - t, color=NAVY, lw=2.6, label="bobot ujung rendah = 1 − t")
    ax.plot(t, t, color=AMBER, lw=2.6, label="bobot ujung tinggi = t")
    ax.plot([0.8, 0.8], [0.2, 0.8], ls=":", color=MUTED, lw=1.2)
    ax.plot([0.8, 0.8], [0.2, 0.8], "o", color=INK, ms=6)
    ax.text(0.83, 0.5, "peringkat 0,8:\n20% rendah, 80% tinggi", fontsize=12, color=INK, va="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.18), ncol=2, frameon=False, fontsize=12)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    comma_axis(ax, "x", 1)
    comma_axis(ax, "y", 1)
    ax.set_xlabel("peringkat kawasan di data latih (0 = terkecil, 1 = terbesar)", fontsize=12)
    ax.set_ylabel("porsi bobot", fontsize=12)
    ax.yaxis.grid(True, color=SOFT)
    fig.tight_layout()
    save(fig, "c1_basis.png")


if __name__ == "__main__":
    s04_shap_top()
    s07_lb()
    s09_trilemma()
    s10_s11_zones()
    c1_basis()
    s08_s12_dependence()
    json.dump(FACTS, open(os.path.join(IMG, "angka_visual.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(FACTS, indent=1, ensure_ascii=False))
