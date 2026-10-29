"""Images and chart data for the final deck v4 (deck/ppt03/BISMILLAH_Final_v4.pptx).

Bar and line charts in the deck are native PowerPoint charts; their numbers are written to img/data_grafik.json and
read by build_deck_v4.js. Images are used only where a native chart is impractical: the 330-zone quadrant map (bubble
scatter with median lines and quadrant labels) and the SHAP dependence plots (20,000-point scatter with a binned mean).
Text inside the images is at least 14 pt at the size the image is placed on its slide.

Every number is checked against the executed notebook final/deliverables/DICEFEST2026_BISMILLAH.ipynb (cell numbers
counted from 0) or against the evidence files:
  - zone aggregates: observed train targets plus the final submission for test rows (sha aec745ab...d66e), as
    explain.zone_indices; asserted against cells 49 and 50 and against ../ppt02/bukti/cek_zona_contoh.json (climate
    COP: Spearman 0.8147, 68 zones change quadrant);
  - SHAP: reports/policy/shap_*.npz on the notebook's own feature frame; mean |SHAP| asserted against cell 45, the size
    contrast against ../ppt01/bukti/cek_kebijakan.json, the share of the derived physical ratios against writeup B
    Table 3;
  - public leaderboard per development step: parsed from the output of cell 37; the private score from
    reports/STATUS.md (Kaggle private leaderboard, 24 Oct 2026);
  - adoption by gas-network share band: train rows weighted by meters, asserted against cek_kebijakan.json;
  - feature-family shares: writeup B Table 3, parsed from reports/WRITEUP_DRAFT_vB.md.

usage: .venv/bin/python deck/ppt03/make_visuals_v4.py   (about 2 minutes; writes deck/ppt03/img/)
"""
import hashlib
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager, patches  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG = os.path.join(HERE, "img")
FONTS = os.path.join(ROOT, "deck/ppt01/fonts")
NB = os.path.join(ROOT, "final/deliverables/DICEFEST2026_BISMILLAH.ipynb")
NB_SRC = os.path.join(ROOT, "final/nbrun_final/dicefest_nb/src")
ZONE_EVIDENCE = os.path.join(ROOT, "deck/ppt02/bukti/cek_zona_contoh.json")
os.environ.setdefault("DICEFEST_DATA_DIR", os.path.join(ROOT, "data/raw"))
sys.path.insert(0, NB_SRC)
os.makedirs(IMG, exist_ok=True)

for f in sorted(os.listdir(FONTS)):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, f))
NAVY, TEAL, AMBER, INK, MUTED = "#16324F", "#1F7A8C", "#D97B29", "#1F2933", "#5F6B7A"
LINE, SOFT, GREY = "#D5DBE3", "#EEF2F6", "#A7B1BD"
QCOL = {"Q1": AMBER, "Q2": TEAL, "Q3": NAVY, "Q4": GREY}
plt.rcParams.update({"font.family": "Carlito", "font.size": 14, "axes.edgecolor": LINE, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "axes.labelsize": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.titlelocation": "left", "figure.dpi": 100, "axes.formatter.use_locale": False})
GAS, EL, ADOPT = "target_macro_gas_kwh", "target_macro_elec_kwh", "target_electric_heating_adoption"
DATA = {}
SHAP_TICKS = [-0.4, -0.2, 0.0, 0.2, 0.4]   # fixed: automatic ticks fall at multiples of 0.25


def idn(x, nd=0, plus=False):
    """Indonesian number format: decimal comma, thousands dot"""
    s = f"{x:+,.{nd}f}" if plus else f"{x:,.{nd}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=250, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def comma_axis(ax, axis="y", nd=0):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: idn(v, nd))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def log_size_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(12, 600)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([20, 50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["20", "50", "100", "200", "500"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# ---- zone aggregates (cells 48 to 50; ../ppt02/bukti/cek_zona_contoh.py) -------------------------------------------
def zones():
    sub_path = os.path.join(ROOT, "submissions/BEST_CURRENT.csv")
    assert hashlib.sha256(open(sub_path, "rb").read()).hexdigest().startswith("aec745ab")
    cols = ["id", "zone_region_id", "bld_dim_06", "env_climate_05"]
    tr = pd.read_csv(os.path.join(ROOT, "data/raw/train.csv"), usecols=cols + [GAS, EL])
    te = pd.read_csv(os.path.join(ROOT, "data/raw/test.csv"), usecols=cols)
    sub = pd.read_csv(sub_path)
    assert (sub["id"].values == te["id"].values).all()
    for c in (GAS, EL):
        te[c] = sub[c].values
    allr = pd.concat([tr, te], ignore_index=True)
    Z = pd.DataFrame({"zone": allr["zone_region_id"].values, "gas": allr[GAS].values, "elec": allr[EL].values,
                      "area": allr["bld_dim_06"].values, "clim": allr["env_climate_05"].values}).groupby("zone").agg(
        gas=("gas", "sum"), elec=("elec", "sum"), area=("area", "sum"), clim=("clim", "mean"))
    Z["eui"] = Z.gas / Z.area
    Z["inc"] = 100 * Z.gas * 0.8 / 3.0 / Z.elec
    r = Z.clim.rank(pct=True).values                    # 0 = mildest climate, 1 = harshest (cek_kebijakan.py 3a)
    cop = 3.5 - 1.0 * (r - r.min()) / (r.max() - r.min())
    Z["inc_c"] = 100 * Z.gas * 0.8 / cop / Z.elec
    for key, inc in (("q0", Z.inc), ("q1", Z.inc_c)):
        xm, ym = Z.eui.median(), inc.median()
        Z[key] = np.where(Z.eui >= xm, np.where(inc >= ym, "Q1", "Q2"), np.where(inc >= ym, "Q3", "Q4"))
    Z["rank_inc0"] = Z.inc.rank(ascending=False, method="min").astype(int)
    Z["rank_inc1"] = Z.inc_c.rank(ascending=False, method="min").astype(int)
    Z["rank_add0"] = (Z.gas * 0.8 / 3.0).rank(ascending=False, method="min").astype(int)
    Z["rank_add1"] = (Z.gas * 0.8 / cop).rank(ascending=False, method="min").astype(int)

    qt = Z.groupby("q0").agg(n=("gas", "size"), gas=("gas", lambda v: v.sum() / 1e9), inc=("inc", "mean"))
    cell49 = {"Q1": (118, 66.095, 102.494), "Q2": (47, 16.299, 90.245), "Q3": (47, 20.607, 99.102),
              "Q4": (118, 38.354, 87.042)}
    for q, (n, gas, inc) in cell49.items():
        assert qt.loc[q, "n"] == n and round(qt.loc[q, "gas"], 3) == gas and round(qt.loc[q, "inc"], 3) == inc, qt
    assert round(Z.gas.sum() / 1e9, 2) == 141.35 and round(Z.inc.median(), 2) == 94.85            # cell 50
    assert round(spearmanr(Z.inc, Z.inc_c).statistic, 4) == 0.8147 and int((Z.q0 != Z.q1).sum()) == 68
    return Z


def peta_kuadran(Z):
    """slide 8: quadrant map of the 330 zones, programme per quadrant as in the slide's table"""
    xm, ym = Z.eui.median(), Z.inc.median()
    fig, ax = plt.subplots(figsize=(6.55, 4.45))
    for q in ("Q4", "Q3", "Q2", "Q1"):
        m = Z.q0 == q
        ax.scatter(Z.eui[m], Z.inc[m], s=10 + 380 * Z.gas[m] / Z.gas.max(), color=QCOL[q], alpha=0.78,
                   edgecolor="white", lw=0.6, zorder=3)
    ax.axvline(xm, color=MUTED, ls="--", lw=1)
    ax.axhline(ym, color=MUTED, ls="--", lw=1)
    lab = {"Q1": ("Q1 · retrofit + gardu (R1, R2)", 0.985, 0.975, "right", "top"),
           "Q2": ("Q2 · pilot pompa panas (R3)", 0.985, 0.03, "right", "bottom"),
           "Q3": ("Q3 · gardu (R2)", 0.015, 0.975, "left", "top"),
           "Q4": ("Q4 · program standar", 0.015, 0.03, "left", "bottom")}
    for q, (t, x, y, ha, va) in lab.items():
        ax.text(x, y, t, transform=ax.transAxes, ha=ha, va=va, fontsize=14, color=QCOL[q] if q != "Q4" else MUTED,
                fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
    ax.set_xlabel("intensitas gas zona (kWh per m²)")
    ax.set_ylabel("kenaikan konsumsi listrik tahunan (%)")
    ax.text(ax.get_xlim()[1], ym + 0.4, "median ", fontsize=14, color=MUTED, ha="right", va="bottom")
    ax.set_ylim(64, 131)
    comma_axis(ax, "y", 0)
    fig.text(0.01, 0.008, "ukuran titik = konsumsi gas zona; garis putus-putus = median", fontsize=14, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "peta_kuadran.png")


def zone_data(Z):
    # slide 9: retrofit indicator of the three candidates, first table of cell 50 (gas kWh, saving_pct)
    cell50 = [("REG_273", 2.311622e9, 7.36), ("REG_270", 1.627015e9, 8.60), ("REG_045", 1.655768e9, 6.69)]
    rows = []
    for z, gas, pct in cell50:
        assert abs(Z.loc[z, "gas"] / gas - 1) < 1e-6, (z, Z.loc[z, "gas"], gas)
        assert Z.loc[z, "q0"] == "Q1" and Z.loc[z, "q1"] == "Q1"
        rows.append({"zona": z, "gwh": round(gas * pct / 100 / 1e6, 1), "pct": pct})
    assert [r["gwh"] for r in rows] == [170.1, 139.9, 110.8]       # ../ppt01/fakta_tambahan.md, slide 10 of v2
    DATA["s9_retrofit"] = rows

    # slide 10: five largest relative increases (second table of cell 50) and their rank under the climate COP
    cell50b = [("REG_290", 121.41), ("REG_259", 115.10), ("REG_275", 114.29), ("REG_260", 113.91), ("REG_270", 113.46)]
    top = Z.inc.sort_values(ascending=False)
    assert [(z, round(v, 2)) for z, v in top.head(5).items()] == cell50b, top.head(5)
    DATA["s10_kenaikan"] = {
        "lima": [{"zona": z, "pct": v, "peringkat_cop_iklim": int(Z.loc[z, "rank_inc1"]), "kuadran": str(Z.loc[z, "q0"])}
                 for z, v in cell50b],
        "median": round(float(Z.inc.median()), 2)}
    # zones in the top ten of both the relative and the absolute increase under both COP scenarios
    both = Z[(Z.rank_inc0 <= 10) & (Z.rank_inc1 <= 10) & (Z.rank_add0 <= 10) & (Z.rank_add1 <= 10)]
    DATA["s10_kenaikan"]["sepuluh_besar_dua_ukuran_dua_skenario"] = sorted(both.index)
    DATA["s10_kenaikan"]["tambahan_absolut_terbesar"] = {
        "zona": Z.rank_add0.idxmin(), "TWh": round(float(Z.gas.max() * 0.8 / 3.0 / 1e9), 3),
        "peringkat_cop_iklim": int(Z.loc[Z.rank_add0.idxmin(), "rank_add1"])}
    DATA["s10_kenaikan"]["sepuluh_besar_persen_dua_skenario"] = sorted(Z[(Z.rank_inc0 <= 10) & (Z.rank_inc1 <= 10)].index)
    ex = json.load(open(ZONE_EVIDENCE))["zona_contoh"]
    for z in ("REG_270", "REG_045", "REG_263", "REG_290", "REG_259", "REG_275"):
        assert int(Z.loc[z, "rank_inc1"]) == ex[z]["peringkat_kenaikan_cop_iklim"]
        assert int(Z.loc[z, "rank_add1"]) == ex[z]["peringkat_tambahan_absolut_cop_iklim"]
    # gardu mapping of slides 2, 8, 10 and 13: the starting zones of R2 sit in Q1 on both scenarios, and the
    # quadrants with an above-median increase are Q1 and Q3 (cell 51, point 3)
    for z in ("REG_263", "REG_270", "REG_045"):
        assert Z.loc[z, "q0"] == "Q1" and Z.loc[z, "q1"] == "Q1", z
    above = Z.inc >= Z.inc.median()
    assert set(Z.q0[above]) == {"Q1", "Q3"} and set(Z.q0[~above]) == {"Q2", "Q4"}


# ---- slide 7: public leaderboard per development step (cell 37) and the private score (Kaggle) --------------------
def leaderboard():
    nb = json.load(open(NB, encoding="utf-8"))
    text = "\n".join("".join(o.get("text", []) or o.get("data", {}).get("text/plain", []))
                     for o in nb["cells"][37].get("outputs", []))
    rows = re.findall(r"^\d+\s+(\S+)\s+(.+?)\s+(0\.\d{5})$", text, re.M)
    assert [(v, float(s)) for v, _, s in rows] == [("v12", 0.25025), ("v15", 0.24954), ("v16", 0.24943),
                                                   ("v19", 0.24934), ("v19vc", 0.24925), ("v19nca", 0.24920),
                                                   ("final", 0.24913)], rows
    # two short lines per step for the category labels (14 pt); Poisson and PLR stay on the backup slides
    label = {"v12": ["gambaran", "kawasan"], "v15": ["koreksi", "dan lainnya"], "v16": ["kawasan", "termirip"],
             "v19": ["objektif", "baru"], "v19vc": ["bobot", "adaptif"], "v19nca": ["model", "NCA"],
             "final": ["penyesuaian", "akhir"]}
    status = open(os.path.join(ROOT, "reports/STATUS.md"), encoding="utf-8").read()
    assert "peringkat 8, 0,24844" in status          # Kaggle private leaderboard, 24 Oct 2026
    DATA["s7_lb"] = {"langkah": [{"versi": v, "notebook": n, "label": label[v], "skor": float(s)} for v, n, s in rows],
                     "privat": 0.24844, "peringkat_privat": 8}


# ---- slide 11 text: adoption by gas-network share band (train rows, weighted by meters) -----------------------------
def adoption_bands():
    tr = pd.read_csv(os.path.join(ROOT, "data/raw/train.csv"), usecols=["hvac_dist_02", "bld_dim_12", ADOPT])
    band = pd.cut(tr["hvac_dist_02"], [-0.01, 40, 60, 80, 100.01], labels=["<=40", "40-60", "60-80", ">80"])
    M = tr["bld_dim_12"].astype(float)
    g = pd.DataFrame({"band": band, "M": M, "eh": tr[ADOPT] / 100 * M}).groupby("band", observed=True).sum()
    ref = json.load(open(os.path.join(ROOT, "deck/ppt01/bukti/cek_kebijakan.json")))["adopsi_per_pita_gas_jaringan"]
    label = {"<=40": "≤40%", "40-60": "40–60%", "60-80": "60–80%", ">80": ">80%"}
    rows = []
    for b in g.index:
        rate = round(100 * float(g.loc[b, "eh"] / g.loc[b, "M"]), 2)
        share = round(100 * float((band == b).mean()), 2)
        assert rate == ref[b]["tingkat_adopsi_pct"] and share == ref[b]["porsi_kawasan_pct"], (b, rate, share)
        rows.append({"pita": label[b], "adopsi": rate, "kawasan": share})
    DATA["s11_adopsi"] = rows


# ---- SHAP of the explanation models (cells 41 to 47) -------------------------------------------------------------------
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
    return idx, sv, imp


def binned(x, s):
    q = pd.Series(s).groupby(pd.qcut(x, 20, duplicates="drop"), observed=True).mean()
    return np.array([iv.mid for iv in q.index]), q.values


def size_bands(x, s, sgn, expected_pct):
    """mean size and SHAP of the smallest and largest 5 % (cek_kebijakan.py, step 5)"""
    bins = pd.qcut(x, 20, labels=False)
    lo, hi = bins == 0, bins == bins.max()
    assert round(float(x[lo].mean()), 1) == 36.6 and round(float(x[hi].mean()), 1) == 173.1
    assert round(100 * float(np.exp(sgn * (s[lo].mean() - s[hi].mean())) - 1), 1) == expected_pct
    return (x[lo].mean(), s[lo].mean()), (x[hi].mean(), s[hi].mean())


def house(ax, x, y, w, color):
    """small house glyph in axes coordinates (x, y = bottom left)"""
    h = w * 0.62
    ax.add_patch(patches.Rectangle((x, y), w, h, transform=ax.transAxes, fc="white", ec=color, lw=1.8, zorder=6))
    ax.add_patch(patches.Polygon([(x - 0.12 * w, y + h), (x + w / 2, y + h + 0.55 * w), (x + 1.12 * w, y + h)],
                                 closed=True, transform=ax.transAxes, fc=color, ec=color, lw=1.2, zorder=6))


def size_panel(ax, x, s, lo, hi, color, ylabel):
    ax.scatter(x, s, s=3, alpha=0.16, color=color, rasterized=True, lw=0)
    bx, by = binned(x, s)
    ax.plot(bx, by, color=INK, lw=2.4, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.8)
    for px, py in (lo, hi):
        ax.scatter([px], [py], s=110, color=NAVY, edgecolor="white", lw=1.5, zorder=7)
    log_size_axis(ax)
    ax.set_ylim(-0.55, 0.65)
    ax.set_yticks(SHAP_TICKS)
    comma_axis(ax, "y", 1)
    ax.set_xlabel("luas lantai per meter (m², skala log)")
    ax.set_ylabel(ylabel)


def dependence(X):
    import explain
    cols = list(X.columns)
    eui = load_shap("eui", X, {"area_per_meter": 0.0895, "env_climate_05": 0.0321, "socio_demog_082": 0.0276,
                               "area_per_epc": 0.0260, "s002_ratio": 0.0250})
    epm = load_shap("epm", X, {"area_per_meter": 0.0895, "meters_per_epc": 0.0294, "socio_demog_005": 0.0222,
                               "socio_demog_038": 0.0196, "socio_demog_061": 0.0163})
    gpm = load_shap("gpm", X, {})
    adopt = load_shap("adopt", X, {"hvac_rest": 3.5653, "hvac_dist_02": 0.9053})

    # slide 3: five features with the largest mean |SHAP| (cell 45); derived features marked as in the "keluarga"
    # column of cell 45 (explain.family)
    DATA["s3_shap_top"] = {key: [{"fitur": f, "nilai": round(float(v), 4),
                                  "turunan": explain.family(f) == "rasio fisik turunan"} for f, v in imp.head(5).items()]
                           for key, (_, _, imp) in (("gas", eui), ("listrik", epm))}
    for key, ratio in (("gas", 2.8), ("listrik", 3.0)):   # "2,8x" and "3,0x" on slide 3 (fakta_tambahan.md)
        top = DATA["s3_shap_top"][key]
        assert top[0]["fitur"] == "area_per_meter" and round(top[0]["nilai"] / top[1]["nilai"], 1) == ratio
    assert [r["fitur"] for r in DATA["s3_shap_top"]["gas"] if r["turunan"]] == ["area_per_meter", "area_per_epc",
                                                                                 "s002_ratio"]
    assert [r["fitur"] for r in DATA["s3_shap_top"]["listrik"] if r["turunan"]] == ["area_per_meter", "meters_per_epc"]
    # slide 3: share of the derived physical ratios in the SHAP attribution, equal to writeup B Table 3
    table3 = {t["nama"]: t["porsi"][DATA["c5_kelompok"]["kelompok"].index("Rasio fisik")]
              for t in DATA["c5_kelompok"]["target"]}
    shares = {}
    for key, name, (_, sv, _) in (("eui", "EUI (intensitas gas)", eui), ("gpm", "Gas per meter", gpm),
                                  ("epm", "Listrik per meter", epm)):
        share = 100 * float(explain.family_shares(sv, cols)["rasio fisik turunan"])
        assert round(share) == table3[name], (key, share, table3[name])
        shares[key] = round(share)
    assert (min(shares.values()), max(shares.values())) == (25, 30), shares
    DATA["s3_rasio_fisik"] = shares

    apm = X["area_per_meter"].values
    ci = cols.index("area_per_meter")

    # slide 9: gas per m2 against house size, with the two 5 % bands of the size contrast
    idx, sv, _ = eui
    x, s = apm[idx], sv[:, ci]
    lo, hi = size_bands(x, s, 1, 66.5)
    fig, ax = plt.subplots(figsize=(5.7, 3.05))
    size_panel(ax, x, s, lo, hi, AMBER, "SHAP (log gas per m²)")
    for (px, py), txt, dx, dy, ha in ((lo, "rumah ±37 m²", 1.45, 0.17, "left"), (hi, "rumah ±173 m²", 0.69, -0.19, "right")):
        ax.annotate(txt, (px, py), xytext=(px * dx, py + dy), fontsize=14, color=NAVY, fontweight="bold", ha=ha,
                    va="center", zorder=7, arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.0))
    house(ax, 0.735, 0.70, 0.065, NAVY)
    house(ax, 0.845, 0.70, 0.11, MUTED)
    ax.text(0.83, 0.585, "kecil vs besar", transform=ax.transAxes, fontsize=14, color=MUTED, ha="center")
    fig.tight_layout(pad=0.4)
    save(fig, "dep_gas_ukuran.png")

    # slide 10: electricity per house against house size (cell 48, point 1: about +75 % for large houses)
    idx, sv, _ = epm
    x, s = apm[idx], sv[:, ci]
    lo, hi = size_bands(x, s, -1, 74.6)
    fig, ax = plt.subplots(figsize=(5.7, 2.75))
    size_panel(ax, x, s, lo, hi, TEAL, "SHAP (log listrik\nper meter)")
    for (px, py), txt, dx, dy, ha in ((lo, "rumah ±37 m²", 1.45, -0.17, "left"), (hi, "rumah ±173 m²", 1.22, -0.26, "left")):
        ax.annotate(txt, (px, py), xytext=(px * dx, py + dy), fontsize=14, color=NAVY, fontweight="bold", ha=ha,
                    va="center", zorder=7, arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.0))
    ax.text(0.03, 0.96, "rumah besar vs kecil:\nsumbangan SHAP +75%", transform=ax.transAxes, fontsize=15, color=TEAL,
            fontweight="bold", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=8)
    fig.tight_layout(pad=0.4)
    save(fig, "dep_listrik_ukuran.png")

    # backup C11: both size panels (gas per m2 and electricity per house)
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.45))
    for ax, (idx, sv, _), color, sgn, pct, ylabel, title in (
            (axs[0], eui, AMBER, 1, 66.5, "SHAP (log gas per m²)", "Gas per m²"),
            (axs[1], epm, TEAL, -1, 74.6, "SHAP (log listrik per meter)", "Listrik per rumah")):
        x, s = apm[idx], sv[:, ci]
        lo, hi = size_bands(x, s, sgn, pct)
        size_panel(ax, x, s, lo, hi, color, ylabel)
        ax.set_title(title, fontsize=16, pad=6)
    fig.tight_layout(w_pad=2.5)
    save(fig, "dep_ukuran_dua.png")

    # slide 11 (main visual): adoption against the gas-network share; annotations quote cell 48, point 4
    idx, sv, _ = adopt
    x, s = X["hvac_dist_02"].values[idx], sv[:, cols.index("hvac_dist_02")]
    bx, by = binned(x, s)
    low = x < 40
    assert by[0] > 4 and np.percentile(s[low], 95) < 13 and np.percentile(s[low], 5) > 0, (by[0], s[low].min())
    assert (by[bx > 80] < 0).all() and (s[x > 85] < 0.5).mean() > 0.95
    fig, ax = plt.subplots(figsize=(5.75, 4.75))
    ax.scatter(x, s, s=4, alpha=0.16, color=TEAL, rasterized=True, lw=0)
    ax.plot(bx, by, color=INK, lw=2.6, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.axvspan(80, 100, color=AMBER, alpha=0.13, lw=0)
    ax.text(24, 11.2, "di bawah 40%:\n+4 sampai +12 poin", fontsize=15, color=TEAL, fontweight="bold", ha="center",
            va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    ax.text(90, 9.6, "di atas 80%:\nsumbangan\nnegatif", fontsize=15, color=AMBER, fontweight="bold", ha="center",
            va="center")
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.5, 14.5)
    ax.set_yticks([-2, 0, 2, 4, 6, 8, 10, 12, 14])
    comma_axis(ax, "y", 0)
    ax.set_xlabel("porsi pemanas gas jaringan di kawasan (%)")
    ax.set_ylabel("sumbangan SHAP adopsi (poin persentase)")
    fig.tight_layout(pad=0.4)
    save(fig, "dep_adopsi_besar.png")


# ---- backup C1: two-knot hat basis of blend.size_basis (same figure as deck v2) ---------------------------------------
def basis_pojok():
    t = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(5.75, 3.0))
    ax.plot(t, 1 - t, color=NAVY, lw=2.6, label="ujung rendah = 1 − t")
    ax.plot(t, t, color=AMBER, lw=2.6, label="ujung tinggi = t")
    ax.plot([0.8, 0.8], [0.2, 0.8], ls=":", color=MUTED, lw=1.2)
    ax.plot([0.8, 0.8], [0.2, 0.8], "o", color=INK, ms=6)
    ax.text(0.84, 0.5, "peringkat 0,8:\n20% rendah,\n80% tinggi", fontsize=14, color=INK, va="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.22), ncol=2, frameon=False, fontsize=14)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    comma_axis(ax, "x", 1)
    comma_axis(ax, "y", 1)
    ax.set_xlabel("peringkat kawasan di data latih")
    ax.set_ylabel("porsi bobot")
    ax.yaxis.grid(True, color=SOFT)
    fig.tight_layout(pad=0.4)
    save(fig, "basis_pojok.png")


# ---- backup C5 and slide 3: feature-family shares, writeup B Table 3 ----------------------------------------------------
def family_table():
    text = open(os.path.join(ROOT, "reports/WRITEUP_DRAFT_vB.md"), encoding="utf-8").read()
    block = text.split("**Tabel 3.**", 1)[1].split("Catatan:", 1)[0]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block.splitlines() if ln.startswith("|")]
    head, body = rows[0], [r for r in rows[2:]]
    DATA["c5_kelompok"] = {"kelompok": head[1:],
                           "target": [{"nama": r[0], "porsi": [int(re.sub(r"\D", "", v)) for v in r[1:]]} for r in body]}
    adopt = DATA["c5_kelompok"]["target"][-1]
    assert adopt["nama"].startswith("Adopsi") and adopt["porsi"][-1] == 81 and adopt["porsi"][0] == 15


if __name__ == "__main__":
    Z = zones()
    peta_kuadran(Z)
    zone_data(Z)
    leaderboard()
    adoption_bands()
    basis_pojok()
    family_table()
    dependence(explain_frame())
    out = os.path.join(IMG, "data_grafik.json")
    json.dump(DATA, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: DATA[k] for k in ("s3_rasio_fisik", "s7_lb")}, indent=1, ensure_ascii=False))
