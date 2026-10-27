"""Visuals for deck/04_struktur_deck.md marked "BUAT BARU" (K1-K4) plus two crops of the notebook's SHAP dependence
figure. Every number below is copied from an output cell of final/deliverables/DICEFEST2026_BISMILLAH.ipynb (cell
numbers in the comments); nothing is recomputed or estimated.

usage: .venv/bin/python deck/make_visuals.py   (writes deck/img/*.png)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
BLUE, RED, INK, MUTED, GRID, GREY = "#3b6ea5", "#c0392b", "#2b2b2b", "#6b6b6b", "#e6e6e6", "#c8c8c8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def id_num(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def k1_crossfit():
    """K1: two-level cross-fitting of the kNN-target features (train_lib.run + Data._knn_features)"""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, color, text="", tc="white", fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="white",
                                    lw=1.5))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs)

    # row 1: ten outer folds, fold 3 is the validation fold
    ax.text(0.0, 5.75, "1. Sepuluh fold luar: satu fold dinilai, sembilan lainnya menjadi bagian latih", color=INK,
            fontsize=9, fontweight="bold")
    for k in range(10):
        box(0.1 + k * 1.03, 4.75, 0.95, 0.7, RED if k == 2 else BLUE, f"fold {k + 1}")
    ax.text(0.1 + 2 * 1.03 + 0.475, 4.55, "baris validasi", ha="center", va="top", color=RED, fontsize=8)
    # row 2: validation rows take neighbours from the nine training folds only
    ax.text(0.0, 3.75, "2. Baris validasi: tetangga hanya dari 9 fold latih (target fold validasi tidak dipakai)",
            color=INK, fontsize=9)
    # row 3: training rows, inner five-fold split of the training part
    ax.text(0.0, 2.95, "3. Baris latih: bagian latih dibagi 5 bagian dalam; tetangga dari 4 bagian lainnya",
            color=INK, fontsize=9, fontweight="bold")
    for j in range(5):
        box(0.1 + j * 2.06, 1.95, 1.98, 0.7, RED if j == 0 else BLUE,
            "baris yang dinilai" if j == 0 else f"bagian dalam {j + 1}")
    for j in (1, 2, 3, 4):  # neighbours come from the four other inner parts
        ax.add_patch(FancyArrowPatch((0.1 + j * 2.06 + 0.99, 1.88), (1.09, 1.88), connectionstyle="arc3,rad=-0.12",
                                     arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.0))
    ax.text(5.3, 1.0, "rata-rata target tetangga yang mirip, dalam jendela ±W nomor subzona", color=MUTED,
            fontsize=8, ha="center")
    # row 4: test rows
    ax.text(0.0, 0.35, "4. Data uji: tetangga dari 9 fold latih model yang sama; prediksi = rata-rata 10 model fold",
            color=INK, fontsize=9)
    save(fig, "baru_k1_crossfit.png")


def k2_shap_bars():
    """K2: top five features by mean |SHAP| for three explanation targets (output of cell 45, In [32])"""
    panels = [
        ("log intensitas gas (EUI)", [("area_per_meter", 0.0895), ("env_climate_05", 0.0321),
                                      ("socio_demog_082", 0.0276), ("area_per_epc", 0.0260), ("s002_ratio", 0.0250)],
         "skala log", 4),
        ("log listrik per meter", [("area_per_meter", 0.0895), ("meters_per_epc", 0.0294),
                                   ("socio_demog_005", 0.0222), ("socio_demog_038", 0.0196),
                                   ("socio_demog_061", 0.0163)], "skala log", 4),
        ("adopsi pemanas listrik (%)", [("hvac_rest", 3.5653), ("hvac_dist_02", 0.9053), ("hvac_dist_09", 0.2691),
                                        ("hvac_dist_01", 0.0968), ("hvac_dist_08", 0.0953)], "poin persentase", 4),
    ]
    fig, axs = plt.subplots(3, 1, figsize=(6.6, 7.4))
    for ax, (title, rows, unit, nd) in zip(axs, panels):
        names = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        colors = [RED if i == len(rows) - 1 else BLUE for i in range(len(rows))]
        ax.barh(names, vals, height=0.62, color=colors)
        for y, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, y, id_num(v, nd), va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(f"rata-rata |SHAP| ({unit})", fontsize=8, color=MUTED)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.text(0.0, -0.02, "Sumber: keluaran sel 45 (In [32]). Skala tiap panel berbeda, jadi panjang batang\ntidak dapat "
             "dibandingkan antarpanel.", fontsize=7.5, color=MUTED)
    plt.tight_layout()
    save(fig, "baru_k2_shap_teratas.png")


def k3_zone_bars():
    """K3: top zones from the two tables printed by cell 50 (In [35])"""
    saving = [("REG_273", "Q1", 7.36), ("REG_270", "Q1", 8.60), ("REG_045", "Q1", 6.69), ("REG_242", "Q2", 7.35),
              ("REG_263", "Q1", 3.43)]
    increase = [("REG_290", "Q1", 121.41), ("REG_259", "Q1", 115.10), ("REG_275", "Q1", 114.29),
                ("REG_260", "Q1", 113.91), ("REG_270", "Q1", 113.46)]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 6.0))
    for ax, rows, title, xlabel, ref in (
            (axs[0], saving, "Lima zona dengan potensi hemat retrofit terbesar (kWh)",
             "potensi hemat retrofit (% konsumsi gas zona)", None),
            (axs[1], increase, "Lima zona dengan kenaikan konsumsi listrik tertinggi",
             "kenaikan konsumsi listrik tahunan bila dielektrifikasi (%)", 94.85)):
        labels = [f"{z} ({q})" for z, q, _ in rows][::-1]
        vals = [v for _, _, v in rows][::-1]
        ax.barh(labels, vals, height=0.62, color=BLUE)
        for y, v in enumerate(vals):
            ax.text(v + max(vals) * 0.015, y, id_num(v, 2), va="center", fontsize=8, color=INK)
        if ref is not None:
            ax.axvline(ref, color=RED, lw=1.2, ls="--")
            ax.set_ylim(-1.3, len(rows) - 0.5)
            ax.text(ref - 2, -0.95, "garis putus-putus: median 330 zona = 94,85%", color=RED, fontsize=7.5, ha="right",
                    va="center")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.text(0.0, -0.03, "Sumber: dua tabel keluaran sel 50 (In [35]); urutan zona atas mengikuti potensi hemat absolut.\n"
             "Q1/Q2 = kuadran pada peta trilema sel 49.", fontsize=7.5, color=MUTED)
    plt.tight_layout()
    save(fig, "baru_k3_zona_teratas.png")


def k4_hat_basis():
    """K4: the two-knot hat basis of blend.size_basis (knots 0 and 1): weights 1 - t and t on the rank t"""
    t = [i / 100 for i in range(101)]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(t, [1 - x for x in t], color=BLUE, lw=2, label="bobot \"ujung rendah\" = 1 − t")
    ax.plot(t, t, color=RED, lw=2, label="bobot \"ujung tinggi\" = t")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8.5)
    ax.set_xlim(-0.03, 1.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.plot([0.8, 0.8], [0.2, 0.8], ls=":", color=MUTED, lw=1)
    ax.plot([0.8, 0.8], [0.2, 0.8], "o", color=INK, ms=5)
    ax.text(0.84, 0.5, "contoh: peringkat 0,8\n20% ujung rendah,\n80% ujung tinggi", ha="left", va="center",
            fontsize=8, color=INK)
    ax.set_xlabel("peringkat kawasan di data latih untuk A, M, atau p (0 = terkecil, 1 = terbesar)", fontsize=8,
                  color=MUTED)
    ax.set_ylabel("porsi bobot", fontsize=8, color=MUTED)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    fig.text(0.0, -0.06, "Tiga variabel × 2 simpul = 2 × 2 × 2 = 8 bobot pojok per model (blend.vc_basis), semuanya "
             "tidak negatif (NNLS).", fontsize=7.5, color=MUTED)
    plt.tight_layout()
    save(fig, "baru_k4_basis_segitiga.png")


def crops():
    """two crops of the cell 47 (In [33]) dependence figure, 1188 x 640 px, 3 x 2 panels"""
    im = Image.open(os.path.join(IMG, "sel47_dependensi_shap.png"))
    w, h = im.size
    im.crop((0, 0, int(w * 2 / 3), h // 2)).save(os.path.join(IMG, "sel47_potongan_luas_per_meter.png"))
    im.crop((int(w * 2 / 3), 0, w, h // 2)).save(os.path.join(IMG, "sel47_potongan_adopsi_gas_jaringan.png"))


if __name__ == "__main__":
    k1_crossfit()
    k2_shap_bars()
    k3_zone_bars()
    k4_hat_basis()
    crops()
    for f in sorted(os.listdir(IMG)):
        print(f, Image.open(os.path.join(IMG, f)).size)
