"""Build the DICE FEST 2026 final pitch deck (team BISMILLAH) as a 16:9 vector PDF.

Source of truth for every number: final/deliverables/WRITEUP_BISMILLAH_B_bab_tetap.pdf and
reports/deck_facts_extra.md. Figures are reused as-is from reports/figures/.

Usage: deck/.venv/bin/python deck/build_deck.py
"""
import os
import re

from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "reports", "figures")
OUT = os.path.join(HERE, "BISMILLAH_Pitch_Deck_v1.pdf")

W, H = 960, 540
MX = 48

CREAM = "#F4EEDF"
CARD = "#FBF8F0"
GREEN = "#1F4D3F"
GREEN2 = "#2F6B58"
SAGE = "#DDE7DC"
MUSTARD = "#E2A62A"
INK = "#23302B"
MUTED = "#5E6F67"
LINE = "#D8CFBA"

import font_roboto

FONT_DIR = os.path.join(os.path.dirname(font_roboto.__file__), "files")
for name, f in [("R", "Roboto-Regular.ttf"), ("B", "Roboto-Bold.ttf"),
                ("I", "Roboto-Italic.ttf"), ("BI", "Roboto-BoldItalic.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, f)))

FOOTER = "Tim BISMILLAH | DICE FEST 2026"


# ---------------------------------------------------------------- text helpers
def _atoms(text, base):
    """Split inline markup (*italic*, **bold**) into words made of (text, font) atoms."""
    segs = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            segs.append((part[2:-2], "BI" if base in ("I", "BI") else "B"))
        elif part.startswith("*"):
            segs.append((part[1:-1], "BI" if base in ("B", "BI") else "I"))
        else:
            segs.append((part, base))
    words, space = [], False
    for seg, font in segs:
        for m in re.finditer(r"(\s*)(\S+)", seg):
            if m.group(1) or space or not words:
                words.append([(m.group(2), font)])
            else:
                words[-1].append((m.group(2), font))
            space = False
        if seg[-1:].isspace():
            space = True
    return words


def _wlen(word, size):
    return sum(pdfmetrics.stringWidth(t, f, size) for t, f in word)


def wrap(text, width, size, base="R"):
    lines, cur, cw = [], [], 0
    sp = pdfmetrics.stringWidth(" ", base, size)
    for word in _atoms(text, base):
        wl = _wlen(word, size)
        add = wl if not cur else sp + wl
        if cur and cw + add > width:
            lines.append(cur)
            cur, cw = [word], wl
        else:
            cur.append(word)
            cw += add
    if cur:
        lines.append(cur)
    return lines


def para(c, x, top, width, text, size=18, color=INK, base="R", lead=None, align="l"):
    """Draw wrapped text with its top edge at `top` (distance from page top). Returns bottom."""
    lead = lead or size * 1.35
    lines = wrap(text, width, size, base)
    sp = pdfmetrics.stringWidth(" ", base, size)
    y = top
    for ln in lines:
        y += lead
        total = sum(_wlen(w, size) for w in ln) + sp * (len(ln) - 1)
        cx = x if align == "l" else (x + (width - total) / 2 if align == "c" else x + width - total)
        c.setFillColor(color)
        for wi, word in enumerate(ln):
            for t, f in word:
                c.setFont(f, size)
                c.drawString(cx, H - y + (lead - size) * 0.28, t)
                cx += pdfmetrics.stringWidth(t, f, size)
            cx += sp
    return y + (lead - size) * 0.3


def height_of(text, width, size, base="R", lead=None):
    lead = lead or size * 1.35
    return len(wrap(text, width, size, base)) * lead


# ---------------------------------------------------------------- shape helpers
def rect(c, x, top, w, h, fill, stroke=None, r=14):
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(1)
    c.roundRect(x, H - top - h, w, h, r, stroke=1 if stroke else 0, fill=1)


def circle(c, cx, cy_top, r, fill):
    c.setFillColor(fill)
    c.circle(cx, H - cy_top, r, stroke=0, fill=1)


def image(c, name, x, top, w, h, card=True):
    """Fit image into the box keeping aspect ratio; image is never cropped or altered."""
    path = os.path.join(FIG, name)
    iw, ih = ImageReader(path).getSize()
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    ox, oy = x + (w - dw) / 2, top + (h - dh) / 2
    if card:
        rect(c, ox - 8, oy - 8, dw + 16, dh + 16, "#FFFFFF", LINE, 12)
    c.drawImage(path, ox, H - oy - dh, dw, dh)
    return ox, oy, dw, dh


_page = {"n": 0}


def base_slide(c, chip, title, number=True, appendix=False):
    _page["n"] += 1
    c.setFillColor(CREAM)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    # corner quarter circle with page number
    c.setFillColor(GREEN)
    c.circle(W, 0, 64, stroke=0, fill=1)
    c.setFillColor(MUSTARD)
    c.circle(W, 0, 30, stroke=0, fill=1)
    c.setFillColor(GREEN)
    c.setFont("B", 11)
    c.drawRightString(W - 9, 9, str(_page["n"]))
    # chip
    label = chip.upper()
    cw = pdfmetrics.stringWidth(label, "B", 11) + 26
    rect(c, MX, 28, cw, 22, GREEN if not appendix else MUSTARD, r=11)
    c.setFillColor("#FFFFFF" if not appendix else INK)
    c.setFont("B", 11)
    c.drawString(MX + 13, H - 43, label)
    # title
    para(c, MX, 56, W - 2 * MX - 30, title, size=28, color=GREEN, base="B", lead=34)
    # footer
    c.setFillColor(MUTED)
    c.setFont("R", 12)
    c.drawString(MX, 16, FOOTER)


def stat(c, x, top, w, number, label, ncolor=GREEN, nsize=36, lsize=18, fill=None):
    nh = nsize * 1.15
    lh = height_of(label, w - 28, lsize, lead=lsize * 1.3)
    h = nh + lh + 26
    if fill:
        rect(c, x, top, w, h, fill)
    c.setFillColor(ncolor)
    c.setFont("B", nsize)
    c.drawString(x + 14, H - (top + 12 + nsize * 0.95), number)
    para(c, x + 14, top + 12 + nh, w - 28, label, size=lsize, color=INK if not fill or fill == CARD else "#FFFFFF",
         lead=lsize * 1.3)
    return top + h


def bullets(c, x, top, w, items, size=18, gap=10, color=INK, dot=MUSTARD):
    y = top
    for it in items:
        circle(c, x + 5, y + size * 0.78, 4.5, dot)
        y = para(c, x + 20, y, w - 20, it, size=size, color=color) + gap
    return y


def table(c, x, top, cols, rows, size=14, header=True, pad=7, zebra=True, bold_first=False):
    """cols: list of widths. rows: list of list of str. Returns bottom."""
    y = top
    for ri, row in enumerate(rows):
        is_h = header and ri == 0
        lines = [wrap(t, cw - 2 * pad, size, "B" if is_h or (bold_first and ci == 0) else "R")
                 for ci, (t, cw) in enumerate(zip(row, cols))]
        rh = max(len(l) for l in lines) * size * 1.3 + 2 * pad - 2
        total_w = sum(cols)
        if is_h:
            rect(c, x, y, total_w, rh, GREEN, r=6)
        elif zebra and ri % 2 == 1:
            rect(c, x, y, total_w, rh, CARD, r=4)
        cx = x
        for ci, (t, cw) in enumerate(zip(row, cols)):
            base = "B" if is_h or (bold_first and ci == 0) else "R"
            col = "#FFFFFF" if is_h else INK
            para(c, cx + pad, y + pad - 3, cw - 2 * pad, t, size=size, color=col, base=base, lead=size * 1.3)
            cx += cw
        y += rh
    return y


# ---------------------------------------------------------------- main slides
def s1(c):
    _page["n"] += 1
    c.setFillColor(CREAM)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(GREEN)
    c.rect(0, 0, 570, H, stroke=0, fill=1)
    circle(c, 570, 86, 62, GREEN)
    circle(c, 570, 86, 28, MUSTARD)
    circle(c, 60, 560, 110, GREEN2)
    rect(c, MX, 44, 232, 24, MUSTARD, r=12)
    c.setFillColor(INK)
    c.setFont("B", 11)
    c.drawString(MX + 14, H - 60, "DICE FEST 2026 MACHINE LEARNING")
    para(c, MX, 150, 470, "Dari Struktur Data ke Peta Kebijakan", size=40, color="#FFFFFF", base="B", lead=48)
    para(c, MX, 150 + 48 * 3 + 14, 450, "Prediksi trilema energi perkotaan dan prioritas program per zona",
         size=20, color="#E9E2CF", lead=28)
    c.setFillColor(MUSTARD)
    c.rect(MX, H - 410, 60, 4, stroke=0, fill=1)
    c.setFillColor("#FFFFFF")
    c.setFont("B", 20)
    c.drawString(MX, H - 445, "Tim BISMILLAH")
    # right column
    rx = 612
    c.setFillColor(MUTED)
    c.setFont("R", 14)
    c.drawString(rx, H - 170, "Skor leaderboard private")
    c.setFillColor(GREEN)
    c.setFont("B", 56)
    c.drawString(rx, H - 228, "0,24844")
    para(c, rx, 236, 300, "peringkat 8", size=20, color=INK, base="B")
    c.setFillColor(LINE)
    c.rect(rx, H - 300, 300, 1.5, stroke=0, fill=1)
    c.setFillColor(MUTED)
    c.setFont("R", 14)
    c.drawString(rx, H - 330, "Skor leaderboard publik")
    c.setFillColor(GREEN)
    c.setFont("B", 34)
    c.drawString(rx, H - 372, "0,24913")
    para(c, rx, 392, 300, "613 ribu kawasan, 330 zona", size=18, color=INK)


def s2(c):
    base_slide(c, "Latar belakang", "Lima target mencerminkan tiga tuntutan transisi energi")
    top = 148
    rows = [("gas total", "emisi"),
            ("listrik total", "keandalan jaringan"),
            ("EUI (gas per m² lantai)", "emisi dan biaya"),
            ("porsi bangunan bersih", "kualitas stok bangunan"),
            ("adopsi pemanas listrik", "kemajuan transisi")]
    y = top
    for t, a in rows:
        rect(c, MX, y, 500, 54, CARD, LINE, 10)
        c.setFillColor(GREEN)
        c.setFont("B", 18)
        c.drawString(MX + 18, H - (y + 33), t)
        rect(c, MX + 270, y + 12, 216, 30, SAGE, r=15)
        c.setFillColor(GREEN)
        c.setFont("R", 15)
        c.drawCentredString(MX + 378, H - (y + 33), a)
        y += 64
    # right: error share
    rx = 600
    rect(c, rx, top, 312, 310, GREEN, r=18)
    c.setFillColor(MUSTARD)
    c.setFont("B", 14)
    c.drawString(rx + 22, H - (top + 36), "SUMBER GALAT MODEL AKHIR")
    shares = [("EUI", "±1/2"), ("Listrik", "±1/4"), ("Gas", "±1/5"), ("Adopsi", "sisanya")]
    yy = top + 60
    for n, v in shares:
        c.setFillColor("#FFFFFF")
        c.setFont("B", 22)
        c.drawString(rx + 22, H - (yy + 28), v)
        c.setFont("R", 18)
        c.drawString(rx + 130, H - (yy + 28), n)
        yy += 58
    para(c, MX, 478, 560, "Metrik: rata-rata RMSE/σ per target; tebakan rata-rata menghasilkan skor sekitar 1,0.",
         size=13, color=MUTED, lead=17)


def s3(c):
    base_slide(c, "Analisis data", "Dua target terjawab oleh struktur data sebelum model dilatih")
    cw, gap, top = 205, 16, 150
    cards = [
        ("7 × 10⁻¹⁶", "selisih terbesar gas = EUI × luas lantai: satu kelompok model untuk keduanya"),
        ("100 ×", "porsi bangunan bersih = bld_dim_02 / bld_dim_01: dihitung dengan rumus, tanpa model"),
        ("92%", "kawasan uji sudah diketahui adopsinya: seragam per subzona"),
        ("0,4997", "AUC adversarial: data uji diambil acak per baris"),
    ]
    for i, (n, t) in enumerate(cards):
        x = MX + i * (cw + gap)
        rect(c, x, top, cw, 262, GREEN if i % 3 == 0 else CARD, None if i % 3 == 0 else LINE, 16)
        light = i % 3 == 0
        c.setFillColor(MUSTARD if light else GREEN)
        c.setFont("B", 30 if len(n) < 8 else 26)
        c.drawString(x + 16, H - (top + 56), n)
        para(c, x + 16, top + 78, cw - 32, t, size=19, color="#FFFFFF" if light else INK, lead=25)
    para(c, MX, 440, 860, "Satu target dihitung dengan rumus, satu hampir seluruhnya disalin, dua lainnya dimodelkan bersama.",
         size=18, color=GREEN, base="B")


def s4(c):
    base_slide(c, "Skema validasi", "Skema validasi yang keliru akan membuang fitur terbaik")
    top = 148
    c.setFillColor(MUTED)
    c.setFont("R", 14)
    c.drawString(MX, H - top - 8, "Galat EUI, dengan rata-rata wilayah")
    rect(c, MX, top + 20, 232, 100, CARD, LINE, 14)
    c.setFillColor(GREEN)
    c.setFont("B", 38)
    c.drawString(MX + 16, H - (top + 72), "0,8331")
    c.setFillColor(MUTED)
    c.setFont("R", 16)
    c.drawString(MX + 16, H - (top + 100), "skema per zona")
    rect(c, MX, top + 130, 232, 100, GREEN, r=14)
    c.setFillColor(MUSTARD)
    c.setFont("B", 38)
    c.drawString(MX + 16, H - (top + 182), "0,6699")
    c.setFillColor("#FFFFFF")
    c.setFont("R", 16)
    c.drawString(MX + 16, H - (top + 210), "acak per baris, sesuai data uji")
    para(c, MX, top + 246, 232, "92% kawasan uji berada di subzona yang ada di data latih.", size=17, color=INK, lead=23)
    image(c, "fig3_progress.png", 320, top + 6, 592, 222)
    para(c, 320, top + 252, 592, "*Holdout set* 30% menduga perubahan skor *leaderboard* secara akurat. Tidak ada keputusan yang diambil dari skor publik.",
         size=16, color=INK, lead=22)


def s5(c):
    base_slide(c, "Rekayasa fitur", "Fitur dibangun dari cara energi dipakai, bukan ukuran kawasan")
    top = 150
    bullets(c, MX, top, 410, [
        "Ukuran rumah: luas lantai per meter dan per sertifikat.",
        "Target per rumah tangga, dengan bobot di skala asli.",
        "Rata-rata target wilayah *out-of-fold*: tanpa kebocoran.",
        "Kawasan termirip (kNN): galat EUI turun 0,0022.",
    ], size=18, gap=10)
    rect(c, MX, 404, 410, 64, GREEN, r=14)
    c.setFillColor(MUSTARD)
    c.setFont("B", 28)
    c.drawString(MX + 16, H - 445, "−0,0022")
    c.setFillColor("#FFFFFF")
    c.setFont("R", 16)
    c.drawString(MX + 140, H - 445, "perbaikan terbesar satu kelompok")
    image(c, "fig5_gains.png", 500, top - 8, 410, 330)
    para(c, 500, top + 340, 380, "Atas: model tunggal. Bawah: penggabungan dan koreksi galat wilayah.", size=13, color=MUTED, lead=17)


def s6(c):
    base_slide(c, "Batas akurasi", "Tiga perempat galat berasal dari variasi antarrumah tangga")
    top = 148
    image(c, "fig2_noise_floor.png", MX + 82, top + 4, 700, 248)
    rect(c, MX, 426, 200, 66, GREEN, r=14)
    c.setFillColor(MUSTARD)
    c.setFont("B", 28)
    c.drawString(MX + 16, H - 469, "a + b/M")
    para(c, MX + 218, 428, 640, "M = jumlah meter di kawasan. Perbaikan datang dari *ensemble* model berpola galat berbeda, bukan dari model yang lebih besar.",
         size=18, color=GREEN, base="B", lead=24)


def s7(c):
    base_slide(c, "Arsitektur", "Tiga jalur untuk lima target, 41 model dengan pola galat berbeda")
    top = 148
    image(c, "fig_arch.png", MX, top + 6, 640, 290)
    rect(c, 720, top, 192, 150, GREEN, r=16)
    c.setFillColor(MUSTARD)
    c.setFont("B", 48)
    c.drawString(736, H - (top + 58), "41")
    c.setFillColor("#FFFFFF")
    c.setFont("R", 16)
    c.drawString(736, H - (top + 88), "model gas/EUI: 28")
    c.drawString(736, H - (top + 110), "model listrik: 13")
    c.drawString(736, H - (top + 132), "dua target: rumus")
    para(c, 720, top + 166, 192, "XGBoost, LightGBM, MLP, NCA; bobot NNLS adaptif", size=16, color=INK, lead=22)
    rect(c, MX, 462, 840, 34, CARD, LINE, 12)
    para(c, MX + 16, 462, 810, "Dilatih ulang dari awal: berkas submisi identik byte (SHA-256 aec745ab).", size=16, color=GREEN, base="B", lead=34)


def s8(c):
    base_slide(c, "Interpretasi SHAP", "Ukuran rumah, periode bangunan, dan iklim menggerakkan konsumsi")
    top = 148
    image(c, "pol_dependence.png", MX, top + 4, 500, 270)
    para(c, MX, top + 292, 500, "Model penjelas: 75% variasi gas per meter, 56% EUI, 49% listrik per meter.",
         size=13, color=MUTED, lead=17)
    tiles = [("+65%", "gas per m² di rumah kecil"), ("+75%", "listrik per rumah di rumah besar"),
             ("+7%", "EUI untuk periode bld_era_15"), ("15%", "pergeseran EUI karena iklim")]
    for i, (n, t) in enumerate(tiles):
        x = 580 + (i % 2) * 170
        y = top + (i // 2) * 138
        rect(c, x, y, 162, 128, GREEN if i in (0, 3) else CARD, None if i in (0, 3) else LINE, 14)
        c.setFillColor(MUSTARD if i in (0, 3) else GREEN)
        c.setFont("B", 32)
        c.drawString(x + 14, H - (y + 46), n)
        para(c, x + 14, y + 56, 138, t, size=15, color="#FFFFFF" if i in (0, 3) else INK, lead=19)
    rect(c, 580, top + 282, 332, 56, SAGE, r=12)
    para(c, 594, top + 288, 308, "Adopsi mengikuti sambungan jaringan gas; sosiodemografi hanya 15%.", size=15, color=GREEN, base="B", lead=19)
    para(c, MX, 478, 860, "SHAP menunjukkan asosiasi, bukan sebab-akibat.", size=13, color=MUTED, lead=17)


def s9(c):
    base_slide(c, "Peta trilema", "Setiap zona membutuhkan urutan program yang berbeda")
    top = 146
    rect(c, MX, top, 210, 64, GREEN, r=12)
    c.setFillColor(MUSTARD)
    c.setFont("B", 26)
    c.drawString(MX + 14, H - (top + 28), "7,4 TWh")
    c.setFillColor("#FFFFFF")
    c.setFont("R", 14)
    c.drawString(MX + 14, H - (top + 50), "potensi retrofit, 5,2% gas")
    rect(c, MX + 224, top, 210, 64, GREEN, r=12)
    c.setFillColor(MUSTARD)
    c.setFont("B", 26)
    c.drawString(MX + 238, H - (top + 28), "+95%")
    c.setFillColor("#FFFFFF")
    c.setFont("R", 14)
    c.drawString(MX + 238, H - (top + 50), "listrik bila elektrifikasi penuh")
    image(c, "pol_trilemma.png", MX + 4, top + 80, 426, 226)
    qs = [("Q1", "118 zona, 47% gas", "retrofit dulu", "+102%"),
          ("Q2", "47 zona, 12% gas", "kandidat heat pump", "+90%"),
          ("Q3", "47 zona, 15% gas", "perkuat gardu dulu", "+99%"),
          ("Q4", "118 zona, 27% gas", "program standar", "+87%")]
    qx, y = 520, top
    for i, (q, a, b, d) in enumerate(qs):
        rect(c, qx, y, 392, 70, GREEN if i == 0 else CARD, None if i == 0 else LINE, 14)
        c.setFillColor(MUSTARD if i == 0 else GREEN)
        c.setFont("B", 28)
        c.drawString(qx + 14, H - (y + 45), q)
        c.setFillColor("#FFFFFF" if i == 0 else INK)
        c.setFont("B", 16)
        c.drawString(qx + 70, H - (y + 29), b)
        c.setFont("R", 14)
        c.drawString(qx + 70, H - (y + 51), a)
        c.setFont("B", 22)
        c.setFillColor(MUSTARD if i == 0 else GREEN)
        c.drawRightString(qx + 380, H - (y + 42), d)
        y += 78
    para(c, MX, 466, 440, "Kenaikan: energi tahunan, 80% gas untuk pemanasan, COP 3.",
         size=13, color=MUTED, lead=17)
    para(c, qx, 466, 392, "Peringkat zona tidak berubah oleh asumsi.", size=13, color=MUTED, lead=17)


def s10(c):
    base_slide(c, "Rekomendasi", "Lima langkah untuk pemerintah kota")
    top = 146
    items = [
        ("Urutkan program per kuadran", "Q1 mendahulukan retrofit: REG_273, REG_270, REG_045."),
        ("Q2 jadi kandidat heat pump", "Kesiapan gardu dikonfirmasi dengan operator."),
        ("Libatkan operator jaringan sejak awal", "Q1 dan Q3, misalnya REG_290 (+121%)."),
        ("Insentif untuk kawasan berjaringan gas", "Porsi gas jaringan di atas 80%."),
        ("Pisahkan target efisiensi dan emisi", "Rumah kecil dan bangunan lama; zona bervolume gas terbesar."),
    ]
    y = top
    for i, (a, b) in enumerate(items):
        rect(c, MX, y, 864, 58, GREEN if i == 0 else CARD, None if i == 0 else LINE, 12)
        circle(c, MX + 34, y + 29, 18, MUSTARD)
        c.setFillColor(INK)
        c.setFont("B", 18)
        c.drawCentredString(MX + 34, H - (y + 36), str(i + 1))
        c.setFillColor("#FFFFFF" if i == 0 else GREEN)
        c.setFont("B", 18)
        c.drawString(MX + 68, H - (y + 25), a)
        c.setFillColor("#E9E2CF" if i == 0 else MUTED)
        c.setFont("R", 15)
        c.drawString(MX + 68, H - (y + 47), b)
        y += 66
    para(c, MX, 482, 800,
         "Keterbatasan: SHAP adalah asosiasi; skenario listrik berupa energi tahunan; ambang kuadran berupa median.",
         size=12, color=MUTED, lead=16)


def s11(c):
    base_slide(c, "Simpulan", "Memahami data lebih berharga daripada memperbesar model")
    top = 150
    cols = [("Data", "Dua identitas fisis, satu sifat data sensus, dan satu uji mekanisme pemisahan data menentukan arah kerja."),
            ("Prosedur", "Setiap gagasan divalidasi di *holdout set* sebelum dipakai; tanpa kebocoran target."),
            ("Kebijakan", "Model yang sama menjadi peta 330 zona: retrofit, heat pump, atau penguatan gardu.")]
    cw = 276
    for i, (a, b) in enumerate(cols):
        x = MX + i * (cw + 18)
        rect(c, x, top, cw, 214, GREEN if i == 2 else CARD, None if i == 2 else LINE, 16)
        c.setFillColor(MUSTARD if i == 2 else GREEN)
        c.setFont("B", 22)
        c.drawString(x + 18, H - (top + 44), a)
        para(c, x + 18, top + 58, cw - 36, b, size=18, color="#FFFFFF" if i == 2 else INK, lead=24)
    rect(c, MX, 384, 864, 86, SAGE, r=14)
    para(c, MX + 18, 392, 830,
         "Reprodusibilitas: seed dikunci, tanpa batas waktu pelatihan, versi pustaka dicatat. Pelatihan ulang menghasilkan submisi identik (SHA-256 aec745ab...d66e); notebook kedua mereproduksinya dari model tersimpan.",
         size=16, color=GREEN, lead=21)


# ---------------------------------------------------------------- appendix
def a1(c):
    base_slide(c, "Lampiran A1", "Pembagian per zona membuat rata-rata target wilayah tampak merusak", appendix=True)
    rows = [["Skema pembagian", "Rata-rata target wilayah", "Gas", "EUI", "Listrik"],
            ["acak per baris (sesuai data uji)", "tanpa", "0,2573", "0,6721", "0,3262"],
            ["acak per baris (sesuai data uji)", "dengan", "0,2562", "0,6699", "0,3255"],
            ["per zona", "tanpa", "0,2708", "0,6923", "0,3302"],
            ["per zona", "dengan", "0,3637", "0,8331", "0,3474"]]
    y = table(c, MX, 150, [330, 210, 100, 100, 110], rows, size=18, pad=11)
    para(c, MX, y + 16, 860, "Galat LightGBM (makin kecil makin baik). Pada skema acak per baris, fitur yang sama menurunkan galat di ketiga target.",
         size=16, color=MUTED, lead=22)
    para(c, MX, y + 60, 860, "Empat indikator pemisahan acak per baris: uji tepat 30%; tersebar merata di zona; subzona baru 8,14% (harapan 8,17%); AUC adversarial 0,4997.",
         size=16, color=INK, lead=22)


def a2(c):
    base_slide(c, "Lampiran A2", "Komposisi ensemble: XGBoost dan MLP beragam memegang bobot terbesar", appendix=True)
    rows = [["Kelompok", "Gas/EUI", "Listrik", "Galat EUI (gas)", "Galat listrik", "Bobot gas / EUI / listrik"],
            ["XGBoost", "16", "3", "0,6564 (0,2486)", "0,3200", "42% / 42% / 16%"],
            ["LightGBM", "5", "4", "0,6566 (0,2488)", "0,3197", "13% / 20% / 23%"],
            ["MLP standar", "2", "1", "0,6677 (0,2522)", "0,3213", "5% / 1% / 3%"],
            ["MLP PLR dan TabM", "3", "2", "0,6584 (0,2485)", "0,3188", "27% / 19% / 25%"],
            ["MLP gas + listrik", "1", "2", "0,6585 (0,2488)", "0,3186", "6% / 11% / 26%"],
            ["NCA", "1", "1", "0,6673 (0,2515)", "0,3217", "8% / 7% / 8%"]]
    y = table(c, MX, 150, [190, 90, 90, 180, 130, 184], rows, size=16, pad=8)
    para(c, MX, y + 14, 860, "Galat terbaik adalah galat model tunggal terbaik di kelompoknya. CatBoost GPU tidak disertakan karena keluarannya tidak identik antar-eksekusi.",
         size=15, color=MUTED, lead=20)


def a3(c):
    base_slide(c, "Lampiran A3", "Peringkat zona tidak berubah oleh asumsi skenario listrik", appendix=True)
    rows = [["Porsi gas untuk pemanasan", "COP 2,5", "COP 3", "COP 3,5"],
            ["70%", "+100%", "+83%", "+71%"],
            ["80%", "+114%", "+95%", "+81%"],
            ["90%", "+128%", "+107%", "+92%"]]
    y = table(c, MX, 150, [300, 160, 160, 160], rows, size=18, pad=11)
    bullets(c, MX, y + 24, 860, [
        "Rumus: gas × porsi pemanasan / COP / listrik; semua zona dikalikan faktor yang sama.",
        "Efisiensi boiler lama sekitar 88% menurunkan semua nilai sekitar 12% (95% menjadi 84%).",
        "Energi tahunan, bukan beban puncak; kapasitas jaringan tidak ada di data.",
    ], size=18, gap=10)


def a4(c):
    base_slide(c, "Lampiran A4", "Adopsi di subzona baru dihitung dari pecahan komposisi pemanas", appendix=True)
    bullets(c, MX, 150, 530, [
        "N = bilangan bulat terkecil yang mengubah semua pecahan menjadi bulat: jumlah rumah tangga subzona.",
        "K = N × (100 − porsi tercatat) / 100: rumah tangga di luar kategori tercatat.",
        "LightGBM binomial cross-entropy berbobot K memprediksi q; adopsi = 100 × K × (1 − q) / N.",
        "Dirata-rata dengan regresi langsung atas porsi nonlistrik.",
    ], size=17, gap=9)
    rect(c, 620, 150, 292, 150, GREEN, r=16)
    c.setFillColor(MUSTARD)
    c.setFont("B", 38)
    c.drawString(638, H - 210, "1,03 pp")
    c.setFillColor("#FFFFFF")
    c.setFont("R", 15)
    c.drawString(638, H - 238, "RMSE (0,143 ternormalisasi)")
    c.drawString(638, H - 262, "8,1% kawasan uji di subzona baru")
    para(c, MX, 474, 860, "Validasi lima fold per subzona, dibobot ulang menurut komposisi subzona baru.", size=14, color=MUTED, lead=19)


def a5(c):
    base_slide(c, "Lampiran A5", "Audit akhir: dua masalah nyata dikoreksi, banyak ide lain tidak berguna", appendix=True)
    bullets(c, MX, 148, 430, [
        "Kawasan 2 sampai 3 meter: prediksi sekitar 20% terlalu rendah; 4 meter: 3% sampai 7%. Hanya 877 kawasan; dikoreksi faktor pengali.",
        "Gas mendekati batas 999.765 kWh: taksiran berlebih, 7 prediksi melampaui maksimum; koreksi normal terpancung, lalu dipotong.",
        "Skor publik membaik 0,00007; 77 kawasan uji kembar disalin dari pasangannya.",
    ], size=16, gap=9)
    rect(c, 520, 148, 392, 300, CARD, LINE, 16)
    c.setFillColor(GREEN)
    c.setFont("B", 16)
    c.drawString(538, H - 178, "Tidak memperbaiki skor")
    bullets(c, 538, 192, 360, [
        "stacking lapis kedua",
        "ensemble khusus subzona baru",
        "koreksi galat nonlinear",
        "validasi silang 20 fold",
        "penelusuran urutan ID, gas kembar, duplikat longgar",
    ], size=16, gap=7)
    para(c, MX, 470, 860, "Pembobotan NNLS adaptif: +0,00009 skor leaderboard. Poisson: −0,0039 galat EUI.", size=13, color=MUTED, lead=17)


def a6(c):
    base_slide(c, "Lampiran A6", "Kontribusi SHAP per kelompok fitur dan zona kandidat", appendix=True)
    rows = [["Target", "Sosiodemografi", "Rasio fisik", "Periode bangunan", "Dimensi bangunan", "Iklim", "Pemanas"],
            ["EUI", "30%", "25%", "11%", "11%", "9%", "3%"],
            ["Gas per meter", "28%", "30%", "11%", "11%", "8%", "3%"],
            ["Listrik per meter", "40%", "26%", "5%", "11%", "4%", "5%"],
            ["Adopsi", "15%", "0%", "0%", "0%", "1%", "81%"]]
    y = table(c, MX, 148, [170, 130, 110, 130, 130, 90, 94], rows, size=15, pad=8)
    bullets(c, MX, y + 18, 860, [
        "Angka 81% pada adopsi sebagian besar identitas akuntansi: dibaca sebagai deskripsi komposisi, bukan pendorong.",
        "Kandidat pertama retrofit: REG_273, REG_270, REG_045 (potensi 6,7% sampai 8,6% gas zona).",
        "Perlu penguatan gardu: REG_290 (+121%), REG_259, REG_275.",
    ], size=17, gap=9)


def a7(c):
    base_slide(c, "Lampiran A7", "Data, metrik, dan batas yang dipakai", appendix=True)
    rows = [["Hal", "Nilai"],
            ["Data", "429 ribu kawasan latih, 184 ribu kawasan uji, 201 kolom fitur, 330 zona"],
            ["Metrik", "rata-rata RMSE/σ; σ = 151.750,04; 41.725,37; 40,65; 12,49; 7,18"],
            ["Perbaikan ensemble", "EUI 0,0083; gas 0,0036; listrik 0,0027 (holdout v16)"],
            ["Potensi retrofit", "7,4 TWh dari 141,4 TWh gas; 30% nilai zona berasal dari prediksi kawasan uji"],
            ["Komputasi", "RTX A4000 16 GB, Python 3.12, sekitar 20 jam pelatihan"]]
    y = table(c, MX, 148, [210, 654], rows, size=17, pad=10)
    para(c, MX, y + 16, 860, "Nama kolom disamarkan panitia; makna kolom adalah dugaan dari pola data.", size=15, color=MUTED, lead=20)


SLIDES = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, a1, a2, a3, a4, a5, a6, a7]


def main():
    c = canvas.Canvas(OUT, pagesize=(W, H), invariant=1)
    c.setTitle("Pitch Deck BISMILLAH, DICE FEST 2026 (v1)")
    c.setAuthor("Tim BISMILLAH")
    for fn in SLIDES:
        fn(c)
        c.showPage()
    c.save()
    print("wrote", OUT, len(SLIDES), "pages")


if __name__ == "__main__":
    main()
