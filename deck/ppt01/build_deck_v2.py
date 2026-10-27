"""Final deck v2 (deck/ppt01): 14 main slides + 8 backup slides for the DICE FEST 2026 final (10 minutes, Zoom).

One content source writes four outputs:
  BISMILLAH_Final_v2.pptx         editable deck with speaker notes (Calibri)
  pratinjau/slide-XX.png + BISMILLAH_Final_v2_pratinjau.pdf
                                  preview rendered with Carlito (same metrics as Calibri) for visual QA
  05_struktur_deck_v2.md          slide-by-slide brief in the format read by check_deck.py
  07_kartu_latihan.md             rehearsal card (timing, first sentence, key numbers)

Numbers: notebook outputs (cell numbers on the slides), deck/ppt01/fakta_tambahan.md for everything else.
usage: <venv with python-pptx and Pillow>/bin/python deck/ppt01/build_deck_v2.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deckkit import SW, check_fit, render, text_width_in, write_pptx  # noqa: E402

IMG = os.path.join(HERE, "img")
OUT_PPTX = os.path.join(HERE, "BISMILLAH_Final_v2.pptx")
PREV = os.path.join(HERE, "pratinjau")
M = 0.6  # left / right margin
CW = SW - 2 * M
WPS = 130 / 60  # speaking rate used for the timing (words per second)


def load_json(name):
    p = os.path.join(HERE, "bukti", name)
    return json.load(open(p)) if os.path.exists(p) else None


LEAK = load_json("uji_kebocoran_notebook.json")
LEAK2 = load_json("uji_kebocoran_oof_dalam.json")  # training rows: labels of one inner part shuffled
VC = load_json("skor_jujur_bobot_adaptif.json")
CEK = load_json("cek_kebijakan.json")  # policy checks after the independent review


def idn(x, nd=0):
    s = f"{x:,.{nd}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


if LEAK:
    LEAK_VAL = LEAK["blok"]["knn_validasi"]["baris_berubah"] + LEAK["blok"]["te_validasi"]["baris_berubah"]
    LEAK_CTRL = LEAK["blok"]["kontrol_bocor_knn_validasi"]["baris_berubah"]
    LEAK_OK = LEAK["lulus"]
    LEAK_NVAL = LEAK["baris_validasi"]
else:
    LEAK_VAL, LEAK_CTRL, LEAK_OK, LEAK_NVAL = None, None, None, 42931
if LEAK2:
    LEAK2_CHANGED = sum(v["berubah_bagian_0"] for v in LEAK2["blok"].values())
    LEAK2_N = next(iter(LEAK2["blok"].values()))["baris_bagian_0"]
    LEAK2_OK = LEAK2["lulus"]
else:
    LEAK2_CHANGED, LEAK2_N, LEAK2_OK = None, 77275, None

# ---------------------------------------------------------------------------------------------------------------
# building blocks
CHIP = {"FAKTA": ("NAVY_T", "NAVY"), "HASIL": ("TEAL_T", "TEAL_D"), "ASUMSI": ("AMBER_T", "AMBER_D"),
        "INTERPRETASI": ("VIOLET_T", "VIOLET")}


def T(x, y, w, h, p, size=18, color="INK", bold=False, align="l", valign="t", lsp=1.0, after=0, role="body",
      italic=False, spacing=None):
    return dict(t="text", x=x, y=y, w=w, h=h, p=p if isinstance(p, list) else [p], size=size, color=color,
                bold=bold, align=align, valign=valign, lsp=lsp, after=after, role=role, italic=italic,
                spacing=spacing)


def R(x, y, w, h, fill=None, line=None, lw=1.0, r=None):
    return dict(t="rect", x=x, y=y, w=w, h=h, fill=fill, line=line, lw=lw, r=r)


def O(x, y, w, h, fill=None, line=None):
    return dict(t="oval", x=x, y=y, w=w, h=h, fill=fill, line=line)


def L(x1, y1, x2, y2, color="MUTED", lw=1.5, arrow=False):
    return dict(t="line", x1=x1, y1=y1, x2=x2, y2=y2, color=color, lw=lw, arrow=arrow)


def I(name, x, y, w, alt, h=None):
    from PIL import Image
    path = os.path.join(IMG, name)
    iw, ih = Image.open(path).size
    return dict(t="img", path=path, x=x, y=y, w=w, h=h if h else w * ih / iw, alt=alt)


def chip(x, y, kind, text, size=12, maxw=None, fill=None):
    """evidence label: FAKTA / HASIL / ASUMSI / INTERPRETASI + source; returns elements (pill + text)"""
    fill0, ink = CHIP[kind]
    fill = fill or fill0
    label = f"**{kind}** · {text}"
    w = (text_width_in(kind, size, bold=True) + text_width_in(f" · {text}", size)) * 1.03 + 0.3
    if maxw and w > maxw:  # wrap inside a taller pill
        lines = 2
        w = maxw
    else:
        lines = 1
    h = 0.32 if lines == 1 else 0.56
    return [R(x, y, w, h, fill=fill, r=0.08),
            T(x + 0.13, y + 0.04, w - 0.24, h - 0.08, label, size=size, color=ink, valign="m", role="sumber")]


def frame(n, kicker, title, istilah=None, dark=False):
    """kicker, action title, glossary line and footer of a content slide"""
    ink, mut = ("WHITE", "LIGHT") if dark else ("INK", "MUTED")
    el = [T(M, 0.38, 9.5, 0.3, kicker, size=12, color="TEAL" if not dark else "LIGHT", bold=True, role="kicker",
            spacing=1.2),
          T(M, 0.68, CW, 1.05, title, size=28, color=ink, bold=True, lsp=0.95, role="title")]
    if istilah:
        el.append(T(M, 6.6, CW, 0.4, f"**Istilah.** {istilah}", size=12, color=mut, role="istilah", lsp=0.95))
    el.append(T(M, 7.08, 6, 0.25, "Tim BISMILLAH · DICE FEST 2026 · Babak final", size=10, color=mut,
                role="footer"))
    el.append(T(SW - M - 1.5, 7.08, 1.5, 0.25, str(n), size=10, color=mut, align="r", role="footer"))
    return el


def stat(x, y, w, big, label, color="AMBER", big_size=40, label_size=15, label_h=0.9):
    return [T(x, y, w, big_size / 72 * 1.25, big, size=big_size, color=color, bold=True, role="angka", lsp=0.9),
            T(x, y + big_size / 72 * 1.22, w, label_h, label, size=label_size, color="INK", lsp=0.98)]


def action_card(x, y, w, rows, label_w=1.55, gap=0.14, size=15):
    """policy action card: label column + text column, one row per item"""
    el, yy = [], y
    for label, text, h in rows:
        el.append(T(x, yy + 0.02, label_w, 0.3, label.upper(), size=11, color="MUTED", bold=True, spacing=0.8,
                    role="label"))
        el.append(T(x + label_w, yy, w - label_w, h, text, size=size, lsp=0.98))
        yy += h + gap
        el.append(L(x, yy - gap / 2, x + w, yy - gap / 2, color="LINE", lw=0.75))
    return el[:-1], yy


SLIDES = []


def slide(id_, short, section, rubric, dur, title, el, notes, istilah=None, sumber=None, visual=None, bg=None):
    SLIDES.append(dict(id=id_, short=short, section=section, rubric=rubric, dur=dur, title=title, el=el,
                       notes=notes, istilah=istilah, sumber=sumber, visual=visual, bg=bg))


# ---------------------------------------------------------------------------------------------------------------
# 1. key message (dark)
el = [T(M, 0.55, CW, 0.3, "DICE FEST 2026 · BABAK FINAL · TIM BISMILLAH", size=13, color="LIGHT", bold=True,
        spacing=1.5, role="kicker"),
      T(M, 1.0, 11.2, 1.75, "Dari energi 613 ribu kawasan ke urutan program untuk 330 zona", size=40,
        color="WHITE", bold=True, lsp=0.92, role="title"),
      T(M, 2.85, 11.0, 0.4, "Trilema transisi energi perkotaan: emisi, keandalan jaringan, keterjangkauan", size=17,
        color="LIGHT")]
cards = [("1", "*Retrofit* dulu di 118 zona Q1 (47% konsumsi gas)"),
         ("2", "Gardu dinilai sebelum subsidi pompa panas"),
         ("3", "Pilot pompa panas di kawasan bergas, mulai zona Q2")]
cw = (CW - 2 * 0.3) / 3
for k, (num, txt) in enumerate(cards):
    x = M + k * (cw + 0.3)
    el += [R(x, 3.55, cw, 1.45, fill="NAVY2", r=0.1), O(x + 0.22, 3.8, 0.5, 0.5, fill="AMBER"),
           T(x + 0.22, 3.8, 0.5, 0.5, num, size=20, color="WHITE", bold=True, align="c", valign="m", role="angka"),
           T(x + 0.9, 3.72, cw - 1.1, 1.15, txt, size=18, color="WHITE", valign="m", lsp=0.98)]
stats = [("0,24844", "skor *leaderboard* privat, peringkat 8"),
         ("Lulus uji label acak", "fitur bertarget, baris validasi dan latih"),
         ("Identik byte", "hasil sama setiap diulang di perangkat sama")]
for k, (big, lab) in enumerate(stats):
    x = M + k * (cw + 0.3)
    el += [T(x, 5.3, cw, 0.5, big, size=26, color="AMBER" if k == 0 else "WHITE", bold=True, role="angka"),
           T(x, 5.82, cw, 0.5, lab, size=14, color="LIGHT")]
el += [T(M, 6.6, CW, 0.4, "**Istilah.** *Retrofit*: perbaikan dinding, atap, dan jendela agar rumah tidak boros "
         "panas. Pompa panas: pemanas listrik yang menghasilkan sekitar 3 kWh panas dari 1 kWh listrik.", size=12,
         color="LIGHT", role="istilah"),
       T(M, 7.08, 6, 0.25, "Tim BISMILLAH · DICE FEST 2026 · Babak final", size=10, color="LIGHT", role="footer"),
       T(SW - M - 1.5, 7.08, 1.5, 0.25, "1", size=10, color="LIGHT", align="r", role="footer")]
slide("1", "Pesan utama", "Pembuka", "Pembuka", 30,
      "Dari energi 613 ribu kawasan ke urutan program untuk 330 zona", el,
      "Selamat pagi, Bapak dan Ibu juri. Kami tim BISMILLAH, dan kami mulai dari kesimpulan. Dari lima besaran energi "
      "untuk 613 ribu kawasan, 184 ribu di antaranya kami prediksi, kami menyusun urutan program untuk 330 zona: "
      "retrofit dulu di zona yang boros gas, kapasitas gardu dinilai sebelum subsidi pompa panas, dan pilot pompa panas "
      "di kawasan bergas. Model di baliknya meraih peringkat 8 di leaderboard privat dan lulus uji kebocoran.",
      istilah="*Retrofit*, pompa panas", sumber="sel 0, 25, 49, 50; Kaggle LB privat; bukti/uji_kebocoran_*.json",
      visual="Slide gelap: judul, tiga kartu rekomendasi bernomor, tiga angka kredibilitas.", bg="NAVY")

# 2. why the problem matters
el = frame(2, "MASALAH", "Elektrifikasi penuh hampir menggandakan konsumsi listrik zona, jadi urutan program menentukan "
           "kesiapan jaringan",
           "COP (*coefficient of performance*): efisiensi pompa panas; COP 3 berarti 1 kWh listrik menghasilkan 3 kWh "
           "panas. Median: nilai tengah dari 330 zona.")
el.append(T(M, 1.95, 6.6, 0.4, "Tiga tuntutan transisi, diukur oleh lima target lomba", size=17, bold=True))
rows = [("TEAL", "Emisi turun", "gas total; adopsi pemanas listrik"),
        ("NAVY", "Jaringan listrik tetap andal", "listrik total"),
        ("AMBER", "Biaya tetap terjangkau", "intensitas gas per m² (EUI); porsi bangunan bersih")]
for k, (c, head, tg) in enumerate(rows):
    y = 2.45 + k * 1.0
    el += [R(M, y, 6.6, 0.86, fill="SOFT", r=0.08), O(M + 0.22, y + 0.23, 0.4, 0.4, fill=c),
           T(M + 0.85, y + 0.08, 5.6, 0.36, head, size=18, bold=True),
           T(M + 0.85, y + 0.45, 5.6, 0.36, tg, size=15, color="MUTED")]
el += chip(M, 5.45, "INTERPRETASI", "pemetaan target ke tuntutan oleh tim")
el += [R(7.6, 1.95, 5.13, 2.8, fill="AMBER_T", r=0.1),
       T(7.9, 2.1, 4.6, 1.02, "+94,85%", size=58, color="AMBER", bold=True, role="angka"),
       T(7.9, 3.08, 4.6, 1.0, "median kenaikan konsumsi listrik tahunan zona bila semua pemanas gas diganti pompa panas",
         size=16, lsp=0.98)]
el += chip(7.9, 4.18, "ASUMSI", "80% gas untuk pemanasan, COP 3 · sel 50", maxw=4.6, fill="WHITE")
el += [T(7.6, 4.97, 2.4, 0.6, "141,35 TWh", size=28, color="NAVY", bold=True, role="angka"),
       T(10.05, 4.97, 2.68, 0.7, "konsumsi gas 330 zona (sel 50)", size=14, color="MUTED", lsp=0.98)]
el += [R(M, 5.98, CW, 0.46, fill="NAVY_T", r=0.06),
       T(M + 0.2, 5.98, CW - 0.4, 0.46, "**Pertanyaan kebijakan:** zona mana di-*retrofit* dulu, mana siap "
         "dielektrifikasi, dan mana yang jaringannya disiapkan lebih dulu?", size=15, color="NAVY", valign="m")]
slide("2", "Masalah", "Masalah", "Pembuka", 35,
      "Elektrifikasi penuh hampir menggandakan konsumsi listrik zona, jadi urutan program menentukan kesiapan jaringan",
      el,
      "Mengapa masalah ini layak diselesaikan? Kota yang beralih dari pemanas gas ke pemanas listrik harus memenuhi "
      "tiga tuntutan sekaligus: emisi turun, jaringan listrik tetap andal, dan biaya tetap terjangkau. Kelima target "
      "lomba mengukur ketiganya. Hasil kami memperlihatkan risikonya. Bila seluruh pemanas gas diganti pompa panas, "
      "konsumsi listrik tahunan zona naik dengan median hampir 95 persen. Jadi pertanyaannya bukan apakah kota "
      "beralih, tetapi zona mana yang didahulukan dan dengan program apa.",
      istilah="COP, median", sumber="sel 49, 50 (In [34], In [35])",
      visual="Tiga baris tuntutan dan target di kiri; angka besar +94,85% dengan label asumsi di kanan; kotak "
             "pertanyaan kebijakan di bawah.")

# 3. data structure
el = frame(3, "DATA DAN FITUR · PRA-PEMROSESAN", "Struktur data menjawab dua target sebelum model dilatih",
           "EUI: konsumsi gas per m² lantai. Subzona: wilayah kecil setingkat unit sensus; satu zona memuat banyak "
           "subzona.")
cards = [("Gas = EUI × luas lantai", "luas lantai = bld_dim_06", "6,7 × 10{^|−16}",
          "galat relatif maksimum", "Satu keluarga model untuk gas dan EUI"),
         ("Porsi bersih = rumus", "100 × bld_dim_02 / bld_dim_01", "100%",
          "baris latih tepat sama", "Dihitung, tidak ditebak"),
         ("Adopsi seragam per subzona", "di 100% subzona data latih", "91,86%",
          "kawasan uji di subzona yang dikenal", "Disalin; model hanya untuk 8,14% sisanya")]
cw = (CW - 2 * 0.3) / 3
for k, (head, sub, big, lab, imp) in enumerate(cards):
    x = M + k * (cw + 0.3)
    el += [R(x, 1.95, cw, 3.9, fill="SOFT", r=0.1),
           T(x + 0.25, 2.12, cw - 0.5, 0.66, head, size=19, bold=True, lsp=0.95),
           T(x + 0.25, 2.8, cw - 0.5, 0.62, sub.replace("`", ""), size=14, color="MUTED", lsp=0.98),
           T(x + 0.25, 3.42, cw - 0.5, 0.75, big, size=40, color="TEAL", bold=True, role="angka"),
           T(x + 0.25, 4.15, cw - 0.5, 0.4, lab, size=15),
           L(x + 0.25, 4.66, x + cw - 0.25, 4.66, color="LINE", lw=1),
           T(x + 0.25, 4.8, cw - 0.5, 0.95, f"{{N|{imp}}}", size=17, lsp=0.98)]
el += chip(M, 6.02, "FAKTA", "pemeriksaan identitas di sel 25 (In [22]), seluruh baris latih")
slide("3", "Struktur data", "Data dan fitur", "Rekayasa fitur dan pra-pemrosesan (25)", 35,
      "Struktur data menjawab dua target sebelum model dilatih", el,
      "Jawaban pertama datang dari struktur data, sebelum model dilatih. Pemeriksaan di sel 25 menemukan tiga hal. Gas "
      "selalu sama dengan EUI dikali luas lantai, dengan galat setara presisi mesin, jadi satu keluarga model cukup "
      "untuk dua kolom. Porsi bangunan bersih tepat sama dengan satu rumus di seluruh baris latih, jadi dihitung, "
      "bukan ditebak. Adopsi pemanas listrik seragam dalam satu subzona, dan 91,86 persen kawasan uji berada di "
      "subzona yang sudah dikenal. Model besar hanya dibutuhkan untuk gas, EUI, dan listrik.",
      istilah="EUI, subzona", sumber="sel 25 (In [22])", visual="Tiga kartu temuan, masing-masing dengan angka bukti "
                                                              "dan implikasi desain.")

# 4. area_per_meter
el = frame(4, "DATA DAN FITUR · FITUR FISIK", "Luas lantai per meter, proksi ukuran rumah, paling berpengaruh di "
           "model penjelas", "SHAP: pembagian setiap tebakan model menjadi sumbangan tiap variabel. Model penjelas: satu "
                          "LightGBM per target yang hanya memakai karakteristik kawasan terukur.")
el += [R(M, 1.95, 4.75, 4.3, fill="SOFT", r=0.1),
       T(M + 0.25, 2.12, 4.3, 0.35, "RUMUS", size=11, color="MUTED", bold=True, spacing=0.8, role="label"),
       T(M + 0.25, 2.42, 4.3, 0.62, ["area_per_meter =", "bld_dim_06 / bld_dim_12"], size=17, color="NAVY",
         bold=True, lsp=0.95),
       T(M + 0.25, 3.1, 4.3, 0.62, "luas lantai dibagi jumlah meter, kira-kira luas rata-rata satu rumah", size=15,
         color="MUTED", lsp=0.98),
       T(M + 0.25, 3.82, 4.3, 1.4, [dict(text="dihitung per baris dari dua kolom input", bullet=True, after=4),
                                    dict(text="tanpa target dan tanpa statistik lintas baris", bullet=True, after=4),
                                    dict(text="rumus sama untuk data latih dan data uji", bullet=True)], size=15),
       T(M + 0.25, 5.15, 4.3, 0.45, "{N|Tidak dapat membocorkan target}", size=18, lsp=0.98)]
el += chip(M + 0.25, 5.72, "FAKTA", "sel 28, bagian 2.1", fill="WHITE")
el.append(I("s04_shap_top.png", 5.65, 1.9, 7.08, "Lima fitur teratas menurut rata-rata |SHAP| untuk log EUI dan log "
                                                 "listrik per meter; area_per_meter peringkat 1 di keduanya"))
el += [T(5.75, 5.6, 4.2, 0.45, "{A|2,8× dan 3,0×} fitur peringkat kedua", size=17, valign="m")]
el += chip(5.75, 6.12, "HASIL", "model penjelas LightGBM · sel 45 (In [32])")
slide("4", "Fitur ukuran rumah", "Data dan fitur", "Rekayasa fitur dan pra-pemrosesan (25)", 35,
      "Luas lantai per meter, proksi ukuran rumah, paling berpengaruh di model penjelas", el,
      "Fitur fisik terkuat kami sangat sederhana: luas lantai dibagi jumlah meteran, kira-kira luas rata-rata satu rumah. "
      "Gas total sebuah kawasan terutama ditentukan oleh banyaknya rumah, sedangkan ukuran rumah menentukan cara "
      "energi dipakai. Pada model penjelas, fitur ini peringkat pertama untuk intensitas gas dan untuk listrik per "
      "rumah, hampir tiga kali fitur kedua. Fitur ini tidak mungkin membocorkan jawaban, karena hanya membagi dua kolom "
      "input yang juga tersedia di data uji, baris demi baris.",
      istilah="SHAP, model penjelas", sumber="sel 28, sel 45 (In [32])",
      visual="Kartu rumus di kiri; dua diagram batang lima fitur teratas (img/s04_shap_top.png) di kanan.")

# 5. kNN out-of-fold, with the label-permutation test
el = frame(5, "DATA DAN FITUR · FITUR SPASIAL", "Kawasan termirip memakai target tetangga tanpa pernah melihat target "
           "baris yang dinilai", "*Out-of-fold*: nilai sebuah baris dihitung hanya dari data yang tidak memuat baris "
                                 "itu. *Fold*: satu dari sepuluh potongan data latih. kNN: rata-rata k kawasan "
                                 "paling mirip.")
x0, y0, wd = M, 1.95, 6.75
el.append(T(x0, y0, wd, 0.35, "{T|1} Sepuluh *fold*: satu dinilai, sembilan untuk mencari tetangga", size=16,
            bold=True))
bw, gp = 0.6, (wd - 10 * 0.6) / 9
for k in range(10):
    x = x0 + k * (bw + gp)
    hot = k == 2
    el += [R(x, y0 + 0.45, bw, 0.5, fill="AMBER" if hot else "NAVY_T", r=0.05),
           T(x, y0 + 0.45, bw, 0.5, str(k + 1), size=13, color="WHITE" if hot else "NAVY", bold=hot, align="c",
             valign="m", role="diagram")]
el.append(T(x0, y0 + 1.55, wd, 0.35, "{T|2} Baris latih: bagian latih dibagi 5, tetangga dari 4 bagian lain", size=16,
            bold=True))
bw5, gp5 = 1.2, (wd - 5 * 1.2) / 4
for k in range(5):
    x = x0 + k * (bw5 + gp5)
    hot = k == 0
    el += [R(x, y0 + 2.0, bw5, 0.55, fill="AMBER" if hot else "NAVY_T", r=0.05),
           T(x, y0 + 2.0, bw5, 0.55, "dihitung" if hot else f"bagian {k + 1}", size=13,
             color="WHITE" if hot else "NAVY", bold=hot, align="c", valign="m", role="diagram")]
    if not hot:
        el.append(L(x + bw5 / 2, y0 + 2.55, x + bw5 / 2, y0 + 2.75, color="MUTED", lw=1.25))
el.append(L(x0 + 4 * (bw5 + gp5) + bw5 / 2, y0 + 2.75, x0 + bw5 / 2 + 0.05, y0 + 2.75, color="MUTED", lw=1.25))
el.append(L(x0 + bw5 / 2, y0 + 2.75, x0 + bw5 / 2, y0 + 2.58, color="MUTED", lw=1.25, arrow=True))
el.append(T(x0, y0 + 2.9, wd, 0.35, "{T|3} Data uji: tetangga dari bagian latih model yang sama", size=16, bold=True))
el += [R(x0, y0 + 3.4, wd, 0.8, fill="SOFT", r=0.08),
       T(x0 + 0.2, y0 + 3.4, wd - 0.4, 0.8, "Jendela ±30 sampai ±300 nomor subzona: {N|99,07%} baris berurutan "
         "berada di zona yang sama (sel 54)", size=15, valign="m", lsp=0.98)]
px, pw = 7.75, SW - M - 7.75
el += [R(px, 1.95, pw, 4.55, fill="TEAL_T", r=0.1),
       T(px + 0.25, 2.1, pw - 0.5, 0.4, "Bukti: uji label acak", size=18, bold=True, color="TEAL_D")]
leak_big = (f"{idn(LEAK_VAL + LEAK2_CHANGED)} baris berubah" if LEAK_VAL is not None and LEAK2_CHANGED is not None
            else "? baris berubah")
el += [T(px + 0.25, 2.55, pw - 0.5, 0.6, leak_big, size=30, color="TEAL", bold=True, role="angka"),
       T(px + 0.25, 3.15, pw - 0.5, 0.95, f"setelah label {idn(LEAK_NVAL)} baris validasi diacak, dan label "
         f"{idn(LEAK2_N)} baris latih satu bagian dalam; fitur uji juga identik", size=15, lsp=0.98),
       L(px + 0.25, 4.15, px + pw - 0.25, 4.15, color="LINE", lw=1),
       T(px + 0.25, 4.25, pw - 0.5, 0.45, (f"{idn(LEAK_CTRL)} baris berubah" if LEAK_CTRL is not None else "?")
         + " pada kontrol", size=20, color="AMBER", bold=True, role="angka"),
       T(px + 0.25, 4.7, pw - 0.5, 0.75, "versi yang sengaja dibuat bocor: uji ini mampu mendeteksi kebocoran",
         size=15, lsp=0.98),
       T(px + 0.25, 5.55, pw - 0.5, 0.4, "Galat EUI model tunggal turun {T|0,0022} (sel 28)", size=15)]
el += chip(px, 6.0, "HASIL", "bukti/uji_kebocoran_notebook.json dan uji_kebocoran_oof_dalam.json", maxw=pw)
slide("5", "Fitur kawasan termirip", "Data dan fitur", "Rekayasa fitur dan pra-pemrosesan (25)", 45,
      "Kawasan termirip memakai target tetangga tanpa pernah melihat target baris yang dinilai", el,
      "Fitur kedua bertanya: berapa konsumsi kawasan lain yang stok bangunannya paling mirip? Pencarian dibatasi pada "
      "jendela nomor subzona, karena nomor subzona ternyata berurutan secara geografis: 99,07 persen baris berurutan "
      "berada di zona yang sama. Fitur ini memakai target tetangga, jadi kami menghitungnya out-of-fold dua lapis, "
      "seperti skema di kiri. Lalu kami mengujinya: label baris validasi kami acak, lalu label satu bagian dalam baris "
      "latih. Fitur yang tidak boleh membaca label itu tetap identik sampai bit terakhir, sedangkan versi yang sengaja "
      "dibuat bocor langsung berubah. Jadi uji ini mampu menangkap kebocoran, dan fitur kami lolos.",
      istilah="*out-of-fold*, *fold*, kNN", sumber="sel 28, sel 54 (In [36]), bukti/uji_kebocoran_*.json",
      visual="Skema *cross-fit* dua lapis (bentuk asli, dapat disunting) di kiri; panel hasil uji label acak di kanan.")

# 6. architecture
el = frame(6, "MODEL · ARSITEKTUR MULTI-OUTPUT", "Lima target, tiga jalur: model besar hanya untuk gas, EUI, dan "
           "listrik", "Galat jujur: galat pada data yang tidak dipakai menyusun bobot gabungan. RMSE/σ: galat dibagi "
                      "sebaran target; 1 berarti setara menebak rata-rata.")
el += [R(M, 3.15, 1.6, 1.3, fill="SOFT", r=0.08),
       T(M + 0.1, 3.15, 1.4, 1.3, ["Data latih + uji", "{m|201 kolom}"], size=15, bold=True, align="c", valign="m")]
lanes = [(1.95, 1.15, "AMBER_T", "Jalur 1 · Gas dan EUI", "28 model: XGBoost, LightGBM, MLP, NCA"),
         (3.25, 1.15, "TEAL_T", "Jalur 2 · Listrik", "13 model, pendorong berbeda dari gas"),
         (4.6, 1.3, "NAVY_T", "Jalur 3 · Tanpa model besar", "porsi bersih: rumus; adopsi: salin subzona (91,86%) "
                                                             "+ 2 LightGBM (8,14%)")]
lx, lw_ = 2.65, 3.75
for y, h, fill, head, sub in lanes:
    el += [R(lx, y, lw_, h, fill=fill, r=0.08),
           T(lx + 0.2, y + 0.12, lw_ - 0.4, 0.38, head, size=17, bold=True),
           T(lx + 0.2, y + 0.5, lw_ - 0.4, h - 0.55, sub, size=14, color="MUTED", lsp=0.98),
           L(M + 1.6, 3.8, lx, y + h / 2, color="MUTED", lw=1.25, arrow=True)]
bx, bw_ = 6.8, 2.85
el.append(R(bx, 1.95, bw_, 2.45, fill="WHITE", line="LINE", lw=1, r=0.08))
for k, txt in enumerate(["Gabungan bobot adaptif", "Koreksi galat wilayah", "Penyesuaian akhir"]):
    y = 2.1 + k * 0.78
    el += [R(bx + 0.18, y, bw_ - 0.36, 0.55, fill="NAVY", r=0.06),
           T(bx + 0.18, y, bw_ - 0.36, 0.55, txt, size=14, color="WHITE", bold=True, align="c", valign="m",
             role="diagram")]
    if k < 2:
        el.append(L(bx + bw_ / 2, y + 0.55, bx + bw_ / 2, y + 0.78, color="MUTED", lw=1.25, arrow=True))
el += [L(lx + lw_, 2.52, bx, 2.52, color="MUTED", lw=1.25, arrow=True),
       L(lx + lw_, 3.82, bx, 3.82, color="MUTED", lw=1.25, arrow=True),
       L(lx + lw_, 5.25, 10.0, 5.25, color="MUTED", lw=1.25, arrow=True)]
ox = 10.1
el.append(T(ox, 1.55, SW - M - ox, 0.4, "GALAT JUJUR SEBELUM KOREKSI (RMSE/σ)", size=11, color="MUTED", bold=True,
            spacing=0.8, role="label"))
outs = [(2.0, "Gas", "0,2449"), (2.55, "EUI", "0,6502"), (3.55, "Listrik", "0,3159"), (4.75, "Porsi bersih", "0"),
        (5.3, "Adopsi", "≈ 0,04")]
for y, name, val in outs:
    el += [T(ox, y, 1.45, 0.45, name, size=16, valign="m"),
           T(ox + 1.45, y, SW - M - ox - 1.45, 0.45, val, size=20, color="NAVY", bold=True, align="r", valign="m",
             role="angka")]
el += [L(bx + bw_, 2.4, ox - 0.05, 2.4, color="MUTED", lw=1.25, arrow=True),
       L(bx + bw_, 3.77, ox - 0.05, 3.77, color="MUTED", lw=1.25, arrow=True)]
el += chip(M, 6.02, "FAKTA", "notebook 56 sel, 17 modul diciutkan, sakelar DICEFEST_TRAIN; submisi identik byte di "
                             "perangkat sama, sha256 aec745ab…d66e (sel 35); galat jujur dari sel 39", maxw=CW)
slide("6", "Arsitektur tiga jalur", "Model", "Arsitektur model (15)", 50,
      "Lima target, tiga jalur: model besar hanya untuk gas, EUI, dan listrik", el,
      "Karena metrik menilai setiap kolom terpisah, setiap target boleh "
      "memakai cara terbaiknya. Jalur pertama untuk gas dan EUI: 28 model, karena keduanya satu besaran. Jalur kedua "
      "untuk listrik: 13 model, karena pendorongnya berbeda. Jalur ketiga hampir tanpa model: porsi bangunan bersih "
      "dihitung dengan rumus, adopsi disalin dari subzonanya, dan dua LightGBM hanya untuk 8 persen kawasan di subzona "
      "baru. Model di dua jalur pertama beragam, lalu digabung, dikoreksi, dan disesuaikan dengan batas data. Angka di "
      "kanan adalah galat jujur gabungan sebelum koreksi; koreksi dan penyesuaian menurunkan skor lebih jauh. Notebook-"
      "nya 56 sel dengan satu sakelar, dan hasilnya identik byte setiap diulang di perangkat yang sama.",
      istilah="galat jujur, RMSE/σ", sumber="sel 23, 31, 35, 39",
      visual="Diagram alur bentuk asli: data, tiga jalur, tiga tahap setelah penggabungan, kolom galat jujur.")

# 7. generalisation evidence
el = frame(7, "MODEL · VALIDASI", "Skor privat dari data yang tidak pernah terlihat sejalan dengan skor publik, "
           "tanda tidak *overfit*", "*Holdout set*: 30% data latih yang disisihkan untuk menguji setiap keputusan. AUC 0,5: model "
                     "tidak lebih baik dari menebak acak.")
el.append(I("s07_lb.png", M, 1.95, 7.2, "Skor leaderboard publik per langkah pengembangan, dari 0,25025 ke 0,24913, dan "
                                       "skor privat 0,24844"))
el += chip(M, 6.12, "FAKTA", "LB publik per langkah: sel 37; LB privat: Kaggle, 24 Okt 2026")
ex = [("0,4997", "TEAL", "AUC validasi adversarial: data latih dan uji tidak dapat dibedakan, jadi validasi acak "
                         "sepuluh *fold* meniru data uji", "sel 55"),
      ("8,14% vs 8,17%", "TEAL", "kawasan uji di subzona baru: teramati vs harapan pengacakan per baris", "sel 54"),
      ("0,24844 vs 0,24913", "AMBER", "privat vs publik: tidak memburuk. Komponen diterima di *holdout set*; "
                                      "*leaderboard* hanya mengonfirmasi arah", "Kaggle")]
rx, rw = 8.15, SW - M - 8.15
for k, (big, c, txt, src) in enumerate(ex):
    y = 1.95 + k * 1.42
    el += [T(rx, y, rw, 0.5, big, size=26, color=c, bold=True, role="angka"),
           T(rx, y + 0.52, rw, 0.8, txt, size=14, lsp=0.98)]
    if k < 2:
        el.append(L(rx, y + 1.33, rx + rw, y + 1.33, color="LINE", lw=0.75))
slide("7", "Teruji di data tersembunyi", "Model", "Arsitektur model (15) dan validitas pra-pemrosesan", 45,
      "Skor privat dari data yang tidak pernah terlihat sejalan dengan skor publik, tanda tidak *overfit*", el,
      "Apakah skor ini bisa dipercaya? Ada tiga bukti. Pertama, data uji diambil acak per baris. Model yang dilatih "
      "untuk membedakan data latih dan data uji hanya mencapai AUC 0,4997, setara menebak acak. Karena itu validasi "
      "acak sepuluh fold meniru data uji. Kedua, "
      "komponen yang diterima dipilih di holdout set internal; leaderboard publik hanya mengonfirmasi arahnya, dan satu "
      "kali kami menolak perubahan karena leaderboard tidak mengonfirmasi. Ketiga, skor privat dari data yang tidak "
      "pernah kami lihat, 0,24844, tidak memburuk dibanding publik 0,24913. Bila model terlalu menyesuaikan diri dengan "
      "leaderboard, skor privat biasanya turun.",
      istilah="*holdout set*, AUC", sumber="sel 37, 54, 55; Kaggle LB privat; reports/DECISIONS.md D011",
      visual="Grafik garis skor publik per langkah dan titik skor privat (img/s07_lb.png); tiga bukti di kanan.")

# 8. SHAP drivers
el = frame(8, "KEBIJAKAN · INTERPRETASI SHAP", "Rumah kecil boros gas per m², rumah besar boros listrik per rumah",
           "Model penjelas: satu LightGBM per target yang hanya memakai karakteristik kawasan terukur. Garis hitam: "
           "rata-rata sumbangan per kelompok ukuran rumah.")
el.append(I("s08_dep_size.png", M, 1.9, 7.4, "Sumbangan SHAP luas lantai per meter: negatif untuk intensitas gas "
                                              "dan positif untuk listrik per rumah pada rumah besar"))
sx, sw_ = 8.75, SW - M - 8.75
el += stat(sx, 1.9, sw_, "+65%", "sumbangan ke gas per m²: 5% kawasan rumah terkecil (±37 m²) vs 5% terbesar "
           "(±173 m²)", color="AMBER", big_size=44, label_h=0.78)
el += stat(sx, 3.3, sw_, "+75%", "sumbangan ke listrik per rumah: 5% rumah terbesar vs 5% terkecil", color="TEAL",
           big_size=44, label_h=0.6)
el += [L(sx, 4.6, sx + sw_, 4.6, color="LINE", lw=0.75),
       T(sx, 4.68, sw_, 1.0, "Kuartil bawah vs atas: sekitar {A|+34%} dan {T|+35%}. Periode bld_era_15: EUI sekitar "
         "+7%; iklim sekitar 15%.", size=14, lsp=0.98)]
el += chip(M, 5.68, "HASIL", "model penjelas, R² data tak terlihat 0,563 (EUI) dan 0,491 (listrik per meter) · sel 41, "
                             "47, 48")
el += chip(M, 6.08, "INTERPRETASI", "SHAP adalah sumbangan pada prediksi model, bukan hubungan "
                                                "sebab-akibat")
slide("8", "Pendorong konsumsi (SHAP)", "Kebijakan", "Kebijakan dari interpretabilitas (35)", 45,
      "Rumah kecil boros gas per m², rumah besar boros listrik per rumah", el,
      "Sekarang bagian kebijakan. Pertanyaannya: apa yang mendorong konsumsi? Kami melatih satu model penjelas per "
      "target yang hanya memakai karakteristik kawasan yang terukur, lalu membacanya dengan SHAP, yaitu pembagian setiap "
      "tebakan menjadi sumbangan tiap variabel. Ukuran rumah mendorong dua arah. Pada lima persen kawasan dengan rumah "
      "terkecil, sumbangan ke gas per meter persegi sekitar 65 persen lebih tinggi daripada lima persen rumah terbesar, "
      "kemungkinan karena dinding dan atapnya relatif luas. Untuk listrik per rumah arahnya terbalik, sekitar 75 "
      "persen. Antara kuartil bawah dan atas, selisihnya sekitar sepertiga. Ini sumbangan pada prediksi, bukan "
      "sebab-akibat.",
      istilah="model penjelas", sumber="sel 41, 45, 47, 48; bukti/cek_kebijakan.json (kontras ukuran)",
      visual="Dua panel dependensi SHAP digambar ulang dari nilai SHAP yang sama dengan sel 47 (img/s08_dep_size.png); "
             "dua angka besar di kanan.")

# 9. trilemma map
el = frame(9, "KEBIJAKAN · PETA ZONA", "Peta 330 zona: 118 zona Q1 memuat 47% gas dan hampir separuh potensi "
           "*retrofit*", "Kuadran: empat kelompok zona yang dibentuk garis median kedua sumbu. Kenaikan listrik di sini "
                         "adalah energi tahunan, bukan beban puncak.")
el.append(I("s09_trilema.png", M, 1.9, 6.45, "Peta trilema 330 zona: intensitas gas terhadap kenaikan konsumsi listrik "
                                            "tahunan, warna menurut kuadran"))
tx, tw = 7.35, SW - M - 7.35
cols = [("Kuadran", 0.85, "l"), ("Zona", 0.6, "r"), ("Porsi gas", 0.85, "r"), ("*Retrofit*", 1.05, "r"),
        ("Listrik", 0.8, "r"), ("Program", 0.95, "r")]
scale = tw / sum(c[1] for c in cols)
rows = [("Q1", "118", "47%", "3,65 TWh", "+102%", "R1 + R2", "AMBER"),
        ("Q2", "47", "12%", "1,07 TWh", "+90%", "R3", "TEAL"),
        ("Q3", "47", "15%", "0,86 TWh", "+99%", "R2", "NAVY"),
        ("Q4", "118", "27%", "1,81 TWh", "+87%", "standar", "MUTED")]
y = 1.95
x = tx
for name, wcol, al in cols:
    el.append(T(x, y, wcol * scale - 0.05, 0.32, name, size=13, color="MUTED", bold=True, align=al, role="tabel"))
    x += wcol * scale
el.append(L(tx, y + 0.38, tx + tw, y + 0.38, color="LINE", lw=1))
for k, r in enumerate(rows):
    yy = y + 0.48 + k * 0.48
    x = tx
    for j, ((name, wcol, al), v) in enumerate(zip(cols, r[:6])):
        txt = f"{{{'A' if r[6] == 'AMBER' else 'T' if r[6] == 'TEAL' else 'N' if r[6] == 'NAVY' else 'm'}|{v}}}" \
            if j == 0 else v
        el.append(T(x, yy, wcol * scale - 0.05, 0.36, txt, size=16, align=al, bold=(k == 0), role="tabel"))
        x += wcol * scale
el += [L(tx, y + 2.42, tx + tw, y + 2.42, color="LINE", lw=1),
       T(tx, y + 2.5, tw, 0.36, "Potensi *retrofit* (batas atas) 7,39 TWh = 5,23% dari 141,35 TWh gas", size=14,
         color="MUTED")]
el += chip(tx, y + 2.92, "HASIL", "sel 48 sampai 50; *retrofit* = batas atas (definisi sel 48)", maxw=tw)
el += [R(tx, 5.25, tw, 1.27, fill="AMBER_T", r=0.08),
       T(tx + 0.2, 5.3, tw - 0.4, 1.17, "{d|ASUMSI} 80% gas untuk pemanasan, COP 3; asumsi seragam lain hanya "
         "menskalakan angka (71% sampai 128%, sel 51). {N|Urutan zona tahan galat prediksi, tetapi peka bila COP berbeda "
         "per iklim} (C4).", size=14, lsp=0.98, valign="m")]
slide("9", "Peta trilema 330 zona", "Kebijakan", "Kebijakan dari interpretabilitas (35)", 50,
      "Peta 330 zona: 118 zona Q1 memuat 47% gas dan hampir separuh potensi *retrofit*", el,
      "Supaya bisa dipakai, data latih dan prediksi kawasan uji kami ringkas menjadi 330 zona. Sumbu mendatar adalah "
      "intensitas gas; sumbu tegak adalah kenaikan konsumsi listrik tahunan bila pemanas gas diganti pompa panas, "
      "dengan asumsi 80 persen gas untuk pemanasan dan efisiensi pompa panas 3. Kuadran pertama berisi 118 zona dengan "
      "47 persen konsumsi gas dan hampir separuh potensi retrofit, yang merupakan batas atas. Asumsi seragam hanya "
      "menggeser angka. Uji yang lebih sulit menunjukkan urutan zona tahan terhadap galat prediksi, tetapi berubah bila "
      "efisiensi pompa panas berbeda per iklim, jadi daftar zona adalah kandidat, bukan keputusan akhir.",
      istilah="kuadran, kenaikan listrik", sumber="sel 48, 49, 50, 51; bukti/cek_kebijakan.json",
      visual="Peta trilema digambar ulang dari data zona sel 49 dengan label kuadran baru (img/s09_trilema.png); tabel "
             "kuadran dan kotak asumsi di kanan.")


# 10 to 12. recommendations
def reco_slide(n, short, kicker, title, img, alt, rows, chip_kind, chip_text, notes, istilah, sumber, visual, dur):
    el = frame(n, kicker, title, istilah)
    el.append(I(img, M, 1.95, 5.75, alt))
    ac, yend = action_card(6.75, 1.98, SW - M - 6.75, rows)
    el += ac
    el += chip(6.75, yend + 0.12, chip_kind, chip_text, maxw=SW - M - 6.75)
    slide(str(n), short, "Kebijakan", "Kebijakan dari interpretabilitas (35)", dur, title, el, notes,
          istilah=istilah.split(":")[0] if istilah else None, sumber=sumber, visual=visual)
    return yend


reco_slide(10, "Rekomendasi 1: retrofit Q1", "KEBIJAKAN · REKOMENDASI 1",
           "*Retrofit* selubung lebih dulu di zona Q1, mulai dari REG_273, REG_270, dan REG_045",
           "s10_retrofit.png", "Lima zona dengan potensi hemat retrofit terbesar dari tabel sel 50",
           [("Siapa", "Pemerintah kota dan penyedia jasa *retrofit*", 0.36),
            ("Apa", "Audit energi, lalu perbaikan dinding, atap, dan jendela di kawasan rumah kecil dan bangunan "
                    "periode bld_era_15, sebelum subsidi pompa panas", 0.95),
            ("Ukuran", "Gas per m² peserta turun dibanding kawasan pembanding", 0.62),
            ("Dasar bukti", "Rumah kecil {A|+65%} gas per m², periode bld_era_15 {A|+7%}; potensi Q1 {A|3,65 TWh} "
                            "(batas atas)", 0.62)],
           "INTERPRETASI", "usulan tim; potensi adalah batas atas, audit energi menentukan target nyata",
           "Rekomendasi pertama: retrofit selubung bangunan lebih dulu di zona kuadran pertama. Dasarnya temuan SHAP tadi: "
           "kawasan rumah kecil dan bangunan periode tertentu memakai gas per meter persegi paling banyak, jadi tagihannya "
           "juga paling berat. Kandidat awalnya REG_273, REG_270, dan REG_045. Pelaksananya pemerintah kota bersama "
           "penyedia jasa retrofit, dimulai dari audit energi, karena angka potensi kami adalah batas atas. Retrofit "
           "dijalankan sebelum subsidi pompa panas, supaya kebutuhan panas turun lebih dulu. Keberhasilannya diukur dari "
           "turunnya gas per meter persegi di kawasan peserta dibanding kawasan pembanding.",
           "Selubung bangunan: dinding, atap, dan jendela yang memisahkan ruang dalam dari udara luar.",
           "sel 47, 48, 49, 50", "Batang lima zona potensi hemat terbesar (img/s10_retrofit.png); kartu tindakan "
                                 "siapa, apa, ukuran, dasar bukti.", 40)
reco_slide(11, "Rekomendasi 2: gardu dulu", "KEBIJAKAN · REKOMENDASI 2",
           "Kapasitas gardu dinilai sebelum subsidi pompa panas dibuka, mulai dari REG_290",
           "s11_kenaikan.png", "Lima zona dengan kenaikan konsumsi listrik tahunan tertinggi dari tabel sel 50",
           [("Siapa", "Operator jaringan distribusi dan pemerintah kota", 0.36),
            ("Apa", "Nilai kapasitas gardu, jadwalkan penguatan, lalu buka subsidi bertahap dengan tarif waktu "
                    "pemakaian", 0.95),
            ("Ukuran", "Tidak ada gardu melampaui kapasitas di setiap tahap subsidi", 0.62),
            ("Dasar bukti", "Kenaikan = gas × 0,8 / 3 / listrik saat ini; lima zona tertinggi naik {A|113% sampai "
                            "121%}. Tambahan absolut terbesar: REG_263, {A|0,76 TWh}", 0.95)],
           "ASUMSI", "energi tahunan, bukan beban puncak; urutan zona peka terhadap COP per iklim (C4), jadi daftar ini "
                     "kandidat penilaian",
           "Rekomendasi kedua: nilai kapasitas gardu sebelum subsidi pompa panas dibuka. Di lima zona tertinggi, konsumsi "
           "listrik tahunan bisa naik 113 sampai 121 persen, jauh di atas median 95 persen, karena gas di zona itu besar "
           "dibanding listriknya hari ini. Dalam tambahan absolut, zona terbesar adalah REG_263, yang juga kandidat "
           "retrofit. Urutan ini peka terhadap efisiensi pompa panas per iklim, jadi daftar ini adalah kandidat penilaian. "
           "Operator jaringan menilai gardu dengan data lokal, lalu subsidi dibuka bertahap dengan tarif berbasis waktu "
           "pemakaian. Ukurannya: tidak ada gardu yang melampaui kapasitas setelah setiap tahap.",
           "Gardu: titik distribusi listrik yang melayani satu lingkungan. Tarif waktu pemakaian: listrik lebih murah di "
           "jam sepi.", "sel 50, 51; bukti/cek_kebijakan.json", "Batang lima zona kenaikan tertinggi dengan garis "
                                                                  "median (img/s11_kenaikan.png); kartu tindakan.", 45)
reco_slide(12, "Rekomendasi 3: kawasan bergas", "KEBIJAKAN · REKOMENDASI 3",
           "Insentif pompa panas untuk kawasan bergas, dimulai sebagai pilot di zona Q2",
           "s12_dep_adopt.png", "Sumbangan SHAP porsi pemanas gas jaringan terhadap adopsi pemanas listrik",
           [("Siapa", "Pemerintah kota bersama penyedia pompa panas", 0.36),
            ("Apa", "Bantuan pasang pompa panas dan bongkar ketel lama sebagai pilot bergiliran; di Q1 setelah "
                    "*retrofit* dan cek gardu", 0.95),
            ("Di mana", "Zona Q2 bergas dengan adopsi rendah, misalnya REG_249 (gas jaringan 85,54%, adopsi 3,75%)",
             0.62),
            ("Dasar bukti", "Adopsi {A|40%} di kawasan bergas di bawah 40%, hanya {A|2,73%} di atas 80%, padahal itu 70% "
                            "kawasan", 0.62),
            ("Ukuran", "Adopsi zona peserta naik dibanding zona pembanding", 0.62)],
           "INTERPRETASI", "pola deskriptif: porsi pemanas berjumlah 100%, bukan tombol (sel 42)",
           "Rekomendasi ketiga menyangkut adopsi. Tingkat adopsi pemanas listrik sekitar 40 persen di kawasan yang "
           "sedikit tersambung gas, tetapi hanya sekitar 3 persen di kawasan bergas di atas 80 persen, padahal kawasan "
           "seperti itu 70 persen dari semua kawasan. Sosiodemografi hanya menyumbang sekitar 15 persen. Karena porsi "
           "pemanas berjumlah 100 persen, ini pola, bukan tombol. Usulan kami: bantuan pasang pompa panas dan bongkar "
           "ketel lama, dimulai sebagai pilot di zona Q2 seperti REG_249 yang kenaikan listriknya lebih rendah; di Q1 "
           "setelah retrofit dan cek gardu.",
           "Pilot bergiliran: program dibuka per zona secara bertahap agar dampaknya dapat dibandingkan dengan zona yang "
           "belum ikut.", "sel 42, 47, 48, 50; bukti/cek_kebijakan.json",
           "Dependensi SHAP adopsi terhadap porsi gas jaringan (img/s12_dep_adopt.png); kartu tindakan.", 40)

# 13. limitations
el = frame(13, "KEBIJAKAN · KETERBATASAN", "Tiga hal belum diketahui, dan setiap rekomendasi sudah "
           "memperhitungkannya")
lim = [("Hubungan sebab-akibat", "SHAP hanya sumbangan pada prediksi.",
        "Program dijalankan sebagai pilot dengan zona pembanding."),
       ("Beban puncak, gardu, dan COP lokal", "Data hanya energi tahunan; urutan zona peka terhadap COP per iklim.",
        "Daftar zona adalah kandidat; gardu dinilai operator dengan data lokal."),
       ("Potensi *retrofit* yang nyata", "7,39 TWh adalah batas atas; periode bangunan saja sekitar separuhnya.",
        "Audit energi di zona pilot menentukan target penghematan.")]
cw = (CW - 2 * 0.3) / 3
for k, (head, what, how) in enumerate(lim):
    x = M + k * (cw + 0.3)
    el += [R(x, 1.95, cw, 4.0, fill="SOFT", r=0.1),
           T(x + 0.25, 2.12, cw - 0.5, 0.3, "BELUM DIKETAHUI", size=11, color="MUTED", bold=True, spacing=0.8,
             role="label"),
           T(x + 0.25, 2.45, cw - 0.5, 0.8, head, size=19, bold=True, lsp=0.95),
           T(x + 0.25, 3.3, cw - 0.5, 1.05, what, size=15, lsp=0.98),
           L(x + 0.25, 4.42, x + cw - 0.25, 4.42, color="LINE", lw=1),
           T(x + 0.25, 4.55, cw - 0.5, 0.3, "PENANGANAN", size=11, color="TEAL", bold=True, spacing=0.8, role="label"),
           T(x + 0.25, 4.88, cw - 0.5, 0.95, f"{{N|{how}}}", size=15, lsp=0.98)]
el += chip(M, 6.12, "HASIL", "definisi potensi sel 48; data tanpa profil beban (bagian 7.2); uji ketahanan "
                             "bukti/cek_kebijakan.json")
slide("13", "Keterbatasan", "Kebijakan", "Kebijakan dari interpretabilitas (35)", 35,
      "Tiga hal belum diketahui, dan setiap rekomendasi sudah memperhitungkannya", el,
      "Kami juga jujur soal batasnya. Pertama, SHAP tidak membuktikan sebab-akibat, jadi semua program kami usulkan "
      "sebagai pilot dengan zona pembanding. Kedua, data hanya memuat energi tahunan, dan urutan zona peka terhadap "
      "efisiensi pompa panas per iklim; jadi daftar zona adalah kandidat yang dinilai operator dengan data lokal. "
      "Ketiga, potensi retrofit 7,39 terawatt-jam adalah batas atas; periode bangunan saja sekitar separuhnya, dan "
      "audit energi yang menentukan angka nyatanya.",
      sumber="sel 48, 51; bukti/cek_kebijakan.json", visual="Tiga kartu: yang belum diketahui, lalu penanganannya.")

# 14. closing (dark)
el = [T(M, 0.55, CW, 0.3, "PENUTUP", size=13, color="LIGHT", bold=True, spacing=1.5, role="kicker"),
      T(M, 0.95, CW, 1.2, "Layak dipilih karena akurat, jujur, dan siap dijalankan", size=36, color="WHITE", bold=True,
        lsp=0.95, role="title")]
cols3 = [("AKURAT", "0,24844", "skor privat, peringkat 8"),
         ("JUJUR", "Lulus uji label acak", "baris validasi dan latih; hasil identik di perangkat sama"),
         ("SIAP DIJALANKAN", "330 zona", "kandidat berurutan, tiga program, ukuran keberhasilan")]
cw = (CW - 2 * 0.3) / 3
for k, (lab, big, txt) in enumerate(cols3):
    x = M + k * (cw + 0.3)
    el += [R(x, 2.25, cw, 1.95, fill="NAVY2", r=0.1),
           T(x + 0.25, 2.42, cw - 0.5, 0.3, lab, size=12, color="LIGHT", bold=True, spacing=1.0, role="label"),
           T(x + 0.25, 2.75, cw - 0.5, 0.6, big, size=28, color="AMBER" if k == 0 else "WHITE", bold=True,
             role="angka"),
           T(x + 0.25, 3.4, cw - 0.5, 0.75, txt, size=15, color="LIGHT", lsp=0.98)]
el.append(T(M, 4.5, CW, 0.4, "Langkah berikutnya", size=18, color="WHITE", bold=True))
steps = ["Minta data beban puncak, kapasitas gardu, dan COP lokal dari operator jaringan",
         "Pilot *retrofit* di REG_273, REG_270, dan REG_045 dengan zona pembanding",
         "Perbarui peta zona setiap tahun dengan data konsumsi terbaru"]
for k, s in enumerate(steps):
    y = 5.0 + k * 0.48
    el += [O(M, y + 0.04, 0.34, 0.34, fill="AMBER"),
           T(M, y + 0.04, 0.34, 0.34, str(k + 1), size=13, color="WHITE", bold=True, align="c", valign="m",
             role="angka"),
           T(M + 0.5, y, CW - 0.5, 0.42, s, size=17, color="WHITE", valign="m")]
el += [T(M, 6.55, CW, 0.4, "Terima kasih. Silakan ajukan pertanyaan.", size=15, color="LIGHT"),
       T(M, 7.08, 6, 0.25, "Tim BISMILLAH · DICE FEST 2026 · Babak final", size=10, color="LIGHT", role="footer"),
       T(SW - M - 1.5, 7.08, 1.5, 0.25, "14", size=10, color="LIGHT", align="r", role="footer")]
slide("14", "Penutup", "Penutup", "Penutup", 30, "Layak dipilih karena akurat, jujur, dan siap dijalankan", el,
      "Jadi, mengapa solusi ini layak dipilih? Akurat: skor privat 0,24844, peringkat 8. Jujur: lulus uji kebocoran, "
      "dan hasilnya identik setiap diulang di perangkat yang sama. Siap dijalankan: 330 zona sudah berurutan sebagai "
      "kandidat, dengan tiga program dan ukuran keberhasilannya. Langkah berikutnya: meminta data beban puncak dan "
      "kapasitas gardu dari operator jaringan, lalu memulai pilot retrofit di tiga zona pertama. Terima kasih, kami siap "
      "menjawab pertanyaan.",
      sumber="Kaggle LB privat; sel 35, 49, 50", visual="Slide gelap: tiga alasan dan tiga langkah berikutnya.",
      bg="NAVY")


# ---------------------------------------------------------------------------------------------------------------
# backup slides
def backup(code, title, el_body, notes, sumber, visual=None):
    el = frame(code, f"CADANGAN {code} · TANYA JAWAB", title) + el_body
    SLIDES.append(dict(id=code, short=title, section="Cadangan", rubric="Tanya jawab", dur=None, title=title, el=el,
                       notes=notes, istilah=None, sumber=sumber, visual=visual, bg=None))


def table(x, y, widths, header, rows, size=15, row_h=0.42, aligns=None, bold_first=False):
    el, xx = [], x
    aligns = aligns or ["l"] + ["r"] * (len(widths) - 1)
    for w, h, a in zip(widths, header, aligns):
        el.append(T(xx, y, w - 0.08, 0.34, h, size=12, color="MUTED", bold=True, align=a, role="tabel"))
        xx += w
    el.append(L(x, y + 0.4, x + sum(widths), y + 0.4, color="LINE", lw=1))
    for k, r in enumerate(rows):
        xx = x
        for j, (w, v, a) in enumerate(zip(widths, r, aligns)):
            el.append(T(xx, y + 0.5 + k * row_h, w - 0.08, row_h, v, size=size, align=a, valign="m",
                        bold=bold_first and j == 0, role="tabel"))
            xx += w
    return el


if VC:
    g, a = VC["global"], VC["adaptif"]
    vc_rows = [("Gas", idn(g["gas"], 5), idn(a["gas"], 5), idn(a["gas"] - g["gas"], 6).replace("-", "−")),
               ("EUI", idn(g["eui"], 5), idn(a["eui"], 5), idn(a["eui"] - g["eui"], 6).replace("-", "−")),
               ("Listrik", idn(g["elec"], 5), idn(a["elec"], 5), idn(a["elec"] - g["elec"], 6).replace("-", "−"))]
    vc_mean = idn(VC["selisih_rata2_5_target"], 6).replace("-", "−")
else:
    vc_rows, vc_mean = [("?", "?", "?", "?")], "?"
body = [I("c1_basis.png", M, 1.95, 6.1, "Basis segitiga dua simpul: bobot ujung rendah 1 minus t dan ujung tinggi t"),
        T(7.0, 1.95, SW - M - 7.0, 1.5, [dict(text="Peringkat luas A, jumlah meter M, dan tebakan p di data latih: 2×2×2 "
                                                  "= 8 bobot pojok per model", bullet=True, after=6),
                                             dict(text="NNLS, semua bobot ≥ 0: 224 bobot gas, 224 EUI, dan 104 listrik "
                                                       "untuk 429.303 baris", bullet=True)], size=15, lsp=0.98)]
body += table(7.0, 3.55, [1.25, 1.35, 1.35, 1.75], ["Skor jujur", "Global", "Adaptif", "Selisih"], vc_rows, size=15)
body += [T(7.0, 5.4, SW - M - 7.0, 0.6, f"Rata-rata 5 target {vc_mean}; LB publik 0,24934 menjadi 0,24925 (sel 37)",
           size=14, color="MUTED", lsp=0.98)]
body += chip(M, 6.12, "HASIL", "bobot di-fit pada 4/5 baris, dinilai pada 1/5 sisanya, KFold 5; global = sel 35 · "
                               "bukti/skor_jujur_bobot_adaptif.json")
backup("C1", "Bobot adaptif tidak kalah di kolom mana pun pada skor jujur, terbesar di EUI", body,
       "Pedoman jawaban (P2): model yang paling akurat di kawasan kecil belum tentu paling akurat di kawasan besar. "
       "Setiap model punya bobot di ujung kecil dan ujung besar untuk luas, meter, dan besar tebakan; kawasan di antaranya "
       "mendapat campuran. Skor jujur bobot adaptif kini dihitung ulang dengan fungsi notebook yang sama (KFold 5): "
       "ketiga kolom membaik, terbesar di EUI, rata-rata lima target sekitar minus 0,000125, sejalan dengan catatan "
       "D012 (minus 0,000124). Catatan jujur: bentuk ini dipilih di baris yang sama, jadi hitungan ulang ini bukan bukti "
       "independen; bukti independennya adalah leaderboard, minus 0,00009. Tiga simpul tidak lebih baik daripada dua.",
       "sel 23, 35, 37; bukti/skor_jujur_bobot_adaptif.json; reports/DECISIONS.md D012")

lam = [("rata-rata galat zona", "0,453"), ("rata-rata galat petak iklim", "0,346"),
       ("rata-rata galat ±30 nomor subzona", "−0,020"), ("kNN full, W 100, k 20", "0,037"),
       ("kNN era, W 30, k 20", "0,126"), ("kNN fc, W 300, k 50", "0,079"), ("kNN full, W 10, k 10", "0,072"),
       ("kNN era, W 300, k 50", "0,124"), ("kNN full, W 3000, k 200", "0,156")]
body = table(M, 1.95, [3.6, 1.1], ["Fitur koreksi (gas)", "λ"], lam, size=14, row_h=0.4)
body += [T(5.9, 1.95, SW - M - 5.9, 3.6, [
    dict(text="Galat latih dirata-rata per zona, petak iklim, ±30 nomor subzona, dan k kawasan termirip", bullet=True,
         after=6),
    dict(text="Dikecilkan n/(n + 30) atau n/(n + 5) bila tetangga sedikit", bullet=True, after=6),
    dict(text="λ dipilih di *holdout set* 30%, lalu dibekukan di `resid_knn.json`".replace("`", ""), bullet=True,
         after=6),
    dict(text="Besar koreksi rms/σ: gas 0,02047; EUI 0,04344; listrik 0,01195 (sel 35)", bullet=True, after=6),
    dict(text="Koreksi kNN menurunkan LB sekitar 0,0004 (0,2499 menjadi 0,24954); koreksi tingkat grup sudah ada "
              "sejak v12", bullet=True)], size=15, lsp=0.98)]
body += [T(5.9, 5.45, SW - M - 5.9, 0.6, "λ negatif kecil muncul karena fitur tetangga saling berkorelasi; yang "
           "divalidasi adalah efek totalnya. Versi ridge diuji di audit akhir dan tidak lebih baik secara konsisten.",
           size=13, color="MUTED", lsp=0.98)]
body += chip(M, 6.12, "FAKTA", "configs/resid_knn.json (ditulis sel 22); postprocess.py K = 30, KSHR = 5")
backup("C2", "Koreksi galat wilayah menggeser tebakan sedikit mengikuti galat tetangga", body,
       "Pedoman jawaban (P3): setelah digabung, galat masih berkelompok; kalau tetangga yang mirip sama-sama ditebak "
       "terlalu rendah, kawasan ini kemungkinan juga terlalu rendah. Target data uji tidak pernah dipakai; λ dipilih "
       "di holdout yang tidak ikut menyusun tetangga. Konstanta K, WIN, dan KSHR diperiksa ulang di audit akhir (D015). "
       "Bila ditanya langkah v12 ke v15 di tabel sel 37: langkah itu menggabungkan beberapa perubahan; koreksi kNN "
       "sendiri sekitar minus 0,0004 di leaderboard.",
       "sel 22, 35, 37; reports/DECISIONS.md D015; reports/JOURNAL.md 15 sampai 16 Okt dan audit akhir")

pp_rows = [("S1: kawasan 2 sampai 3 meter", "+20,1%", "+18,4%", "+18,7%"),
           ("S1: kawasan 4 meter", "+6,4%", "+3,2%", "+7,1%"),
           ("S3: faktor k", "0,505", "0,657", "tidak"),
           ("S3: a0 dan b0", "0,0081; 0,393", "0,0084; 0,387", "tidak"),
           ("S4: batas", "batas latih", "batas latih", "batas latih")]
body = table(M, 1.95, [3.0, 1.65, 1.65, 1.35], ["Langkah", "Gas", "EUI", "Listrik"], pp_rows, size=14, row_h=0.42)
body += [T(8.75, 1.95, SW - M - 8.75, 3.9, [
    dict(text="S1: kawasan 2 sampai 3 meter rata-rata ditebak sekitar 20% terlalu rendah oleh semua keluarga model",
         bullet=True, after=6),
    dict(text="S3: tebakan gas dekat batas 999.765 kWh ditarik turun (normal terpotong)", bullet=True, after=6),
    dict(text="S4: dipotong ke batas data latih; 77 kawasan duplikat disalin", bullet=True, after=6),
    dict(text="Asal: S1 dari prediksi OOF penuh, k pada S3 dari *holdout*, batas S4 dari data latih", bullet=True,
         after=6),
    dict(text="LB 0,24920 menjadi 0,24913; perkiraan *holdout* sama", bullet=True)], size=15, lsp=0.98)]
body += [T(M, 4.7, 8.0, 1.2, "Reprodusibilitas: *seed* dikunci, mode deterministik, *thread* BLAS dibatasi satu. "
           "Sidik sha256 aec745ab…d66e tercetak di sel 35, sama dengan berkas yang dipilih (perangkat sama).",
           size=15, lsp=0.98)]
body += chip(M, 6.12, "FAKTA", "resid_knn.json bagian postproc (sel 22); sha256 sel 35; LB sel 37")
backup("C3", "Penyesuaian akhir menangani dua batas data, dan hasilnya identik setiap diulang di perangkat sama", body,
       "Pedoman jawaban (P5): assemble menjalankan urutan tetap: gabung, koreksi, sesuaikan batas, salin duplikat, "
       "periksa, tulis. Yang di-fit di dalamnya hanya bobot gabungan. S1 berasal dari prediksi out-of-fold penuh, k pada "
       "S3 dari holdout, dan batas S4 dari data latih; semuanya dibekukan di satu berkas konfigurasi (D015). Efeknya "
       "0,00007 di leaderboard, sama dengan perkiraan holdout.",
       "sel 22, 35, 37; reports/DECISIONS.md D015")

sens = [("70%", "+99,6%", "+83,0%", "+71,1%"), ("80%", "+113,8%", "+94,8%", "+81,3%"),
        ("90%", "+128,0%", "+106,7%", "+91,5%")]
body = table(M, 1.95, [2.4, 1.6, 1.6, 1.6], ["Porsi gas untuk pemanasan", "COP 2,5", "COP 3", "COP 3,5"], sens, size=18,
             row_h=0.55)
nz, cz = (CEK["derau_prediksi"], CEK["cop_per_iklim"]) if CEK else ({}, {})
body += [T(M, 4.3, 7.2, 1.7, [
    dict(text="Rumus: gas × porsi pemanasan / COP / listrik. Asumsi seragam mengalikan semua zona dengan faktor yang "
              "sama, jadi urutan zona tetap secara aljabar, bukan karena diuji.", after=8),
    dict(text="Median 94,85% (sel 50) dikali (porsi / 0,8) × (3 / COP); *writeup* Tabel 4 memuat nilai yang sama dalam "
              "bilangan bulat (+71% sampai +128%).")], size=14, color="MUTED", lsp=0.98)]
body += [R(8.15, 1.95, SW - M - 8.15, 3.95, fill="TEAL_T", r=0.1),
         T(8.4, 2.1, SW - M - 8.65, 0.35, "UJI KETAHANAN URUTAN ZONA", size=11, color="TEAL_D", bold=True, spacing=0.8,
           role="label"),
         T(8.4, 2.5, SW - M - 8.65, 1.6, [
             dict(text=f"**Galat prediksi** baris uji, 200 simulasi: Spearman {idn(nz.get('spearman_median', 0), 3)}; "
                       f"median {idn(nz.get('pindah_kuadran_median', 0))} zona pindah kuadran; 5 zona teratas tetap.")],
           size=14, lsp=0.98),
         T(8.4, 4.0, SW - M - 8.65, 1.8, [
             dict(text=f"**COP 2,5 sampai 3,5 menurut iklim zona:** Spearman {idn(cz.get('spearman', 0), 2)}; "
                       f"{cz.get('zona_pindah_kuadran', '?')} zona pindah kuadran; "
                       f"{cz.get('irisan_5_teratas', '?')} dari 5 zona teratas tetap.")], size=14, color="AMBER_D",
           lsp=0.98)]
body += chip(M, 6.12, "ASUMSI", "energi tahunan; COP per iklim adalah uji tekanan, bukan data · bukti/cek_kebijakan.json")
backup("C4", "Urutan zona tahan galat prediksi, tetapi peka terhadap COP yang berbeda per iklim", body,
       "Pedoman jawaban: asumsi seragam hanya menskalakan semua zona, jadi tidak menguji apa pun. Kami menguji dua hal yang "
       "bisa mengubah urutan. Galat prediksi pada baris uji hampir tidak berpengaruh karena zona merata-ratakan banyak "
       "kawasan. COP yang berbeda per iklim berpengaruh besar, sehingga daftar zona kami sebut kandidat dan operator "
       "menilai dengan data lokal. Beban puncak bisa jauh lebih besar karena pemanasan menumpuk di bulan terdingin dan "
       "COP turun saat dingin.",
       "sel 50, 51; bukti/cek_kebijakan.json; writeup Tabel 4")

val = [("acak per baris", "tanpa", "0,2573", "0,6721", "0,3262"), ("acak per baris", "dengan", "0,2562", "0,6699",
                                                                       "0,3255"),
       ("per zona", "tanpa", "0,2708", "0,6923", "0,3302"), ("per zona", "dengan", "0,3637", "0,8331", "0,3474")]
body = table(M, 1.95, [2.2, 2.1, 1.3, 1.3, 1.3], ["Skema", "Rata-rata wilayah", "Gas", "EUI", "Listrik"], val, size=16,
             row_h=0.5, aligns=["l", "l", "r", "r", "r"])
body += [T(M, 4.65, 8.2, 1.3, "Data uji diambil acak per baris (AUC 0,4997; subzona baru 8,14% vs 8,17%). Skema per "
           "zona menguji situasi yang tidak terjadi di data uji dan membuat fitur rata-rata wilayah tampak merusak.",
           size=15, lsp=0.98),
         T(9.3, 1.95, SW - M - 9.3, 3.5, "Pada skema yang sesuai data uji, fitur yang sama menurunkan galat di ketiga "
           "target. Skema yang keliru akan membuang fitur yang berguna.", size=17, color="NAVY", bold=True, lsp=0.98)]
body += chip(M, 6.12, "FAKTA", "writeup Tabel 1 (LightGBM, dijalankan saat pengembangan); sel 54 dan 55")
backup("C5", "Validasi per zona keliru untuk data ini karena data uji diambil acak per baris", body,
       "Pedoman jawaban: untuk data spasial, pembagian per wilayah sering dianjurkan. Namun 92% kawasan uji berada di "
       "subzona yang juga ada di data latih, dan validasi adversarial tidak bisa membedakan latih dan uji. Jadi validasi "
       "acak per baris adalah skema yang meniru data uji.", "writeup Tabel 1; sel 54, 55")

body = [T(M, 1.95, 6.3, 3.9, [
    dict(text="Kolom komposisi pemanas adalah pecahan jumlah rumah tangga N. Bilangan bulat terkecil yang membuat semua "
              "pecahan bulat adalah N.", bullet=True, after=8),
    dict(text="K = rumah tangga di luar kategori pemanas yang tercatat. Model menebak peluang rumah tangga K bukan "
              "pengguna pemanas listrik.", bullet=True, after=8),
    dict(text="Dua LightGBM (q_xent, r_reg) dirata-rata; validasi 5 *fold* per subzona, dibobot ulang ke komposisi "
              "subzona baru.", bullet=True)], size=15, lsp=0.98)]
body += [R(7.3, 1.95, SW - M - 7.3, 3.9, fill="TEAL_T", r=0.1)]
body += stat(7.6, 2.15, SW - M - 7.9, "1,03 poin", "RMSE pada subzona baru (0,143 setelah dibagi σ adopsi)",
             color="TEAL", big_size=36, label_h=0.7)
body += stat(7.6, 3.75, SW - M - 7.9, "8,14%", "kawasan uji yang memakai model ini; 91,86% sisanya disalin",
             color="NAVY", big_size=36, label_h=0.7)
body += chip(M, 6.12, "HASIL", "sel 35 (CV rmse unseen-weighted 1,0303) dan sel 39")
backup("C6", "Adopsi di subzona baru dimodelkan sebagai peluang dari jumlah rumah tangga", body,
       "Pedoman jawaban: adopsi seragam per subzona, jadi kawasan di subzona yang dikenal cukup disalin. Untuk subzona "
       "baru, struktur cacah rumah tangga dipakai: N dari pecahan komposisi pemanas, K rumah tangga di luar kategori "
       "tercatat. Galatnya dekat batas derau binomial.", "sel 28, 35, 39")

if LEAK:
    lk = LEAK["blok"]
    names = [("knn_validasi", "kNN-target, baris validasi"), ("knn_uji", "kNN-target, data uji"),
             ("knn_latih_oof", "kNN-target, baris latih (OOF dalam)"), ("te_validasi", "rata-rata wilayah, validasi"),
             ("te_uji", "rata-rata wilayah, data uji"), ("te_latih_oof", "rata-rata wilayah, baris latih"),
             ("kontrol_bocor_knn_validasi", "KONTROL: tetangga termasuk validasi")]
    lrows = [(lab, idn(lk[k]["kolom"]), idn(lk[k]["baris"]), idn(lk[k]["baris_berubah"])) for k, lab in names
             if k != "knn_latih_oof" and k != "te_latih_oof"]
else:
    lrows = [("uji validasi belum selesai", "?", "?", "?")]
if LEAK2:
    lk2 = LEAK2["blok"]
    lrows[4:4] = [("kNN-target, baris latih bagian 0", idn(lk2["knn_latih_oof"]["kolom"]),
                   idn(lk2["knn_latih_oof"]["baris_bagian_0"]), idn(lk2["knn_latih_oof"]["berubah_bagian_0"])),
                  ("rata-rata wilayah, baris latih bagian 0", idn(lk2["te_latih_oof"]["kolom"]),
                   idn(lk2["te_latih_oof"]["baris_bagian_0"]), idn(lk2["te_latih_oof"]["berubah_bagian_0"]))]
body = table(M, 1.95, [3.85, 0.9, 1.2, 1.4], ["Blok fitur (fold 0 dari 10)", "Kolom", "Baris", "Berubah"], lrows,
             size=14, row_h=0.4)
body += [T(8.35, 1.95, SW - M - 8.35, 2.5, [
    dict(text="Label kelima target diacak (1) di baris validasi dan (2) di satu bagian dalam baris latih; fitur "
              "dihitung ulang dengan modul notebook yang sama.", after=6),
    dict(text="Lulus bila blok yang tidak boleh membaca label itu identik bit, dan kontrol bocor berubah. Cakupan: satu "
              "fold, kNN-target dan rata-rata wilayah.")], size=13, lsp=0.98),
         T(8.35, 4.6, SW - M - 8.35, 1.3, "Bobot dari prediksi *out-of-fold* (0,25096) setara dengan bobot yang di-fit "
           "di *holdout set* (0,25095); ensemble pengembangan 13 Okt (D002).", size=13, color="NAVY", lsp=0.98)]
body += chip(M, 6.12, "HASIL", "bukti/uji_kebocoran_notebook.json dan uji_kebocoran_oof_dalam.json; tests/"
                               "test_no_leakage.py (LK1)")
backup("C7", "Uji label acak: fitur bertarget tidak berubah saat label baris yang dinilai diacak", body,
       "Pedoman jawaban (P1 dan P4): untuk fitur kNN-target dan rata-rata wilayah, baris validasi dan uji tidak membaca "
       "label validasi, dan baris latih tidak membaca label bagian dalamnya sendiri; keduanya diuji dengan label acak. "
       "Cakupannya satu fold; koreksi galat wilayah hanya memakai galat baris latih untuk baris uji. Untuk pergeseran "
       "bobot (P1): di holdout, prediksi berasal dari rata-rata model fold seperti data uji, dan bobot dari prediksi "
       "out-of-fold memberi skor setara dengan bobot yang di-fit di holdout (bukti pengembangan, ensemble 13 Okt). "
       "Catatan jujur: XGBoost dan LightGBM memakai early stopping pada fold validasinya sendiri (sel 36), sehingga galat "
       "out-of-fold sedikit optimistis; keputusan diambil dari holdout.",
       "bukti/uji_kebocoran_*.json; reports/DECISIONS.md D002; sel 36")

r2 = [("log intensitas gas (EUI)", "0,563"), ("log gas per meter", "0,747"), ("log listrik per meter", "0,491"),
      ("adopsi pemanas listrik", "0,991")]
body = table(M, 1.95, [4.0, 1.2], ["Model penjelas", "R²"], r2, size=16, row_h=0.48)
body += [T(M, 4.4, 5.3, 1.5, "R² adopsi tinggi karena porsi jenis pemanas berjumlah 100%, bukan karena perilaku "
           "terjelaskan (identitas komposisi).", size=14, color="MUTED", lsp=0.98)]
body += [T(6.4, 1.95, SW - M - 6.4, 3.9, [
    dict(text="Gabungan 41 model memakai rata-rata target wilayah, yang bukan tuas kebijakan, dan sulit dibaca.",
         bullet=True, after=8),
    dict(text="Model penjelas hanya memakai karakteristik terukur: bangunan, periode, komposisi pemanas, "
              "sosiodemografi, iklim, agregat kawasan.", bullet=True, after=8),
    dict(text="Porsi kontribusi (dibaca dari grafik sel 43): adopsi, komposisi pemanas sekitar 81% dan sosiodemografi "
              "sekitar 15%; listrik per meter, sosiodemografi sekitar 40%.", bullet=True)], size=15, lsp=0.98)]
body += chip(M, 6.12, "HASIL", "sel 40 sampai 43; LightGBM 800 putaran, TreeSHAP eksak")
backup("C8", "SHAP dibaca dari model penjelas terpisah karena gabungan 41 model memakai fitur yang bukan tuas kebijakan",
       body,
       "Pedoman jawaban: kecocokan model penjelas dilaporkan supaya batas keyakinannya terlihat. SHAP dihitung dengan "
       "TreeSHAP eksak pada sampel 20.000 baris latih untuk tabel dan grafik; peta zona memakai kontribusi pada seluruh "
       "613 ribu kawasan.", "sel 40, 41, 42, 43, 45")


# ---------------------------------------------------------------------------------------------------------------
def words(s):
    return len(re.findall(r"\S+", s))


def main():
    main_slides = [s for s in SLIDES if s["dur"]]
    total = sum(s["dur"] for s in main_slides)
    print(f"{len(main_slides)} main slides, {len(SLIDES) - len(main_slides)} backup, planned {total // 60} min "
          f"{total % 60} s")
    for s in main_slides:
        w = words(s["notes"])
        print(f"  slide {s['id']:>2}: {s['dur']:3d} s planned, {w:3d} words = {w / WPS:5.1f} s at 130 wpm")
    issues = check_fit(SLIDES)
    print("text fit issues:", len(issues))
    for i in issues:
        print("  ", i)
    for s in SLIDES:
        s["notes_full"] = s["notes"] + (f"\n\nSumber: {s['sumber']}" if s.get("sumber") else "")
    os.makedirs(PREV, exist_ok=True)
    for f in os.listdir(PREV):
        os.remove(os.path.join(PREV, f))
    ims = []
    for k, s in enumerate(SLIDES, 1):
        ims.append(render(s, os.path.join(PREV, f"slide-{k:02d}.png")).convert("RGB"))
    ims[0].save(os.path.join(HERE, "BISMILLAH_Final_v2_pratinjau.pdf"), save_all=True, append_images=ims[1:],
                resolution=144)
    write_pptx([dict(s, notes=s["notes_full"], doc_title="BISMILLAH: Dari prediksi energi ke urutan program 330 zona")
                for s in SLIDES], OUT_PPTX)
    write_brief(main_slides)
    write_card(main_slides)
    print("wrote", OUT_PPTX)


def screen_text(s, roles=("body",)):
    """on-screen text of a slide for the brief: prose (role body) by default; big numbers, table cells and diagram
    labels are listed separately as visual elements"""
    out = []
    for e in s["el"]:
        if e["t"] == "text" and e.get("role") in roles:
            for p in e["p"]:
                t = p if isinstance(p, str) else p["text"]
                t = re.sub(r"\{[a-zA-Z^]\|(.+?)\}", r"\1", t)
                if t.strip():
                    out.append(t.strip())
    return out


def write_brief(main_slides):
    head = open(os.path.join(HERE, "brief_kepala.md")).read()
    lines = [head.rstrip(), "", "## C. Isi per slide", ""]
    for s in main_slides:
        m, sec = divmod(s["dur"], 60)
        dur = (f"{m} menit " if m else "") + (f"{sec} detik" if sec else "")
        lines += [f"### Slide {s['id']}. {s['short']} ({dur.strip()})", f"- **Bagian rubrik:** {s['rubric']}",
                  f"- **Judul:** {s['title']}"]
        body = [t for t in screen_text(s) if t != s["title"]]
        lines.append("- **Isi di layar:** " + " | ".join(body))
        lines.append(f"- **Visual:** {s['visual']} Angka besar, sel tabel, dan label diagram: "
                     + " | ".join(screen_text(s, ("angka", "tabel", "label", "diagram"))))
        if s.get("istilah"):
            lines.append(f"- **Istilah:** {s['istilah']} (penjelasan satu kalimat tertulis di kaki slide)")
        lines.append(f"- **Sumber:** {s['sumber']}")
        lines.append(f"- **Narasi pembicara (±{s['dur']} detik):** {s['notes']}")
        lines.append("")
    lines += ["## D. Slide cadangan (tanya jawab)", ""]
    for s in SLIDES:
        if s["dur"]:
            continue
        lines += [f"#### {s['id']}. {s['title']}",
                  "- Isi di layar: " + " | ".join(screen_text(s, ("body", "angka", "tabel", "label", "diagram"))),
                  f"- {s['notes']}", f"- Sumber: {s['sumber']}", ""]
    tail = open(os.path.join(HERE, "brief_ekor.md")).read()
    lines.append(tail.rstrip())
    open(os.path.join(HERE, "05_struktur_deck_v2.md"), "w").write("\n".join(lines) + "\n")


ROLES = [  # three-person split: one anchor (opening, closing, slides, timer, Q&A routing) and two content speakers
    ("Orang 1 · Jangkar", ["1", "2", "14"],
     "Membuka dan menutup presentasi. Selama Orang 2 dan Orang 3 berbicara, ia berbagi layar dan memindahkan slide, "
     "memegang pengatur waktu, dan menyiapkan slide cadangan. Saat tanya jawab ia mengarahkan pertanyaan. Pilih anggota "
     "yang paling lancar berbicara dan memahami cerita dari awal sampai akhir."),
    ("Orang 2 · Teknis", ["3", "4", "5", "6", "7"],
     "Menyampaikan data, fitur, arsitektur, dan validasi, lalu menjawab pertanyaan teknis. Pilih anggota yang paling "
     "paham model dan validasi."),
    ("Orang 3 · Kebijakan", ["8", "9", "10", "11", "12", "13"],
     "Menyampaikan SHAP, peta zona, tiga rekomendasi, dan keterbatasan, lalu menjawab pertanyaan kebijakan. Pilih "
     "anggota yang paling nyaman menjelaskan hasil kepada pembuat kebijakan."),
]
HANDOFF = {"2": "Untuk menjawabnya, [nama Orang 2] akan memulai dari datanya.",
           "7": "Dengan model yang sudah teruji itu, [nama Orang 3] akan masuk ke bagian kebijakan.",
           "13": "[nama Orang 1] akan menutup presentasi kami."}


def must_say(notes):
    found = re.findall(r"REG_\d+|(?<!sel )(?<![\d,.])\d+(?:[.,]\d+)*(?: persen)?", notes)
    return sorted({t for t in found if t.startswith("REG_") or "persen" in t or "," in t or int(t) >= 10},
                  key=notes.find)


def mmss(sec):
    return f"{sec // 60}:{sec % 60:02d}"


def write_card(main_slides):
    by_id = {s["id"]: s for s in main_slides}
    ends, cum = {}, 0
    for s in main_slides:
        cum += s["dur"]
        ends[s["id"]] = cum
    total = cum
    L = ["# Kartu Latihan: Deck Final v2 (10 menit, 3 orang)", "",
         "Dibuat otomatis oleh `build_deck_v2.py` dari narasi slide yang sama dengan PPTX. Kecepatan acuan 130 kata per "
         f"menit. Total rencana {total // 60} menit {total % 60} detik, menyisakan {600 - total} detik untuk serah "
         "terima dan jeda. Latih minimal dua kali penuh dengan pengatur waktu, lalu catat waktu tiap slide.", "",
         "## A. Peran tim", ""]
    for name, ids, desc in ROLES:
        secs = sum(by_id[i]["dur"] for i in ids)
        L += [f"**{name}**: slide {', '.join(ids)} ({secs // 60} menit {secs % 60} detik). {desc}", ""]
    L += ["## B. Urutan bicara dan serah terima", "",
          f"- **0:00** Orang 1: slide 1 dan 2",
          f"- **{mmss(ends['2'])}** Orang 2: slide 3 sampai 7",
          f"- **{mmss(ends['7'])}** Orang 3: slide 8 sampai 13",
          f"- **{mmss(ends['13'])}** Orang 1: slide 14",
          f"- **{mmss(total)}** tanya jawab, dipandu Orang 1", "",
          "Kalimat serah terima, diucapkan di akhir slide sebelum pergantian (sekitar 3 detik):", ""]
    L += [f"- Akhir slide {k}: \"{v}\"" for k, v in HANDOFF.items()]
    L += ["", "Orang 1 memindahkan slide tepat setelah kalimat terakhir tiap slide. Kalimat terakhir itu tertulis di "
              "naskah bagian C, jadi Orang 1 cukup mengikuti naskah.", "", "## C. Naskah per orang", ""]
    for name, ids, _ in ROLES:
        L += [f"### {name}", ""]
        for i in ids:
            s = by_id[i]
            L += [f"#### Slide {i} · {s['short']} ({s['dur']} detik, selesai di {mmss(ends[i])})", "", s["notes"]
                  + (f" *{HANDOFF[i]}*" if i in HANDOFF else ""), ""]
            nums = must_say(s["notes"])
            if nums:
                L += [f"**Angka wajib:** {', '.join(nums[:7])}", ""]
            L += ["Latihan 1: ........ detik. Latihan 2: ........ detik.", ""]
    L += ["## D. Tanya jawab (10 menit)", "",
          "1. Juri bertanya. Orang 1 mengulang inti pertanyaan dalam satu kalimat dan menyebut siapa yang menjawab. "
          "Pengulangan ini memberi waktu berpikir dan mencegah dua orang bicara bersamaan di Zoom.",
          "2. Penjawab memakai urutan inti, bukti (angka dan nomor slide atau sel), lalu batasan, maksimal sekitar 45 "
          "detik.",
          "3. Bila perlu bukti, Orang 1 membuka slide cadangan. Di mode slideshow PowerPoint, ketik nomor slide lalu "
          "Enter: C1 = 15, C2 = 16, C3 = 17, C4 = 18, C5 = 19, C6 = 20, C7 = 21, C8 = 22.",
          "4. Anggota lain tidak memotong. Bila ingin menambah, tunggu jawaban selesai lalu tambahkan satu kalimat.",
          "5. Bila tidak tahu, katakan terus terang, sebut yang sudah diketahui, lalu jadikan langkah berikutnya.", "",
          "**Pembagian penjawab:**", "",
          "- **Orang 2:** 06 nomor 1 sampai 7 dan 13; cadangan C1, C2, C3, C5, C7.",
          "- **Orang 3:** 06 nomor 8 sampai 12 dan 14; cadangan C4, C6, C8.",
          "- **Orang 1:** pertanyaan umum dan peran AI (06 bagian C). Ia juga menengahi bila jawaban terlalu panjang.",
          "",
          "## E. Daftar periksa Zoom", "",
          "- Hanya Orang 1 yang berbagi layar, dari PowerPoint mode presenter, sepanjang presentasi. Orang 2 menyiapkan "
          "PPTX dan PDF yang sama dan siap mengambil alih bila koneksi Orang 1 putus.",
          "- Nama tampilan: \"BISMILLAH · Nama · peran\". Kamera menyala, latar netral, mikrofon dibisukan saat tidak "
          "bicara.",
          "- Grup chat privat tim dipakai untuk mengirim angka cepat saat tanya jawab; 06 terbuka di HP masing-masing.",
          "- Uji audio dan berbagi layar 15 menit sebelum mulai, termasuk satu kali pindah slide.", "",
          "## F. Alternatif dua pembicara", "",
          "Bila satu anggota tidak ingin memaparkan: Orang 2 membawakan slide 1 sampai 7 dan Orang 3 slide 8 sampai 14. "
          "Orang 1 tetap menjadi operator slide, pemegang waktu, dan pengarah tanya jawab, serta menjawab pertanyaan "
          "umum. Serah terima hanya satu kali (slide 8). Kekurangannya, juri melihat satu anggota tidak ikut "
          "memaparkan. Bila panduan lomba mewajibkan semua anggota berbicara, pakai pembagian tiga orang.", ""]
    open(os.path.join(HERE, "07_kartu_latihan.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
