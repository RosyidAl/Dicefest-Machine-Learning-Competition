"""Deck v5: edits a copy of deck v4 with python-pptx (brief of 29 October 2026, tasks A to H).

deck/ppt03/BISMILLAH_Final_v4.pptx is only read. The script copies it to BISMILLAH_Final_v5.pptx, edits the copy,
and checks that the sha256 of v4 is unchanged afterwards.

Every number written here is traced in laporan_v5.md (fact table) to a notebook output or to a separate check script:
  - notebook: execution numbers In [n] stored in final/deliverables/DICEFEST2026_BISMILLAH.ipynb, whose 56 cells are
    identical to final/DICEFEST2026_BISMILLAH.ipynb (that file stores no outputs and no execution counts);
  - check scripts: deck/ppt01/bukti/cek_kebijakan.py and deck/ppt02/bukti/cek_zona_contoh.py. The absolute additions
    under the climate-dependent COP are not stored in any JSON; they come from a read-only run of the expressions on
    lines 64 and 65 of cek_zona_contoh.py (Z.gas * 0.8 / 3.0 and Z.gas * 0.8 / cop).

Text is replaced paragraph by paragraph. A new run copies the run properties (font, size, colour, weight) of a run
that already exists in the same box, or, where a box changes role, of a named sibling box; no default formatting is
introduced. Native charts keep their style and get new data through chart.replace_data.

usage: <python with python-pptx 1.0.2, lxml, xlsxwriter> deck/ppt04/edit_v5.py
"""
import copy
import hashlib
import re
import shutil
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V4 = ROOT / "deck" / "ppt03" / "BISMILLAH_Final_v4.pptx"
V5 = HERE / "BISMILLAH_Final_v5.pptx"
V4_SHA256 = "e9177bf62000c434a6871cde55d9740ddb8451fe258c93543cabfd147361c057"
NBSP = " "
P14 = "{http://schemas.microsoft.com/office/powerpoint/2010/main}"
LOG = []        # (slide number, shape, change), printed at the end and copied into laporan_v5.md


# ----------------------------------------------------------------------------------------------------------- helpers
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type == 6:      # group
            yield from walk(sh.shapes)


def get(slide, name):
    hits = [s for s in walk(slide.shapes) if s.name == name]
    assert len(hits) == 1, f"{name}: {len(hits)} shapes"
    return hits[0]


def get_all(slide, name):
    """same-named shapes (e.g. the three chain arrows), top to bottom"""
    return sorted((s for s in walk(slide.shapes) if s.name == name), key=lambda s: s.top)


def place(shape, x=None, y=None, w=None, h=None):
    for attr, v in (("left", x), ("top", y), ("width", w), ("height", h)):
        if v is not None:
            setattr(shape, attr, Inches(v))


def text_of(shape):
    return [p.text for p in shape.text_frame.paragraphs]


def write(shape, paras, src=None):
    """Rewrite the paragraphs of a text box.

    paras: one item per paragraph, either a string (one run) or a list of (text, (p, r)) runs. A paragraph copies the
    pPr of the original paragraph with the same index (or the last one); a run copies the rPr of run r of original
    paragraph p (default: run 0 of the same paragraph). src: a sibling box whose original paragraphs and runs are
    used as the formatting source instead of this box's own."""
    body = shape.text_frame._txBody
    old = [copy.deepcopy(p) for p in (src or shape).text_frame._txBody.findall(qn("a:p"))]
    for p in body.findall(qn("a:p")):
        body.remove(p)
    for i, para in enumerate(paras):
        k = min(i, len(old) - 1)
        if isinstance(para, str):
            para = [(para, (k, 0))]
        p = copy.deepcopy(old[k])
        for child in list(p):
            if child.tag in (qn("a:r"), qn("a:br"), qn("a:fld")):
                p.remove(child)
        end = p.find(qn("a:endParaRPr"))
        for text, (pi, ri) in para:
            r = etree.Element(qn("a:r"))
            r.append(copy.deepcopy(old[pi].findall(qn("a:r"))[ri].find(qn("a:rPr"))))
            etree.SubElement(r, qn("a:t")).text = text
            (end.addprevious(r) if end is not None else p.append(r))
        body.append(p)


def clone(slide, src, name, x, y, w, h):
    """a new text box in the style of an existing one (deep copy of its XML), placed at x, y, w, h"""
    el = copy.deepcopy(src._element)
    ids = [int(e.get("id")) for e in slide.shapes._spTree.iter(qn("p:cNvPr"))]
    cnv = el.find(".//" + qn("p:cNvPr"))
    cnv.set("id", str(max(ids) + 1))
    cnv.set("name", name)
    slide.shapes._spTree.insert_element_before(el, "p:extLst")
    shp = get(slide, name)
    place(shp, x, y, w, h)
    return shp


def delete(shape):
    shape._element.getparent().remove(shape._element)


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def plot_box(frame):
    """inner plot area of a chart frame in inches (manual layout, layoutTarget inner, as written by build_deck_v4.js)"""
    ml = frame.chart._chartSpace.find(".//" + qn("c:plotArea") + "/" + qn("c:layout") + "/" + qn("c:manualLayout"))
    assert ml.find(qn("c:layoutTarget")).get("val") == "inner"
    f = {k: float(ml.find(qn("c:" + k)).get("val")) for k in ("x", "y", "w", "h")}
    X, Y, W, H = (Emu(v).inches for v in (frame.left, frame.top, frame.width, frame.height))
    return X + f["x"] * W, Y + f["y"] * H, f["w"] * W, f["h"] * H


def log(n, shape, change):
    LOG.append((n, shape, change))


# ----------------------------------------------------------------------------------------------------- slide 1 (G)
def slide1(s):
    # speaker names: INPUT says "akan saya atur nanti", so a visible placeholder in a new box styled like the
    # assumption text (Text 14, 15 pt, accent4), under it in the free right column
    box = clone(s, get(s, "Text 14"), "pembicara", 6.85, 6.42, 5.88, 0.55)
    write(box, ["Pembicara: [PERLU DIISI: nama dan peran pembicara]"])
    log(1, "pembicara (baru)", "kotak baru gaya Text 14: [PERLU DIISI: nama dan peran pembicara]")


# ----------------------------------------------------------------------------------------------------- slide 5 (E)
def slide5(s):
    box = get(s, "Text 24")
    assert text_of(box)[0].startswith("galat EUI model tunggal −0,0022;"), text_of(box)
    # "terbesar" is not used: the regional target mean gives an equal EUI gain (laporan_v5.md, task E)
    write(box, [f"perbaikan galat EUI model tunggal dari kelompok fitur ini; prediksi untuk 30%{NBSP}nilai{NBSP}peta{NBSP}zona "
                "memakainya"])
    log(5, "Text 24 (pita KONTRIBUSI)", "−0,0022 diganti frasa tanpa 'terbesar'; angkanya pindah ke catatan")


# ----------------------------------------------------------------------------------------------------- slide 6 (C)
def slide6(s):
    # C.1: one line on the notebook, styled like the error header (Text 27: 14 pt, accent5, right-aligned), in the free
    # row right of the FAKTA DATA label
    box = clone(s, get(s, "Text 27"), "baris notebook", 3.733, 5.4, 9.0, 0.3)
    # "tanpa pelatihan" needs the 41 members' stored predictions (notebook title cell), hence "dari prediksi tersimpan"
    write(box, ["56 sel notebook, 17 modul, satu sakelar pelatihan (DICEFEST_TRAIN); jalan ulang dari prediksi "
                f"tersimpan ±5{NBSP}jam"])
    box.text_frame._txBody.find(qn("a:bodyPr")).set("anchor", "ctr")
    log(6, "baris notebook (baru)", "56 sel, 17 modul, sakelar DICEFEST_TRAIN, ±5 jam")
    # C.3 finding: the identity holds in the training data only (predictions are not forced to satisfy it)
    fact = get(s, "Text 50")
    runs = [r.text for r in fact.text_frame.paragraphs[1].runs]
    assert runs == ["galat relatif maks 6,7 × 10", "−16"], runs
    first = [(r.text, (0, i)) for i, r in enumerate(fact.text_frame.paragraphs[0].runs)]
    write(fact, [first, [(runs[0], (1, 0)), (runs[1], (1, 1)), (" di data latih", (1, 0))]])
    log(6, "Text 50 (fakta 1)", "ditambah 'di data latih' (identitas tidak ditegakkan pada prediksi)")


# ----------------------------------------------------------------------------------------------------- slide 7 (C)
STEP2 = ["koreksi galat,", "adopsi, MLP,", "salin duplikat,", "tanpa CatBoost"]


def slide7(s):
    label = get(s, "Text 6")
    assert text_of(label) == ["koreksi", "dan lainnya"], text_of(label)
    write(label, STEP2)
    cx = Emu(label.left).inches + Emu(label.width).inches / 2
    place(label, x=cx - 0.625, w=1.25, h=1.0)
    log(7, "Text 6 (label langkah 2)", "'koreksi dan lainnya' diganti isi C2: " + " ".join(STEP2))
    frame = get(s, "grafik skor per langkah")
    chart = frame.chart
    plot = chart.plots[0]
    cats = [str(c) for c in plot.categories]
    vals = list(plot.series[0].values)
    assert cats[1] == "koreksi dan lainnya"
    cats[1] = " ".join(STEP2)
    cd = CategoryChartData(number_format="General")
    cd.categories = cats
    cd.add_series(plot.series[0].name, vals)
    chart.replace_data(cd)
    log(7, "grafik skor per langkah", "kategori 2 diganti (data grafik dan workbook tertanam)")


# ---------------------------------------------------------------------------------------------------- slide 10 (A)
# absolute addition of annual electricity, GWh: COP 3 from cek_kebijakan.json "tambahan_absolut" (= In [35] gas x 0.8 / 3),
# COP per climate from the read-only run of cek_zona_contoh.py lines 64-65; ranks from cek_zona_contoh.json
ZONES = [("REG_263", 757, 805, "1 → 1"), ("REG_045", 442, 525, "3 → 3"), ("REG_270", 434, 514, "4 → 4")]
INTERP10 = (f"peringkat tambahan absolut stabil di kedua skenario COP (REG_263{NBSP}1→1, REG_045{NBSP}3→3, "
            f"REG_270{NBSP}4→4), sedangkan peringkat persen tidak (REG_290{NBSP}1→47): titik awal menurut nilai absolut")


def slide10(s):
    frame = get(s, "grafik kenaikan tertinggi")
    chart = frame.chart
    assert [str(c) for c in chart.plots[0].categories][:5] == ["REG_290", "REG_259", "REG_275", "REG_260", "REG_270"]
    cd = CategoryChartData(number_format="General")
    cd.categories = [z for z, *_ in ZONES]
    cd.add_series("COP 3", [a for _, a, _, _ in ZONES])
    cd.add_series("COP per iklim", [b for _, _, b, _ in ZONES])
    chart.replace_data(cd)
    space = chart._chartSpace
    bar = space.find(".//" + qn("c:barChart"))
    assert bar.find(qn("c:barDir")).get("val") == "bar" and bar.find(qn("c:gapWidth")).get("val") == "35"
    for ser, colour in zip(bar.findall(qn("c:ser")), ("D97B29", "A7B1BD")):   # accent1 = COP 3, accent4 = per iklim
        for dpt in ser.findall(qn("c:dPt")):
            ser.remove(dpt)
        ser.find(qn("c:spPr")).find(qn("a:solidFill")).find(qn("a:srgbClr")).set("val", colour)
    vmax = 1000
    space.find(".//" + qn("c:valAx") + "/" + qn("c:scaling") + "/" + qn("c:max")).set("val", str(vmax))
    frame.name = "grafik tambahan absolut titik awal"
    frame._element.find(".//" + qn("p:cNvPr")).set(
        "descr", "Tambahan konsumsi listrik tahunan tiga titik awal R2 (REG_263, REG_045, REG_270) dalam GWh, "
                 "COP 3 dan COP per iklim")
    log(10, "grafik (asli PowerPoint)", "lima zona persen diganti tambahan absolut 3 zona x 2 skenario COP, GWh")

    # labels: the chart has no axis text; positions follow the fixed inner plot area (clustered bars, gap 35%)
    px, py, pw, ph = plot_box(frame)
    band = ph / len(ZONES)
    t = band / (2 + 0.35)
    rows = [("Text 4", "Text 5", "Text 14", "Text 6"), ("Text 7", "Text 8", "Text 17", "Text 9"),
            ("Text 10", "Text 11", "Text 20", "Text 12")]
    val_fmt = get(s, "Text 5")            # bold tx1: COP 3 values (as the v4 value labels)
    alt_fmt = get(s, "Text 20")           # accent5 regular: COP per iklim values (as the v4 median label)
    name_fmt, rank_fmt = get(s, "Text 4"), get(s, "Text 6")
    fmt = {k: copy.deepcopy(v.text_frame._txBody) for k, v in
           (("val", val_fmt), ("alt", alt_fmt), ("name", name_fmt), ("rank", rank_fmt))}

    class Src:                             # formatting source with the original text body of a sibling box
        def __init__(self, body):
            self.text_frame = type("TF", (), {"_txBody": body})()

    for (zone, a, b, rank), (n_box, v_box, a_box, r_box) in zip(ZONES, rows):
        i = [z for z, *_ in ZONES].index(zone)
        top = py + i * band
        y1, y2 = top + 0.175 * t + t / 2, top + 0.175 * t + 1.5 * t     # centres of the two bars
        nb, vb, ab, rb = (get(s, n) for n in (n_box, v_box, a_box, r_box))
        write(nb, [zone], src=Src(fmt["name"]))
        place(nb, 0.55, (y1 + y2) / 2 - 0.13, 1.5, 0.26)
        write(vb, [f"{a}{NBSP}GWh"], src=Src(fmt["val"]))
        place(vb, px + pw * a / vmax + 0.05, y1 - 0.13, 1.0, 0.26)
        write(ab, [f"{b}{NBSP}GWh"], src=Src(fmt["alt"]))
        place(ab, px + pw * b / vmax + 0.05, y2 - 0.13, 1.0, 0.26)
        write(rb, [rank], src=Src(fmt["rank"]))
        place(rb, 4.95, (y1 + y2) / 2 - 0.13, 1.35, 0.26)
    for name in ("Text 13", "Text 15", "Text 16", "Text 18", "Text 19"):
        delete(get(s, name))
    for box in (get(s, n) for n in ("Text 5", "Text 8", "Text 11", "Text 14", "Text 17", "Text 20")):
        box.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT       # next to the bar end
    title = get(s, "Text 2")
    assert text_of(title) == ["Kenaikan listrik tahunan, lima zona tertinggi (COP 3)", "jingga = titik awal"]
    write(title, ["Tambahan listrik tahunan, tiga titik awal (GWh)",
                  f"jingga{NBSP}={NBSP}COP 3 · abu-abu{NBSP}={NBSP}COP 2,5–3,5 per iklim"])
    head = get(s, "Text 3")
    assert text_of(head) == ["peringkat COP per iklim"]
    write(head, ["peringkat", "COP 3 → iklim"], src=None)
    log(10, "label grafik", "12 kotak label dipakai ulang (zona, nilai dua skenario, peringkat), 5 kotak dihapus")

    # chain texts that pointed at the old chart or lost the annual-energy limit, and the zone order of the chart
    temuan = get(s, "Text 24")
    assert text_of(temuan) == ["SHAP: sumbangan listrik per rumah +75% di rumah besar vs kecil; lima zona tertinggi +113% "
                               "sampai +121%"]
    write(temuan, ["SHAP: sumbangan listrik per rumah +75% di rumah besar vs kecil; lima zona dengan kenaikan persen "
                   "tertinggi: +113% sampai +121%"])
    rekom = get(s, "Text 34")
    assert text_of(rekom) == ["gardu dinilai per zona sebelum subsidi; daftar ini skrining, bukan bukti kapasitas"]
    write(rekom, ["gardu dinilai per zona sebelum subsidi; skrining energi tahunan, bukan bukti kapasitas"])
    langkah = get(s, "Text 39")
    assert text_of(langkah) == ["operator menilai mulai REG_263, REG_270, REG_045 → penguatan bila perlu → subsidi "
                                "bertahap"]
    write(langkah, ["operator menilai mulai REG_263, REG_045, REG_270 → penguatan bila perlu → subsidi bertahap"])
    log(10, "Text 24, 34, 39", "TEMUAN menyebut dasar persen; REKOMENDASI memuat 'energi tahunan'; LANGKAH urut peringkat")

    # A.2: interpretation box holds one sentence; the chain is re-spaced so it fits in four lines at 14 pt
    interp = get(s, "Text 29")
    write(interp, [INTERP10])
    log(10, "Text 29 (INTERPRETASI)", "satu kalimat peringkat absolut vs persen")
    chain = [("rantai TEMUAN", "label TEMUAN", "Text 23", "Text 24", 0.93),
             ("rantai INTERPRETASI", "label INTERPRETASI", "Text 28", "Text 29", 1.15),
             ("rantai REKOMENDASI", "label REKOMENDASI", "Text 33", "Text 34", 0.70),
             ("rantai LANGKAH", "label LANGKAH", "Text 38", "Text 39", 0.93)]
    respace(s, chain, top=1.85, bottom=6.20)


def respace(s, chain, top, bottom):
    """stack the chain boxes from top to bottom with equal gaps; labels and texts keep their offsets in the box,
    arrows sit in the middle of each gap"""
    gap = (bottom - top - sum(c[-1] for c in chain)) / (len(chain) - 1)
    assert gap >= 0.17, gap
    arrows = get_all(s, "panah rantai")
    assert len(arrows) == len(chain) - 1
    y = top
    for k, (box, lab, labtxt, txt, h) in enumerate(chain):
        b = get(s, box)
        old_y = Emu(b.top).inches
        place(b, y=y, h=h)
        for name in (lab, labtxt, txt):
            el = get(s, name)
            place(el, y=y + Emu(el.top).inches - old_y)
        tb = get(s, txt)
        place(tb, h=h - 0.16)
        if k < len(chain) - 1:
            place(arrows[k], y=y + h + (gap - Emu(arrows[k].height).inches) / 2)
        y += h + gap


# ---------------------------------------------------------------------------------------------------- slide 11 (B)
TEMUAN11 = (f"sumbangan SHAP adopsi turun saat gas jaringan naik; tingkat adopsi 9,46% (gas jaringan 60–80%) "
            f"vs{NBSP}2,73% (>80%, 70% kawasan)")
Q2_REASON = "Q2 dipilih: intensitas gas di atas median, kenaikan listrik tahunan (%) di bawah median"


def slide11(s):
    temuan = get(s, "Text 5")
    assert "40,24% (gas jaringan ≤40%)" in text_of(temuan)[0]
    write(temuan, [TEMUAN11])
    log(11, "Text 5 (TEMUAN)", "kontras 40,24% vs 2,73% diganti 9,46% (pita 60–80%) vs 2,73% (opsi 1)")
    detail = get(s, "Text 16")
    assert text_of(detail) == ["REG_033: gas jaringan 83,21%, adopsi 4,99%"]
    write(detail, [Q2_REASON, "REG_033: gas jaringan 83,21%, adopsi 4,99%"])
    log(11, "Text 16 (REKOMENDASI)", "ditambah satu kalimat alasan memilih Q2 (tanpa 'paling kecil')")
    chain = [("rantai TEMUAN", "label TEMUAN", "Text 4", "Text 5", 0.93),
             ("rantai INTERPRETASI", "label INTERPRETASI", "Text 9", "Text 10", 0.70),
             ("rantai REKOMENDASI", "label REKOMENDASI", "Text 14", "Text 15", 1.42),
             ("rantai LANGKAH", "label LANGKAH", "Text 20", "Text 21", 0.70)]
    old_box = Emu(get(s, "rantai REKOMENDASI").top).inches
    old_detail = Emu(detail.top).inches - old_box
    respace(s, chain, top=1.85, bottom=6.20)
    new_box = Emu(get(s, "rantai REKOMENDASI").top).inches
    place(get(s, "Text 15"), h=0.5)
    place(detail, y=new_box + old_detail - 0.02, h=0.76)
    detail.text_frame._txBody.find(qn("a:bodyPr")).set("anchor", "t")


# ------------------------------------------------------------------------------------------------- footers (D)
FOOTERS = {     # D: numbers not printed by the notebook point to the separate check scripts (laporan_v5.md lists file/line)
    3: "Sumber: notebook bagian 2, In [31], [32]; ±37/±173\u00a0m²: skrip cek terpisah",
    4: "Sumber: notebook bagian 2, In [6]; 96,5%: skrip cek terpisah",
    5: "Sumber: notebook bagian 1 dan 2; uji label acak: skrip cek terpisah",
    8: "Sumber: notebook bagian 7.2 dan 7.3, In [34], [35]; Spearman: skrip cek terpisah",
    9: "Sumber: notebook bagian 7.1 (+65%), In [33], [34], [35]; +34%: skrip cek terpisah",
    10: "Sumber: notebook bagian 7.1 dan 7.3, In [32], [33], [35]; peringkat dan COP per iklim: skrip cek terpisah",
    11: "Sumber: notebook bagian 7.1 dan 7.3, In [33]–[35]; adopsi per kelompok dan REG_033: skrip cek terpisah",
}


def footers(prs):
    for n, text in FOOTERS.items():
        s = prs.slides[n - 1]
        box = get(s, "sel notebook")
        old = text_of(box)[0]
        write(box, [text])
        right = Emu(box.left).inches + Emu(box.width).inches
        place(box, x=right - 8.4, w=8.4)               # right-aligned text; the box grows to the left only
        log(n, "sel notebook (kaki)", f"'{old}' -> '{text}'")


# --------------------------------------------------------------------------------------------- C12 and sections (F)
# From the notebook only: code of In [6] (train_lib: XGBoost/LightGBM get NaN as is), In [7] (nn.preprocess: MLP/NCA
# fill 0 after scaling, add NaN flags, clip at ±6 sd), part 3 (gas never above 999.765 kWh in train; areas with 2 to 4
# meters under-predicted), In [25] (10 of the 41 members predict a log form) and part 2.3 (blend on the original
# scale). The notebook states no reason for the log targets: the In [23] sentence explains the log axis of the plot.
C12_TITLE = "Tidak ada pencilan yang dibuang"
C12_ROWS = [("Aspek", "Penanganan"),
            ("Nilai hilang", "XGBoost/LightGBM membaca NaN; MLP/NCA mengisi 0 plus penanda"),
            ("Pencilan", "tidak dibuang; fitur MLP/NCA dipotong ±6 simpangan baku"),
            ("Batas data", f"tebakan kawasan 2–4 meter dinaikkan; gas dekat 999.765{NBSP}kWh ditarik turun"),
            ("Target log", "10 dari 41 anggota menebak log, digabung di skala asli; "
                           "[PERLU DIISI: alasan log target]")]
C12_NOTES = "\n\n".join([
    "[Cadangan C12 · penjawab: Orang 2]",
    "Penjawab: Orang 2 (teknis), [PERLU DIISI: nama Orang 2 sesuai slide 1]",
    "Pertanyaan juri:\n"
    "- \"Bagaimana nilai hilang dan pencilan ditangani?\"\n"
    "- \"Apakah ada data yang dibuang?\"\n"
    "- \"Mengapa sebagian anggota menebak logaritma target?\"",
    "Pedoman jawaban:\n"
    "1. Nilai hilang: XGBoost dan LightGBM menerima NaN apa adanya (In [6]). MLP dan NCA mengisi NaN dengan 0 setelah "
    "standardisasi, setara rata-rata data latih, ditambah kolom penanda (In [7]). Kolom bernilai konstan dibuang "
    "(bagian 2).\n"
    "2. Pencilan: tidak ada baris yang dibuang sebagai pencilan; model ketiga target konsumsi memakai semua 429.303 baris "
    "latih (In [22]) dan target tidak dipotong saat pelatihan. Hanya nilai baku fitur MLP dan NCA yang dipotong di ±6 simpangan baku (In [7]).\n"
    "3. Batas data, bukan pencilan: gas latih tidak pernah di atas 999.765 kWh dan kawasan 2 sampai 4 meter ditebak "
    "terlalu rendah secara sistematis (bagian 3). Setelah prediksi, S1 menaikkan tebakan kawasan 2 sampai 4 meter, S3 "
    "menarik turun tebakan gas di dekat batas, S4 memotong ke rentang data latih, dan adopsi dipotong 0 sampai 100 (In [13], In [20]; "
    "faktornya di C8); "
    "leaderboard publik 0,24920 menjadi 0,24913 (In [28]).\n"
    "4. Target log: 10 dari 41 anggota menebak bentuk log; prediksinya dikembalikan dengan exp lalu digabung dengan bobot "
    "NNLS di skala asli (bagian 2, In [25]). Alasan memakai log: [PERLU DIISI: alasan log target]. Notebook hanya "
    "menampilkan sebaran gas, listrik, dan EUI dalam skala log10 karena rentangnya lebar (In [23]).",
    "Sumber: sel 4 (In [2] common.base_features, common.constant_cols), sel 8 (In [6] train_lib), sel 9 (In [7] "
    "nn.preprocess), sel 15 (In [13] pipeline._assemble, S1 sampai S4), sel 22 (In [20] resid_knn.json), sel 25 (In [22]), sel 26 dan 27 (In [23]), sel 28 (bagian 2), "
    "sel 30 (bagian 3), sel 31 (In [25]), sel 37 (In [28])",
])


def c12_table(s, src_slide):
    """the C9 table (navy header row, 15 pt rows) copied and resized to two columns over the full width"""
    el = copy.deepcopy(get(src_slide, "tabel skor adopsi")._element)
    ids = [int(e.get("id")) for e in s.shapes._spTree.iter(qn("p:cNvPr"))]
    cnv = el.find(".//" + qn("p:cNvPr"))
    cnv.set("id", str(max(ids) + 1))
    cnv.set("name", "tabel pra-pemrosesan")
    s.shapes._spTree.insert_element_before(el, "p:extLst")
    tbl = el.find(".//" + qn("a:tbl"))
    cols = tbl.find(qn("a:tblGrid")).findall(qn("a:gridCol"))
    assert len(cols) == 2
    cols[0].set("w", str(Inches(2.2)))
    cols[1].set("w", str(Inches(12.133 - 2.2)))
    rows = tbl.findall(qn("a:tr"))
    assert len(rows) == 4
    # C9's last row is a highlighted summary row (bold, tinted): drop it, and alternate copies of the two plain rows
    tbl.remove(rows[3])
    rows[2].addnext(copy.deepcopy(rows[1]))
    tbl.findall(qn("a:tr"))[3].addnext(copy.deepcopy(rows[2]))
    rows = tbl.findall(qn("a:tr"))
    row_h = 0.72
    for tr, cells in zip(rows, C12_ROWS):
        tr.set("h", str(Inches(row_h)))
        for tc, text in zip(tr.findall(qn("a:tc")), cells):
            p = tc.find(".//" + qn("a:p"))
            runs = p.findall(qn("a:r"))
            for extra in runs[1:]:
                p.remove(extra)
            runs[0].find(qn("a:t")).text = text
            p.find(qn("a:pPr")).set("algn", "l")
    shp = get(s, "tabel pra-pemrosesan")
    place(shp, 0.6, 1.85, 12.133, row_h * len(rows))
    return shp


def add_c12(prs):
    tmpl = prs.slides[23]                                         # C11: same layout CADANGAN and naming
    s = prs.slides.add_slide(tmpl.slide_layout)
    for ph in list(s.placeholders):
        delete(ph)
    for sh in tmpl.shapes:                                        # kicker, title, slide number: copies of C11
        if sh.is_placeholder:
            s.shapes._spTree.insert_element_before(copy.deepcopy(sh._element), "p:extLst")
    for sh in s.placeholders:
        if sh.placeholder_format.type == 1:
            write(sh, [C12_TITLE])
        elif sh.placeholder_format.idx == 100:
            assert text_of(sh) == ["CADANGAN · C11"]
            write(sh, ["CADANGAN · C12"])
        else:
            for fld in sh._element.iter(qn("a:fld")):
                fld.find(qn("a:t")).text = str(len(prs.slides))
    c12_table(s, prs.slides[21])
    set_notes(s, C12_NOTES)
    # on-screen words as cek_pptx.py counts them: every whitespace token with a letter or digit, label and number included
    shown = ["CADANGAN · C12", str(len(prs.slides)), C12_TITLE] + [c for row in C12_ROWS for c in row]
    words = sum(1 for t in " ".join(shown).replace(NBSP, " ").split() if re.search(r"\w", t))
    assert words <= 60, words
    # sections: C12 joins "Cadangan tanya jawab"
    sec = [e for e in prs.part._element.iter(P14 + "section") if e.get("name") == "Cadangan tanya jawab"]
    assert len(sec) == 1
    new_id = prs.slides._sldIdLst[-1].get("id")
    etree.SubElement(sec[0].find(P14 + "sldIdLst"), P14 + "sldId").set("id", new_id)
    log(25, "slide baru C12", f"layout CADANGAN, tabel gaya C9, {words} kata di layar termasuk penanda, judul, dan nomor; "
                              "section 'Cadangan tanya jawab'")
    return s


# --------------------------------------------------------------------------------------------------------- C2 (F)
C2_AFTER = ["Setelah koreksi dan penyesuaian akhir: skor ", "leaderboard", " publik 0,24913 (rata-rata lima target)"]


def slide_c2(s):
    """F: the error after correction. The notebook prints per-target errors only before correction (In [27], [28]);
    after correction and the final adjustment only the public leaderboard score of all five targets exists (In [28])."""
    box = get(s, "Text 22")
    assert text_of(box) == ["Koreksi rms/σ: gas 0,02047; EUI 0,04344; listrik 0,01195"]
    write(box, [text_of(box)[0], [(C2_AFTER[0], (0, 0)), (C2_AFTER[1], (0, 0)), (C2_AFTER[2], (0, 0))]])
    box.text_frame.paragraphs[1].runs[1].font.italic = True        # English term in italics, as on slides 7, 14, 20, 21
    place(box, h=0.85)
    log(15, "Text 22 (C2)", "ditambah skor setelah koreksi dan penyesuaian akhir (0,24913, In [28])")
    # C2 quotes the slide 7 label; the label changes, so the quote changes with it
    old = ('Langkah "koreksi dan lainnya" di grafik slide 7 (0,25025 menjadi 0,24954) juga memuat model adopsi, '
           'MLP dua keluaran, salinan duplikat, dan pencabutan CatBoost; bagian kNN sendiri sekitar 0,0004.')
    new = ('Langkah "' + " ".join(STEP2) + '" di grafik slide 7 (0,25025 menjadi 0,24954) memuat koreksi galat '
           'tetangga (kNN), model adopsi, MLP dua keluaran, salinan duplikat, dan pencabutan CatBoost; bagian koreksi kNN '
           'sendiri sekitar 0,0004. Galat per target setelah koreksi tidak dicetak di notebook; yang tersedia hanya skor '
           'leaderboard rata-rata lima target.')
    ts = [t for t in s.notes_slide.notes_text_frame._txBody.iter(qn("a:t")) if t.text and old in t.text]
    assert len(ts) == 1
    ts[0].text = ts[0].text.replace(old, new)
    log(15, "catatan C2", "kutipan label slide 7 disamakan; catatan galat setelah koreksi")




SPOKEN = {}      # slide number -> narration (word count)


# ------------------------------------------------------------------------------------------------- notes (H)
# Narration only (the timing line, the speaker role and the "Sumber" line stay as in v4 and are not counted): 950 to
# 1,000 words for 9:30, at most 110 words per minute on each slide's v4 time allocation. Every number spoken here is on
# the v5 slide; number words that name visible counts (tiga fakta, empat kuadran, tiga batas) are kept.
NARASI = {
1: "Bapak dan Ibu juri, kami tim BISMILLAH. Kota yang beralih dari pemanas gas ke pemanas listrik menghadapi trilema: emisi harus turun, jaringan tetap andal, dan biaya tetap terjangkau. Bila semua pemanas gas diganti pompa panas, median kenaikan konsumsi listrik tahunan zona mencapai 94,85 persen.",
2: "Tesis kami: transisi 330 zona perlu diurutkan per kuadran, bukan diseragamkan. Pertama, demi biaya dan emisi, retrofit selubung didahulukan di 118 zona Q1 yang memuat 47 persen konsumsi gas. Kedua, demi keandalan, gardu di zona Q1 dan Q3 dinilai sebelum subsidi pompa panas dibuka. Ketiga, agar transisi terukur, kawasan bergas zona Q2 mulai dengan pilot. Rekan kami akan menunjukkan dasar analisisnya.",
3: "Dasar pertama: fitur. Fitur utamanya sederhana: luas lantai kawasan dibagi jumlah meteran. Satu meteran kira-kira satu rumah tangga, jadi ini mendekati luas satu rumah. Fitur yang diam-diam membawa jawaban membuat skor tampak bagus, tetapi gagal di data baru. Risiko itu dikendalikan sejak desain: fitur ini hanya membagi dua kolom input baris yang sama, tanpa target, dengan rumus sama di data latih dan uji. Pada model penjelas, model terpisah yang dibaca dengan SHAP, rasio fisik turunan menyumbang 25 sampai 30 persen, dan fitur ini peringkat pertama, 2,8 dan 3,0 kali fitur kedua. Inilah dasar rekomendasi retrofit.",
4: "Fitur kedua bersifat spasial. Untuk setiap kawasan, kami mencari kawasan dengan stok bangunan paling mirip di sekitarnya, lalu memakai rata-rata konsumsinya. Kedekatan diukur dari nomor subzona: 96,5 persen pasangan subzona bernomor berurutan berada di zona yang sama, jadi penomoran itu menjadi pendekatan lokasi. Karena memakai target tetangga, fitur ini berisiko bocor sehingga dihitung out-of-fold dua lapis. Lapis luar: baris yang dinilai hanya melihat tetangga dari sembilan fold lain. Lapis dalam: baris latih hanya melihat tetangga dari empat bagian lain, jadi tidak ada baris memakai targetnya sendiri.",
5: "Desain itu kami uji, bukan kami asumsikan. Label 42.931 baris validasi kami acak, lalu fitur dihitung ulang: fitur validasi dan kawasan uji tetap identik. Pada baris latih, label bagian dalam diacak: fitur bagian itu identik, bagian lain berubah karena memakainya sebagai tetangga, sesuai rancangan. Versi yang sengaja dibuat bocor berubah, jadi uji ini mampu menangkap kebocoran. Cakupannya dua blok fitur pada satu fold. Fitur ini menurunkan galat EUI model tunggal, dan prediksi untuk 30 persen nilai peta zona memakainya.",
6: "Arsitekturnya mengikuti tiga fakta data. Pertama, di data latih, gas selalu sama dengan EUI dikali luas lantai, jadi keduanya ditangani satu jalur dengan 28 model. Kedua, porsi bangunan bersih tepat sama dengan satu rumus di 100 persen baris latih, jadi tanpa model. Ketiga, adopsi seragam per subzona, jadi adopsi 91,86 persen kawasan uji disalin dan 8,14 persen sisanya ditebak 2 LightGBM. Listrik punya jalur sendiri dengan 13 model. Model tiap jalur digabung berbobot adaptif, dikoreksi galat tetangga, lalu disesuaikan dengan batas data. Galat di kanan dihitung sebelum koreksi. Totalnya 43 model: 41 di-blend dan 2 terpisah. Semua ada di notebook 56 sel dengan satu sakelar pelatihan.",
7: "Apakah arsitektur ini bekerja di data baru? Skor publik turun di setiap langkah. Skor privat dari data tersembunyi 0,24844, peringkat 8, tidak memburuk. Validasi meniru kondisi uji karena data latih dan uji tidak dapat dibedakan: AUC 0,4997. Berkas submisinya identik setiap diulang di perangkat yang sama. Rekan kami akan melanjutkan ke kebijakan.",
8: "Sekarang kebijakan. Nilai aktual kawasan latih dan prediksi kawasan uji kami ringkas menjadi 330 zona. Ini peta kuadran, bukan peta geografis: sumbu mendatar intensitas gas zona, sumbu tegak kenaikan konsumsi listrik tahunan bila pemanas gas diganti pompa panas. Garis median membagi empat kuadran. Kuadran satu berisi 118 zona di atas median pada kedua sumbu dan memuat 47 persen konsumsi gas. Penilaian gardu menyasar kuadran satu dan tiga, yang kenaikan listriknya di atas median. Urutan zona tahan derau prediksi acak, tetapi berubah bila COP pompa panas berbeda per iklim. Jadi, peta ini daftar kandidat.",
9: "Rekomendasi pertama menyangkut retrofit. Temuan SHAP pada model penjelas: kawasan rumah terkecil punya sumbangan gas per meter persegi sekitar 65 persen lebih tinggi daripada rumah terbesar, dan satu periode bangunan sekitar 7 persen. Ini asosiasi, bukan sebab-akibat; kemungkinan karena dinding dan atap rumah kecil relatif luas. Temuan model mendukung rekomendasi ini: retrofit selubung lebih dulu di zona kuadran satu. Karena potensinya baru indikator, langkah pertamanya audit energi di REG_273, REG_270, dan REG_045, lalu pilot dengan pembanding. Pelaksananya pemerintah kota bersama penyedia jasa retrofit.",
10: "Rekomendasi kedua menyangkut keandalan jaringan. Temuan SHAP: ukuran rumah adalah fitur teratas untuk listrik per rumah, dan sumbangannya di rumah besar sekitar 75 persen lebih tinggi. Bila COP berbeda per iklim, peringkat kenaikan persen berubah: REG_290 turun dari 1 ke 47. Peringkat tambahan absolut teratas stabil, jadi titik awal dipilih menurut nilai absolut: REG_263, REG_045, dan REG_270. Ini energi tahunan; beban puncak dan kapasitas gardu tidak ada di data. Temuan model mendukung rekomendasi ini sebagai daftar skrining: gardu dinilai per zona sebelum subsidi. Pelaksananya operator jaringan dan pemerintah kota.",
11: "Rekomendasi ketiga menyangkut adopsi. Temuan SHAP: sumbangan porsi gas jaringan pada adopsi menjadi negatif di atas 80 persen. Tingkat adopsinya 9,46 persen pada gas jaringan 60 sampai 80 persen, tetapi 2,73 persen di atas 80 persen, padahal kelompok itu 70 persen kawasan. Porsi jenis pemanas berjumlah tetap, jadi ini pola komposisi, bukan sebab-akibat. Temuan model mendukung rekomendasi ini: pilot pompa panas di kuadran dua seperti REG_033, yang intensitas gasnya tinggi tetapi persentase kenaikan listriknya di bawah median. Pilot didahului cek gardu dan memakai pembanding. Pelaksananya pemerintah kota dan penyedia pompa panas. Rekan kami akan menutup.",
12: "Ada tiga batas yang kami akui. Pertama, data tanpa beban puncak dan kapasitas gardu, jadi penilaian gardu tetap tugas operator. Kedua, COP lokal, ambang median, dan makna kolom belum pasti, jadi daftar zona hanya kandidat. Ketiga, SHAP bukan sebab-akibat dan potensi retrofit baru indikator, jadi perlu audit dan pilot. Rekomendasi ini dasar prioritas, bukan pengganti audit teknis atau validasi lapangan.",
13: "Trilema tidak selesai dengan program seragam. Retrofit di kuadran satu menjawab biaya dan emisi, penilaian gardu di kuadran satu dan tiga menjaga keandalan, pilot di kuadran dua menguji transisi. Langkah pertama: audit energi, data gardu dari operator, dan satu pilot dengan pembanding. Terima kasih.",
}
EXTRA = {   # unspoken lines before "Sumber"
    5: "Bila ditanya (tidak diucapkan): angka kontribusi ini −0,0022 pada galat EUI model tunggal (notebook bagian 2), "
       "diukur sebagai tambahan di atas model yang sudah memuat rata-rata target wilayah dan agregat kawasan. Kata "
       "\"terbesar\" tidak dipakai karena dalam ablasi pengembangan, rata-rata target wilayah memberi perbaikan setara "
       "atau lebih besar.",
    9: "Bila ditanya (tidak diucapkan): +65% di notebook bagian 7.1 adalah pembulatan dari 66,5% "
       "(deck/ppt01/bukti/cek_kebijakan.json); notebook menamai pitanya sekitar 30 dan 300 m² per meter (titik tengah "
       "interval), sedangkan grafik memakai rata-rata pita ±37 dan ±173 m².",
}
SUMBER_ADD = {   # appended to the v4 "Sumber" line
    3: "; skrip cek terpisah di baris kaki = deck/ppt01/bukti/cek_kebijakan.py baris 163-176",
    4: "; skrip cek terpisah di baris kaki = deck/ppt02/bukti/cek_zona_contoh.py baris 97-104",
    6: "; sel 0 (cara menjalankan: tanpa pelatihan sekitar 5 jam), sel 2 (In [1]: sakelar DICEFEST_TRAIN), sel 3 (17 modul Python)",
    7: "; reports/cv_lb_history.csv baris v12r sampai v15ncrkd (isi langkah kedua: model adopsi, MLP dua keluaran, "
       "salinan duplikat, pencabutan CatBoost, koreksi kNN)",
    8: "; skrip cek terpisah di baris kaki = deck/ppt01/bukti/cek_kebijakan.py baris 84-124",
    9: "; skrip cek terpisah di baris kaki = deck/ppt01/bukti/cek_kebijakan.py baris 163-176 (antarkuartil 33,7)",
    10: "; tambahan absolut COP 3 (757, 442, 434 GWh): cek_kebijakan.json tambahan_absolut = sel 50 gas × 0,8 / 3; "
        "COP per iklim (805, 525, 514 GWh) dan peringkat: deck/ppt02/bukti/cek_zona_contoh.py baris 55-65, dijalankan read-only",
    11: "; sel 49 dan 50 (setiap zona Q2 di bawah median 94,85%; rata-rata Q2 +90%, Q4 +87%); "
        "skrip cek terpisah di baris kaki = deck/ppt01/bukti/cek_kebijakan.py baris 44-52 dan deck/ppt02/bukti/cek_zona_contoh.py",
}


def main_notes(prs):
    for n, narasi in NARASI.items():
        s = prs.slides[n - 1]
        old = s.notes_slide.notes_text_frame.text
        lines = [ln for ln in old.split("\n") if ln.strip()]
        head, sumber = lines[0], next(ln for ln in lines if ln.startswith("Sumber:"))
        assert re.fullmatch(r"\[\d\d:\d\d–\d\d:\d\d · \d+ detik · Orang \d \(\w+\)\]", head), head
        parts = [head, narasi] + ([EXTRA[n]] if n in EXTRA else []) + [sumber + SUMBER_ADD.get(n, "")]
        set_notes(s, "\n\n".join(parts))
        SPOKEN[n] = narasi


# ============================================================================================================ main
def main():
    assert sha256(V4) == V4_SHA256, "deck v4 differs from the inventoried file"
    shutil.copyfile(V4, V5)
    prs = Presentation(str(V5))
    assert len(prs.slides) == 24
    s = prs.slides
    slide1(s[0])
    slide5(s[4])
    slide6(s[5])
    slide7(s[6])
    slide10(s[9])
    slide11(s[10])
    footers(prs)
    main_notes(prs)
    slide_c2(s[14])
    add_c12(prs)
    prs.save(str(V5))
    assert sha256(V4) == V4_SHA256, "deck v4 changed"
    print(f"wrote {V5.relative_to(ROOT)}: {len(Presentation(str(V5)).slides)} slides; v4 sha256 unchanged")
    for n, shp, ch in LOG:
        print(f"  slide {n:2d} | {shp} | {ch}")


if __name__ == "__main__":
    main()
