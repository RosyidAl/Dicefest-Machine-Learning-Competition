"""4:3 slide PDFs of the presentation study documents (written to deck/ppt01/pdf_hp/).

Every page is a 10 x 7.5 in slide, the PowerPoint 4:3 size: a title band with the section name, Carlito body text at
18 pt (the deck font), and a footer with the document name and page number. Headings of level 2 to 4 each start a new
slide; a section that runs past one slide continues on the next with "(lanjutan)" after its title. When 17 or 16 pt
saves a slide, the section is set at that size instead, like PowerPoint's autofit (15 pt only when that keeps the
section on one slide); a trial layout counts the slides. Each document opens with a cover slide and a table of
contents whose entries are links, and the PDF outline follows the headings.

Sections about one deck slide ("Slide N" in 05 and 07, "C1." to "C8." in 05) show that slide's preview
(pratinjau/slide-NN.png, written by build_deck_v2.py) at the top right, with the text flowing around it.

Markdown tables with at most four columns and no cell longer than 180 characters stay tables, with the header row
repeated on continuation slides. Wider tables become one card per row (first cell as the card title, the other cells
as "column: value" lines); a card never splits across slides.

Needs markdown-it-py, reportlab, and Pillow (the project's pdfenv). Run build_deck_v2.py first for the previews:
    pdfenv/bin/python deck/ppt01/build_pdf_hp.py
"""
import html
import io
import os
import re

from markdown_it import MarkdownIt
from PIL import Image as PILImage
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.fonts import addMapping
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, ImageAndFlowables, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.flowables import _listWrapOn

rl_config.invariant = 1  # no timestamps or random ids: rebuilding unchanged sources gives identical files

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.dirname(HERE)
OUT = os.path.join(HERE, "pdf_hp")
PREVIEW = os.path.join(HERE, "pratinjau")
N_PREVIEW = 22  # 14 main slides + backup C1 to C8
# source, output, short title for the footer, status note shown on the cover (or None)
DOCS = [
    (os.path.join(HERE, "05_struktur_deck_v2.md"), "05_struktur_deck_v2.pdf", "05 Struktur deck v2", None),
    (os.path.join(HERE, "07_kartu_latihan.md"), "07_kartu_latihan.pdf", "07 Kartu latihan", None),
    (os.path.join(HERE, "06_qna_juri_v2.md"), "06_qna_juri_v2.pdf", "06 Tanya jawab v2", None),
    (os.path.join(DECK, "03_qna_juri.md"), "03_qna_juri.pdf", "03 Tanya jawab rinci",
     "Rujukan rinci untuk P1 sampai P5. Status bagian RISIKO sudah diperbarui di 06 Tanya jawab v2."),
    (os.path.join(DECK, "02_naskah_kebijakan.md"), "02_naskah_kebijakan.pdf", "02 Naskah kebijakan",
     "Rujukan untuk tanya jawab: pakai bagian A sampai C (arti kolom hvac dan bacaan grafik SHAP). Naskah lisan "
     "bagian E sudah digantikan narasi slide 8 sampai 13 di 05 Struktur deck v2."),
    (os.path.join(HERE, "fakta_tambahan.md"), "fakta_tambahan.pdf", "Fakta tambahan", None),
]

FONTS = os.path.join(HERE, "fonts")
for name, f in (("carl", "Carlito-Regular.ttf"), ("carlb", "Carlito-Bold.ttf"), ("carli", "Carlito-Italic.ttf"),
                ("carlbi", "Carlito-BoldItalic.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, f)))
pdfmetrics.registerFont(TTFont("mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
for bold, italic, face in ((0, 0, "carl"), (0, 1, "carli"), (1, 0, "carlb"), (1, 1, "carlbi")):
    addMapping("carl", bold, italic, face)
    addMapping("mono", bold, italic, "mono")

W, H = 10 * inch, 7.5 * inch
MX = 0.6 * inch                                    # side margin
FW = W - 2 * MX                                    # body width
BODY_Y, BODY_TOP = 0.55 * inch, H - 1.74 * inch    # body frame; the title band sits above it
FRAME_H = BODY_TOP - BODY_Y
HEAD_TOP = H - 0.42 * inch                         # top of the kicker line
SIZES = (18, 17, 16, 15)                           # body sizes, largest first
CODE = 0.84                                        # DejaVu Sans Mono looks bigger than Carlito at the same size
PAD = 6
HEX = {"navy": "#16324F", "teal": "#1F7A8C", "ink": "#1F2933", "muted": "#5F6B7A", "line": "#D5DBE3",
       "pale": "#F3F6F9", "head": "#E3EAF2", "note": "#FBEEDF", "note_ink": "#7A3F0C", "white": "#FFFFFF",
       "cover_kick": "#8CC9D6", "cover_soft": "#C9D6E3"}
C = {k: colors.HexColor(v) for k, v in HEX.items()}


def ps(name, size, font="carl", color="ink", lead=1.3, **kw):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=size * lead, textColor=C[color],
                          bulletFontName="carl", bulletFontSize=size, **kw)


FIXED = {"kicker": ps("kicker", 12.5, font="carlb", color="teal", lead=1.2),
         "covk": ps("covk", 13, font="carlb", color="cover_kick"),
         "covt": ps("covt", 34, font="carlb", color="white", lead=1.15),
         "covh": ps("covh", 14, color="cover_soft"), "note": ps("note", 15, color="note_ink")}
_STY = {}


def sty(body):
    """paragraph styles for one body size; tables, cards, and spacing scale with it"""
    if body not in _STY:
        k = body / SIZES[0]
        cell, card = 15.5 * k, 16 * k
        s = {"size": body, "cell": cell, "card": card,
             "body": ps(f"body{body}", body, spaceAfter=9 * k, allowWidows=0),
             "quote": ps(f"quote{body}", body - 1, spaceAfter=4 * k),
             "tcell": ps(f"tcell{body}", cell, lead=1.25),
             "hcell": ps(f"hcell{body}", cell - 1, font="carlb", color="white", lead=1.2),
             "cardt": ps(f"cardt{body}", card, font="carlb", color="navy", lead=1.25),
             "cardb": ps(f"cardb{body}", card, spaceAfter=3 * k),
             "code": ps(f"code{body}", 13 * k, font="mono", color="navy")}
        for depth in range(4):
            for ordered in (0, 1):
                for last in (0, 1):
                    ind = 22 * k * depth
                    s[depth, ordered, last] = ps(f"li{body}{depth}{ordered}{last}", body, allowWidows=0,
                                                 leftIndent=ind + (24 if ordered else 17) * k, bulletIndent=ind + 2,
                                                 spaceAfter=(11 if last else 5) * k)
        _STY[body] = s
    return _STY[body]


MD = MarkdownIt("commonmark").enable("table")


def esc(s):
    return html.escape(s, quote=False)


def inline(tok, size, accent=True):
    """inline token -> reportlab paragraph markup; accent colours bold runs and code navy"""
    if isinstance(tok, str):
        return tok
    out = []
    for c in tok.children or []:
        t = c.type
        if t == "strong_open":
            out.append(f'<font color="{HEX["navy"]}"><b>' if accent else "<b>")
        elif t == "strong_close":
            out.append("</b></font>" if accent else "</b>")
        elif t == "em_open":
            out.append("<i>")
        elif t == "em_close":
            out.append("</i>")
        elif t == "code_inline":
            col = f' color="{HEX["navy"]}"' if accent else ""
            out.append(f'<font face="mono" size="{size * CODE:.1f}"{col}>{esc(c.content)}</font>')
        elif t == "softbreak":
            out.append(" ")
        elif t == "hardbreak":
            out.append("<br/>")
        elif t in ("text", "html_inline"):
            out.append(esc(c.content))
    return "".join(out)


def plain(tok):
    if isinstance(tok, str):
        return tok
    return "".join(" " if c.type == "softbreak" else c.content for c in tok.children or []
                   if c.type in ("text", "code_inline", "softbreak"))


def is_label(tok):
    """a paragraph that introduces what follows: ends with a colon, or is bold from start to end"""
    ch = [c for c in tok.children or [] if not (c.type == "text" and not c.content.strip())]
    return plain(tok).rstrip().endswith(":") or bool(ch) and ch[0].type == "strong_open" and \
        ch[-1].type == "strong_close"


def cell_metrics(tok, size, bold=False):
    """one-line width of a cell and its widest unbreakable word, with code spans in the mono font"""
    segs = []
    for c in tok.children or []:
        if c.type == "strong_open":
            bold = True
        elif c.type == "strong_close":
            bold = False
        elif c.type == "code_inline":
            segs.append((c.content, "mono", size * CODE))
        elif c.type in ("text", "html_inline", "softbreak"):
            segs.append((" " if c.type == "softbreak" else c.content, "carlb" if bold else "carl", size))
    words, cur = [], 0.0
    for text, font, sz in segs:
        for k, part in enumerate(text.split(" ")):
            if k:
                words.append(cur)
                cur = 0.0
            cur += pdfmetrics.stringWidth(part, font, sz)
    words.append(cur)
    return sum(pdfmetrics.stringWidth(t, f, s) for t, f, s in segs), max(words)


def col_widths(metrics):
    """metrics[k]: (one-line width, widest word) of every cell in column k. Widths follow the one-line widths, but
    no column gets less than its widest word (capped at 150 pt; reportlab splits longer words)"""
    nat = [max(n for n, _ in col) + 2 * PAD + 2 for col in metrics]
    mn = [min(150, max(54, max(w for _, w in col) + 2 * PAD + 2)) for col in metrics]
    if sum(nat) <= FW:
        return [n * FW / sum(nat) for n in nat]
    w, fixed = [0.0] * len(nat), set()
    while True:
        free = [k for k in range(len(nat)) if k not in fixed]
        room = FW - sum(w[k] for k in fixed)
        for k in free:
            w[k] = room * nat[k] / sum(nat[j] for j in free)
        low = [k for k in free if w[k] < mn[k]]
        if not low:
            return w
        for k in low:
            w[k] = mn[k]
            fixed.add(k)


def table(toks, s):
    hdr, rows, cur, in_head = [], [], None, False
    for t in toks:
        if t.type == "thead_open":
            in_head = True
        elif t.type == "thead_close":
            in_head = False
        elif t.type == "tr_open":
            cur = []
        elif t.type == "tr_close":
            (hdr if in_head else rows).append(cur)
        elif t.type == "inline":
            cur.append(t)
    hdr = hdr[0] if hdr else []
    ncol = max([len(hdr)] + [len(r) for r in rows])
    if ncol <= 4 and max((len(c.content) for r in rows for c in r), default=0) <= 180:
        metrics = [[cell_metrics(h, s["cell"] - 1, bold=True) for h in hdr[k:k + 1]] +
                   [cell_metrics(r[k], s["cell"]) for r in rows if k < len(r)] for k in range(ncol)]
        data = [[Paragraph(inline(h, s["cell"] - 1, accent=False), s["hcell"]) for h in hdr]] if hdr else []
        data += [[Paragraph(inline(c, s["cell"]), s["tcell"]) for c in r] + [""] * (ncol - len(r)) for r in rows]
        t = Table(data, colWidths=col_widths(metrics), repeatRows=1 if hdr else 0, hAlign="LEFT", spaceAfter=10)
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), PAD),
                 ("RIGHTPADDING", (0, 0), (-1, -1), PAD), ("TOPPADDING", (0, 0), (-1, -1), 4),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LINEBELOW", (0, 0), (-1, -1), 0.5, C["line"])]
        if hdr:
            style += [("BACKGROUND", (0, 0), (-1, 0), C["navy"]),
                      ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C["white"], C["pale"]])]
        t.setStyle(TableStyle(style))
        return [t]
    cards = []
    for r in rows:
        lines = [Paragraph((f'<font color="{HEX["navy"]}"><b>{inline(h, s["card"], accent=False)}:</b></font> '
                            if h.content.strip() else "") + inline(c, s["card"]), s["cardb"])
                 for h, c in zip(hdr[1:], r[1:]) if c.content.strip()]
        t = Table([[Paragraph(inline(r[0], s["card"]), s["cardt"])], [lines]], colWidths=[FW], hAlign="LEFT",
                  spaceAfter=9 * s["size"] / SIZES[0])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), C["head"]), ("BACKGROUND", (0, 1), (0, 1), C["pale"]),
                               ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        cards.append(KeepTogether([t]))
    return cards


def box(flowables, bar=None):
    """tinted single-cell box, optionally with a coloured bar on the left (quotes)"""
    t = Table([[flowables]], colWidths=[FW], hAlign="LEFT", spaceAfter=10)
    style = [("BACKGROUND", (0, 0), (-1, -1), C["pale"]), ("LEFTPADDING", (0, 0), (-1, -1), 12),
             ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    if bar:
        style.append(("LINEBEFORE", (0, 0), (0, -1), 3, C[bar]))
    t.setStyle(TableStyle(style))
    return t


def parse(md_text, body):
    """markdown -> (title token, blocks); a block is ("h", level, inline token) or ("f", flowable)"""
    s = sty(body)
    toks = MD.parse(md_text)
    title, out, lists, quote, first_in_item, last_item = None, [], [], None, False, None

    def emit(f):
        (out.append(("f", f)) if quote is None else quote.append(f))

    i = 0
    while i < len(toks):
        t = toks[i]
        if t.type == "heading_open":
            lvl, inl = int(t.tag[1]), toks[i + 1]
            if lvl == 1:
                title = inl
            else:
                out.append(("h", lvl, inl))
            i += 3
            continue
        if t.type == "table_open":
            j = i
            while toks[j].type != "table_close":
                j += 1
            for f in table(toks[i:j + 1], s):
                emit(f)
            i = j + 1
            continue
        if t.type in ("bullet_list_open", "ordered_list_open"):
            lists.append([t.type == "ordered_list_open", int(t.attrGet("start") or 1) - 1])
        elif t.type in ("bullet_list_close", "ordered_list_close"):
            ordered = lists.pop()[0]
            if not lists and last_item is not None:  # more room after the whole list than between its items
                last_item.style = s[last_item._depth, ordered, 1]
        elif t.type == "list_item_open":
            lists[-1][1] += 1
            first_in_item = True
        elif t.type == "inline":
            if lists:
                ordered, n = lists[-1]
                bullet = (f"{n}." if ordered else "•") if first_in_item else None
                last_item = Paragraph(inline(t, body), s[len(lists) - 1, ordered, 0], bulletText=bullet)
                last_item._depth = len(lists) - 1
                f = last_item
                first_in_item = False
            else:
                f = Paragraph(inline(t, body), s["quote" if quote is not None else "body"])
            if is_label(t):
                f.keepWithNext = 1
            emit(f)
        elif t.type == "blockquote_open":
            quote = []
        elif t.type == "blockquote_close":
            q, quote = quote, None
            emit(box(q, bar="teal"))
        elif t.type in ("fence", "code_block"):
            emit(box([Paragraph(esc(line).replace(" ", "&nbsp;") or "&nbsp;", s["code"])
                      for line in t.content.rstrip("\n").split("\n")]))
        i += 1
    return title, out


class Marker(Flowable):
    """Zero-size flowable that sets the slide header and registers the section's bookmark and outline entry."""

    def __init__(self, state, kicker, title, key=None, level=0):
        super().__init__()
        self.state, self.kicker, self.title, self.key, self.level = state, kicker, title, key, level

    def wrap(self, aw, ah):
        return 0, 0

    def draw(self):
        self.state.update(kicker=self.kicker, title=self.title, fresh=True)
        if self.key:
            self.canv.bookmarkPage(self.key)
            self.canv.addOutlineEntry(plain(self.title), self.key, level=self.level)
            self.state["pages"][self.key] = self.canv.getPageNumber()


class FramedImage(Image):
    def draw(self):
        super().draw()
        self.canv.setStrokeColor(C["line"])
        self.canv.setLineWidth(0.8)
        self.canv.rect(getattr(self, "_offs_x", 0), getattr(self, "_offs_y", 0), self.drawWidth, self.drawHeight)


def slide_number(text):
    """deck slide shown by a section heading: "Slide N ..." or backup "C1." to "C8." (slides 15 to 22)"""
    m = re.match(r"Slide (\d+)\b", text)
    if m:
        return int(m.group(1))
    m = re.match(r"C([1-8])\. ", text)
    return 14 + int(m.group(1)) if m else None


def preview_images():
    """JPEG copies of the previews (smaller PDFs). They live at a fixed path, because reportlab names an embedded
    image after its file path: a random temporary folder would make every rebuild differ."""
    out, jpg = {}, os.path.join(PREVIEW, "jpg")
    os.makedirs(jpg, exist_ok=True)
    for n in range(1, N_PREVIEW + 1):
        src = os.path.join(PREVIEW, f"slide-{n:02d}.png")
        assert os.path.exists(src), f"{src} missing: run build_deck_v2.py first"
        im = PILImage.open(src).convert("RGB")
        im.thumbnail((1400, 1400))
        out[n] = os.path.join(jpg, f"slide-{n:02d}.jpg")
        im.save(out[n], quality=90)
    return out


_MEASURE = Canvas(io.BytesIO(), pagesize=(W, H))
NOPAD = {"leftPadding": 0, "rightPadding": 0, "topPadding": 0, "bottomPadding": 0}


def count_slides(F):
    """slides the flowables take when laid out from the top of a fresh slide (a trial build; it uses them up)"""
    doc = BaseDocTemplate(io.BytesIO(), pagesize=(W, H))
    doc.addPageTemplates([PageTemplate("s", [Frame(MX, BODY_Y, FW, FRAME_H, id="b", **NOPAD)])])
    doc.build(list(F))
    return doc.page


def section(trial, final, a, b, img):
    """flowables of blocks a..b-1 at the largest body size that needs the fewest slides. The trial versions are
    laid out to count slides; the final ones, untouched, go into the document."""
    def flow(versions, size):
        F = [blk[1] for blk in versions[size][a:b]]
        if img:
            F = [ImageAndFlowables(FramedImage(img, width=4.0 * inch, height=4.0 * inch * 9 / 16), F,
                                   imageLeftPadding=14, imageRightPadding=0, imageBottomPadding=10)]
        return F

    best = (count_slides(flow(trial, SIZES[0])), SIZES[0])
    if best[0] > 1:
        for size in SIZES[1:]:
            n = count_slides(flow(trial, size))
            if n < best[0] and (size >= 16 or n == 1):  # 15 pt only to keep a section on one slide
                best = (n, size)
    return flow(final, best[1])


def draw_header(canv, state):
    k = Paragraph(inline(state["kicker"], 12.5, accent=False), FIXED["kicker"])
    k.wrapOn(canv, FW, 100)
    assert len(k.blPara.lines) == 1, f"kicker longer than one line: {plain(state['kicker'])}"
    k.drawOn(canv, MX, HEAD_TOP - k.height)
    top = HEAD_TOP - k.height - 6
    for size, max_lines in ((26, 2), (24, 2), (22, 2), (20, 2), (18, 3), (16, 3)):
        title = inline(state["title"], size, accent=False)
        if not state["fresh"]:
            title += f' <font color="{HEX["muted"]}">(lanjutan)</font>'
        p = Paragraph(title, ps("title", size, font="carlb", color="navy", lead=1.12))
        p.wrapOn(canv, FW, 1000)
        if len(p.blPara.lines) <= max_lines and p.height <= top - (BODY_TOP + 14):
            break
    else:
        raise AssertionError(f"slide title does not fit: {plain(state['title'])}")
    p.drawOn(canv, MX, top - p.height)


def draw_footer(canv, page, total, short, dark=False):
    canv.setFont("carl", 10.5)
    canv.setFillColor(C["cover_soft"] if dark else C["muted"])
    canv.drawString(MX, 0.3 * inch, f"Tim BISMILLAH · {short}")
    canv.drawRightString(W - MX, 0.3 * inch, f"{page} / {total}" if total else str(page))


def toc_table(entries, pages):
    """table of contents: one linked row per heading, page number on the right; 14 pt when it is long"""
    size, pad = (15, 3) if len(entries) <= 14 else (14, 2)
    row = [ps(f"toc{size}_{d}", size, font="carlb" if d == 0 else "carl", color="navy" if d == 0 else "ink",
              leftIndent=20 * d, lead=1.18) for d in range(3)]
    num = ps(f"tocn{size}", size, color="muted", lead=1.18, alignment=TA_RIGHT)
    rows = []
    for key, depth, tok in entries:
        n = str(pages.get(key, "")) if pages else ""
        rows.append([Paragraph(f'<a href="#{key}">{inline(tok, size, accent=False)}</a>', row[min(depth, 2)]),
                     Paragraph(f'<a href="#{key}">{n}</a>', num)])
    t = Table(rows, colWidths=[FW - 54, 54], hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), pad),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), pad + 1),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.4, C["line"])]))
    return t


def story(state, title, short, note, trial, final, images, pages):
    """cover, table of contents, optional introduction, then one or more slides per heading"""
    cover = [Paragraph("DICE FEST 2026 · Babak final 30 Oktober 2026 · Tim BISMILLAH", FIXED["covk"]),
             Spacer(1, 14), Paragraph(inline(title, 34, accent=False), FIXED["covt"])]
    if note:
        n = Table([[Paragraph(f"<b>Status.</b> {esc(note)}", FIXED["note"])]], colWidths=[FW], hAlign="LEFT")
        n.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C["note"]), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
        cover += [Spacer(1, 24), n]
    cover += [Spacer(1, 24), Paragraph("Daftar isi ada di slide berikutnya: ketuk judul bagian untuk melompat ke sana.",
                                       FIXED["covh"])]
    lift = max(0.0, (H - 1.9 * inch - _listWrapOn(cover, FW, _MEASURE)[1]) / 2 - 0.25 * inch)
    S = [Spacer(1, lift)] + cover + [NextPageTemplate("slide"), PageBreak()]

    blocks = final[SIZES[0]]
    entries, body, stack, i = [], [], [], 0
    while i < len(blocks) and blocks[i][0] == "f":
        i += 1
    if i:
        entries.append(("intro", 0, "Pengantar"))
        body += [Marker(state, short, "Pengantar", "intro", 0)] + section(trial, final, 0, i, None)
    while i < len(blocks):
        _, lvl, tok = blocks[i]
        i += 1
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        key = f"s{len(entries)}"
        entries.append((key, len(stack), tok))
        if body and not isinstance(body[-1], Marker):  # consecutive headings share a slide
            body.append(PageBreak())
        body.append(Marker(state, stack[-1][1] if stack else short, tok, key, len(stack)))
        stack.append((lvl, tok))
        a = i
        while i < len(blocks) and blocks[i][0] == "f":
            i += 1
        if i > a:
            n = slide_number(plain(tok))
            body += section(trial, final, a, i, images[n] if n else None)
    return S + [Marker(state, short, "Daftar isi"), toc_table(entries, pages), PageBreak()] + body


def build_pdf(path, text, short, note, images, pages=None, total=None):
    state = {"kicker": "", "title": "", "fresh": True, "pages": {}}
    title = parse(text, SIZES[0])[0]
    trial, final = ({size: parse(text, size)[1] for size in SIZES} for _ in range(2))

    def on_cover(canv, doc):
        canv.setFillColor(C["navy"])
        canv.rect(0, 0, W, H, stroke=0, fill=1)
        draw_footer(canv, doc.page, total, short, dark=True)

    def on_slide_start(canv, doc):
        state["fresh"] = False

    def on_slide_end(canv, doc):
        draw_header(canv, state)
        draw_footer(canv, doc.page, total, short)

    doc = BaseDocTemplate(path, pagesize=(W, H), title=plain(title), author="Tim BISMILLAH",
                          subject="Bahan belajar presentasi final DICE FEST 2026")
    doc.addPageTemplates([
        PageTemplate("cover", [Frame(MX, 0.9 * inch, FW, H - 1.9 * inch, id="cover", **NOPAD)], onPage=on_cover),
        PageTemplate("slide", [Frame(MX, BODY_Y, FW, FRAME_H, id="body", **NOPAD)],
                     onPage=on_slide_start, onPageEnd=on_slide_end)])
    doc.build(story(state, title, short, note, trial, final, images, pages))
    return state["pages"], doc.page


def build(src, name, short, note, images):
    text = open(src, encoding="utf-8").read()
    assert "—" not in text, f"em-dash in {src}"
    path = os.path.join(OUT, name)
    # pass 1 finds the page of every section; pass 2 prints them in the table of contents and as "n / total".
    # Flowables keep layout state, so each pass parses afresh.
    pages, total = build_pdf(path, text, short, note, images)
    assert build_pdf(path, text, short, note, images, pages, total) == (pages, total), \
        "page numbers moved between passes"
    return path, total


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    imgs = preview_images()
    for src, name, short, note in DOCS:
        print("wrote %s (%d slides)" % build(src, name, short, note, imgs))
