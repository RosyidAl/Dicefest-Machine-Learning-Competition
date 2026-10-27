"""Phone-friendly PDFs of the presentation study documents (written to deck/ppt01/pdf_hp/).

Pages are 100 x 178 mm, close to the shape of a phone screen, so a page fits the screen width without zooming. Text is
Carlito 11.5 pt (the deck font). Markdown tables with more than two columns or long cells are rewritten as one card per
row (first cell as the card title, the other cells as "column: value" lines), so nothing needs horizontal scrolling.

Needs markdown-it-py and xhtml2pdf (the project's pdfenv):
    pdfenv/bin/python deck/ppt01/build_pdf_hp.py
"""
import html
import os

from markdown_it import MarkdownIt
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import pisa
from xhtml2pdf.default import DEFAULT_FONT

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.dirname(HERE)
OUT = os.path.join(HERE, "pdf_hp")
# source, output, short title for the footer, status note shown at the top (or None)
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
for bold, italic, face in ((0, 0, "carl"), (0, 1, "carli"), (1, 0, "carlb"), (1, 1, "carlbi")):
    addMapping("carl", bold, italic, face)
DEFAULT_FONT["carl"] = "carl"
pdfmetrics.registerFont(TTFont("mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
DEFAULT_FONT["mono"] = "mono"

CSS = """
@page { size: 100mm 178mm; margin: 9mm 7mm 13mm 7mm;
        @frame footer { -pdf-frame-content: foot; bottom: 4mm; margin-left: 7mm; margin-right: 7mm; height: 6mm; } }
body { font-family: carl; font-size: 11.5pt; color: #1F2933; }
p { margin: 0 0 6pt 0; line-height: 1.38; }
li { margin: 0 0 3pt 0; line-height: 1.35; }
ul, ol { margin: 0 0 6pt 0; }
h1 { font-size: 17pt; color: #16324F; line-height: 1.15; margin: 0 0 8pt 0; }
h2 { font-size: 14.5pt; color: #16324F; line-height: 1.2; margin: 14pt 0 5pt 0; -pdf-keep-with-next: true; }
h3 { font-size: 12.5pt; color: #1F7A8C; line-height: 1.2; margin: 11pt 0 4pt 0; -pdf-keep-with-next: true; }
h4 { font-size: 12pt; color: #1F7A8C; line-height: 1.2; margin: 9pt 0 3pt 0; -pdf-keep-with-next: true; }
code { font-family: mono; font-size: 9.5pt; color: #16324F; }
.mono { font-family: mono; font-size: 9pt; color: #16324F; line-height: 1.3; margin: 0; }
.codebox { background-color: #EEF2F6; padding: 4pt; margin: 0 0 6pt 0; }
blockquote { background-color: #EEF2F6; padding: 5pt; margin: 0 0 6pt 0; }
.note { background-color: #FBEEDF; color: #9A5212; padding: 5pt; font-size: 10.5pt; margin: 0 0 8pt 0;
        line-height: 1.3; }
table.card { width: 100%; margin: 0 0 6pt 0; }
td.ch { background-color: #E3EAF2; color: #16324F; font-weight: bold; padding: 4pt 5pt 3pt 5pt; font-size: 11.5pt;
        line-height: 1.25; }
td.cb { background-color: #F7F9FB; padding: 4pt 5pt 4pt 5pt; font-size: 11pt; line-height: 1.32; }
table.plain { width: 100%; margin: 0 0 6pt 0; }
table.plain td { border-bottom: 0.5pt solid #D5DBE3; padding: 3pt 3pt; font-size: 11pt; line-height: 1.3;
                 vertical-align: top; }
table.plain td.th { color: #5F6B7A; font-weight: bold; font-size: 10pt; }
hr { color: #D5DBE3; }
"""

MD = MarkdownIt("commonmark").enable("table")


def inline(tok, env):
    return MD.renderer.renderInline(tok.children or [], MD.options, env)


def render_table(toks, env):
    """table tokens -> plain two-column table, or one card per row for wider tables"""
    header, rows, cur, in_head = [], [], None, False
    for t in toks:
        if t.type == "thead_open":
            in_head = True
        elif t.type == "thead_close":
            in_head = False
        elif t.type == "tr_open":
            cur = []
        elif t.type == "tr_close":
            (header if in_head else rows).append(cur)
        elif t.type == "inline":
            cur.append(inline(t, env))
    header = header[0] if header else []
    longest = max((len(c) for r in rows for c in r), default=0)
    if len(header) <= 2 and longest <= 70:
        out = ['<table class="plain">']
        if header:
            out.append("<tr>" + "".join(f'<td class="th">{h}</td>' for h in header) + "</tr>")
        out += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
        return "\n".join(out + ["</table>"])
    out = []
    for r in rows:
        title = r[0] if r else ""
        lines = [f"<b>{h}:</b> {c}" if h else c for h, c in zip(header[1:], r[1:]) if c.strip()]
        out.append(f'<table class="card"><tr><td class="ch">{title}</td></tr>'
                   f'<tr><td class="cb">{"<br/>".join(lines)}</td></tr></table>')
    return "\n".join(out)


def to_html(md_text):
    env = {}
    toks = MD.parse(md_text, env)
    out, buf, i = [], [], 0
    while i < len(toks):
        t = toks[i]
        if t.type == "table_open":
            out.append(MD.renderer.render(buf, MD.options, env))
            buf = []
            j = i
            while toks[j].type != "table_close":
                j += 1
            out.append(render_table(toks[i:j + 1], env))
            i = j + 1
            continue
        if t.type in ("fence", "code_block"):  # long code lines would run off a narrow page: wrap them as text
            out.append(MD.renderer.render(buf, MD.options, env))
            buf = []
            lines = "".join(f'<p class="mono">{html.escape(l) or "&nbsp;"}</p>' for l in t.content.rstrip().split("\n"))
            out.append(f'<div class="codebox">{lines}</div>')
            i += 1
            continue
        buf.append(t)
        i += 1
    out.append(MD.renderer.render(buf, MD.options, env))
    return "\n".join(out)


def build(src, name, short, note):
    text = open(src, encoding="utf-8").read()
    assert "—" not in text, f"em-dash in {src}"
    body = to_html(text)
    if note:  # status note right under the document title
        k = body.find("</h1>")
        body = body[:k + 5] + f'<div class="note">{html.escape(note)}</div>' + body[k + 5:]
    page = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}"
            f"<div id='foot' style='font-size:8.5pt; color:#5F6B7A;'>Tim BISMILLAH · {html.escape(short)} · "
            "hal <pdf:pagenumber/></div></body></html>")
    path = os.path.join(OUT, name)
    with open(path, "wb") as fh:
        res = pisa.CreatePDF(page, dest=fh, encoding="utf-8")
    assert not res.err, (name, res.err)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for src, name, short, note in DOCS:
        print("wrote", build(src, name, short, note))
