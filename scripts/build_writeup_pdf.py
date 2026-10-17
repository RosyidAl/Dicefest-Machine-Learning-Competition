"""Render reports/WRITEUP_DRAFT.md (+ reports/figures) to an A4 PDF.

Needs markdown-it-py and xhtml2pdf, which are NOT part of the modelling environment. Run it from a separate venv:
    python3 -m venv pdfenv && pdfenv/bin/pip install xhtml2pdf markdown-it-py
    pdfenv/bin/python scripts/build_writeup_pdf.py [out.pdf]
"""
import os
import re
import sys

from markdown_it import MarkdownIt
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import pisa
from xhtml2pdf.default import DEFAULT_FONT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "reports", "WRITEUP_DRAFT.md")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reports", "WRITEUP.pdf")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# DejaVu covers the non-ASCII characters of the text (superscripts, arrows, approx. signs)
pdfmetrics.registerFont(TTFont("dv", FONT))
pdfmetrics.registerFont(TTFont("dvb", FONT_B))
for bold, italic in ((0, 0), (0, 1), (1, 0), (1, 1)):
    addMapping("dv", bold, italic, "dvb" if bold else "dv")
DEFAULT_FONT["dv"] = "dv"

CSS = f"""
@page {{ size: a4 portrait; margin: 2cm 2cm 2cm 2cm;
        @frame footer {{ -pdf-frame-content: foot; bottom: 1cm; margin-left: 2cm; margin-right: 2cm; height: 0.6cm; }} }}
body {{ font-family: dv; font-size: 9.5pt; line-height: 1.35; }}
h1 {{ font-size: 15pt; margin-bottom: 2pt; }}
h2 {{ font-size: 12pt; margin-top: 10pt; margin-bottom: 3pt; border-bottom: 0.5pt solid #888; }}
p {{ margin: 3pt 0; text-align: justify; }}
li {{ margin: 1pt 0; }}
table {{ border-collapse: collapse; margin: 4pt 0; }}
th, td {{ border: 0.5pt solid #999; padding: 2pt 3pt; font-size: 8pt; }}
th {{ background-color: #e8e8e8; }}
img {{ width: 16.5cm; }}
.cap {{ font-size: 8pt; color: #444; margin-bottom: 6pt; }}
code {{ font-family: dv; font-size: 8.5pt; }}
"""


def figure(m):
    """markdown image -> centred image with a caption below (xhtml2pdf has no figcaption)"""
    alt, path = m.group(1), m.group(2)
    full = os.path.join(ROOT, "reports", path)
    return f'<div><img src="{full}"/></div><div class="cap">{alt}</div>'


md = open(SRC, encoding="utf-8").read()
assert "—" not in md, "em-dash in writeup"
html_body = MarkdownIt("commonmark").enable("table").render(md)
html_body = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)" /></p>',
                   lambda m: figure(re.match(r"(.*)\|(.*)", f"{m.group(2)}|{m.group(1)}")), html_body)
html = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}"
        "<div id='foot' style='text-align:center; font-size:8pt;'><pdf:pagenumber/></div></body></html>")
with open(OUT, "wb") as fh:
    res = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
assert not res.err, res.err
print("wrote", OUT)
