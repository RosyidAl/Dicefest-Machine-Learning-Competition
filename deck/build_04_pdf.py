"""Build deck/04_struktur_deck.pdf from deck/04_struktur_deck.md: A4 landscape, one slide per page (label, statement
title, main visual with its notebook cell, on-screen text, glossary, speaker narration), then the script-trimming notes
and, on the last page, the timing summary. Content comes only from the Markdown file; images from deck/img/.

usage: deck/.venv/bin/python deck/build_04_pdf.py
"""
import os
import re

import font_roboto
from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Image, KeepInFrame, Paragraph, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "04_struktur_deck.md")
OUT = os.path.join(HERE, "04_struktur_deck.pdf")
ROBOTO = os.path.join(os.path.dirname(font_roboto.__file__), "files")
DEJAVU = "/usr/share/fonts/truetype/dejavu"

INK, MUTED, BLUE, RED = HexColor("#2b2b2b"), HexColor("#6b6b6b"), HexColor("#3b6ea5"), HexColor("#c0392b")
CARD, LINE, HEAD = HexColor("#f3f6fa"), HexColor("#d9dee5"), HexColor("#e9eef5")
W, H = landscape(A4)
M = 30
LEFT_W, GAP = 452, 18
RIGHT_W = W - 2 * M - LEFT_W - GAP

for name, path in (("Roboto", f"{ROBOTO}/Roboto-Regular.ttf"), ("Roboto-Bold", f"{ROBOTO}/Roboto-Bold.ttf"),
                   ("Roboto-Italic", f"{ROBOTO}/Roboto-Italic.ttf"),
                   ("Roboto-BoldItalic", f"{ROBOTO}/Roboto-BoldItalic.ttf"),
                   ("DejaVu", f"{DEJAVU}/DejaVuSans.ttf"), ("Mono", f"{DEJAVU}/DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, path))
pdfmetrics.registerFontFamily("Roboto", normal="Roboto", bold="Roboto-Bold", italic="Roboto-Italic",
                              boldItalic="Roboto-BoldItalic")
ROBOTO_CMAP = set(TTFont("chk", f"{ROBOTO}/Roboto-Regular.ttf").face.charToGlyph)


def style(size, color=INK, font="Roboto", leading=None, **kw):
    return ParagraphStyle(f"s{size}{font}", fontName=font, fontSize=size, leading=leading or size * 1.3,
                          textColor=color, **kw)


S_LABEL = style(8.5, BLUE, "Roboto-Bold")
S_TITLE = style(18, INK, "Roboto-Bold", 22.5)
S_CAP = style(7.5, BLUE, "Roboto-Bold", 9.5, spaceAfter=2)
S_VDESC = style(8.3, MUTED, leading=10.8)
S_BULLET = style(11.5, INK, leading=15, leftIndent=12, bulletIndent=0, spaceAfter=4)
S_GLOSS = style(7.9, MUTED, leading=10.2, spaceAfter=1.5)
S_NARR = style(9.2, INK, leading=12.6)
S_CELL = style(7.6, INK, leading=9.4)
S_CELL_B = style(7.6, INK, "Roboto-Bold", 9.4)
S_BODY = style(10.5, INK, leading=14.5, spaceAfter=4)
S_BODY_BULLET = style(10.5, INK, leading=14.5, leftIndent=14, bulletIndent=2, spaceAfter=3)
S_BODY_BULLET2 = style(10, INK, leading=13.5, leftIndent=30, bulletIndent=18, spaceAfter=2)
S_TCELL = style(10, INK, leading=12.5)
S_TCELL_B = style(10, INK, "Roboto-Bold", 12.5)


def fixglyphs(markup):
    """characters missing from Roboto (e.g. the arrow) are drawn with DejaVu; tags are left alone"""
    out = []
    for part in re.split(r"(<[^>]+>)", markup):
        if part.startswith("<"):
            out.append(part)
            continue
        out.append("".join(ch if ord(ch) < 128 or ord(ch) in ROBOTO_CMAP else f'<font name="DejaVu">{ch}</font>'
                           for ch in part))
    return "".join(out)


def inline(text):
    t = text.replace("\\|", "|").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", lambda m: f'<font name="Mono">{m.group(1)}</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<i>\1</i>", t)
    return fixglyphs(t)


def sections(md):
    """(heading, lines) for every '## ' section, ignoring headings inside code blocks"""
    out, cur, code = [], None, False
    for ln in md.split("\n"):
        if ln.startswith("```"):
            code = not code
        if not code and ln.startswith("## "):
            cur = (ln[3:].strip(), [])
            out.append(cur)
        elif cur is not None:
            cur[1].append(ln)
    return out


FIELD = re.compile(r"^- \*\*(.+?):\*\*\s*(.*)$")
SLIDE = re.compile(r"^(?:Slide (\d+)|Cadangan (C\d+))\. (.+?) \((.+?)\)$")


def fields_of(lines):
    f, cur = {}, None
    for ln in lines:
        m = FIELD.match(ln)
        if m:
            cur = m.group(1)
            f[cur] = [m.group(2)] if m.group(2) else []
        elif cur is not None:
            f[cur].append(ln)
    return f


def table_rows(lines):
    rows = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("|") and not set(s.replace("|", "").strip()) <= set("-: "):
            rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows


def make_table(rows, width, cell=S_CELL, cell_b=S_CELL_B, col_w=None, bold_last=False):
    data = [[Paragraph(inline(c), cell_b if (i == 0 or (bold_last and i == len(rows) - 1)) else cell) for c in r]
            for i, r in enumerate(rows)]
    if col_w is None:
        lens = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
        tot = sum(min(x, 60) + 6 for x in lens)
        col_w = [width * (min(x, 60) + 6) / tot for x in lens]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEAD), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def fit_image(path, max_w, max_h):
    iw, ih = PILImage.open(path).size
    w = max_w
    h = w * ih / iw
    if h > max_h:
        h = max_h
        w = h * iw / ih
    return Image(path, width=w, height=h)


def glossary(text):
    """split 'Term: definition. Term: definition.' into items (a piece without ':' belongs to the previous item)"""
    items = []
    for piece in re.split(r"(?<=\.)\s+", text.strip()):
        if items and ":" not in piece:
            items[-1] += " " + piece
        else:
            items.append(piece)
    return items


def header(c, label, title):
    p = Paragraph(inline(label), S_LABEL)
    _, h = p.wrap(W - 2 * M, 30)
    p.drawOn(c, M, H - M - h)
    y = H - M - h - 5
    t = Paragraph(inline(title), S_TITLE)
    _, th = t.wrap(W - 2 * M, 100)
    t.drawOn(c, M, y - th)
    y -= th + 7
    c.setFillColor(BLUE)
    c.rect(M, y, 64, 2.6, stroke=0, fill=1)
    return y - 12


def footer(c, page, total):
    c.setFont("Roboto", 7.5)
    c.setFillColor(MUTED)
    c.drawString(M, 15, "Tim BISMILLAH · DICE FEST 2026 · Struktur pitch deck (dibangun dari deck/04_struktur_deck.md)")
    c.drawRightString(W - M, 15, f"halaman {page} dari {total}")


def place(c, flow, x, y_top, w, h):
    f = Frame(x, y_top - h, w, h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0)
    f.addFromList([KeepInFrame(w, h, flow, mode="shrink")], c)


def slide_page(c, head, lines, page, total, n_main):
    m = SLIDE.match(head)
    num, cnum, topic, dur = m.groups()
    f = fields_of(lines)
    if num:
        label = f"SLIDE {num} DARI {n_main} · {f.get('Bagian rubrik', [''])[0]} · {dur}"
    else:
        label = f"SLIDE CADANGAN {cnum} · {topic} · {dur}"
    top = header(c, label, f["Judul"][0])
    body_h = top - 30
    # left column: the main visual
    vis = f["Visual utama"]
    desc = " ".join(x.strip() for x in vis if x.strip() and not x.strip().startswith(("![", "|")))
    img = next((re.search(r"\((.+?)\)", x).group(1) for x in vis if x.strip().startswith("![")), None)
    rows = table_rows(vis)
    left = [Paragraph("VISUAL UTAMA", S_CAP), Paragraph(inline(desc), S_VDESC), Spacer(1, 6)]
    if img:
        left.append(fit_image(os.path.join(HERE, img), LEFT_W, body_h - 60 - (150 if rows else 0)))
        left.append(Spacer(1, 6))
    if rows:
        left.append(make_table(rows, LEFT_W))
    place(c, left, M, top, LEFT_W, body_h)
    # right column: on-screen text, glossary, narration or answer guide
    right = [Paragraph("ISI TEKS DI SLIDE", S_CAP)]
    right += [Paragraph(inline(x.strip()[2:]), S_BULLET, bulletText="•") for x in f.get("Isi", [])
              if x.strip().startswith("- ")]
    right.append(Spacer(1, 4))
    right.append(Paragraph("ISTILAH (CATATAN KECIL DI SLIDE)", S_CAP))
    right += [Paragraph(inline(g), S_GLOSS) for g in glossary(" ".join(f.get("Istilah", [])))]
    right.append(Spacer(1, 7))
    nkey = next((k for k in f if k.startswith("Narasi pembicara")), None)
    if nkey:
        cap, text = nkey.upper(), " ".join(x.strip() for x in f[nkey] if x.strip())
    else:
        cap, text = "PEDOMAN JAWABAN", " ".join(x.strip() for x in f.get("Pedoman jawaban", []) if x.strip())
    card = Table([[Paragraph(inline(cap), S_CAP)], [Paragraph(inline(text), S_NARR)]], colWidths=[RIGHT_W])
    card.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CARD), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (0, 0), 7),
                              ("BOTTOMPADDING", (0, -1), (-1, -1), 8), ("TOPPADDING", (0, 1), (-1, -1), 1),
                              ("LINEBEFORE", (0, 0), (0, -1), 2.5, BLUE)]))
    right.append(card)
    place(c, right, M + LEFT_W + GAP, top, RIGHT_W, body_h)
    footer(c, page, total)
    c.showPage()


def text_page(c, title, lines, page, total):
    top = header(c, "CATATAN UNTUK TIM", title)
    blocks, blank = [], True  # (style, bullet, text); a non-bullet line right after text continues that block
    for ln in lines:
        s = ln.rstrip()
        if not s.strip():
            blank = True
            continue
        if s.lstrip().startswith("- "):
            nested = len(s) - len(s.lstrip()) >= 2
            blocks.append([S_BODY_BULLET2 if nested else S_BODY_BULLET, "–" if nested else "•", s.lstrip()[2:]])
        elif blank or not blocks:
            blocks.append([S_BODY, None, s.strip()])
        else:
            blocks[-1][2] += " " + s.strip()
        blank = False
    flow = [Paragraph(inline(t), st, bulletText=bu) for st, bu, t in blocks]
    place(c, flow, M, top, W - 2 * M, top - 30)
    footer(c, page, total)
    c.showPage()


def summary_page(c, title, lines, page, total):
    top = header(c, "RINGKASAN", f"{title}: total 8 menit 45 detik dari batas 9,5 menit")
    blocks, cur = [], []
    for ln in lines:
        if ln.strip().startswith("|"):
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    width = W - 2 * M
    t1 = make_table(table_rows(blocks[0]), width, S_TCELL, S_TCELL_B, [48, 230, 190, 120, width - 588], True)
    t2 = make_table(table_rows(blocks[1]), width, S_TCELL, S_TCELL_B, [270, 150, 170, width - 590])
    flow = [Paragraph("WAKTU PER SLIDE", S_CAP), t1, Spacer(1, 14), Paragraph("WAKTU PER BAGIAN RUBRIK", S_CAP), t2]
    place(c, flow, M, top, width, top - 30)
    footer(c, page, total)
    c.showPage()


def main():
    secs = sections(open(SRC, encoding="utf-8").read())
    slides = [(h, l) for h, l in secs if SLIDE.match(h)]
    n_main = sum(1 for h, _ in slides if h.startswith("Slide "))
    trim = next((h, l) for h, l in secs if h.startswith("Bagian naskah"))
    summ = next((h, l) for h, l in secs if h.startswith("Ringkasan waktu"))
    total = len(slides) + 2
    c = canvas.Canvas(OUT, pagesize=landscape(A4))
    c.setTitle("Struktur pitch deck final DICE FEST 2026, Tim BISMILLAH")
    c.setAuthor("Tim BISMILLAH")
    for i, (h, l) in enumerate(slides, 1):
        slide_page(c, h, l, i, total, n_main)
    text_page(c, trim[0], trim[1], len(slides) + 1, total)
    summary_page(c, summ[0], summ[1], total, total)
    c.save()
    print(f"wrote {OUT}: {total} pages ({n_main} main slides, {len(slides) - n_main} backup)")


if __name__ == "__main__":
    main()
