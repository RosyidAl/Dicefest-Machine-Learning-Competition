#!/usr/bin/env python3
"""render_pptx.py: visual QA renderer for .pptx decks built by pptxgenjs 4.0.1 (no LibreOffice).

Renders slides to PNG with Pillow and reports layout problems that PowerPoint would show:

  text_overflow   laid-out text is taller than the space inside its box (by more than 0.02 in)
  word_split      a single word is wider than its line and had to be broken
  out_of_bounds   a shape, picture or rendered text line extends beyond the slide edge
  table_overflow  table rows grew past the table frame, or past the slide bottom
  unsupported     an element (geometry, chart type, image format) could not be drawn
  tight_fit       advisory: the text fits, but one line fits with less than 1 pt to spare and one more
                  wrapped line would overflow (small metric differences in PowerPoint could tip it)

usage:
  render_pptx.py DECK.pptx --out DIR [--width 1600] [--pdf FILE.pdf] [--sheet FILE.jpg]
                 [--slides 1,3,5-7] [--strict] [--fonts DIR] [--no-kern] [--debug]

Writes DIR/slide-01.png ... (numbered by slide position), DIR/laporan.json (issues, counts,
warnings), and optionally a combined PDF (page = slide size) and a contact sheet (4 thumbnails
per row). The same report is printed. --strict exits with status 1 when any text_overflow,
word_split, table_overflow or out_of_bounds issue exists (tight_fit and unsupported do not
count). --no-kern measures without pair kerning, to see whether a borderline wrap depends on it.
--debug outlines every text area in magenta. Run with the Python that has Pillow (with raqm),
lxml: e.g. the scratch venv.

Text model (the part that matters most for QA):
  * Every latin typeface is measured and drawn with Carlito, which is metric-compatible with
    Calibri; typefaces other than Calibri/Carlito are listed in a warning. Glyphs missing from
    Carlito fall back to DejaVu Sans when it is installed; NBSP is measured as a space.
  * Widths come from the font's design units (font loaded at 2048 px = units per em) through
    HarfBuzz (Pillow raqm layout). Ligatures are off (Office does not apply them); pair kerning
    is on when the run size is at least the inherited `kern` threshold (pptxgenjs decks inherit
    kern="1200", i.e. 12 pt and above, from defaultTextStyle/otherStyle). Carlito's kerning pairs
    approximate Calibri's, which is why tight_fit exists.
  * Line height = 1.2207 x size x spcPct (Calibri ascent+descent+lineGap per em), or the exact
    spcPts value; mixed sizes use the largest. Extra spacing goes above the text; the baseline
    sits one descent above the bottom of the line box. spcBef of the first paragraph and spcAft
    of the last paragraph are ignored (Apache POI and the ppt01 deckkit model do the same; see
    APPLY_FIRST_SPACE_BEFORE / APPLY_LAST_SPACE_AFTER).
  * Lines break after spaces, after a hyphen between a letter/digit and a letter, after an en
    dash inside a word and at zero-width spaces; not at "/" (conservative). Trailing spaces hang
    past the margin; a word wider than the line is broken by character and reported. Tabs go to
    the next tab stop (a:tabLst, else defTabSz). Superscript/subscript runs are drawn at 2/3 size.
  * Inheritance: run/paragraph -> shape lstStyle -> p:style fontRef color -> layout placeholder
    lstStyle -> master placeholder lstStyle -> master titleStyle/bodyStyle/otherStyle (by
    placeholder type; otherStyle for plain shapes) -> presentation defaultTextStyle. bodyPr
    attributes (insets, anchor, wrap, autofit) and missing a:xfrm/geometry/fill/line inherit from
    the layout and master placeholders. Note: pptxgenjs writes no algn for a placeholder unless
    `align` is given, so title placeholders inherit titleStyle algn="ctr" and render centered, as
    in PowerPoint. Placeholder shapes on layouts/masters are never drawn (prompt text included);
    their non-placeholder shapes are drawn beneath the slide content (unless showMasterSp="0").
  * Text uses the preset's text rectangle (roundRect, ellipse, triangle, arrows, chevron, ...)
    minus the insets; flipV turns text upside down, flipH does not mirror it.
  * normAutofit/spAutoFit are noted in the issue detail but text is not shrunk, because
    PowerPoint only applies them after the text is edited.

Known approximations: gradient fills use the average stop color; pattern fills use the
foreground color; effects (shadow, glow, reflection) are ignored; table styles referenced by GUID
are not drawn (only explicit cell formatting); charts other than bar/column are drawn as labelled
placeholder boxes; chart auto-layout (no manualLayout), axis auto-scaling, tick-label gaps and
data-label offsets approximate PowerPoint's; group rotation is ignored; vertical text (vert) is
drawn horizontally; arrowheads are sized 2/3/5 x max(line width, 2 pt).
"""
from __future__ import annotations

import argparse
import colorsys
import io
import json
import math
import os
import posixpath
import re
import struct
import sys
import zipfile
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal

from lxml import etree
from PIL import Image, ImageChops, ImageDraw, ImageFont
from PIL import features as pil_features

# ----------------------------------------------------------------------------------------------
# Units and constants
# ----------------------------------------------------------------------------------------------
EMU_IN = 914400
EMU_PT = 12700
SS = 2  # supersampling factor; slides are drawn at SS x the output size, then downsampled

LINE_RATIO = 2500 / 2048  # Calibri/Carlito (ascent + descent + lineGap) per em = 1.2207
DESC_RATIO = 550 / 2048  # usWinDescent per em
SUPSUB_SCALE = 2 / 3
APPLY_FIRST_SPACE_BEFORE = False  # PowerPoint/Apache POI ignore spcBef of the first paragraph
APPLY_LAST_SPACE_AFTER = False  # ... and spcAft of the last one (also the ppt01 deckkit model)

TOL_TEXT = 0.02 * EMU_IN
TOL_OOB = 0.01 * EMU_IN
TOL_TABLE = 0.05 * EMU_IN
FIT_EPS = 0.01  # pt
TIGHT_PT = 1.0  # a line that fits with less spare width than this may wrap in PowerPoint (kerning differences)

DEF_LINS, DEF_TINS = 91440, 45720
DEF_LINE_W = 9525

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR_CANDIDATES = [os.path.normpath(os.path.join(HERE, "..", "ppt01", "fonts")),
                       "/root/Dicefest/deck/ppt01/fonts"]
FALLBACK_FONTS = {False: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  True: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}
KNOWN_FACES = {"calibri", "carlito"}
OVERFLOW_KINDS = {"text_overflow", "word_split", "table_overflow", "out_of_bounds"}

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}
R_EMBED = "{%s}embed" % NS["r"]
R_ID = "{%s}id" % NS["r"]

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=True)


def lname(el):
    t = el.tag
    return t.rsplit("}", 1)[-1] if isinstance(t, str) else ""


def find(el, path):
    return None if el is None else el.find(path, NS)


def findall(el, path):
    return [] if el is None else el.findall(path, NS)


def pct(v, default=None):
    """OOXML percentage ('90000' = 90 %, or '90%') -> fraction."""
    if v is None:
        return default
    v = v.strip()
    try:
        return float(v[:-1]) / 100.0 if v.endswith("%") else int(v) / 100000.0
    except ValueError:
        return default


def to_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


def is_true(v, default=False):
    if v is None:
        return default
    return v in ("1", "true", "on")


def inch(emu):
    return emu / EMU_IN


# ----------------------------------------------------------------------------------------------
# Package access
# ----------------------------------------------------------------------------------------------
class Package:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        self.names = set(self.zf.namelist())
        self._xml, self._rels = {}, {}

    def blob(self, name):
        return self.zf.read(name)

    def xml(self, name):
        if name not in self._xml:
            self._xml[name] = etree.fromstring(self.zf.read(name), _PARSER) if name in self.names else None
        return self._xml[name]

    def rels(self, name):
        if name not in self._rels:
            d, base = {}, posixpath.dirname(name)
            rp = posixpath.join(base, "_rels", posixpath.basename(name) + ".rels")
            if rp in self.names:
                for r in etree.fromstring(self.zf.read(rp), _PARSER):
                    tgt = r.get("Target", "")
                    if r.get("TargetMode") == "External":
                        part = None
                    elif tgt.startswith("/"):
                        part = tgt.lstrip("/")
                    else:
                        part = posixpath.normpath(posixpath.join(base, tgt))
                    d[r.get("Id")] = (r.get("Type", "").rsplit("/", 1)[-1], part)
            self._rels[name] = d
        return self._rels[name]

    def target(self, name, rid):
        v = self.rels(name).get(rid)
        return v[1] if v else None

    def first(self, name, rtype):
        for t, p in self.rels(name).values():
            if t == rtype:
                return p
        return None


# ----------------------------------------------------------------------------------------------
# Theme and colors
# ----------------------------------------------------------------------------------------------
OFFICE_COLORS = {
    "dk1": (0, 0, 0), "lt1": (255, 255, 255), "dk2": (0x44, 0x54, 0x6A), "lt2": (0xE7, 0xE6, 0xE6),
    "accent1": (0x44, 0x72, 0xC4), "accent2": (0xED, 0x7D, 0x31), "accent3": (0xA5, 0xA5, 0xA5),
    "accent4": (0xFF, 0xC0, 0x00), "accent5": (0x5B, 0x9B, 0xD5), "accent6": (0x70, 0xAD, 0x47),
    "hlink": (0x05, 0x63, 0xC1), "folHlink": (0x95, 0x4F, 0x72),
}
DEFAULT_CLR_MAP = {"bg1": "lt1", "tx1": "dk1", "bg2": "lt2", "tx2": "dk2", "accent1": "accent1",
                   "accent2": "accent2", "accent3": "accent3", "accent4": "accent4", "accent5": "accent5",
                   "accent6": "accent6", "hlink": "hlink", "folHlink": "folHlink"}
PRESET_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 128, 0),
    "blue": (0, 0, 255), "yellow": (255, 255, 0), "gray": (128, 128, 128), "grey": (128, 128, 128),
    "orange": (255, 165, 0), "purple": (128, 0, 128), "darkGray": (169, 169, 169),
    "lightGray": (211, 211, 211), "darkBlue": (0, 0, 139), "darkRed": (139, 0, 0),
    "darkGreen": (0, 100, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
}
COLOR_TAGS = {"srgbClr", "schemeClr", "sysClr", "prstClr", "scrgbClr", "hslClr"}


def hex_rgb(s, default=(0, 0, 0)):
    try:
        s = s.strip().lstrip("#")
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except (AttributeError, ValueError, IndexError):
        return default


def color_child(el):
    if el is None:
        return None
    for c in el:
        if lname(c) in COLOR_TAGS:
            return c
    return None


class Theme:
    def __init__(self, root):
        self.colors = dict(OFFICE_COLORS)
        self.major, self.minor = "Calibri Light", "Calibri"
        self.ln_styles, self.fill_styles = [], []
        if root is None:
            return
        cs = find(root, "a:themeElements/a:clrScheme")
        for el in (cs if cs is not None else []):
            c = color_child(el)
            if c is None:
                continue
            if lname(c) == "srgbClr":
                self.colors[lname(el)] = hex_rgb(c.get("val"))
            elif lname(c) == "sysClr":
                self.colors[lname(el)] = hex_rgb(c.get("lastClr") or ("000000" if c.get("val") == "windowText" else "FFFFFF"))
        fs = find(root, "a:themeElements/a:fontScheme")
        if fs is not None:
            mj, mn = find(fs, "a:majorFont/a:latin"), find(fs, "a:minorFont/a:latin")
            if mj is not None and mj.get("typeface"):
                self.major = mj.get("typeface")
            if mn is not None and mn.get("typeface"):
                self.minor = mn.get("typeface")
        fmt = find(root, "a:themeElements/a:fmtScheme")
        self.ln_styles = list(find(fmt, "a:lnStyleLst") if find(fmt, "a:lnStyleLst") is not None else [])
        self.fill_styles = list(find(fmt, "a:fillStyleLst") if find(fmt, "a:fillStyleLst") is not None else [])


class ColorCtx:
    def __init__(self, theme, clr_map):
        self.theme = theme
        self.map = clr_map

    def resolve(self, el, ph=None):
        """color element -> (r, g, b, alpha); ph is the placeholder color for schemeClr phClr."""
        if el is None:
            return None
        tag, alpha = lname(el), 1.0
        if tag == "srgbClr":
            rgb = hex_rgb(el.get("val"))
        elif tag == "schemeClr":
            name = el.get("val", "tx1")
            if name == "phClr":
                rgb = tuple(ph[:3]) if ph else (0, 0, 0)
                alpha = ph[3] if ph else 1.0
            else:
                rgb = self.theme.colors.get(self.map.get(name, name)) or self.theme.colors.get(name) or (0, 0, 0)
        elif tag == "sysClr":
            rgb = hex_rgb(el.get("lastClr") or ("000000" if el.get("val") == "windowText" else "FFFFFF"))
        elif tag == "prstClr":
            rgb = PRESET_COLORS.get(el.get("val"), (0, 0, 0))
        elif tag == "scrgbClr":
            def lin(v):
                v = max(0.0, min(1.0, pct(v, 0.0)))
                return round(255 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055))
            rgb = (lin(el.get("r")), lin(el.get("g")), lin(el.get("b")))
        elif tag == "hslClr":
            h = to_int(el.get("hue")) / 60000.0 / 360.0
            r, g, b = colorsys.hls_to_rgb(h, pct(el.get("lum"), 0.0), pct(el.get("sat"), 0.0))
            rgb = (round(r * 255), round(g * 255), round(b * 255))
        else:
            rgb = (0, 0, 0)
        r, g, b = [c / 255.0 for c in rgb]
        for t in el:
            n = lname(t)
            v = pct(t.get("val"), 0.0)
            if n == "alpha":
                alpha = v
            elif n == "alphaMod":
                alpha *= v
            elif n == "alphaOff":
                alpha += v
            elif n in ("lumMod", "lumOff", "satMod", "satOff", "hueMod", "hueOff", "lum", "sat", "hue"):
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                if n == "lumMod":
                    l *= v
                elif n == "lumOff":
                    l += v
                elif n == "lum":
                    l = v
                elif n == "satMod":
                    s *= v
                elif n == "satOff":
                    s += v
                elif n == "sat":
                    s = v
                elif n == "hueMod":
                    h = (h * v) % 1.0
                elif n == "hueOff":
                    h = (h + to_int(t.get("val")) / 60000.0 / 360.0) % 1.0
                elif n == "hue":
                    h = (to_int(t.get("val")) / 60000.0 / 360.0) % 1.0
                r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
            elif n == "tint":
                r, g, b = [c * v + (1 - v) for c in (r, g, b)]
            elif n == "shade":
                r, g, b = [c * v for c in (r, g, b)]
            elif n == "inv":
                r, g, b = 1 - r, 1 - g, 1 - b
            elif n == "gray":
                y = 0.3 * r + 0.59 * g + 0.11 * b
                r = g = b = y
        clamp = lambda c: max(0, min(255, round(c * 255)))
        return (clamp(r), clamp(g), clamp(b), max(0.0, min(1.0, alpha)))


NOFILL = "none"


def parse_fill(parent, cc, ph=None):
    """First fill child of parent -> None (not specified), NOFILL, rgba tuple or ('blip', el)."""
    if parent is None:
        return None
    for el in parent:
        n = lname(el)
        if n == "noFill":
            return NOFILL
        if n == "solidFill":
            c = color_child(el)
            return cc.resolve(c, ph) if c is not None else (0, 0, 0, 1.0)
        if n == "gradFill":
            cols = [cc.resolve(color_child(gs), ph) for gs in findall(el, "a:gsLst/a:gs") if color_child(gs) is not None]
            if not cols:
                return NOFILL
            return tuple(round(sum(c[i] for c in cols) / len(cols)) for i in range(3)) + (sum(c[3] for c in cols) / len(cols),)
        if n == "pattFill":
            c = color_child(find(el, "a:fgClr"))
            return cc.resolve(c, ph) if c is not None else (0, 0, 0, 1.0)
        if n == "blipFill":
            return ("blip", el)
        if n == "grpFill":
            return NOFILL
    return None


@dataclass
class LineSpec:
    width: float  # EMU
    color: tuple
    dash: str = "solid"
    head: tuple = None  # (type, w, len)
    tail: tuple = None


def parse_line(ln, cc, ph=None, style_color=None, style_width=None):
    """a:ln (or None) plus the p:style lnRef color/width -> LineSpec or None (no line)."""
    if ln is None:
        return LineSpec(style_width or DEF_LINE_W, style_color) if style_color is not None else None
    fill = parse_fill(ln, cc, ph)
    if fill == NOFILL:
        return None
    color = fill if isinstance(fill, tuple) and len(fill) == 4 and not isinstance(fill[0], str) else style_color
    if color is None:
        return None
    w = to_int(ln.get("w"), style_width or DEF_LINE_W) if ln.get("w") is not None else (style_width or DEF_LINE_W)
    pd = find(ln, "a:prstDash")
    spec = LineSpec(w, color, pd.get("val", "solid") if pd is not None else "solid")
    for tag in ("headEnd", "tailEnd"):
        e = find(ln, "a:" + tag)
        if e is not None and e.get("type", "none") != "none":
            setattr(spec, "head" if tag == "headEnd" else "tail", (e.get("type"), e.get("w", "med"), e.get("len", "med")))
    return spec


def style_refs(sp, cc, theme):
    """p:style -> (fill rgba|None, line color|None, line width|None, font color element|None)."""
    st = find(sp, "p:style")
    if st is None:
        return None, None, None, None
    fill = lc = lw = fc = None
    fr = find(st, "a:fillRef")
    if fr is not None and to_int(fr.get("idx")) > 0 and color_child(fr) is not None:
        fill = cc.resolve(color_child(fr))
    lr = find(st, "a:lnRef")
    if lr is not None and to_int(lr.get("idx")) > 0 and color_child(lr) is not None:
        lc = cc.resolve(color_child(lr))
        i = to_int(lr.get("idx")) - 1
        if 0 <= i < len(theme.ln_styles):
            lw = to_int(theme.ln_styles[i].get("w"), DEF_LINE_W)
    fo = find(st, "a:fontRef")
    if fo is not None:
        fc = color_child(fo)
    return fill, lc, lw, fc


# ----------------------------------------------------------------------------------------------
# Fonts and measurement
# ----------------------------------------------------------------------------------------------
def read_cmap(path):
    """Set of code points covered by a TrueType font (cmap formats 4 and 12)."""
    d = open(path, "rb").read()
    n = struct.unpack(">H", d[4:6])[0]
    tabs = {}
    for i in range(n):
        tag, _, off, ln_ = struct.unpack(">4sIII", d[12 + 16 * i:28 + 16 * i])
        tabs[tag.decode("latin-1")] = (off, ln_)
    cps = set()
    if "cmap" not in tabs:
        return cps
    off = tabs["cmap"][0]
    nt = struct.unpack(">H", d[off + 2:off + 4])[0]
    for i in range(nt):
        _, _, so = struct.unpack(">HHI", d[off + 4 + 8 * i:off + 12 + 8 * i])
        st = off + so
        fmt = struct.unpack(">H", d[st:st + 2])[0]
        if fmt == 4:
            segx2 = struct.unpack(">H", d[st + 6:st + 8])[0]
            seg = segx2 // 2
            ends = struct.unpack(">%dH" % seg, d[st + 14:st + 14 + segx2])
            starts = struct.unpack(">%dH" % seg, d[st + 16 + segx2:st + 16 + 2 * segx2])
            for s, e in zip(starts, ends):
                if s != 0xFFFF:
                    cps.update(range(s, e + 1))
        elif fmt == 12:
            ng = struct.unpack(">I", d[st + 12:st + 16])[0]
            for g in range(ng):
                s, e, _ = struct.unpack(">III", d[st + 16 + 12 * g:st + 28 + 12 * g])
                cps.update(range(s, e + 1))
    return cps


@dataclass(frozen=True)
class RunStyle:
    size: float = 18.0  # pt (nominal)
    bold: bool = False
    italic: bool = False
    color: tuple = (0, 0, 0, 1.0)  # None = no fill (invisible)
    underline: bool = False
    strike: bool = False
    baseline: float = 0.0  # fraction of size, + = superscript
    spc: float = 0.0  # pt added after each character
    kern: bool = False
    caps: bool = False
    face: str = "+mn-lt"

    @property
    def eff_size(self):
        return self.size * (SUPSUB_SCALE if self.baseline else 1.0)


class FontBook:
    FILES = {(False, False): "Carlito-Regular.ttf", (True, False): "Carlito-Bold.ttf",
             (False, True): "Carlito-Italic.ttf", (True, True): "Carlito-BoldItalic.ttf"}
    REF = 2048  # load size in px = units per em, so advances are exact design units

    def __init__(self, font_dir):
        self.paths = {k: os.path.join(font_dir, f) for k, f in self.FILES.items()}
        for p in self.paths.values():
            if not os.path.exists(p):
                raise SystemExit("font not found: %s (use --fonts DIR)" % p)
        self.raqm = pil_features.check("raqm")
        self.engine = ImageFont.Layout.RAQM if self.raqm else ImageFont.Layout.BASIC
        self.ref = {k: ImageFont.truetype(p, self.REF, layout_engine=self.engine) for k, p in self.paths.items()}
        self.cmap = read_cmap(self.paths[(False, False)])
        self.fb_paths = {k: p for k, p in FALLBACK_FONTS.items() if os.path.exists(p)}
        self.fb_ref = {k: ImageFont.truetype(p, self.REF, layout_engine=self.engine) for k, p in self.fb_paths.items()}
        self.major, self.minor = "Calibri Light", "Calibri"
        self.kerning = True  # False: ignore the kern threshold (sensitivity check)
        self.unknown, self.missing = {}, {}
        self._w, self._draw = {}, {}

    def note_face(self, name):
        if not name:
            name = "+mn-lt"
        real = {"+mn-lt": self.minor, "+mj-lt": self.major}.get(name, name)
        if real.startswith("+"):
            real = self.minor
        if real.strip().lower() not in KNOWN_FACES:
            self.unknown[real] = self.unknown.get(real, 0) + 1

    def feats(self, kern):
        if not self.raqm:
            return None
        f = ["-liga", "-clig", "-calt", "-dlig"]
        return f if (kern and self.kerning) else f + ["-kern"]

    def chunks(self, text):
        """Split text into (substring, use_fallback) by Carlito glyph coverage."""
        text = text.replace("\u00a0", " ")  # Carlito lacks NBSP; Calibri gives it the advance of a space
        if all(ord(c) in self.cmap for c in text) or not self.fb_ref:
            return [(text, False)]
        out = []
        for ch in text:
            fb = ord(ch) not in self.cmap
            if fb:
                self.missing[ch] = self.missing.get(ch, 0) + 1
            if out and out[-1][1] == fb:
                out[-1] = (out[-1][0] + ch, fb)
            else:
                out.append((ch, fb))
        return out

    def width_em(self, text, bold, italic, kern):
        kern = kern and self.kerning
        key = (text, bold, italic, kern)
        w = self._w.get(key)
        if w is None:
            w = 0.0
            for chunk, fb in self.chunks(text):
                f = self.fb_ref[bold if bold in self.fb_ref else False] if fb else self.ref[(bold, italic)]
                w += f.getlength(chunk, features=self.feats(kern)) / self.REF
            self._w[key] = w
        return w

    def width(self, text, st):
        """advance width in pt of text drawn in style st"""
        if not text:
            return 0.0
        t = text.upper() if st.caps else text
        return self.width_em(t, st.bold, st.italic, st.kern) * st.eff_size + st.spc * len(t)

    def draw_font(self, bold, italic, px, fb=False):
        key = (bold, italic, round(px * 8) / 8, fb)
        f = self._draw.get(key)
        if f is None:
            path = self.fb_paths.get(bold, self.fb_paths.get(False)) if fb else self.paths[(bold, italic)]
            f = ImageFont.truetype(path, max(1.0, key[2]), layout_engine=self.engine)
            self._draw[key] = f
        return f


# ----------------------------------------------------------------------------------------------
# Text styles (lstStyle / pPr / rPr) and inheritance
# ----------------------------------------------------------------------------------------------
def parse_ppr(el):
    d = {}
    if el is None:
        return d
    for k in ("marL", "marR", "indent", "defTabSz"):
        if el.get(k) is not None:
            d[k] = to_int(el.get(k))
    if el.get("algn"):
        d["algn"] = el.get("algn")
    for sub in ("lnSpc", "spcBef", "spcAft"):
        s = find(el, "a:" + sub)
        if s is not None:
            sp, st = find(s, "a:spcPct"), find(s, "a:spcPts")
            if sp is not None:
                d[sub] = ("pct", pct(sp.get("val"), 1.0))
            elif st is not None:
                d[sub] = ("pts", to_int(st.get("val")) / 100.0)
    for c in el:
        n = lname(c)
        if n == "buNone":
            d["bullet"] = ("none",)
        elif n == "buChar":
            d["bullet"] = ("char", c.get("char", "\u2022"))
        elif n == "buAutoNum":
            d["bullet"] = ("auto", c.get("type", "arabicPeriod"), to_int(c.get("startAt"), 1))
        elif n == "buBlip":
            d["bullet"] = ("char", "\u2022")
        elif n == "buFont":
            d["buFont"] = c.get("typeface")
        elif n == "buFontTx":
            d["buFont"] = None
        elif n == "buSzPct":
            d["buSz"] = ("pct", pct(c.get("val"), 1.0))
        elif n == "buSzPts":
            d["buSz"] = ("pts", to_int(c.get("val")) / 100.0)
        elif n == "buSzTx":
            d["buSz"] = None
        elif n == "buClr":
            d["buClr"] = color_child(c)
        elif n == "buClrTx":
            d["buClr"] = None
        elif n == "tabLst":
            d["tabs"] = sorted(to_int(t.get("pos")) for t in findall(c, "a:tab"))
    return d


def parse_rpr(el):
    d = {}
    if el is None:
        return d
    if el.get("sz"):
        d["sz"] = to_int(el.get("sz"), 1800) / 100.0
    for k in ("b", "i"):
        if el.get(k) is not None:
            d[k] = is_true(el.get(k))
    if el.get("u") is not None:
        d["u"] = el.get("u") != "none"
    if el.get("strike") is not None:
        d["strike"] = el.get("strike") != "noStrike"
    if el.get("baseline") is not None:
        d["baseline"] = pct(el.get("baseline"), 0.0)
    if el.get("spc") is not None:
        d["spc"] = to_int(el.get("spc")) / 100.0
    if el.get("kern") is not None:
        d["kern"] = to_int(el.get("kern")) / 100.0
    if el.get("cap") is not None:
        d["cap"] = el.get("cap")
    for c in el:
        n = lname(c)
        if n == "solidFill":
            d["color"] = color_child(c)
        elif n == "noFill":
            d["color"] = NOFILL
        elif n == "gradFill":
            gs = find(c, "a:gsLst/a:gs")
            d["color"] = color_child(gs)
        elif n == "latin":
            d["latin"] = c.get("typeface")
    return d


_LST_CACHE = {}


def parse_lst(el):
    """a:lstStyle / p:titleStyle / p:bodyStyle / p:otherStyle / p:defaultTextStyle -> {level: (ppr, rpr)}"""
    if el is None:
        return None
    key = id(el)
    hit = _LST_CACHE.get(key)
    if hit is not None and hit[0] is el:
        return hit[1]
    levels = {}
    defp = find(el, "a:defPPr")
    base = (parse_ppr(defp), parse_rpr(find(defp, "a:defRPr"))) if defp is not None else ({}, {})
    for lvl in range(1, 10):
        lv = find(el, "a:lvl%dpPr" % lvl)
        if lv is not None:
            p = dict(base[0])
            p.update(parse_ppr(lv))
            r = dict(base[1])
            r.update(parse_rpr(find(lv, "a:defRPr")))
            levels[lvl] = (p, r)
        elif defp is not None:
            levels[lvl] = base
    _LST_CACHE[key] = (el, levels)
    return levels


def resolve_level(chain, lvl):
    p, r = {}, {}
    for src in reversed(chain):
        if src and lvl in src:
            p.update(src[lvl][0])
            r.update(src[lvl][1])
    return p, r


@dataclass
class Para:
    ppr: dict
    runs: list  # (kind, text, RunStyle); kind in {"text", "br"}
    end: RunStyle
    lvl: int = 0
    lines: list = field(default_factory=list)


def make_style(r, ctx):
    col = r.get("color")
    if isinstance(col, str) and col == NOFILL:
        rgba = None
    elif col is None:
        rgba = (0, 0, 0, 1.0)
    else:
        rgba = ctx.cc.resolve(col)
    size = r.get("sz", 18.0)
    kmin = r.get("kern")
    return RunStyle(size=size, bold=r.get("b", False), italic=r.get("i", False), color=rgba,
                    underline=r.get("u", False), strike=r.get("strike", False), baseline=r.get("baseline", 0.0),
                    spc=r.get("spc", 0.0), kern=bool(kmin) and size >= kmin - 1e-9, caps=r.get("cap") == "all",
                    face=r.get("latin") or "+mn-lt")


def parse_paragraphs(ctx, txbody, chain):
    paras = []
    for p in findall(txbody, "a:p"):
        ppr_el = find(p, "a:pPr")  # pptxgenjs may repeat a:pPr inside a paragraph; only the first counts
        lvl = to_int(ppr_el.get("lvl"), 0) if ppr_el is not None else 0
        base_p, base_r = resolve_level(chain, lvl + 1)
        ppr = dict(base_p)
        ppr.update(parse_ppr(ppr_el))
        if isinstance(ppr.get("buClr"), etree._Element):
            ppr["buClr"] = ctx.cc.resolve(ppr["buClr"])
        runs, end_r = [], dict(base_r)
        for el in p:
            n = lname(el)
            if n in ("r", "fld", "br"):
                r = dict(base_r)
                rpr = find(el, "a:rPr")
                r.update(parse_rpr(rpr))
                hl = find(rpr, "a:hlinkClick")
                if hl is not None and not any(x.get("val") == "tx" for x in hl.iter() if lname(x) == "hlinkClr"):
                    r["color"] = etree.Element("{%s}schemeClr" % NS["a"], val="hlink")
                    r["u"] = True
                if n == "br":
                    runs.append(("br", "", make_style(r, ctx)))
                    continue
                text = "".join(t.text or "" for t in findall(el, "a:t"))
                if n == "fld" and (el.get("type") or "").lower() == "slidenum":
                    text = str(ctx.slide_no)
                runs.append(("text", text, make_style(r, ctx)))
            elif n == "endParaRPr":
                end_r.update(parse_rpr(el))
        paras.append(Para(ppr, runs, make_style(end_r, ctx), lvl))
    return paras


def plain_text(paras):
    return " ".join("".join(t for k, t, _ in p.runs if k == "text") for p in paras).strip()


# ----------------------------------------------------------------------------------------------
# Text layout
# ----------------------------------------------------------------------------------------------
@dataclass
class Frag:
    text: str
    st: RunStyle
    w: float
    x: float = 0.0
    gap: bool = False  # ends a word (justification gap after it)


@dataclass
class Line:
    frags: list
    left: float
    right: float
    para: int
    bullet: Frag = None
    width: float = 0.0  # from left to the end of the last non-space character
    height: float = 0.0
    descent: float = 0.0
    nat: float = 0.0  # single-spaced natural height
    top: float = 0.0
    last: bool = False
    forced: bool = False
    br_style: RunStyle = None


@dataclass
class TextLayout:
    lines: list
    height: float
    splits: list
    left: float = 0.0  # min x of any drawn text (pt)
    right: float = 0.0  # max x of any drawn text (pt)


HYPHENS = "-\u2010"


def make_atoms(runs):
    """runs -> atoms: ('word', core_pieces, trail_pieces) | ('br', style) | ('tab', style)."""
    atoms, chars = [], []

    def group(seq):
        out = []
        for ch, st in seq:
            if out and out[-1][1] is st:
                out[-1][0].append(ch)
            else:
                out.append(([ch], st))
        return [("".join(cs), st) for cs, st in out]

    def flush():
        if not chars:
            return
        s = "".join(c for c, _ in chars)
        n = len(s)
        cuts = [0]
        for i in range(n - 1):
            c, nx = s[i], s[i + 1]
            if c == " " and nx != " ":
                cuts.append(i + 1)
            elif c in HYPHENS and i > 0 and s[i - 1].isalnum() and nx.isalpha():
                cuts.append(i + 1)
            elif c == "\u2013" and i > 0 and s[i - 1] != " " and nx != " ":
                cuts.append(i + 1)
            elif c == "\u200b":
                cuts.append(i + 1)
        cuts.append(n)
        for a, b in zip(cuts, cuts[1:]):
            seg = chars[a:b]
            k = len(seg)
            while k > 0 and seg[k - 1][0] == " ":
                k -= 1
            atoms.append(("word", group(seg[:k]), group(seg[k:])))
        chars.clear()

    for kind, text, st in runs:
        if kind == "br":
            flush()
            atoms.append(("br", st))
            continue
        for ch in text:
            if ch == "\t":
                flush()
                atoms.append(("tab", st))
            elif ch in "\r\n\v":
                flush()
                atoms.append(("br", st))
            else:
                chars.append((ch, st))
    flush()
    return atoms


def roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def autonum_text(scheme, n):
    if scheme.startswith("alphaLc"):
        core = chr(ord("a") + (n - 1) % 26) * (1 + (n - 1) // 26)
    elif scheme.startswith("alphaUc"):
        core = chr(ord("A") + (n - 1) % 26) * (1 + (n - 1) // 26)
    elif scheme.startswith("romanLc"):
        core = roman(n).lower()
    elif scheme.startswith("romanUc"):
        core = roman(n)
    else:
        core = str(n)
    if scheme.endswith("ParenBoth"):
        return "(%s)" % core
    if scheme.endswith("ParenR"):
        return core + ")"
    if scheme.endswith("Period"):
        return core + "."
    return core


SYMBOL_BULLETS = {"\xa7": "\u25a0", "\xd8": "\u27a2", "\xfc": "\u2713", "q": "\u25a1", "v": "\u2756", "n": "\u25a0", "l": "\u25cf"}


def layout_text(paras, width, fonts, wrap=True):
    """Lay out paragraphs in a text area `width` pt wide. Returns TextLayout (all values in pt)."""
    lines, splits = [], []
    auto = {}
    for pi, para in enumerate(paras):
        ppr = para.ppr
        marL = ppr.get("marL", 0) / EMU_PT
        indent = ppr.get("indent", 0) / EMU_PT
        right = width - ppr.get("marR", 0) / EMU_PT
        has_text = any(k == "text" and t for k, t, _ in para.runs)
        first_st = next((st for k, t, st in para.runs if k == "text" and t), para.end)
        # bullet / autonumber
        bullet = None
        b = ppr.get("bullet", ("none",))
        if b[0] == "auto" and has_text:
            prev = auto.get(para.lvl)
            num = prev[1] + 1 if prev and prev[0] == b[1] else b[2]
            auto[para.lvl] = (b[1], num)
        elif has_text:
            auto.pop(para.lvl, None)
        for k in [k for k in auto if k > para.lvl]:
            del auto[k]
        if has_text and b[0] in ("char", "auto"):
            if b[0] == "char":
                btxt = b[1]
                if (ppr.get("buFont") or "").lower() in ("wingdings", "symbol", "wingdings 2", "wingdings 3"):
                    btxt = SYMBOL_BULLETS.get(btxt, "\u2022")
            else:
                btxt = autonum_text(b[1], auto[para.lvl][1])
            bsz = ppr.get("buSz")
            size = first_st.size * (bsz[1] if bsz and bsz[0] == "pct" else 1.0) if not (bsz and bsz[0] == "pts") else bsz[1]
            bcol = ppr.get("buClr") if isinstance(ppr.get("buClr"), tuple) else first_st.color
            bst = replace(first_st, size=size, color=bcol, underline=False, strike=False, baseline=0.0, spc=0.0)
            bullet = Frag(btxt, bst, fonts.width(btxt, bst))
        first_left = marL + indent
        if bullet:
            bullet.x = first_left
            start = max(marL, first_left + bullet.w) if indent < 0 else first_left + bullet.w
        else:
            start = first_left
        tab = (ppr.get("defTabSz") or EMU_IN) / EMU_PT
        tabs = [t / EMU_PT for t in ppr.get("tabs", [])]

        para_lines = []
        cur = Line([], start, right, pi, bullet=bullet)
        x = start

        def close(line, forced=False, br_st=None):
            line.forced = forced
            line.br_style = br_st
            para_lines.append(line)

        def newline(line):
            nonlocal x
            close(line)
            nl = Line([], marL, right, pi)
            x = marL
            return nl

        def place(pieces, gap_after=False):
            nonlocal x
            for t, st in pieces:
                w = fonts.width(t, st)
                cur.frags.append(Frag(t, st, w, x))
                x += w
            if gap_after and cur.frags:
                cur.frags[-1].gap = True

        for atom in make_atoms(para.runs):
            if atom[0] == "br":
                close(cur, True, atom[1])
                cur = Line([], marL, right, pi)
                x = marL
                continue
            if atom[0] == "tab":
                stops = [t for t in tabs if t > x + 1e-6]
                nx = stops[0] if stops else (math.floor(x / tab + 1e-9) + 1) * tab
                if wrap and cur.frags and nx > right + FIT_EPS:
                    cur = newline(cur)
                    nx = (math.floor(x / tab + 1e-9) + 1) * tab
                cur.frags.append(Frag("", atom[1], nx - x, x, gap=False))
                x = nx
                cur.width = x - cur.left
                continue
            core, trail = atom[1], atom[2]
            cw = sum(fonts.width(t, st) for t, st in core)
            if wrap and cur.frags and x + cw > right + FIT_EPS:
                cur = newline(cur)
            if wrap and core and x + cw > right + FIT_EPS:
                # the word is wider than an empty line: break it by character (PowerPoint does too)
                word = "".join(t for t, _ in core)
                splits.append({"word": word, "width_pt": cw, "line_pt": right - x})
                chars = [(ch, st) for t, st in core for ch in t]
                i = 0
                while i < len(chars):
                    j = i + 1
                    while j < len(chars):
                        seg = chars[i:j + 1]
                        sw = sum(fonts.width(ch, st) for ch, st in seg)
                        if x + sw > right + FIT_EPS:
                            break
                        j += 1
                    piece = chars[i:j]
                    grouped = []
                    for ch, st in piece:
                        if grouped and grouped[-1][1] is st:
                            grouped[-1] = (grouped[-1][0] + ch, st)
                        else:
                            grouped.append((ch, st))
                    place(grouped)
                    cur.width = x - cur.left
                    i = j
                    if i < len(chars):
                        cur = newline(cur)
                place(trail, gap_after=bool(trail))
                continue
            place(core)
            cur.width = x - cur.left
            place(trail, gap_after=bool(trail))
        close(cur)
        para_lines[-1].last = True
        # line metrics
        lnspc = ppr.get("lnSpc", ("pct", 1.0))
        for ln_ in para_lines:
            sizes = [f.st.size for f in ln_.frags if f.text] + ([ln_.bullet.st.size] if ln_.bullet else [])
            if not sizes:
                sizes = [(ln_.br_style or para.end).size if not ln_.frags else max(f.st.size for f in ln_.frags)]
            m = max(sizes)
            ln_.nat = LINE_RATIO * m
            ln_.height = ln_.nat * lnspc[1] if lnspc[0] == "pct" else lnspc[1]
            ln_.descent = DESC_RATIO * m
        # horizontal alignment
        algn = ppr.get("algn", "l")
        for ln_ in para_lines:
            free = ln_.right - ln_.left - ln_.width
            shift = 0.0
            if algn == "ctr":
                shift = free / 2
            elif algn == "r":
                shift = free
            elif algn in ("just", "dist", "thaiDist") and (algn != "just" or not (ln_.last or ln_.forced)) and free > 0:
                gaps = [f for f in ln_.frags[:-1] if f.gap]
                if gaps:
                    extra, acc = free / len(gaps), 0.0
                    for f in ln_.frags:
                        f.x += acc
                        if f.gap:
                            acc += extra
            for f in ln_.frags:
                f.x += shift
        lines.extend(para_lines)
        para.lines = para_lines
    # vertical positions
    y = 0.0
    for pi, para in enumerate(paras):
        pl = para.lines
        bef = para.ppr.get("spcBef")
        if bef and (pi > 0 or APPLY_FIRST_SPACE_BEFORE):
            y += bef[1] * pl[0].nat if bef[0] == "pct" else bef[1]
        for ln_ in pl:
            ln_.top = y
            y += ln_.height
        aft = para.ppr.get("spcAft")
        if aft and (pi < len(paras) - 1 or APPLY_LAST_SPACE_AFTER):
            y += aft[1] * pl[-1].nat if aft[0] == "pct" else aft[1]
    lo, hi = None, None
    for ln_ in lines:
        xs = [(f.x, f.x + f.w) for f in ln_.frags if f.text.strip()]
        if ln_.bullet:
            xs.append((ln_.bullet.x, ln_.bullet.x + ln_.bullet.w))
        for a, b2 in xs:
            lo = a if lo is None else min(lo, a)
            hi = b2 if hi is None else max(hi, b2)
    return TextLayout(lines, y, splits, lo or 0.0, hi or 0.0)


# ----------------------------------------------------------------------------------------------
# Raster canvas (all drawing goes through 8-bit masks pasted with a solid color)
# ----------------------------------------------------------------------------------------------
DASHES = {"solid": None, "dash": (4, 3), "sysDash": (3, 1), "sysDot": (1, 1), "dot": (1, 3),
          "lgDash": (8, 3), "dashDot": (4, 3, 1, 3), "lgDashDot": (8, 3, 1, 3), "lgDashDotDot": (8, 3, 1, 3, 1, 3),
          "sysDashDot": (3, 1, 1, 1), "sysDashDotDot": (3, 1, 1, 1, 1, 1)}


def dash_split(pts, pattern):
    """Split a polyline into 'on' sub-polylines following a dash pattern (lengths in px)."""
    out, cur = [], []
    pi, left, on = 0, pattern[0], True
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        if on and not cur:
            cur = [(x0, y0)]
        while seg - pos > left:
            pos += left
            px, py = x0 + (x1 - x0) * pos / seg, y0 + (y1 - y0) * pos / seg
            if on:
                cur.append((px, py))
                out.append(cur)
                cur = []
            else:
                cur = [(px, py)]
            on = not on
            pi = (pi + 1) % len(pattern)
            left = pattern[pi]
        left -= seg - pos
        if on:
            cur.append((x1, y1))
    if on and len(cur) > 1:
        out.append(cur)
    return out


class Canvas:
    def __init__(self, img, k, ox=0.0, oy=0.0):
        self.img = img  # RGB (slide) or RGBA (rotated layer)
        self.k = k  # px per EMU
        self.ox, self.oy = ox, oy  # EMU position of the image's top-left corner

    def X(self, x):
        return (x - self.ox) * self.k

    def Y(self, y):
        return (y - self.oy) * self.k

    def P(self, pts):
        return [(self.X(x), self.Y(y)) for x, y in pts]

    def _paste(self, mask, x0, y0, rgba):
        if rgba is None or rgba[3] <= 0.001:
            return
        W, H = self.img.size
        cx0, cy0 = max(0, -x0), max(0, -y0)
        cx1, cy1 = min(mask.width, W - x0), min(mask.height, H - y0)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        if (cx0, cy0, cx1, cy1) != (0, 0, mask.width, mask.height):
            mask = mask.crop((cx0, cy0, cx1, cy1))
        a = rgba[3]
        if a < 0.999:
            mask = mask.point(lambda v, a=a: int(v * a + 0.5))
        color = tuple(rgba[:3]) if self.img.mode == "RGB" else tuple(rgba[:3]) + (255,)
        self.img.paste(color, (x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1), mask)

    @staticmethod
    def _bbox(polys, pad):
        xs = [p[0] for poly in polys for p in poly]
        ys = [p[1] for poly in polys for p in poly]
        return math.floor(min(xs) - pad), math.floor(min(ys) - pad), math.ceil(max(xs) + pad), math.ceil(max(ys) + pad)

    def fill_polys(self, polys, rgba, holes=()):
        polys = [p for p in polys if len(p) >= 3]
        if not polys or rgba is None:
            return
        x0, y0, x1, y1 = self._bbox(polys, 1)
        W, H = self.img.size
        if x1 < 0 or y1 < 0 or x0 > W or y0 > H or x1 - x0 > 20 * W:
            return
        mask = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
        d = ImageDraw.Draw(mask)
        for poly in polys:
            d.polygon([(x - x0, y - y0) for x, y in poly], fill=255)
        for poly in holes:
            if len(poly) >= 3:
                d.polygon([(x - x0, y - y0) for x, y in poly], fill=0)
        self._paste(mask, x0, y0, rgba)

    def stroke(self, paths, width_px, rgba, dash="solid"):
        """paths: list of (points_px, closed)."""
        if rgba is None or not paths:
            return
        w = max(1.0, width_px)
        pattern = DASHES.get(dash, (4, 3)) if dash else None
        segs = []
        for pts, closed in paths:
            if len(pts) < 2:
                continue
            seq = list(pts) + [pts[0]] if closed else list(pts)
            segs.extend(dash_split(seq, [v * max(w, 2.0) for v in pattern]) if pattern else [seq])
        segs = [s for s in segs if len(s) >= 2]
        if not segs:
            return
        x0, y0, x1, y1 = self._bbox(segs, w + 2)
        mask = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
        d = ImageDraw.Draw(mask)
        iw = max(1, int(round(w)))
        for s in segs:
            d.line([(x - x0, y - y0) for x, y in s], fill=255, width=iw, joint="curve" if iw > 2 else None)
        self._paste(mask, x0, y0, rgba)

    def text(self, x, y, s, font, rgba, feats):
        """draw s with its baseline-left at (x, y) px"""
        if not s or rgba is None:
            return
        try:
            l, t, r, b = font.getbbox(s, anchor="ls", features=feats)
        except Exception:
            l, t, r, b = font.getbbox(s, anchor="ls")
        x0, y0 = math.floor(x + l) - 2, math.floor(y + t) - 2
        w, h = math.ceil(x + r) + 2 - x0, math.ceil(y + b) + 2 - y0
        if w <= 0 or h <= 0:
            return
        mask = Image.new("L", (w, h), 0)
        kw = {"features": feats} if feats else {}
        ImageDraw.Draw(mask).text((x - x0, y - y0), s, font=font, fill=255, anchor="ls", **kw)
        self._paste(mask, x0, y0, rgba)

    def image(self, im, x0, y0):
        """paste an RGBA image with its top-left at integer px (x0, y0)"""
        W, H = self.img.size
        cx0, cy0 = max(0, -x0), max(0, -y0)
        cx1, cy1 = min(im.width, W - x0), min(im.height, H - y0)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        im = im.crop((cx0, cy0, cx1, cy1))
        if self.img.mode == "RGBA":
            self.img.alpha_composite(im, (x0 + cx0, y0 + cy0))
        else:
            self.img.paste(im.convert("RGB"), (x0 + cx0, y0 + cy0), im.getchannel("A"))


# ----------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------
def arc(cx, cy, rx, ry, a0, a1, step=6.0):
    n = max(2, int(abs(a1 - a0) / step) + 1)
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


class PathBuilder:
    def __init__(self):
        self.paths, self.cur, self.pt = [], None, (0.0, 0.0)

    def move(self, x, y):
        self.end(False)
        self.cur = [(x, y)]
        self.pt = (x, y)

    def line(self, x, y):
        if self.cur is None:
            self.cur = [self.pt]
        self.cur.append((x, y))
        self.pt = (x, y)

    def arc_to(self, wr, hr, st, sw):
        """OOXML arcTo: angles in degrees, clockwise, from the current point"""
        x, y = self.pt
        cx, cy = x - wr * math.cos(math.radians(st)), y - hr * math.sin(math.radians(st))
        pts = arc(cx, cy, wr, hr, st, st + sw)
        for p in pts[1:]:
            self.line(*p)

    def cubic(self, p1, p2, p3, n=16):
        x0, y0 = self.pt
        for i in range(1, n + 1):
            t = i / n
            mt = 1 - t
            self.line(mt ** 3 * x0 + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
                      mt ** 3 * y0 + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1])

    def quad(self, p1, p2, n=12):
        x0, y0 = self.pt
        for i in range(1, n + 1):
            t = i / n
            mt = 1 - t
            self.line(mt * mt * x0 + 2 * mt * t * p1[0] + t * t * p2[0], mt * mt * y0 + 2 * mt * t * p1[1] + t * t * p2[1])

    def close(self):
        self.end(True)

    def end(self, closed):
        if self.cur and len(self.cur) > 1:
            self.paths.append((self.cur, closed))
        self.cur = None


def adj_values(geom):
    d = {}
    for gd in findall(geom, "a:avLst/a:gd"):
        m = re.match(r"\s*val\s+(-?\d+)", gd.get("fmla", ""))
        if m:
            d[gd.get("name")] = int(m.group(1))
    return d


SUPPORTED_PRST = {"rect", "roundRect", "ellipse", "triangle", "rtTriangle", "rightArrow", "leftArrow", "upArrow",
                  "downArrow", "chevron", "homePlate", "rightBrace", "leftBrace", "line", "straightConnector1",
                  "bentConnector2", "bentConnector3", "curvedConnector3", "diamond", "parallelogram", "trapezoid",
                  "hexagon", "octagon", "pentagon", "plus", "flowChartProcess", "flowChartAlternateProcess",
                  "rightBracket", "leftBracket", "snip1Rect", "round2SameRect", "frame"}
LINE_PRST = {"line", "straightConnector1", "bentConnector2", "bentConnector3", "curvedConnector3", "rightBrace",
             "leftBrace", "rightBracket", "leftBracket"}


def preset_geometry(prst, w, h, adj):
    """-> (paths [(pts, closed)], text_rect (l, t, r, b)) in local EMU, or None if unsupported."""
    ss = min(w, h)
    hc, vc = w / 2, h / 2
    pb = PathBuilder()
    rect = (0, 0, w, h)
    A = lambda name, default: adj.get(name, default)
    if prst in ("rect", "flowChartProcess"):
        return [([(0, 0), (w, 0), (w, h), (0, h)], True)], rect
    if prst in ("roundRect", "flowChartAlternateProcess", "round2SameRect"):
        a = max(0, min(A("adj", 16667) if prst == "roundRect" else (16667 if prst != "round2SameRect" else A("adj1", 16667)), 50000))
        r = ss * a / 100000
        if prst == "round2SameRect":
            pts = arc(r, r, r, r, 180, 270) + arc(w - r, r, r, r, 270, 360) + [(w, h), (0, h)]
        else:
            pts = arc(r, r, r, r, 180, 270) + arc(w - r, r, r, r, 270, 360) + arc(w - r, h - r, r, r, 0, 90) + arc(r, h - r, r, r, 90, 180)
        il = r * 0.29289
        return [(pts, True)], (il, il, w - il, h - il)
    if prst == "ellipse":
        return [(arc(hc, vc, hc, vc, 0, 360, 4), True)], (w * 0.14645, h * 0.14645, w * 0.85355, h * 0.85355)
    if prst == "triangle":
        a = max(0, min(A("adj", 50000), 100000))
        return [([(0, h), (w * a / 100000, 0), (w, h)], True)], (w * a / 200000, vc, w * a / 200000 + hc, h)
    if prst == "rtTriangle":
        return [([(0, h), (0, 0), (w, h)], True)], (w / 12, h * 7 / 12, w * 7 / 12, h * 11 / 12)
    if prst in ("rightArrow", "leftArrow"):
        a1 = max(0, min(A("adj1", 50000), 100000))
        a2 = max(0, min(A("adj2", 50000), 100000 * w / ss if ss else 0))
        dx1 = ss * a2 / 100000
        x1 = w - dx1
        dy1 = h * a1 / 200000
        y1, y2 = vc - dy1, vc + dy1
        pts = [(0, y1), (x1, y1), (x1, 0), (w, vc), (x1, h), (x1, y2), (0, y2)]
        x2 = x1 + (y1 * dx1 / vc if vc else 0)
        tr = (0, y1, x2, y2)
        if prst == "leftArrow":
            pts = [(w - x, y) for x, y in pts]
            tr = (w - x2, y1, w, y2)
        return [(pts, True)], tr
    if prst in ("upArrow", "downArrow"):
        a1 = max(0, min(A("adj1", 50000), 100000))
        a2 = max(0, min(A("adj2", 50000), 100000 * h / ss if ss else 0))
        dy1 = ss * a2 / 100000
        dx1 = w * a1 / 200000
        x1, x2 = hc - dx1, hc + dx1
        pts = [(x1, h), (x1, dy1), (0, dy1), (hc, 0), (w, dy1), (x2, dy1), (x2, h)]
        y2 = dy1 - (x1 * dy1 / hc if hc else 0)
        tr = (x1, max(0, y2), x2, h)
        if prst == "downArrow":
            pts = [(x, h - y) for x, y in pts]
            tr = (x1, 0, x2, h - max(0, y2))
        return [(pts, True)], tr
    if prst == "chevron":
        a = max(0, min(A("adj", 50000), 100000 * w / ss if ss else 0))
        x1 = ss * a / 100000
        x2 = w - x1
        return [([(0, 0), (x2, 0), (w, vc), (x2, h), (0, h), (x1, vc)], True)], ((x1, 0, x2, h) if x2 > x1 else rect)
    if prst == "homePlate":
        a = max(0, min(A("adj", 50000), 100000 * w / ss if ss else 0))
        x1 = w - ss * a / 100000
        return [([(0, 0), (x1, 0), (w, vc), (x1, h), (0, h)], True)], (0, 0, (x1 + w) / 2, h)
    if prst in ("rightBrace", "leftBrace"):
        a2 = max(0, min(A("adj2", 50000), 100000))
        q2 = min(100000 - a2, a2)
        max1 = (q2 / 2) * h / ss if ss else 0
        a1 = max(0, min(A("adj1", 8333), max1))
        y1 = ss * a1 / 100000
        y3 = h * a2 / 100000
        wd2 = w / 2
        pb.move(0, 0)
        pb.arc_to(wd2, y1, 270, 90)
        pb.line(hc, y3 - y1)
        pb.arc_to(wd2, y1, 180, -90)
        pb.arc_to(wd2, y1, 270, -90)
        pb.line(hc, h - y1)
        pb.arc_to(wd2, y1, 0, 90)
        pb.end(False)
        paths = pb.paths
        if prst == "leftBrace":
            paths = [([(w - x, y) for x, y in pts], c) for pts, c in paths]
        return paths, rect
    if prst in ("rightBracket", "leftBracket"):
        a = max(0, min(A("adj", 8333), 50000 * h / ss if ss else 0))
        y1 = ss * a / 100000
        pb.move(0, 0)
        pb.arc_to(w, y1, 270, 90)
        pb.line(w, h - y1)
        pb.arc_to(w, y1, 0, 90)
        pb.end(False)
        paths = pb.paths
        if prst == "leftBracket":
            paths = [([(w - x, y) for x, y in pts], c) for pts, c in paths]
        return paths, rect
    if prst in ("line", "straightConnector1"):
        return [([(0, 0), (w, h)], False)], rect
    if prst == "bentConnector2":
        return [([(0, 0), (w, 0), (w, h)], False)], rect
    if prst == "bentConnector3":
        x1 = w * A("adj1", 50000) / 100000
        return [([(0, 0), (x1, 0), (x1, h), (w, h)], False)], rect
    if prst == "curvedConnector3":
        pb.move(0, 0)
        pb.cubic((w / 2, 0), (w / 2, h), (w, h))
        pb.end(False)
        return pb.paths, rect
    if prst == "diamond":
        return [([(hc, 0), (w, vc), (hc, h), (0, vc)], True)], (w / 4, h / 4, w * 3 / 4, h * 3 / 4)
    if prst == "parallelogram":
        x2 = ss * max(0, min(A("adj", 25000), 100000 * w / ss if ss else 0)) / 100000
        return [([(0, h), (x2, 0), (w, 0), (w - x2, h)], True)], (x2 / 2, 0, w - x2 / 2, h)
    if prst == "trapezoid":
        x2 = ss * max(0, min(A("adj", 25000), 50000 * w / ss if ss else 0)) / 100000
        return [([(0, h), (x2, 0), (w - x2, 0), (w, h)], True)], (x2 * 2 / 3, h / 3, w - x2 * 2 / 3, h)
    if prst == "hexagon":
        x1 = ss * max(0, min(A("adj", 25000), 50000 * w / ss if ss else 0)) / 100000
        return [([(0, vc), (x1, 0), (w - x1, 0), (w, vc), (w - x1, h), (x1, h)], True)], (x1 / 2, h / 4, w - x1 / 2, h * 3 / 4)
    if prst == "octagon":
        x1 = ss * max(0, min(A("adj", 29289), 50000)) / 100000
        pts = [(0, x1), (x1, 0), (w - x1, 0), (w, x1), (w, h - x1), (w - x1, h), (x1, h), (0, h - x1)]
        return [(pts, True)], (x1 / 2, x1 / 2, w - x1 / 2, h - x1 / 2)
    if prst == "pentagon":
        pts = [(hc, 0), (w, h * 0.38197), (w * 0.80902, h), (w * 0.19098, h), (0, h * 0.38197)]
        return [(pts, True)], (w * 0.19098, h * 0.38197, w * 0.80902, h)
    if prst == "plus":
        x1 = ss * max(0, min(A("adj", 25000), 50000)) / 100000
        pts = [(0, x1), (x1, x1), (x1, 0), (w - x1, 0), (w - x1, x1), (w, x1), (w, h - x1), (w - x1, h - x1),
               (w - x1, h), (x1, h), (x1, h - x1), (0, h - x1)]
        return [(pts, True)], (0, x1, w, h - x1)
    if prst == "snip1Rect":
        x1 = ss * max(0, min(A("adj", 16667), 50000)) / 100000
        return [([(0, 0), (w - x1, 0), (w, x1), (w, h), (0, h)], True)], (0, x1 / 2, w - x1 / 2, h)
    if prst == "frame":
        x1 = ss * max(0, min(A("adj1", 12500), 50000)) / 100000
        inner = [(x1, x1), (w - x1, x1), (w - x1, h - x1), (x1, h - x1)]
        return [([(0, 0), (w, 0), (w, h), (0, h)], True), (inner, "hole")], (x1, x1, w - x1, h - x1)
    return None


def custom_geometry(geom, w, h):
    pb = PathBuilder()
    paths = []
    for path in findall(geom, "a:pathLst/a:path"):
        pw, ph = to_int(path.get("w"), 0), to_int(path.get("h"), 0)
        sx = w / pw if pw else 1.0
        sy = h / ph if ph else 1.0
        P = lambda el: (to_int(el.get("x")) * sx, to_int(el.get("y")) * sy)
        pb = PathBuilder()
        for cmd in path:
            n = lname(cmd)
            pts = [P(p) for p in findall(cmd, "a:pt")]
            if n == "moveTo" and pts:
                pb.move(*pts[0])
            elif n == "lnTo" and pts:
                pb.line(*pts[0])
            elif n == "cubicBezTo" and len(pts) == 3:
                pb.cubic(*pts)
            elif n == "quadBezTo" and len(pts) == 2:
                pb.quad(*pts)
            elif n == "arcTo":
                pb.arc_to(to_int(cmd.get("wR")) * sx, to_int(cmd.get("hR")) * sy,
                          to_int(cmd.get("stAng")) / 60000.0, to_int(cmd.get("swAng")) / 60000.0)
            elif n == "close":
                pb.close()
        pb.end(False)
        fill_ok = path.get("fill") != "none"
        paths.extend([(p, c, fill_ok) for p, c in pb.paths])
    return paths


@dataclass
class Xform:
    """shape box in slide EMU plus rotation/flip; maps local shape coords to slide coords"""
    x: float
    y: float
    w: float
    h: float
    rot: float = 0.0  # degrees clockwise
    flip_h: bool = False
    flip_v: bool = False

    def map(self, pts):
        cx, cy = self.w / 2, self.h / 2
        c, s = math.cos(math.radians(self.rot)), math.sin(math.radians(self.rot))
        out = []
        for x, y in pts:
            if self.flip_h:
                x = self.w - x
            if self.flip_v:
                y = self.h - y
            dx, dy = x - cx, y - cy
            out.append((self.x + cx + dx * c - dy * s, self.y + cy + dx * s + dy * c))
        return out

    def bbox(self):
        pts = self.map([(0, 0), (self.w, 0), (self.w, self.h), (0, self.h)])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)


def read_xfrm(xfrm, T):
    """a:xfrm/p:xfrm + group transform T=(sx, sy, tx, ty) -> Xform"""
    off, ext = find(xfrm, "a:off"), find(xfrm, "a:ext")
    x, y = to_int(off.get("x")) if off is not None else 0, to_int(off.get("y")) if off is not None else 0
    w, h = to_int(ext.get("cx")) if ext is not None else 0, to_int(ext.get("cy")) if ext is not None else 0
    sx, sy, tx, ty = T
    return Xform(tx + x * sx, ty + y * sy, w * sx, h * sy, to_int(xfrm.get("rot")) / 60000.0,
                 is_true(xfrm.get("flipH")), is_true(xfrm.get("flipV")))


# ----------------------------------------------------------------------------------------------
# Deck model: masters, layouts, placeholders
# ----------------------------------------------------------------------------------------------
def ph_of(el):
    for nv in el:
        if lname(nv).startswith("nv") and lname(nv).endswith("Pr"):
            ph = find(nv, "p:nvPr/p:ph")
            if ph is not None:
                return ph
            return None
    return None


def index_placeholders(tree):
    out = []
    for el in tree.iter():
        if lname(el) in ("sp", "pic", "graphicFrame"):
            ph = ph_of(el)
            if ph is not None:
                out.append((el, ph.get("type", "obj"), ph.get("idx")))
    return out


TYPE_EQUIV = {"ctrTitle": ("ctrTitle", "title"), "title": ("title", "ctrTitle"), "subTitle": ("subTitle", "body"),
              "obj": ("obj", "body"), "body": ("body", "obj"), "pic": ("pic", "body", "obj"),
              "chart": ("chart", "body", "obj"), "tbl": ("tbl", "body", "obj")}


def find_ph(phs, ptype, idx):
    if idx is not None:
        for sp, t, i in phs:
            if i == idx:
                return sp
    for want in TYPE_EQUIV.get(ptype, (ptype,)):
        for sp, t, i in phs:
            if t == want:
                return sp
    return None


class MasterInfo:
    def __init__(self, deck, part):
        pkg = deck.pkg
        self.part = part
        root = pkg.xml(part)
        self.tree = find(root, "p:cSld/p:spTree")
        self.bg = find(root, "p:cSld/p:bg")
        cm = find(root, "p:clrMap")
        self.clr_map = dict(cm.attrib) if cm is not None else dict(DEFAULT_CLR_MAP)
        tp = pkg.first(part, "theme")
        self.theme = Theme(pkg.xml(tp) if tp else None)
        tx = find(root, "p:txStyles")
        self.title_lst = parse_lst(find(tx, "p:titleStyle"))
        self.body_lst = parse_lst(find(tx, "p:bodyStyle"))
        self.other_lst = parse_lst(find(tx, "p:otherStyle"))
        self.phs = index_placeholders(self.tree) if self.tree is not None else []


class LayoutInfo:
    def __init__(self, deck, part):
        pkg = deck.pkg
        self.part = part
        root = pkg.xml(part)
        self.tree = find(root, "p:cSld/p:spTree")
        self.bg = find(root, "p:cSld/p:bg")
        self.show_master = is_true(root.get("showMasterSp"), True)
        self.clr_override = find(root, "p:clrMapOvr/a:overrideClrMapping")
        mp = pkg.first(part, "slideMaster")
        self.master = deck.master(mp)
        self.phs = index_placeholders(self.tree) if self.tree is not None else []


@dataclass
class Ctx:
    deck: "Deck"
    slide_no: int
    slide_part: str
    layout: LayoutInfo
    master: MasterInfo
    cc: ColorCtx
    fonts: FontBook
    canvas: Canvas
    debug: bool = False


# ----------------------------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------------------------
class Deck:
    def __init__(self, path, font_dir, debug=False):
        self.path = path
        self.pkg = Package(path)
        pres_part = "ppt/presentation.xml"
        pres = self.pkg.xml(pres_part)
        if pres is None:
            raise SystemExit("not a presentation: %s" % path)
        sz = find(pres, "p:sldSz")
        self.cx = to_int(sz.get("cx"), 12192000) if sz is not None else 12192000
        self.cy = to_int(sz.get("cy"), 6858000) if sz is not None else 6858000
        self.first_num = to_int(pres.get("firstSlideNum"), 1)
        self.slides = []
        for sid in findall(pres, "p:sldIdLst/p:sldId"):
            t = self.pkg.target(pres_part, sid.get(R_ID))
            if t:
                self.slides.append(t)
        self.default_lst = parse_lst(find(pres, "p:defaultTextStyle"))
        self._masters, self._layouts, self._images = {}, {}, {}
        self.fonts = FontBook(font_dir)
        self.debug = debug
        self.issues, self.warnings = [], {}
        self._seen = set()
        if self.slides:
            lp = self.pkg.first(self.slides[0], "slideLayout")
            if lp:
                th = self.layout(lp).master.theme
                self.fonts.major, self.fonts.minor = th.major, th.minor

    # -- bookkeeping -------------------------------------------------------------------------
    def master(self, part):
        if part not in self._masters:
            self._masters[part] = MasterInfo(self, part)
        return self._masters[part]

    def layout(self, part):
        if part not in self._layouts:
            self._layouts[part] = LayoutInfo(self, part)
        return self._layouts[part]

    def warn(self, msg):
        self.warnings[msg] = self.warnings.get(msg, 0) + 1

    def issue(self, ctx, owner_part, el, name, kind, detail, text=""):
        key = None
        if owner_part != ctx.slide_part:  # layout/master shapes: report once
            key = (owner_part, name, kind, detail)
            if key in self._seen:
                return
            self._seen.add(key)
            name = "%s [%s]" % (name, posixpath.basename(owner_part))
        self.issues.append({"slide": ctx.slide_no, "shape": name, "kind": kind, "detail": detail,
                            "text": (text or "")[:50]})

    def image(self, part):
        if part not in self._images:
            im = None
            try:
                im = Image.open(io.BytesIO(self.pkg.blob(part)))
                im.load()
                im = im.convert("RGBA")
            except Exception:
                im = None
            self._images[part] = im
        return self._images[part]

    # -- slide -------------------------------------------------------------------------------
    def render(self, index, width_px):
        part = self.slides[index]
        sroot = self.pkg.xml(part)
        layout = self.layout(self.pkg.first(part, "slideLayout"))
        master = layout.master
        clr_map = dict(master.clr_map)
        for ovr in (layout.clr_override, find(sroot, "p:clrMapOvr/a:overrideClrMapping")):
            if ovr is not None:
                clr_map.update(ovr.attrib)
        H = round(width_px * self.cy / self.cx)
        img = Image.new("RGB", (width_px * SS, H * SS), (255, 255, 255))
        k = width_px * SS / self.cx
        ctx = Ctx(self, index + self.first_num, part, layout, master, ColorCtx(master.theme, clr_map), self.fonts,
                  Canvas(img, k), self.debug)
        # background: slide -> layout -> master
        for bg, owner in ((find(sroot, "p:cSld/p:bg"), part), (layout.bg, layout.part), (master.bg, master.part)):
            if bg is not None:
                self.draw_background(ctx, bg, owner)
                break
        show = is_true(sroot.get("showMasterSp"), True)
        ident = (1.0, 1.0, 0.0, 0.0)
        if show and layout.show_master and master.tree is not None:
            self.render_tree(ctx, master.tree, master.part, "master", ident)
        if show and layout.tree is not None:
            self.render_tree(ctx, layout.tree, layout.part, "layout", ident)
        tree = find(sroot, "p:cSld/p:spTree")
        if tree is not None:
            self.render_tree(ctx, tree, part, "slide", ident)
        return img.resize((width_px, H), Image.Resampling.LANCZOS), is_true(sroot.get("show"), True)

    def draw_background(self, ctx, bg, owner):
        cv = ctx.canvas
        full = [(0, 0), (self.cx, 0), (self.cx, self.cy), (0, self.cy)]
        bgpr, bgref = find(bg, "p:bgPr"), find(bg, "p:bgRef")
        fill = None
        if bgpr is not None:
            fill = parse_fill(bgpr, ctx.cc)
        elif bgref is not None:
            c = color_child(bgref)
            fill = ctx.cc.resolve(c) if c is not None else None
        if isinstance(fill, tuple) and fill and fill[0] == "blip":
            self.draw_blip(ctx, fill[1], owner, Xform(0, 0, self.cx, self.cy), None)
        elif isinstance(fill, tuple):
            cv.fill_polys([cv.P(full)], fill[:3] + (1.0,))

    def render_tree(self, ctx, tree, part, owner, T):
        for el in tree:
            n = lname(el)
            if n in ("sp", "cxnSp", "pic", "graphicFrame", "grpSp") and is_hidden(el):
                continue
            try:
                if n == "sp":
                    self.render_sp(ctx, el, part, owner, T)
                elif n == "cxnSp":
                    self.render_sp(ctx, el, part, owner, T, connector=True)
                elif n == "pic":
                    self.render_pic(ctx, el, part, owner, T)
                elif n == "graphicFrame":
                    self.render_frame(ctx, el, part, owner, T)
                elif n == "grpSp":
                    self.render_group(ctx, el, part, owner, T)
                elif n == "AlternateContent":
                    alt = find(el, "mc:Fallback")
                    if alt is None:
                        alt = find(el, "mc:Choice")
                    if alt is not None:
                        self.render_tree(ctx, alt, part, owner, T)
                elif n in ("nvGrpSpPr", "grpSpPr", "extLst", ""):
                    continue
                else:
                    self.issue(ctx, part, el, n, "unsupported", "element p:%s not rendered" % n)
            except Exception as e:  # never let one shape kill the run
                nm = shape_name(el)
                self.issue(ctx, part, el, nm, "unsupported", "render error: %s: %s" % (type(e).__name__, e))

    def render_group(self, ctx, grp, part, owner, T):
        xfrm = find(grp, "p:grpSpPr/a:xfrm")
        T2 = T
        if xfrm is not None:
            off, ext = find(xfrm, "a:off"), find(xfrm, "a:ext")
            choff, chext = find(xfrm, "a:chOff"), find(xfrm, "a:chExt")
            if None not in (off, ext, choff, chext):
                cw, ch = to_int(chext.get("cx")), to_int(chext.get("cy"))
                gx = to_int(ext.get("cx")) / cw if cw else 1.0
                gy = to_int(ext.get("cy")) / ch if ch else 1.0
                sx, sy, tx, ty = T
                # child x -> off + (x - chOff) * g, then the outer transform
                T2 = (sx * gx, sy * gy, tx + sx * (to_int(off.get("x")) - to_int(choff.get("x")) * gx),
                      ty + sy * (to_int(off.get("y")) - to_int(choff.get("y")) * gy))
            if to_int(xfrm.get("rot")) != 0 or is_true(xfrm.get("flipH")) or is_true(xfrm.get("flipV")):
                self.warn("group rotation/flip ignored")
        self.render_tree(ctx, grp, part, owner, T2)

    # -- p:sp / p:cxnSp ------------------------------------------------------------------------
    def render_sp(self, ctx, sp, part, owner, T, connector=False):
        cv = ctx.canvas
        name = shape_name(sp)
        ph = ph_of(sp)
        if ph is not None and owner != "slide":
            return  # layout/master placeholders are templates; PowerPoint does not draw them on slides
        inh, ptype, ltype = [], None, None
        if ph is not None:
            ptype = ph.get("type", "obj")
            lsp = find_ph(ctx.layout.phs, ptype, ph.get("idx"))
            ltype = ptype
            if lsp is not None:
                inh.append(lsp)
                lph = ph_of(lsp)
                ltype = lph.get("type", "obj") if lph is not None else ptype
            msp = find_ph(ctx.master.phs, ltype, None)
            if msp is not None:
                inh.append(msp)
        spPr = find(sp, "p:spPr")
        xfrm = find(spPr, "a:xfrm")
        for s in inh:
            if xfrm is None:
                xfrm = find(s, "p:spPr/a:xfrm")
        if xfrm is None:
            if find(sp, "p:txBody") is not None and plain_text(parse_paragraphs(ctx, find(sp, "p:txBody"), [])):
                self.issue(ctx, part, sp, name, "unsupported", "shape without position (no a:xfrm)")
            return
        xf = read_xfrm(xfrm, T)
        geom = find(spPr, "a:prstGeom")
        cust = find(spPr, "a:custGeom")
        for s in inh:
            if geom is None and cust is None:
                geom, cust = find(s, "p:spPr/a:prstGeom"), find(s, "p:spPr/a:custGeom")
        s_fill, s_lc, s_lw, s_fc = style_refs(sp, ctx.cc, ctx.master.theme)
        fill = parse_fill(spPr, ctx.cc)
        ln_el = find(spPr, "a:ln")
        for s in inh:
            if fill is None:
                fill = parse_fill(find(s, "p:spPr"), ctx.cc)
            if ln_el is None:
                ln_el = find(s, "p:spPr/a:ln")
        if fill is None:
            fill = s_fill
        line = parse_line(ln_el, ctx.cc, style_color=s_lc, style_width=s_lw)
        # geometry
        prst = geom.get("prst", "rect") if geom is not None else ("custom" if cust is not None else "rect")
        paths, trect, fillable = [], (0, 0, xf.w, xf.h), True
        if cust is not None and geom is None:
            cpaths = custom_geometry(cust, xf.w, xf.h)
            paths = [(p, c) for p, c, _ in cpaths]
            fill_paths = [p for p, c, f in cpaths if f]
        else:
            g = preset_geometry(prst, xf.w, xf.h, adj_values(geom))
            if g is None:
                self.issue(ctx, part, sp, name, "unsupported", "prstGeom '%s' drawn as dashed bounding box" % prst)
                g = ([([(0, 0), (xf.w, 0), (xf.w, xf.h), (0, xf.h)], True)], (0, 0, xf.w, xf.h))
                fill, line = NOFILL, LineSpec(9525, (200, 0, 120, 1.0), "dash")
            paths, trect = g
            fillable = prst not in LINE_PRST
            fill_paths = [p for p, c in paths if c is True] if fillable else []
        if isinstance(fill, tuple) and fill and fill[0] == "blip":
            self.draw_blip(ctx, fill[1], part, xf, paths)
        elif isinstance(fill, tuple) and fillable:
            holes = [cv.P(xf.map(p)) for p, c in paths if c == "hole"]
            cv.fill_polys([cv.P(xf.map(p)) for p in fill_paths], fill, holes)
        if line is not None:
            self.draw_outline(ctx, paths, xf, line)
        # bounds
        self.check_bounds(ctx, part, sp, name, xf, plain_text_quick(sp))
        # text
        tx = find(sp, "p:txBody")
        if tx is not None:
            chain = [parse_lst(find(tx, "a:lstStyle"))]
            if s_fc is not None:
                chain.append({lv: ({}, {"color": s_fc}) for lv in range(1, 10)})
            for s in inh:
                chain.append(parse_lst(find(s, "p:txBody/a:lstStyle")))
            if ph is not None:
                ptype2 = ltype if (ph.get("type") is None and ltype) else ptype
                if ptype2 in ("title", "ctrTitle"):
                    chain.append(ctx.master.title_lst)
                elif ptype2 in ("dt", "ftr", "sldNum", "hdr"):
                    chain.append(ctx.master.other_lst)
                else:
                    chain.append(ctx.master.body_lst)
            else:
                chain.append(ctx.master.other_lst)
            chain.append(self.default_lst)
            bodies = [find(tx, "a:bodyPr")] + [find(s, "p:txBody/a:bodyPr") for s in inh]
            self.render_text(ctx, part, sp, name, tx, chain, bodies, xf, trect)

    def draw_outline(self, ctx, paths, xf, line):
        cv = ctx.canvas
        wpx = line.width * cv.k
        for pts, closed in paths:
            mp = xf.map(pts)
            if not closed and (line.head or line.tail) and len(mp) >= 2:
                if line.tail:
                    mp = self.arrow(ctx, mp, line.tail, line, at_end=True)
                if line.head:
                    mp = self.arrow(ctx, mp, line.head, line, at_end=False)
            cv.stroke([(cv.P(mp), closed)], wpx, line.color, line.dash)

    def arrow(self, ctx, pts, spec, line, at_end):
        """draw an arrowhead at one end of pts (slide EMU); return the polyline shortened under it"""
        cv = ctx.canvas
        typ, ws, ls = spec
        fac = {"sm": 2.0, "med": 3.0, "lg": 5.0}
        base = max(line.width, 2 * EMU_PT)
        aw, al = base * fac.get(ws, 3.0), base * fac.get(ls, 3.0)
        (x1, y1), (x0, y0) = (pts[-1], pts[-2]) if at_end else (pts[0], pts[1])
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1:
            return pts
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        px, py = -uy, ux
        bx, by = x1 - ux * al, y1 - uy * al
        left, right = (bx + px * aw / 2, by + py * aw / 2), (bx - px * aw / 2, by - py * aw / 2)
        if typ == "arrow":
            cv.stroke([(cv.P([left, (x1, y1), right]), False)], line.width * cv.k, line.color)
            return pts
        if typ == "oval":
            cv.fill_polys([cv.P(arc(x1, y1, aw / 2, al / 2, 0, 360, 15))], line.color)
        elif typ == "diamond":
            cv.fill_polys([cv.P([(x1 + ux * al / 2, y1 + uy * al / 2), (x1 + px * aw / 2, y1 + py * aw / 2),
                                 (x1 - ux * al / 2, y1 - uy * al / 2), (x1 - px * aw / 2, y1 - py * aw / 2)])], line.color)
            return pts
        else:  # triangle, stealth
            tip = [(x1, y1), left, right] if typ != "stealth" else [(x1, y1), left, (x1 - ux * al * 0.6, y1 - uy * al * 0.6), right]
            cv.fill_polys([cv.P(tip)], line.color)
        cut = (bx, by) if typ != "oval" else (x1, y1)
        return pts[:-1] + [cut] if at_end else [cut] + pts[1:]

    def check_bounds(self, ctx, part, el, name, xf, text):
        x0, y0, x1, y1 = xf.bbox()
        msgs = []
        if x0 < -TOL_OOB:
            msgs.append("%.2f in past left edge" % inch(-x0))
        if y0 < -TOL_OOB:
            msgs.append("%.2f in past top edge" % inch(-y0))
        if x1 > self.cx + TOL_OOB:
            msgs.append("%.2f in past right edge" % inch(x1 - self.cx))
        if y1 > self.cy + TOL_OOB:
            msgs.append("%.2f in past bottom edge" % inch(y1 - self.cy))
        if msgs:
            self.issue(ctx, part, el, name, "out_of_bounds", "shape extends " + ", ".join(msgs), text)
        return bool(msgs)

    # -- text body ---------------------------------------------------------------------------
    def body_props(self, bodies):
        d = {"lIns": DEF_LINS, "tIns": DEF_TINS, "rIns": DEF_LINS, "bIns": DEF_TINS, "anchor": "t",
             "wrap": "square", "autofit": None, "vert": "horz"}
        for bp in reversed([b for b in bodies if b is not None]):
            for k in ("lIns", "tIns", "rIns", "bIns"):
                if bp.get(k) is not None:
                    d[k] = to_int(bp.get(k))
            for k in ("anchor", "wrap", "vert"):
                if bp.get(k):
                    d[k] = bp.get(k)
            if find(bp, "a:normAutofit") is not None:
                d["autofit"] = "normAutofit"
            elif find(bp, "a:spAutoFit") is not None:
                d["autofit"] = "spAutoFit"
            elif find(bp, "a:noAutofit") is not None:
                d["autofit"] = None
        return d

    def render_text(self, ctx, part, el, name, tx, chain, bodies, xf, trect):
        paras = parse_paragraphs(ctx, tx, chain)
        text = plain_text(paras)
        if not text:
            return
        bp = self.body_props(bodies)
        l, t, r, b = trect
        # flipV = 180 deg rotation + flipH: text turns upside down (draw_layout) and the text
        # rectangle mirrors horizontally when exactly one flip is set; text itself is never mirrored
        if xf.flip_h != xf.flip_v:
            l, r = xf.w - r, xf.w - l
        area_w = (r - l) - bp["lIns"] - bp["rIns"]
        area_h = (b - t) - bp["tIns"] - bp["bIns"]
        if bp["vert"] not in ("horz", None):
            self.warn("vertical text (vert=%s) drawn horizontally" % bp["vert"])
        wrap = bp["wrap"] != "none"
        lay = layout_text(paras, area_w / EMU_PT, ctx.fonts, wrap)
        need = lay.height * EMU_PT
        anchor = bp["anchor"]
        dy = (area_h - need) / 2 if anchor == "ctr" else (area_h - need if anchor == "b" else 0.0)
        ox, oy = l + bp["lIns"], t + bp["tIns"] + dy  # local coords of the text block origin
        self.draw_layout(ctx, lay, xf, ox, oy)
        if ctx.debug:
            cv = ctx.canvas
            box = [(l + bp["lIns"], t + bp["tIns"]), (r - bp["rIns"], t + bp["tIns"]),
                   (r - bp["rIns"], b - bp["bIns"]), (l + bp["lIns"], b - bp["bIns"])]
            cv.stroke([(cv.P(xf.map(box)), True)], 1, (255, 0, 255, 0.8))
        # checks
        if need - area_h > TOL_TEXT:
            note = " (%s set; PowerPoint applies it only after an edit)" % bp["autofit"] if bp["autofit"] else ""
            self.issue(ctx, part, el, name, "text_overflow",
                       "text needs %.2f in, space %.2f in (+%.2f in)%s" % (inch(need), inch(area_h), inch(need - area_h), note), text)
        elif wrap:
            # advisory: the box fits, but a line fits so narrowly that PowerPoint could wrap it, and one
            # more line would overflow
            for i, ln_ in enumerate(lay.lines):
                words = [f for f in ln_.frags if f.text.strip()]
                slack = ln_.right - ln_.left - ln_.width
                if len(words) < 2 or slack >= TIGHT_PT:
                    continue
                extra = max(l2.height for l2 in lay.lines if l2.para == ln_.para) * EMU_PT
                if need + extra - area_h > TOL_TEXT:
                    self.issue(ctx, part, el, name, "tight_fit",
                               "line %d fits with %.2f pt to spare; one more wrapped line would overflow by %.2f in"
                               % (i + 1, slack, inch(need + extra - area_h)), text)
                    break
        for s in lay.splits:
            self.issue(ctx, part, el, name, "word_split",
                       "word '%s' is %.2f in wide, line %.2f in" % (s["word"][:40], s["width_pt"] / 72, max(0, s["line_pt"]) / 72), text)
        if (xf.rot + (180 if xf.flip_v else 0)) % 360 == 0:
            tx0, ty0 = xf.x + ox + lay.left * EMU_PT, xf.y + oy
            tx1, ty1 = xf.x + ox + lay.right * EMU_PT, xf.y + oy + need
            if xf.x >= -TOL_OOB and xf.y >= -TOL_OOB and xf.x + xf.w <= self.cx + TOL_OOB and xf.y + xf.h <= self.cy + TOL_OOB:
                msgs = []
                if tx0 < -TOL_OOB:
                    msgs.append("%.2f in past left edge" % inch(-tx0))
                if ty0 < -TOL_OOB:
                    msgs.append("%.2f in past top edge" % inch(-ty0))
                if tx1 > self.cx + TOL_OOB:
                    msgs.append("%.2f in past right edge" % inch(tx1 - self.cx))
                if ty1 > self.cy + TOL_OOB:
                    msgs.append("%.2f in past bottom edge" % inch(ty1 - self.cy))
                if msgs:
                    self.issue(ctx, part, el, name, "out_of_bounds", "text lines extend " + ", ".join(msgs), text)

    def draw_layout(self, ctx, lay, xf, ox, oy):
        """draw a TextLayout whose top-left is at local (ox, oy) of the shape box xf"""
        cv = ctx.canvas
        angle = (xf.rot + (180 if xf.flip_v else 0)) % 360
        if angle == 0:
            self._draw_lines(ctx, cv, lay, xf.x + ox, xf.y + oy)
            return
        # rotated: draw into an unrotated RGBA layer of the shape box, rotate, composite at the center
        pad = max(xf.w, xf.h)
        k = cv.k
        layer = Image.new("RGBA", (max(1, int((xf.w + 2 * pad) * k)), max(1, int((xf.h + 2 * pad) * k))), (0, 0, 0, 0))
        lc = Canvas(layer, k, xf.x - pad, xf.y - pad)
        self._draw_lines(ctx, lc, lay, xf.x + ox, xf.y + oy)
        rot = layer.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        cxp, cyp = cv.X(xf.x + xf.w / 2), cv.Y(xf.y + xf.h / 2)
        cv.image(rot, int(round(cxp - rot.width / 2)), int(round(cyp - rot.height / 2)))

    def _draw_lines(self, ctx, cv, lay, x0, y0):
        fonts, ptk = ctx.fonts, EMU_PT * cv.k  # px per pt
        for ln_ in lay.lines:
            base = y0 + (ln_.top + ln_.height - ln_.descent) * EMU_PT
            items = ([ln_.bullet] if ln_.bullet else []) + ln_.frags
            for f in items:
                if not f.text or f.st.color is None:
                    continue
                st = f.st
                if f is not ln_.bullet:
                    fonts.note_face(st.face)
                t = f.text.upper() if st.caps else f.text
                px = st.eff_size * ptk
                yb = base - st.baseline * st.size * EMU_PT
                fx = x0 + f.x * EMU_PT
                if st.spc:
                    xx = fx
                    for ch in t:
                        self._glyphs(ctx, cv, xx, yb, ch, st, px)
                        xx += fonts.width(ch, replace(st, caps=False)) * EMU_PT
                else:
                    self._glyphs(ctx, cv, fx, yb, t, st, px)
                if st.underline or st.strike:
                    th = max(0.05 * st.eff_size * EMU_PT, EMU_PT * 0.5)
                    for yy in ([yb + 0.1 * st.eff_size * EMU_PT] if st.underline else []) + ([yb - 0.3 * st.eff_size * EMU_PT] if st.strike else []):
                        cv.fill_polys([cv.P([(fx, yy), (fx + f.w * EMU_PT, yy), (fx + f.w * EMU_PT, yy + th), (fx, yy + th)])], st.color)

    def _glyphs(self, ctx, cv, x, y, text, st, px):
        fonts = ctx.fonts
        xx = x
        for chunk, fb in fonts.chunks(text):
            font = fonts.draw_font(st.bold, st.italic, px, fb)
            cv.text(cv.X(xx), cv.Y(y), chunk, font, st.color, fonts.feats(st.kern))
            xx += fonts.width_em(chunk, st.bold, st.italic, st.kern) * st.eff_size * EMU_PT

    # -- pictures ----------------------------------------------------------------------------
    def render_pic(self, ctx, pic, part, owner, T):
        name = shape_name(pic)
        ph = ph_of(pic)
        if ph is not None and owner != "slide":
            return
        spPr = find(pic, "p:spPr")
        xfrm = find(spPr, "a:xfrm")
        if xfrm is None and ph is not None:
            lsp = find_ph(ctx.layout.phs, ph.get("type", "obj"), ph.get("idx"))
            xfrm = find(lsp, "p:spPr/a:xfrm")
        if xfrm is None:
            return
        xf = read_xfrm(xfrm, T)
        geom = find(spPr, "a:prstGeom")
        prst = geom.get("prst", "rect") if geom is not None else "rect"
        g = preset_geometry(prst, xf.w, xf.h, adj_values(geom)) if prst != "rect" else None
        self.draw_blip(ctx, find(pic, "p:blipFill"), part, xf, g[0] if g else None, name=name, el=pic)
        line = parse_line(find(spPr, "a:ln"), ctx.cc)
        if line is not None:
            self.draw_outline(ctx, g[0] if g else [([(0, 0), (xf.w, 0), (xf.w, xf.h), (0, xf.h)], True)], xf, line)
        self.check_bounds(ctx, part, pic, name, xf, "")

    def draw_blip(self, ctx, blipfill, part, xf, clip_paths, name="", el=None):
        cv = ctx.canvas
        blip = find(blipfill, "a:blip")
        target = self.pkg.target(part, blip.get(R_EMBED)) if blip is not None else None
        im = self.image(target) if target else None
        if im is None:
            self.issue(ctx, part, el if el is not None else blipfill, name or "image", "unsupported",
                       "image %s could not be decoded" % (posixpath.basename(target) if target else "(missing)"))
            box = xf.map([(0, 0), (xf.w, 0), (xf.w, xf.h), (0, xf.h)])
            cv.fill_polys([cv.P(box)], (220, 220, 220, 1.0))
            cv.stroke([(cv.P(box), True)], 2, (200, 0, 120, 1.0), "dash")
            return
        # srcRect selects the part of the image (fractions; negative = padding, as pptxgenjs 'contain' writes)
        # that is stretched onto the fillRect of the box
        W, H = im.size
        src = find(blipfill, "a:srcRect")
        l_, t_, r_, b_ = [pct(src.get(k), 0.0) if src is not None else 0.0 for k in ("l", "t", "r", "b")]
        fr = find(blipfill, "a:stretch/a:fillRect")
        fl, ft, frr, fb = [pct(fr.get(k), 0.0) if fr is not None else 0.0 for k in ("l", "t", "r", "b")]
        wpx, hpx = max(1, round(xf.w * cv.k)), max(1, round(xf.h * cv.k))
        fx0, fy0, fx1, fy1 = fl * wpx, ft * hpx, wpx * (1 - frr), hpx * (1 - fb)
        dw, dh = 1 - l_ - r_, 1 - t_ - b_
        u0, u1, v0, v1 = max(0.0, l_), min(1.0, 1 - r_), max(0.0, t_), min(1.0, 1 - b_)
        if dw <= 0 or dh <= 0 or fx1 <= fx0 or fy1 <= fy0 or u1 <= u0 or v1 <= v0:
            return
        bx0, bx1 = fx0 + (u0 - l_) / dw * (fx1 - fx0), fx0 + (u1 - l_) / dw * (fx1 - fx0)
        by0, by1 = fy0 + (v0 - t_) / dh * (fy1 - fy0), fy0 + (v1 - t_) / dh * (fy1 - fy0)
        piece = im.crop((round(u0 * W), round(v0 * H), max(round(u0 * W) + 1, round(u1 * W)), max(round(v0 * H) + 1, round(v1 * H))))
        piece = piece.resize((max(1, round(bx1 - bx0)), max(1, round(by1 - by0))), Image.Resampling.LANCZOS)
        if piece.size == (wpx, hpx) and round(bx0) == 0 and round(by0) == 0:
            im = piece
        else:
            im = Image.new("RGBA", (wpx, hpx), (0, 0, 0, 0))
            im.paste(piece, (round(bx0), round(by0)))
        amf = find(blip, "a:alphaModFix")
        if amf is not None:
            a = pct(amf.get("amt"), 1.0)
            im.putalpha(im.getchannel("A").point(lambda v, a=a: int(v * a + 0.5)))
        if clip_paths:
            mask = Image.new("L", im.size, 0)
            d = ImageDraw.Draw(mask)
            for pts, closed in clip_paths:
                if closed:
                    d.polygon([(x * cv.k, y * cv.k) for x, y in pts], fill=255)
            im.putalpha(ImageChops.multiply(im.getchannel("A"), mask))
        if xf.flip_h:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if xf.flip_v:
            im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if xf.rot % 360:
            im = im.rotate(-xf.rot, resample=Image.Resampling.BICUBIC, expand=True)
        cxp, cyp = cv.X(xf.x + xf.w / 2), cv.Y(xf.y + xf.h / 2)
        cv.image(im, int(round(cxp - im.width / 2)), int(round(cyp - im.height / 2)))

    # -- graphic frames ----------------------------------------------------------------------
    def render_frame(self, ctx, gf, part, owner, T):
        name = shape_name(gf)
        if ph_of(gf) is not None and owner != "slide":
            return
        xfrm = find(gf, "p:xfrm")
        if xfrm is None:
            return
        xf = read_xfrm(xfrm, T)
        gd = find(gf, "a:graphic/a:graphicData")
        uri = gd.get("uri", "") if gd is not None else ""
        if uri.endswith("/table"):
            self.render_table(ctx, gf, find(gd, "a:tbl"), part, name, xf)
        elif uri.endswith("/chart"):
            ch = find(gd, "c:chart")
            cpart = self.pkg.target(part, ch.get(R_ID)) if ch is not None else None
            self.check_bounds(ctx, part, gf, name, xf, "")
            if cpart and self.pkg.xml(cpart) is not None:
                ChartRenderer(self, ctx, part, gf, name, xf, cpart).render()
            else:
                self.placeholder_box(ctx, xf, "chart part missing")
                self.issue(ctx, part, gf, name, "unsupported", "chart part not found")
        else:
            self.placeholder_box(ctx, xf, uri.rsplit("/", 1)[-1] or "graphicFrame")
            self.issue(ctx, part, gf, name, "unsupported", "graphicFrame %s not rendered" % (uri.rsplit("/", 1)[-1] or "?"))

    def placeholder_box(self, ctx, xf, label):
        cv = ctx.canvas
        box = xf.map([(0, 0), (xf.w, 0), (xf.w, xf.h), (0, xf.h)])
        cv.fill_polys([cv.P(box)], (240, 240, 240, 1.0))
        cv.stroke([(cv.P(box), True)], 2, (200, 0, 120, 1.0), "dash")
        st = RunStyle(size=12, color=(200, 0, 120, 1.0))
        lay = layout_text([Para({"algn": "ctr"}, [("text", label, st)], st)], xf.w / EMU_PT, ctx.fonts)
        self._draw_lines(ctx, cv, lay, xf.x, xf.y + (xf.h - lay.height * EMU_PT) / 2)

    def render_table(self, ctx, gf, tbl, part, name, xf):
        cv = ctx.canvas
        grid = [to_int(g.get("w")) for g in findall(tbl, "a:tblGrid/a:gridCol")]
        rows = findall(tbl, "a:tr")
        if not grid or not rows:
            return
        if find(tbl, "a:tblPr/a:tableStyleId") is not None:
            self.warn("table style (tableStyleId) not rendered; only explicit cell formatting is drawn")
        ncols = len(grid)
        colx = [xf.x]
        for g in grid:
            colx.append(colx[-1] + g)
        row_min = [to_int(tr.get("h")) for tr in rows]
        cells = []
        chain = [None, ctx.master.other_lst, self.default_lst]
        all_text = []
        for r, tr in enumerate(rows):
            c = 0
            for tc in findall(tr, "a:tc"):
                if c >= ncols:
                    break
                gs, rs = max(1, to_int(tc.get("gridSpan"), 1)), max(1, to_int(tc.get("rowSpan"), 1))
                if is_true(tc.get("hMerge")) or is_true(tc.get("vMerge")):
                    c += 1
                    continue
                tcpr = find(tc, "a:tcPr")
                g = lambda k, d: to_int(tcpr.get(k), d) if tcpr is not None and tcpr.get(k) is not None else d
                mar = (g("marL", 91440), g("marR", 91440), g("marT", 45720), g("marB", 45720))
                cw = sum(grid[c:c + gs])
                txb = find(tc, "a:txBody")
                paras = []
                if txb is not None:
                    chain[0] = parse_lst(find(txb, "a:lstStyle"))
                    paras = parse_paragraphs(ctx, txb, list(chain))
                lay = layout_text(paras, (cw - mar[0] - mar[1]) / EMU_PT, ctx.fonts) if paras else None
                text = plain_text(paras) if paras else ""
                all_text.append(text)
                need = (lay.height * EMU_PT if lay else 0) + mar[2] + mar[3]
                cells.append(dict(r=r, c=c, rs=min(rs, len(rows) - r), gs=min(gs, ncols - c), tcpr=tcpr, mar=mar,
                                  lay=lay, text=text, need=need))
                c += 1  # every a:tc is one grid column; spanned columns follow as hMerge cells
        row_h = list(row_min)
        for cell in cells:
            if cell["rs"] == 1:
                row_h[cell["r"]] = max(row_h[cell["r"]], cell["need"])
        for cell in cells:
            if cell["rs"] > 1:
                tot = sum(row_h[cell["r"]:cell["r"] + cell["rs"]])
                if cell["need"] > tot:
                    row_h[cell["r"] + cell["rs"] - 1] += cell["need"] - tot
        rowy = [xf.y]
        for h in row_h:
            rowy.append(rowy[-1] + h)
        # fills, text, then borders on top
        for cell in cells:
            x0, x1 = colx[cell["c"]], colx[min(ncols, cell["c"] + cell["gs"])]
            y0, y1 = rowy[cell["r"]], rowy[cell["r"] + cell["rs"]]
            cell["box"] = (x0, y0, x1, y1)
            fill = parse_fill(cell["tcpr"], ctx.cc)
            if isinstance(fill, tuple) and fill and not isinstance(fill[0], str):
                cv.fill_polys([cv.P([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])], fill)
        for cell in cells:
            if not cell["text"]:
                continue
            x0, y0, x1, y1 = cell["box"]
            ml, mr, mt, mb = cell["mar"]
            lay = cell["lay"]
            need = lay.height * EMU_PT
            area = (y1 - y0) - mt - mb
            anchor = cell["tcpr"].get("anchor", "t") if cell["tcpr"] is not None else "t"
            dy = (area - need) / 2 if anchor == "ctr" else (area - need if anchor == "b" else 0)
            self._draw_lines(ctx, cv, lay, x0 + ml, y0 + mt + dy)
            for s in lay.splits:
                self.issue(ctx, part, gf, name, "word_split",
                           "cell r%dc%d: word '%s' is %.2f in wide, line %.2f in" % (cell["r"] + 1, cell["c"] + 1, s["word"][:40], s["width_pt"] / 72, max(0, s["line_pt"]) / 72), cell["text"])
        for cell in cells:
            x0, y0, x1, y1 = cell["box"]
            for tag, seg in (("lnL", [(x0, y0), (x0, y1)]), ("lnR", [(x1, y0), (x1, y1)]),
                             ("lnT", [(x0, y0), (x1, y0)]), ("lnB", [(x0, y1), (x1, y1)])):
                ln_ = parse_line(find(cell["tcpr"], "a:" + tag), ctx.cc)
                if ln_ is not None:
                    cv.stroke([(cv.P(seg), False)], ln_.width * cv.k, ln_.color, ln_.dash)
        # checks
        total = rowy[-1] - xf.y
        intended = max(xf.h, sum(row_min))
        text = " | ".join(t for t in all_text if t)
        if total - intended > TOL_TABLE:
            note = " (frame is the pptxgenjs default 1 in: no h/rowH given)" if xf.h == EMU_IN and sum(row_min) < EMU_IN else ""
            self.issue(ctx, part, gf, name, "table_overflow", "rows grew to %.2f in, frame %.2f in (+%.2f in)%s"
                       % (inch(total), inch(intended), inch(total - intended), note), text)
        if rowy[-1] > self.cy + TOL_OOB:
            self.issue(ctx, part, gf, name, "table_overflow", "table ends %.2f in below the slide bottom" % inch(rowy[-1] - self.cy), text)
        self.check_bounds(ctx, part, gf, name, Xform(xf.x, xf.y, colx[-1] - xf.x, min(total, xf.h) if total else xf.h), text)


def is_hidden(el):
    for nv in el:
        if lname(nv).startswith("nv"):
            c = find(nv, "p:cNvPr")
            return c is not None and is_true(c.get("hidden"))
    return False


def shape_name(el):
    for nv in el:
        if lname(nv).startswith("nv"):
            c = find(nv, "p:cNvPr")
            if c is not None:
                return c.get("name", "") or ("id %s" % c.get("id", "?"))
    return lname(el)


def plain_text_quick(sp):
    return " ".join("".join(t.text or "" for t in findall(p, ".//a:t")) for p in findall(sp, "p:txBody/a:p")).strip()


# ----------------------------------------------------------------------------------------------
# Charts (bar/column)
# ----------------------------------------------------------------------------------------------
def format_number(v, code):
    """Excel-style number formatting for the codes charts use."""
    if v is None:
        return ""
    if not code or code.strip().lower() == "general":
        if float(v).is_integer() and abs(v) < 1e15:
            return str(int(v))
        s = "%.10g" % v
        return s
    sections = code.split(";")
    sec = sections[0]
    neg = v < 0
    if neg and len(sections) > 1 and sections[1]:
        sec, v, neg = sections[1], -v, False
    toks, i = [], 0
    while i < len(sec):
        ch = sec[i]
        if ch == '"':
            j = sec.find('"', i + 1)
            j = len(sec) if j < 0 else j
            toks.append(("lit", sec[i + 1:j]))
            i = j + 1
        elif ch == "\\" and i + 1 < len(sec):
            toks.append(("lit", sec[i + 1]))
            i += 2
        elif ch == "[":
            j = sec.find("]", i)
            i = len(sec) if j < 0 else j + 1
        elif ch == "_":
            toks.append(("lit", " "))
            i += 2
        elif ch == "*":
            i += 2
        elif ch in "0#?.,":
            j = i
            while j < len(sec) and sec[j] in "0#?.,":
                j += 1
            toks.append(("num", sec[i:j]))
            i = j
        elif ch == "%":
            toks.append(("pct", "%"))
            i += 1
        else:
            toks.append(("lit", ch))
            i += 1
    num = next((t for k, t in toks if k == "num"), None)
    scale = 100 ** sum(1 for k, _ in toks if k == "pct")
    val = abs(v) * scale
    if num is None:
        body = ""
    else:
        ip, _, dp = num.partition(".")
        trailing = len(ip) - len(ip.rstrip(","))
        val /= 1000 ** trailing
        ip = ip.rstrip(",")
        dmin, dmax = dp.count("0"), len(dp.replace(",", ""))
        q = Decimal(str(val)).quantize(Decimal(1).scaleb(-dmax), rounding=ROUND_HALF_UP)
        s = format(q, "f")
        a, _, b = s.partition(".")
        b = b[:dmax]
        while len(b) > dmin and b.endswith("0"):
            b = b[:-1]
        imin = ip.count("0")
        a = a.lstrip("0") or ""
        if len(a) < imin:
            a = "0" * (imin - len(a)) + a
        if "," in ip and a:
            a = "{:,}".format(int(a))
        body = a + ("." + b if b else "")
    out, done = "", False
    for k, t in toks:
        if k == "num":
            if not done:
                out += body
                done = True
        elif k == "pct":
            out += "%"
        else:
            out += t
    return ("-" if neg and body.strip("0.,") else "") + out


def nice_step(raw):
    if raw <= 0:
        return 1.0
    e = math.floor(math.log10(raw))
    f = raw / 10 ** e
    nf = 1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10
    return nf * 10 ** e


SERIES_COLORS = ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6"]


class ChartRenderer:
    PAD = 0.07 * EMU_IN
    GAP = 0.06 * EMU_IN

    def __init__(self, deck, ctx, part, gf, name, xf, cpart):
        self.deck, self.ctx, self.part, self.gf, self.name, self.xf, self.cpart = deck, ctx, part, gf, name, xf, cpart
        self.cv = ctx.canvas
        self.root = deck.pkg.xml(cpart)

    def style(self, txpr, default_size=10.0, base=None):
        r = dict(base or {})
        r.update(parse_rpr(find(txpr, "a:p/a:pPr/a:defRPr")))
        r.setdefault("sz", default_size)
        rot = find(txpr, "a:bodyPr")
        return make_style(r, self.ctx), (to_int(rot.get("rot")) / 60000.0 if rot is not None and rot.get("rot") else 0.0)

    def block(self, text, st, width_emu, algn="ctr", wrap=True):
        return layout_text([Para({"algn": algn}, [("text", text, st)], st)], max(1.0, width_emu / EMU_PT), self.ctx.fonts, wrap)

    def draw_block(self, lay, x, y):
        self.deck._draw_lines(self.ctx, self.cv, lay, x, y)

    def rect(self, x0, y0, x1, y1, fill=None, line=None):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if isinstance(fill, tuple) and not isinstance(fill[0], str):
            self.cv.fill_polys([self.cv.P(pts)], fill)
        if line is not None:
            self.cv.stroke([(self.cv.P(pts), True)], line.width * self.cv.k, line.color, line.dash)

    def seg(self, a, b, line):
        if line is not None:
            self.cv.stroke([(self.cv.P([a, b]), False)], line.width * self.cv.k, line.color, line.dash)

    def render(self):
        deck, ctx, cc = self.deck, self.ctx, self.ctx.cc
        xf = self.xf
        X0, Y0, CW, CH = xf.x, xf.y, xf.w, xf.h
        root = self.root
        sp = find(root, "c:spPr")
        self.rect(X0, Y0, X0 + CW, Y0 + CH, parse_fill(sp, cc), parse_line(find(sp, "a:ln"), cc))
        chart = find(root, "c:chart")
        pa = find(chart, "c:plotArea")
        bars = findall(pa, "c:barChart")
        others = sorted({lname(e) for e in pa if lname(e).endswith("Chart") and lname(e) != "barChart"})
        lines = findall(pa, "c:lineChart")
        if not bars and lines and others == ["lineChart"] and len(lines) == 1:
            self.render_line(pa, lines[0])
            return
        if not bars:
            self.deck.placeholder_box(ctx, xf, "chart: %s" % (", ".join(others) or "empty"))
            self.deck.issue(ctx, self.part, self.gf, self.name, "unsupported", "chart type %s not rendered" % (", ".join(others) or "?"))
            return
        if others or len(bars) > 1:
            self.deck.issue(ctx, self.part, self.gf, self.name, "unsupported", "combo chart: only the first barChart is drawn")
        bc = bars[0]
        axes = {}
        for ax in pa:
            if lname(ax) in ("catAx", "valAx", "dateAx", "serAx"):
                i = find(ax, "c:axId")
                if i is not None:
                    axes[i.get("val")] = ax
        ids = [a.get("val") for a in findall(bc, "c:axId")]
        cat_ax = axes.get(ids[0]) if ids else None
        val_ax = axes.get(ids[1]) if len(ids) > 1 else None
        g = lambda el, path, d=None: (find(el, path).get("val", d) if find(el, path) is not None else d)
        bar_dir = g(bc, "c:barDir", "col")
        grouping = g(bc, "c:grouping", "clustered")
        stacked = grouping in ("stacked", "percentStacked")
        gap = to_int(g(bc, "c:gapWidth", "150"), 150) / 100.0
        overlap = to_int(g(bc, "c:overlap", "100" if stacked else "0"), 0) / 100.0
        chart_dl = find(bc, "c:dLbls")
        series = []
        for si, ser in enumerate(findall(bc, "c:ser")):
            name = "".join(v.text or "" for v in findall(ser, "c:tx//c:v")) or "Series %d" % (si + 1)
            idx = to_int(g(ser, "c:idx", str(si)), si)
            fill = parse_fill(find(ser, "c:spPr"), cc)
            if fill is None:
                fill = cc.resolve(etree.fromstring('<a:schemeClr xmlns:a="%s" val="%s"/>' % (NS["a"], SERIES_COLORS[idx % 6])))
            dpts = {}
            for dp in findall(ser, "c:dPt"):
                f = parse_fill(find(dp, "c:spPr"), cc)
                if f is not None:
                    dpts[to_int(g(dp, "c:idx", "0"))] = f
            cats = self.cat_values(find(ser, "c:cat"))
            vals, vfmt = self.num_values(find(ser, "c:val"))
            series.append(dict(name=name, fill=fill, dpts=dpts, cats=cats, vals=vals, vfmt=vfmt,
                               dl=find(ser, "c:dLbls"), line=parse_line(find(ser, "c:spPr/a:ln"), cc)))
        if not series:
            return
        ncat = max(max(len(s["vals"]) for s in series), len(series[0]["cats"]))
        cats = series[0]["cats"] + [""] * (ncat - len(series[0]["cats"]))
        horiz = bar_dir == "bar"

        def axinfo(ax):
            d = dict(deleted=True, orient="minMax", mn=None, mx=None, major=None, fmt=None, linked=False,
                     style=None, rot=0.0, line=None, grid=None, tick="none", lblpos="nextTo")
            if ax is None:
                return d
            d["deleted"] = is_true(g(ax, "c:delete", "0"))
            d["orient"] = g(ax, "c:scaling/c:orientation", "minMax")
            mn, mx = g(ax, "c:scaling/c:min"), g(ax, "c:scaling/c:max")
            d["mn"] = float(mn) if mn is not None else None
            d["mx"] = float(mx) if mx is not None else None
            mu = g(ax, "c:majorUnit")
            d["major"] = float(mu) if mu is not None else None
            nf = find(ax, "c:numFmt")
            if nf is not None:
                d["fmt"], d["linked"] = nf.get("formatCode"), is_true(nf.get("sourceLinked"))
            d["style"], d["rot"] = self.style(find(ax, "c:txPr"))
            d["line"] = parse_line(find(ax, "c:spPr/a:ln"), cc) if find(ax, "c:spPr/a:ln") is not None else LineSpec(9525, (134, 134, 134, 1.0))
            gl = find(ax, "c:majorGridlines")
            if gl is not None:
                d["grid"] = parse_line(find(gl, "c:spPr/a:ln"), cc) if find(gl, "c:spPr/a:ln") is not None else LineSpec(9525, (217, 217, 217, 1.0))
            d["tick"] = g(ax, "c:majorTickMark", "out")
            d["lblpos"] = g(ax, "c:tickLblPos", "nextTo")
            return d

        CA, VA = axinfo(cat_ax), axinfo(val_ax)
        # value scale
        if grouping == "percentStacked":
            lo, hi = 0.0, 1.0
            dmin, dmax = 0.0, 1.0
        else:
            if stacked:
                pos = [sum(max(0.0, s["vals"][i] or 0.0) for s in series if i < len(s["vals"])) for i in range(ncat)]
                neg = [sum(min(0.0, s["vals"][i] or 0.0) for s in series if i < len(s["vals"])) for i in range(ncat)]
                dmin, dmax = min(neg + [0.0]), max(pos + [0.0])
            else:
                allv = [v for s in series for v in s["vals"] if v is not None]
                dmin, dmax = min(allv + [0.0]), max(allv + [0.0])
        lo = VA["mn"] if VA["mn"] is not None else (0.0 if dmin >= 0 else dmin)
        hi = VA["mx"] if VA["mx"] is not None else dmax
        major = VA["major"] or nice_step(((hi if VA["mx"] is not None else dmax + 0.05 * (dmax - lo)) - lo) / 6 or 1)
        if VA["mx"] is None:
            hi = math.ceil((dmax + 0.05 * (dmax - lo)) / major - 1e-9) * major if dmax > lo else lo + major
        if VA["mn"] is None and dmin < 0:
            lo = math.floor((dmin - 0.05 * (hi - dmin)) / major + 1e-9) * major
        if hi <= lo:
            hi = lo + 1.0
        ticks = []
        t = lo
        while t <= hi + 1e-9 * max(1.0, abs(hi)) and len(ticks) < 200:
            ticks.append(t)
            t = lo + major * len(ticks)
        vfmt = VA["fmt"] if (VA["fmt"] and not VA["linked"]) else series[0]["vfmt"]
        if grouping == "percentStacked" and not VA["fmt"]:
            vfmt = "0%"
        # title and legend
        x_l, y_t, x_r, y_b = X0 + self.PAD, Y0 + self.PAD, X0 + CW - self.PAD, Y0 + CH - self.PAD
        title_el = find(chart, "c:title")
        title_lay = None
        if title_el is not None and not is_true(g(chart, "c:autoTitleDeleted", "0")):
            runs = findall(title_el, "c:tx/c:rich/a:p")
            ttext = " ".join("".join(t.text or "" for t in findall(p, ".//a:t")) for p in runs).strip()
            if not ttext and len(series) == 1:
                ttext = series[0]["name"]
            tst, _ = self.style(find(title_el, "c:tx/c:rich"), 18.0, {"b": True})
            rp = find(title_el, "c:tx/c:rich/a:p/a:r/a:rPr")
            if rp is not None:
                tst = make_style({**{"sz": tst.size, "b": tst.bold}, **parse_rpr(rp)}, ctx)
            if ttext:
                title_lay = self.block(ttext, tst, CW * 0.9)
                self.draw_block(title_lay, X0 + CW * 0.05, y_t)
                if not is_true(g(title_el, "c:overlay", "0")):
                    y_t += title_lay.height * EMU_PT + self.GAP
        leg = find(chart, "c:legend")
        if leg is not None:
            lst, _ = self.style(find(leg, "c:txPr"), 10.0)
            pos = g(leg, "c:legendPos", "r")
            y_t, y_b, x_l, x_r = self.legend(series, lst, pos, x_l, y_t, x_r, y_b, overlay=is_true(g(leg, "c:overlay", "0")))
        # plot area
        ml = find(pa, "c:layout/c:manualLayout")
        cst, vst = CA["style"], VA["style"]
        cat_lbl = not CA["deleted"] and CA["lblpos"] != "none"
        val_lbl = not VA["deleted"] and VA["lblpos"] != "none"
        vlabels = [format_number(tk, vfmt) for tk in ticks] if val_lbl else []
        vlw = max([ctx.fonts.width(s, vst) for s in vlabels] + [0]) * EMU_PT
        vlh = LINE_RATIO * vst.size * EMU_PT if val_lbl else 0
        manual = None
        if ml is not None and all(find(ml, "c:" + k) is not None for k in ("x", "y", "w", "h")):
            fx, fy, fw, fh = [float(find(ml, "c:" + k).get("val")) for k in ("x", "y", "w", "h")]
            manual = (X0 + fx * CW, Y0 + fy * CH, X0 + (fx + fw) * CW, Y0 + (fy + fh) * CH)
            if g(ml, "c:xMode", "edge") != "edge" or g(ml, "c:yMode", "edge") != "edge":
                self.deck.warn("chart manualLayout xMode/yMode 'factor' treated as 'edge'")
        inner = manual is not None and g(ml, "c:layoutTarget", "outer") == "inner"
        px0, py0, px1, py1 = manual if manual else (x_l, y_t, x_r, y_b)
        if not inner:
            if horiz:
                if cat_lbl:
                    cw_max = min(max([ctx.fonts.width(c, cst) for c in cats] + [0]) * EMU_PT, 0.4 * CW)
                    px0 += cw_max + self.GAP
                if val_lbl:
                    py1 -= vlh + self.GAP
            else:
                if val_lbl:
                    px0 += vlw + self.GAP
                if cat_lbl:
                    bw = (px1 - px0) / max(1, ncat)
                    hmax = max([self.block(c, cst, bw).height for c in cats] + [0]) * EMU_PT
                    py1 -= hmax + self.GAP
        pw, ph = px1 - px0, py1 - py0
        if pw <= 0 or ph <= 0:
            return
        self.rect(px0, py0, px1, py1, parse_fill(find(pa, "c:spPr"), cc), parse_line(find(pa, "c:spPr/a:ln"), cc))
        cat_rev = CA["orient"] == "maxMin"
        val_rev = VA["orient"] == "maxMin"

        def cpos(tc):  # category coordinate (0..ncat) -> slide EMU along the category axis
            f = tc / max(1, ncat)
            if horiz:
                return py0 + f * ph if cat_rev else py1 - f * ph
            return px1 - f * pw if cat_rev else px0 + f * pw

        def vpos(v):
            f = (min(max(v, lo), hi) - lo) / (hi - lo)
            if horiz:
                return px1 - f * pw if val_rev else px0 + f * pw
            return py0 + f * ph if val_rev else py1 - f * ph

        # gridlines
        if VA["grid"] is not None:
            for tk in ticks:
                p = vpos(tk)
                self.seg((p, py0), (p, py1), VA["grid"]) if horiz else self.seg((px0, p), (px1, p), VA["grid"])
        if CA["grid"] is not None:
            for i in range(ncat + 1):
                p = cpos(i)
                self.seg((px0, p), (px1, p), CA["grid"]) if horiz else self.seg((p, py0), (p, py1), CA["grid"])
        # bars
        nser = 1 if stacked else len(series)
        ov = 1.0 if stacked else max(-1.0, min(1.0, overlap))
        bw = 1.0 / (nser - (nser - 1) * ov + gap)
        base_v = min(max(0.0, lo), hi)
        labels = []
        cum_pos, cum_neg = [0.0] * ncat, [0.0] * ncat
        tot = [sum(abs(s["vals"][i] or 0.0) for s in series if i < len(s["vals"])) for i in range(ncat)]
        for k, s in enumerate(series):
            for i in range(ncat):
                v = s["vals"][i] if i < len(s["vals"]) else None
                if v is None:
                    continue
                vv = v / tot[i] if grouping == "percentStacked" and tot[i] else v
                if stacked:
                    a = cum_pos[i] if vv >= 0 else cum_neg[i]
                    b = a + vv
                    if vv >= 0:
                        cum_pos[i] = b
                    else:
                        cum_neg[i] = b
                    t0 = i + gap * bw / 2
                else:
                    a, b = base_v, vv
                    t0 = i + gap * bw / 2 + k * bw * (1 - ov)
                c0, c1 = cpos(t0), cpos(t0 + bw)
                v0, v1 = vpos(a), vpos(b)
                fill = s["dpts"].get(i, s["fill"])
                if horiz:
                    x0, x1, y0, y1 = min(v0, v1), max(v0, v1), min(c0, c1), max(c0, c1)
                else:
                    x0, x1, y0, y1 = min(c0, c1), max(c0, c1), min(v0, v1), max(v0, v1)
                self.rect(x0, y0, x1, y1, fill, s["line"])
                labels.append((s, i, v, vv, (x0, y0, x1, y1), a, b))
        # axes
        if not CA["deleted"] and CA["line"] is not None:
            p = vpos(base_v)
            self.seg((p, py0), (p, py1), CA["line"]) if horiz else self.seg((px0, p), (px1, p), CA["line"])
        if not VA["deleted"] and VA["line"] is not None:
            p = cpos(0)
            self.seg((px0, p), (px1, p), VA["line"]) if horiz else self.seg((p, py0), (p, py1), VA["line"])
        # category labels
        if cat_lbl:
            for i, c in enumerate(cats):
                if not c:
                    continue
                cc0, cc1 = cpos(i), cpos(i + 1)
                mid = (cc0 + cc1) / 2
                if horiz:
                    avail = max(0.2 * EMU_IN, px0 - X0 - self.GAP)
                    lay = self.block(c, cst, avail, "r")
                    self.draw_block(lay, px0 - self.GAP - avail, mid - lay.height * EMU_PT / 2)
                else:
                    bwid = abs(cc1 - cc0)
                    lay = self.block(c, cst, bwid, "ctr")
                    self.draw_block(lay, mid - bwid / 2, py1 + self.GAP)
        if val_lbl:
            for tk, s in zip(ticks, vlabels):
                p = vpos(tk)
                if horiz:
                    w = ctx.fonts.width(s, vst) * EMU_PT
                    lay = self.block(s, vst, w + EMU_PT, "ctr", wrap=False)
                    yy = (py1 + self.GAP) if not cat_rev else (py0 - self.GAP - vlh)
                    self.draw_block(lay, p - (w + EMU_PT) / 2, yy)
                else:
                    lay = self.block(s, vst, vlw + EMU_PT, "r", wrap=False)
                    self.draw_block(lay, px0 - self.GAP - vlw - EMU_PT, p - vlh / 2)
        # data labels
        for s, i, v, vv, (x0, y0, x1, y1), a, b in labels:
            dl = s["dl"]
            get = lambda tag, d=None: g(dl, tag) if find(dl, tag) is not None else (g(chart_dl, tag) if find(chart_dl, tag) is not None else d)
            if is_true(get("c:delete", "0")) or not is_true(get("c:showVal", "0")):
                continue
            src = dl if find(dl, "c:txPr") is not None else chart_dl
            st, _ = self.style(find(src, "c:txPr"), 10.0)
            nf = find(dl, "c:numFmt") if find(dl, "c:numFmt") is not None else find(chart_dl, "c:numFmt")
            code = nf.get("formatCode") if nf is not None and not is_true(nf.get("sourceLinked")) else s["vfmt"]
            txt = format_number(vv if grouping == "percentStacked" else v, code if grouping != "percentStacked" or code else "0%")
            posn = get("c:dLblPos", "ctr" if stacked else "outEnd")
            w = ctx.fonts.width(txt, st) * EMU_PT
            hgt = LINE_RATIO * st.size * EMU_PT
            lay = self.block(txt, st, w + EMU_PT, "l", wrap=False)
            g2 = 0.04 * EMU_IN
            if horiz:
                end, start = (x1, x0) if (b >= a) != val_rev else (x0, x1)
                d = 1 if end >= start else -1
                yy = (y0 + y1) / 2 - hgt / 2
                if posn == "outEnd":
                    xx = end + g2 if d > 0 else end - g2 - w
                elif posn == "inEnd":
                    xx = end - g2 - w if d > 0 else end + g2
                elif posn == "inBase":
                    xx = start + g2 if d > 0 else start - g2 - w
                else:
                    xx = (x0 + x1) / 2 - w / 2
                self.draw_block(lay, xx, yy)
            else:
                end, start = (y0, y1) if (b >= a) != val_rev else (y1, y0)
                d = -1 if end <= start else 1
                xx = (x0 + x1) / 2 - w / 2
                if posn == "outEnd":
                    yy = end - g2 - hgt if d < 0 else end + g2
                elif posn == "inEnd":
                    yy = end + g2 if d < 0 else end - g2 - hgt
                elif posn == "inBase":
                    yy = start - g2 - hgt if d < 0 else start + g2
                else:
                    yy = (y0 + y1) / 2 - hgt / 2
                self.draw_block(lay, xx, yy)

    def render_line(self, pa, lc):
        """line chart (c:lineChart) with circle markers; axis text, gridlines and data labels are not drawn, so this
        covers charts whose labels are separate text boxes (build_deck_v4.js)"""
        cc, xf = self.ctx.cc, self.xf
        X0, Y0, CW, CH = xf.x, xf.y, xf.w, xf.h
        g = lambda el, path, d=None: (find(el, path).get("val", d) if el is not None and find(el, path) is not None else d)
        axes = {}
        for ax in pa:
            if lname(ax) in ("catAx", "valAx", "dateAx"):
                i = find(ax, "c:axId")
                if i is not None:
                    axes[i.get("val")] = ax
        ids = [a.get("val") for a in findall(lc, "c:axId")]
        val_ax = axes.get(ids[1]) if len(ids) > 1 else None
        for ax in (axes.get(ids[0]) if ids else None, val_ax):
            if ax is not None and not is_true(g(ax, "c:delete", "0")):
                self.deck.issue(self.ctx, self.part, self.gf, self.name, "unsupported", "line chart axis text not drawn")
        series = []
        for ser in findall(lc, "c:ser"):
            vals, _ = self.num_values(find(ser, "c:val"))
            mk = find(ser, "c:marker")
            series.append(dict(vals=vals, line=parse_line(find(ser, "c:spPr/a:ln"), cc), symbol=g(mk, "c:symbol", "none"),
                               size=to_int(g(mk, "c:size", "5"), 5), fill=parse_fill(find(mk, "c:spPr"), cc) if mk is not None else None,
                               edge=parse_line(find(mk, "c:spPr/a:ln"), cc) if mk is not None else None))
        if not series:
            return
        allv = [v for s in series for v in s["vals"] if v is not None]
        mn, mx = g(val_ax, "c:scaling/c:min"), g(val_ax, "c:scaling/c:max")
        lo = float(mn) if mn is not None else min(allv + [0.0])
        hi = float(mx) if mx is not None else max(allv)
        if hi <= lo:
            hi = lo + 1.0
        between = g(val_ax, "c:crossBetween", "between") == "between"
        ml = find(pa, "c:layout/c:manualLayout")
        if ml is not None and all(find(ml, "c:" + k) is not None for k in ("x", "y", "w", "h")):
            fx, fy, fw, fh = [float(find(ml, "c:" + k).get("val")) for k in ("x", "y", "w", "h")]
            px0, py0, px1, py1 = X0 + fx * CW, Y0 + fy * CH, X0 + (fx + fw) * CW, Y0 + (fy + fh) * CH
        else:
            px0, py0, px1, py1 = X0 + self.PAD, Y0 + self.PAD, X0 + CW - self.PAD, Y0 + CH - self.PAD
        n = max(len(s["vals"]) for s in series)

        def xpos(i):
            return px0 + (i + 0.5) * (px1 - px0) / n if between else px0 + i * (px1 - px0) / max(1, n - 1)

        def ypos(v):
            return py1 - (min(max(v, lo), hi) - lo) / (hi - lo) * (py1 - py0)

        for s in series:
            pts = [(xpos(i), ypos(v)) for i, v in enumerate(s["vals"]) if v is not None]
            if s["line"] is not None and len(pts) > 1:
                self.cv.stroke([(self.cv.P(pts), False)], s["line"].width * self.cv.k, s["line"].color, s["line"].dash)
            if s["symbol"] != "none":
                if s["symbol"] != "circle":
                    self.deck.issue(self.ctx, self.part, self.gf, self.name, "unsupported", "marker %s drawn as a circle" % s["symbol"])
                r = s["size"] * EMU_PT / 2
                for x, y in pts:
                    circ = arc(x, y, r, r, 0, 360)
                    self.cv.fill_polys([self.cv.P(circ)], s["fill"])
                    if s["edge"] is not None:
                        self.cv.stroke([(self.cv.P(circ), True)], s["edge"].width * self.cv.k, s["edge"].color, s["edge"].dash)

    def legend(self, series, st, pos, x_l, y_t, x_r, y_b, overlay):
        fonts = self.ctx.fonts
        key = 0.6 * st.size * EMU_PT
        hgt = LINE_RATIO * st.size * EMU_PT
        items = [(s["name"], s["fill"], fonts.width(s["name"], st) * EMU_PT) for s in series]
        sp = 0.12 * EMU_IN
        if pos in ("b", "t"):
            total = sum(key + self.GAP + w for _, _, w in items) + sp * (len(items) - 1)
            x = (x_l + x_r) / 2 - total / 2
            y = y_b - hgt if pos == "b" else y_t
            for nm, fill, w in items:
                self.rect(x, y + (hgt - key) / 2, x + key, y + (hgt + key) / 2, fill)
                self.draw_block(self.block(nm, st, w + EMU_PT, "l", False), x + key + self.GAP, y)
                x += key + self.GAP + w + sp
            if not overlay:
                if pos == "b":
                    y_b -= hgt + self.GAP
                else:
                    y_t += hgt + self.GAP
        else:
            wmax = max(w for _, _, w in items) + key + self.GAP
            x = x_r - wmax if pos in ("r", "tr") else x_l
            y = (y_t + y_b) / 2 - hgt * len(items) / 2 if pos != "tr" else y_t
            for nm, fill, w in items:
                self.rect(x, y + (hgt - key) / 2, x + key, y + (hgt + key) / 2, fill)
                self.draw_block(self.block(nm, st, w + EMU_PT, "l", False), x + key + self.GAP, y)
                y += hgt
            if not overlay:
                if pos in ("r", "tr"):
                    x_r -= wmax + self.GAP
                else:
                    x_l += wmax + self.GAP
        return y_t, y_b, x_l, x_r

    def cat_values(self, cat):
        if cat is None:
            return []
        lvl = find(cat, ".//c:lvl")
        holder = lvl if lvl is not None else (find(cat, ".//c:strCache") if find(cat, ".//c:strCache") is not None else find(cat, ".//c:numCache"))
        if holder is None:
            return []
        fmt = find(holder, "c:formatCode")
        pts = {}
        for pt in findall(holder, "c:pt"):
            v = find(pt, "c:v")
            txt = v.text if v is not None and v.text is not None else ""
            if lname(holder) == "numCache":
                try:
                    txt = format_number(float(txt), fmt.text if fmt is not None else "General")
                except ValueError:
                    pass
            pts[to_int(pt.get("idx"))] = txt
        n = max(pts) + 1 if pts else 0
        pc = find(holder, "c:ptCount")
        if pc is not None:
            n = max(n, to_int(pc.get("val")))
        return [pts.get(i, "") for i in range(n)]

    def num_values(self, val):
        nc = find(val, ".//c:numCache") if val is not None else None
        if nc is None:
            nc = find(val, ".//c:numLit") if val is not None else None
        if nc is None:
            return [], "General"
        fmt = find(nc, "c:formatCode")
        pts = {}
        for pt in findall(nc, "c:pt"):
            v = find(pt, "c:v")
            try:
                pts[to_int(pt.get("idx"))] = float(v.text)
            except (TypeError, ValueError, AttributeError):
                pass
        n = max(pts) + 1 if pts else 0
        pc = find(nc, "c:ptCount")
        if pc is not None:
            n = max(n, to_int(pc.get("val")))
        return [pts.get(i) for i in range(n)], (fmt.text if fmt is not None and fmt.text else "General")


# ----------------------------------------------------------------------------------------------
# Outputs and CLI
# ----------------------------------------------------------------------------------------------
def parse_slides(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [s for s in out if s < 1 or s > n]
    if bad:
        raise SystemExit("--slides: no such slide %s (deck has %d)" % (bad, n))
    return [s - 1 for s in sorted(set(out))]


def contact_sheet(images, numbers, path, cols=4, thumb_w=400, fonts=None):
    th = round(thumb_w * images[0].height / images[0].width)
    pad, lab = 14, 30
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (pad + cols * (thumb_w + pad), pad + rows * (th + lab + pad)), (210, 214, 220))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(fonts.paths[(True, False)], 18) if fonts else ImageFont.load_default()
    for i, (im, n) in enumerate(zip(images, numbers)):
        r, c = divmod(i, cols)
        x, y = pad + c * (thumb_w + pad), pad + r * (th + lab + pad)
        sheet.paste(im.resize((thumb_w, th), Image.Resampling.LANCZOS), (x, y + lab))
        d.rectangle([x - 1, y + lab - 1, x + thumb_w, y + lab + th], outline=(120, 120, 120))
        d.text((x, y + lab - 6), "Slide %d" % n, font=font, fill=(20, 30, 40), anchor="ls")
    sheet.save(path, quality=88)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a pptxgenjs .pptx to PNG and report text overflow.")
    ap.add_argument("deck")
    ap.add_argument("--out", required=True, help="output directory for slide-NN.png and laporan.json")
    ap.add_argument("--width", type=int, default=1600, help="PNG width in px (default 1600)")
    ap.add_argument("--pdf", help="also write all rendered slides into this PDF")
    ap.add_argument("--sheet", help="also write a contact sheet (4 per row) to this image file")
    ap.add_argument("--slides", help="slides to render, e.g. 1,3,5-7 (default: all)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any overflow/out-of-bounds issue exists")
    ap.add_argument("--fonts", help="directory with Carlito-{Regular,Bold,Italic,BoldItalic}.ttf")
    ap.add_argument("--no-kern", action="store_true",
                    help="measure without pair kerning (sensitivity check for borderline wraps)")
    ap.add_argument("--debug", action="store_true", help="outline text areas in magenta")
    args = ap.parse_args(argv)

    font_dir = args.fonts or next((d for d in FONT_DIR_CANDIDATES if os.path.isdir(d)), FONT_DIR_CANDIDATES[0])
    deck = Deck(args.deck, font_dir, debug=args.debug)
    deck.fonts.kerning = not args.no_kern
    os.makedirs(args.out, exist_ok=True)
    picks = parse_slides(args.slides, len(deck.slides))
    images, numbers, hidden = [], [], []
    for i in picks:
        img, shown = deck.render(i, args.width)
        n = i + deck.first_num
        img.save(os.path.join(args.out, "slide-%02d.png" % n))
        images.append(img)
        numbers.append(n)
        if not shown:
            hidden.append(n)
    if args.pdf and images:
        dpi = args.width / (deck.cx / EMU_IN)
        images[0].save(args.pdf, "PDF", resolution=dpi, save_all=True, append_images=images[1:])
    if args.sheet and images:
        contact_sheet(images, numbers, args.sheet, fonts=deck.fonts)

    if deck.fonts.unknown:
        deck.warn("font substitution: %s drawn and measured with Carlito" % ", ".join(sorted(deck.fonts.unknown)))
    if deck.fonts.missing:
        miss = "".join(sorted(deck.fonts.missing))
        deck.warn("glyphs missing from Carlito (%s) drawn with %s" % (miss, "DejaVu Sans" if deck.fonts.fb_ref else "notdef boxes"))
    if not deck.fonts.raqm:
        deck.warn("Pillow has no raqm: kerning/ligature control unavailable, widths approximate")
    if hidden:
        deck.warn("hidden slides rendered: %s" % ", ".join(map(str, hidden)))

    issues = sorted(deck.issues, key=lambda d: (d["slide"], d["kind"], d["shape"]))
    counts = {}
    for it in issues:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    report = {
        "deck": os.path.abspath(args.deck),
        "slide_size_in": [round(deck.cx / EMU_IN, 3), round(deck.cy / EMU_IN, 3)],
        "width_px": args.width,
        "slides_rendered": numbers,
        "counts": counts,
        "issues": issues,
        "warnings": ["%s (x%d)" % (m, c) if c > 1 else m for m, c in sorted(deck.warnings.items())],
    }
    with open(os.path.join(args.out, "laporan.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print("rendered %d slide(s) of %s -> %s (%dx%d px)" % (len(numbers), os.path.basename(args.deck), args.out,
                                                         args.width, round(args.width * deck.cy / deck.cx)))
    if issues:
        print("issues: %d (%s)" % (len(issues), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
        for it in issues:
            print("  slide %02d | %-22s | %-14s | %s | %s" % (it["slide"], it["shape"][:22], it["kind"], it["detail"],
                                                             json.dumps(it["text"], ensure_ascii=False)))
    else:
        print("issues: none")
    for w in report["warnings"]:
        print("warning: " + w)
    if args.strict and any(it["kind"] in OVERFLOW_KINDS for it in issues):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
