/*
 * Builds deck/ppt03/BISMILLAH_Final_v4.pptx with pptxgenjs 4.0.1 (structured deck: theme, layouts with named
 * placeholders, sections, speaker notes). v4 revises v3 (deck/ppt02) as listed in part G4 of 01_struktur_deck_v4.md:
 * every on-screen text is at least 14 pt (footers 12 pt) and each main slide prints its notebook cells in the footer.
 *
 * Sources (nothing is typed twice where the markdown already has it):
 *   build/isi_v4.json   titles, on-screen text ("Isi Teks"), narration, timing, sources, notebook cells and backup
 *                       evidence, exported from 01_struktur_deck_v4.md by isi_deck.py
 *   img/data_grafik.json numbers behind the native bar and line charts, checked against the notebook by
 *                       make_visuals_v4.py
 *   img/*.png           quadrant map and SHAP dependence plots (make_visuals_v4.py)
 *
 * Native charts carry no axis text: every label is a text box placed from the chart's fixed inner plot area
 * (manual layout), so numbers keep the Indonesian decimal comma whatever the locale of the presenting computer.
 *
 * usage:
 *   python3 deck/ppt03/isi_deck.py
 *   NODE_PATH=<dir>/node_modules PPTX_SKILL_DIR=<pptx skill dir> node deck/ppt03/build_deck_v4.js
 * (node_modules: pptxgenjs@4.0.1 react react-dom react-icons sharp; the pptx skill provides scripts/apply_theme.js)
 */
'use strict';
const fs = require('fs');
const path = require('path');
const PptxGenJS = require('pptxgenjs');
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const sharp = require('sharp');
const fa = require('react-icons/fa6');

const HERE = __dirname;
const ISI = JSON.parse(fs.readFileSync(path.join(HERE, 'build', 'isi_v4.json'), 'utf8'));
const DATA = JSON.parse(fs.readFileSync(path.join(HERE, 'img', 'data_grafik.json'), 'utf8'));
const OUT = path.join(HERE, 'BISMILLAH_Final_v4.pptx');
const SKILL = process.env.PPTX_SKILL_DIR;
if (!SKILL || !fs.existsSync(path.join(SKILL, 'scripts', 'apply_theme.js'))) {
  throw new Error('set PPTX_SKILL_DIR to the pptx skill directory (it holds scripts/apply_theme.js)');
}
const { applyTheme } = require(path.join(SKILL, 'scripts', 'apply_theme.js'));

// ---- theme: the v2 palette (design system B4 of the structure document) ---------------------------------------
const THEME = {
  name: 'BISMILLAH Final v4',
  headFontFace: 'Calibri',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '1F2933', // ink
    lt1: 'FFFFFF',
    dk2: '16324F', // navy: structure
    lt2: 'EEF2F6', // soft card fill
    accent1: 'D97B29', // amber: gas, the key number
    accent2: '1F7A8C', // teal: electricity, validation
    accent3: '5B4A86', // violet: interpretation, cost
    accent4: 'A7B1BD', // grey: lanes without big models, Q4
    accent5: '5F6B7A', // muted text
    accent6: 'B83A3A', // red: the deliberately leaky control
    hlink: '1F7A8C',
    folHlink: '5B4A86',
  },
};
const HEX = THEME.colors;
const LINE_HEX = 'D5DBE3';
const MEDIAN_HEX = 'C3D0DE';

const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = 'Tim BISMILLAH';
pres.company = 'Tim BISMILLAH';
pres.title = 'DICE FEST 2026 Babak Final: Tim BISMILLAH';
pres.subject = 'Presentasi final v4 (top-down, 9 menit 30 detik)';

const C = pres.SchemeColor;
const INK = C.text1, NAVY = C.text2, WHITE = C.background1, SOFT = C.background2;
const AMBER = C.accent1, TEAL = C.accent2, VIOLET = C.accent3, GREY = C.accent4, MUTED = C.accent5, RED = C.accent6;
const SH = pres.ShapeType;

const X0 = 0.6, X1 = 12.733, W = X1 - X0; // content box
const Y0 = 1.85; // content top (below a two-line title)

const SECTIONS = ['I. Pembukaan', 'II. Rekayasa fitur', 'III. Arsitektur model', 'IV. Kebijakan publik',
  'V. Keterbatasan dan penutup', 'Cadangan tanya jawab'];
const KICKER = {
  I: 'I · RINGKASAN EKSEKUTIF', II: 'II · REKAYASA FITUR', III: 'III · ARSITEKTUR MODEL', IV: 'IV · KEBIJAKAN PUBLIK',
  V: 'V · KETERBATASAN DAN PENUTUP',
};

// ---- text helpers ------------------------------------------------------------------------------------------------
/** non-breaking spaces where a line break would split a figure from its unit or an equation */
function typo(t) {
  return t.replace(/(\d) (TWh|GWh|kWh|m²)/g, '$1\u00a0$2').replace(/ = /g, '\u00a0=\u00a0').replace(/≈ /g, '≈\u00a0')
    .replace(/\bk kawasan/g, 'k\u00a0kawasan').replace(/(\d) dari (\d)/g, '$1\u00a0dari\u00a0$2');
}

/** inline markup of the markdown (*italic*, **bold**, `code`, ^superscript^) -> pptxgenjs runs */
function runs(text, base = {}) {
  text = typo(text);
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\^[^^]+\^)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), options: { ...base } });
    const t = m[0];
    if (t.startsWith('**')) out.push({ text: t.slice(2, -2), options: { ...base, bold: true } });
    else if (t.startsWith('*')) out.push({ text: t.slice(1, -1), options: { ...base, italic: true } });
    else if (t.startsWith('^')) out.push({ text: t.slice(1, -1), options: { ...base, superscript: true } });
    else out.push({ text: t.slice(1, -1), options: { ...base } });
    last = m.index + t.length;
  }
  if (last < text.length) out.push({ text: text.slice(last), options: { ...base } });
  // "\n" inside a text is a soft line break (<a:br/>) within the same paragraph
  return out.flatMap((r) => r.text.split('\n').map((t, i) => ({ text: t,
    options: i ? { ...r.options, softBreakBefore: true } : r.options })));
}

/**
 * paragraphs [[text or runs, paragraph options], ...] -> one runs array with line breaks between paragraphs.
 * pptxgenjs starts a new paragraph whenever `align` changes from one run to the next, and at every run with `bullet`,
 * so `align` goes on every run of a paragraph and the other paragraph options only on its first run.
 */
function paras(list) {
  const out = [];
  list.forEach(([content, popt = {}], i) => {
    const r = typeof content === 'string' ? runs(content, popt.run || {}) : content;
    r.forEach((run, j) => {
      const o = { ...run.options };
      if (popt.align !== undefined) o.align = popt.align;
      if (j === 0) {
        for (const k of ['bullet', 'indentLevel', 'paraSpaceBefore', 'paraSpaceAfter']) {
          if (popt[k] !== undefined) o[k] = popt[k];
        }
      }
      if (j === r.length - 1 && i < list.length - 1) o.breakLine = true;
      out.push({ text: run.text, options: o });
    });
  });
  return out;
}

const plain = (t) => t.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1').replace(/`([^`]+)`/g, '$1')
  .replace(/\^([^^]+)\^/g, '$1');

/** text box; content is a markup string or runs */
function T(s, content, o) {
  const base = { fontSize: o.size || 16, color: o.color || INK, bold: !!o.bold, italic: !!o.italic };
  const r = typeof content === 'string' ? runs(content, {}) : content;
  s.addText(r, {
    x: o.x, y: o.y, w: o.w, h: o.h, isTextBox: true, ...base,
    align: o.align || 'left', valign: o.valign || 'top', margin: o.margin === undefined ? 0 : o.margin,
    paraSpaceAfter: o.after || 0, lineSpacingMultiple: o.lsp || undefined, charSpacing: o.spacing || undefined,
    objectName: o.name || undefined, fit: 'none', wrap: true,
  });
}

/** shape (no text) */
function R(s, o) {
  const opt = { x: o.x, y: o.y, w: o.w, h: o.h, objectName: o.name || undefined };
  opt.fill = o.fill ? { color: o.fill, transparency: o.alpha || 0 } : { type: 'none' };
  opt.line = o.line ? { color: o.line, width: o.lw || 1, dashType: o.dash || 'solid' } : { type: 'none' };
  if (o.r) opt.rectRadius = o.r;
  if (o.flipH) opt.flipH = true;
  if (o.flipV) opt.flipV = true;
  if (o.rotate) opt.rotate = o.rotate;
  s.addShape(o.shape || (o.r ? SH.roundRect : SH.rect), opt);
}

/** straight line or arrow from (x1, y1) to (x2, y2) */
function L(s, x1, y1, x2, y2, o = {}) {
  const line = { color: o.color || GREY, width: o.lw || 1.5, dashType: o.dash || 'solid' };
  if (o.arrow !== false) line.endArrowType = 'triangle';
  s.addShape(SH.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001),
    flipH: x2 < x1, flipV: y2 < y1, line, objectName: o.name || undefined,
  });
}

/** small label chip (evidence labels, chain labels) */
function chip(s, text, x, y, o = {}) {
  const size = o.size || 14;
  const w = o.w || 0.16 + text.length * size * 0.0092;
  const h = o.h || 0.34;
  R(s, { x, y, w, h, fill: o.fill || NAVY, alpha: o.alpha || 0, line: o.line, r: 0.06, name: `label ${text}` });
  T(s, text, { x, y, w, h, size, bold: true, color: o.color || WHITE, align: 'center', valign: 'middle', spacing: 0.5 });
  return w;
}

// ---- icons (react-icons rendered to PNG with sharp) ------------------------------------------------------------
const iconCache = {};
async function iconData(name, hex) {
  const key = `${name}-${hex}`;
  if (!iconCache[key]) {
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { size: 256 }))
      .replace(/currentColor/g, `#${hex}`);
    const png = await sharp(Buffer.from(svg)).resize(256, 256, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png().toBuffer();
    iconCache[key] = 'image/png;base64,' + png.toString('base64');
  }
  return iconCache[key];
}

async function icon(s, name, x, y, size, hex, alt) {
  s.addImage({ data: await iconData(name, hex), x, y, w: size, h: size, altText: alt || name });
}

/** icon in a filled circle (the deck's recurring motif) */
async function badge(s, name, cx, cy, d, fill, o = {}) {
  R(s, { shape: SH.ellipse, x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill, line: o.ring, lw: o.ringW || 2,
    name: o.name || `ikon ${name}` });
  const k = d * (o.scale || 0.5);
  await icon(s, name, cx - k / 2, cy - k / 2, k, o.hex || HEX.lt1, o.alt);
}

/** number in a filled circle */
function numberBadge(s, n, cx, cy, d, fill, color = WHITE) {
  R(s, { shape: SH.ellipse, x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill, name: `nomor ${n}` });
  T(s, String(n), { x: cx - d / 2, y: cy - d / 2, w: d, h: d, size: Math.round(d * 34), bold: true, color, align: 'center',
    valign: 'middle' });
}

// ---- native bar charts with text labels --------------------------------------------------------------------------
/**
 * One-series bar chart without axis text. o: frame {x, y, w, h}, plot {x, y, w, h} in inches inside the slide,
 * cats, vals, colors (hex per bar), max, dir 'bar' (horizontal, first category on top) or 'col', gap (%).
 * Returns the geometry of every bar so labels can be placed exactly.
 */
function bars(s, o) {
  const gap = o.gap === undefined ? 70 : o.gap;
  const layout = { x: (o.plot.x - o.x) / o.w, y: (o.plot.y - o.y) / o.h, w: o.plot.w / o.w, h: o.plot.h / o.h };
  s.addChart(pres.charts.BAR, [{ name: o.series || 'nilai', labels: o.cats, values: o.vals }], {
    x: o.x, y: o.y, w: o.w, h: o.h, barDir: o.dir || 'bar', barGapWidthPct: gap, chartColors: o.colors,
    catAxisOrientation: o.dir === 'col' ? 'minMax' : 'maxMin', catAxisHidden: true, valAxisHidden: true,
    valAxisMinVal: 0, valAxisMaxVal: o.max, valGridLine: { style: 'none' }, catGridLine: { style: 'none' },
    catAxisLineShow: false, valAxisLineShow: false, showLegend: false, showValue: false, showTitle: false,
    layout, objectName: o.name || undefined, altText: o.alt || undefined,
  });
  const n = o.cats.length;
  return o.vals.map((v, i) => {
    if (o.dir === 'col') {
      const band = o.plot.w / n, thick = band / (1 + gap / 100);
      const top = o.plot.y + o.plot.h * (1 - v / o.max);
      return { cx: o.plot.x + band * (i + 0.5), x0: o.plot.x + band * (i + 0.5) - thick / 2, thick, top,
        bottom: o.plot.y + o.plot.h, band };
    }
    const band = o.plot.h / n, thick = band / (1 + gap / 100);
    return { cy: o.plot.y + band * (i + 0.5), x0: o.plot.x, x1: o.plot.x + o.plot.w * v / o.max, thick, band };
  });
}

// ---- number formatting ------------------------------------------------------------------------------------------
function idn(x, nd, plus = false) {
  const s = Math.abs(x).toFixed(nd).replace('.', ',');
  const [a, b] = s.split(',');
  const int = a.replace(/\B(?=(\d{3})+(?!\d))/g, '.');
  return (x < 0 ? '−' : plus ? '+' : '') + int + (b !== undefined ? ',' + b : '');
}

// ---- content access with structure checks -----------------------------------------------------------------------
const MAIN = Object.fromEntries(ISI.utama.map((d) => [d.n, d]));
const BACK = Object.fromEntries(ISI.cadangan.map((d) => [d.id, d]));
/** the i-th bullet of a slide's on-screen text; optionally checks its label prefix */
function item(d, i, prefix) {
  const b = d.isi.filter((x) => x.jenis === 'butir')[i];
  if (!b) throw new Error(`slide ${d.n}: no bullet ${i}`);
  if (prefix && !b.teks.startsWith(prefix)) throw new Error(`slide ${d.n}, bullet ${i}: expected "${prefix}", got "${b.teks}"`);
  return prefix ? b.teks.slice(prefix.length).replace(/^\s*·\s*/, '') : b.teks;
}
function table(blocks, i = 0) {
  const t = blocks.filter((x) => x.jenis === 'tabel')[i];
  if (!t) throw new Error('missing table');
  return t.baris;
}

// ---- layouts ---------------------------------------------------------------------------------------------------------
const FOOTER = 'Tim BISMILLAH · DICE FEST 2026 · Babak final';
function defineLayouts() {
  for (const [title, th] of [['GELAP', 1.3], ['PENUTUP', 1.8]]) defineDark(title, th);
  for (const [title, footer] of [['ISI', FOOTER], ['CADANGAN', 'Cadangan tanya jawab · ' + FOOTER]]) {
    pres.defineSlideMaster({
      title,
      background: { color: WHITE },
      margin: [0.5, 0.6, 0.5, 0.6],
      objects: [
        { placeholder: { options: { name: 'kicker', type: 'body', x: X0, y: 0.28, w: W, h: 0.3, fontSize: 14, bold: true,
          color: TEAL, align: 'left', valign: 'top', margin: 0 }, text: '' } },
        { placeholder: { options: { name: 'title', type: 'title', x: X0, y: 0.6, w: W, h: 1.05,
          fontSize: title === 'ISI' ? 28 : 26, bold: true, color: NAVY, align: 'left', valign: 'top', margin: 0 }, text: '' } },
        { text: { text: footer, options: { x: X0, y: 7.0, w: 6.2, h: 0.3, fontSize: 12, color: MUTED, margin: 0,
          valign: 'top', isTextBox: true } } },
      ],
      slideNumber: { x: 12.233, y: 7.0, w: 0.5, h: 0.3, fontSize: 12, color: MUTED, align: 'right', margin: 0 },
    });
  }
}

/** dark layouts: GELAP (opening slide) and PENUTUP (closing slide, title of up to three lines) */
function defineDark(title, titleH) {
  pres.defineSlideMaster({
    title,
    background: { color: NAVY },
    margin: [0.5, 0.6, 0.5, 0.6],
    objects: [
      { placeholder: { options: { name: 'kicker', type: 'body', x: X0, y: 0.45, w: W, h: 0.34, fontSize: 14, bold: true,
        color: AMBER, align: 'left', valign: 'top', margin: 0 }, text: '' } },
      { placeholder: { options: { name: 'title', type: 'title', x: X0, y: 0.88, w: W, h: titleH, fontSize: 32, bold: true,
        color: WHITE, align: 'left', valign: 'top', margin: 0 }, text: '' } },
      { text: { text: FOOTER, options: { x: X0, y: 7.0, w: 6.2, h: 0.3, fontSize: 12, color: GREY, margin: 0,
        valign: 'top', isTextBox: true } } },
    ],
    slideNumber: { x: 12.233, y: 7.0, w: 0.5, h: 0.3, fontSize: 12, color: GREY, align: 'right', margin: 0 },
  });
}

let currentSection = -1;
function newSlide(master, sectionIdx, kicker, lines) {
  if (sectionIdx !== currentSection) {
    pres.addSection({ title: SECTIONS[sectionIdx] });
    currentSection = sectionIdx;
  }
  const s = pres.addSlide({ masterName: master, sectionTitle: SECTIONS[sectionIdx] });
  s.addText(kicker, { placeholder: 'kicker' });
  const t = [];
  lines.forEach((ln, i) => runs(ln).forEach((r, j) => t.push({ text: r.text,
    options: { ...r.options, ...(i > 0 && j === 0 ? { softBreakBefore: true } : {}) } })));
  s.addText(t, { placeholder: 'title' });
  return s;
}

function mainNotes(s, d) {
  const lines = [`[${d.mulai}–${d.selesai} · ${d.detik} detik · ${d.pembicara}]`, '', plain(d.narasi), '',
    `Sumber: ${plain(d.sumber)}`];
  s.addNotes(lines.join('\n'));
}

function backupNotes(s, b) {
  const lines = [`[Cadangan ${b.id} · penjawab: ${b.penjawab}]`, '', 'Pertanyaan juri:',
    ...b.pertanyaan.map((q) => `- "${q}"`), '', `Pedoman jawaban: ${plain(b.pedoman)}`, '', `Sumber: ${plain(b.sumber)}`];
  s.addNotes(lines.join('\n'));
}

// ======================================================================================================================
// v4 helpers: footer cells, quadrant glyph, line chart, implementer row
// ======================================================================================================================
const QPOS = { Q3: [0, 0], Q1: [1, 0], Q4: [0, 1], Q2: [1, 1] };   // [column, row] as on the quadrant map of slide 8
const QFILL = { Q1: AMBER, Q2: TEAL, Q3: NAVY, Q4: GREY };

/** notebook markers of a main slide (section numbers, execution numbers In [n]), in the footer left of the slide number */
function footerCells(s, d, dark = false) {
  T(s, `Notebook: ${d.sel}`, { x: 5.6, y: 7.0, w: 6.5, h: 0.3, size: 12, color: dark ? GREY : MUTED, align: 'right',
    name: 'sel notebook' });
}

/** 2 x 2 quadrant glyph laid out as the quadrant map (gas intensity across, electricity increase up) */
function quadGlyph(s, x, y, lit, dark = false, q = 0.42) {
  const g = 0.05;
  for (const [name, [c, r]] of Object.entries(QPOS)) {
    const on = lit.includes(name);
    const qx = x + c * (q + g), qy = y + r * (q + g);
    const inverse = on && dark && QFILL[name] === NAVY;   // a navy square would vanish on the navy card
    R(s, { x: qx, y: qy, w: q, h: q, fill: on ? (inverse ? WHITE : QFILL[name]) : WHITE, alpha: on ? 0 : (dark ? 85 : 0),
      line: on ? undefined : (dark ? undefined : LINE_HEX), lw: 0.75, r: 0.05, name: `kuadran ${name}${on ? ' (sasaran)' : ''}` });
    T(s, name, { x: qx, y: qy, w: q, h: q, size: 14, bold: true, color: inverse ? NAVY : on ? WHITE : (dark ? GREY : MUTED),
      align: 'center', valign: 'middle' });
  }
  return 2 * q + g;
}

/**
 * One-series native line chart without axis text (as bars()): o.frame {x, y, w, h}, o.plot inner plot area, o.vals,
 * o.min, o.max, o.color (hex). PowerPoint puts line points at the centre of each category band (crossBetween
 * "between"); returns those centres so labels can be placed exactly.
 */
function lineChart(s, o) {
  const f = o.frame, p = o.plot;
  const layout = { x: (p.x - f.x) / f.w, y: (p.y - f.y) / f.h, w: p.w / f.w, h: p.h / f.h };
  s.addChart(pres.charts.LINE, [{ name: o.series || 'skor', labels: o.cats, values: o.vals }], {
    x: f.x, y: f.y, w: f.w, h: f.h, chartColors: [o.color], lineSize: 2.5, lineDataSymbol: 'circle', lineDataSymbolSize: 10,
    lineDataSymbolLineColor: 'FFFFFF', lineDataSymbolLineSize: 1.5, catAxisHidden: true, valAxisHidden: true,
    valAxisMinVal: o.min, valAxisMaxVal: o.max, valGridLine: { style: 'none' }, catGridLine: { style: 'none' },
    catAxisLineShow: false, valAxisLineShow: false, showLegend: false, showValue: false, showTitle: false, layout,
    objectName: o.name || undefined, altText: o.alt || undefined,
  });
  const band = p.w / o.vals.length;
  const y = (v) => p.y + p.h * (1 - (v - o.min) / (o.max - o.min));
  return { pts: o.vals.map((v, i) => ({ cx: p.x + band * (i + 0.5), cy: y(v) })), band, y };
}

/** "Pelaksana:" line under the policy chain of slides 9 to 11 */
async function implementer(s, d, i, y) {
  const who = item(d, i, 'PELAKSANA');
  const x = 6.6;
  await badge(s, 'FaUserGroup', x + 0.24, y + 0.24, 0.48, NAVY, { name: 'ikon pelaksana' });
  T(s, [{ text: 'Pelaksana: ', options: { bold: true, color: NAVY } }, ...runs(who, {})],
    { x: x + 0.62, y, w: X1 - x - 0.62, h: 0.48, size: 15, valign: 'middle', name: 'pelaksana' });
}

// ======================================================================================================================
// Main slides
// ======================================================================================================================
async function slide1() {
  const d = MAIN[1];
  const s = newSlide('GELAP', 0, 'DICE FEST 2026 · BABAK FINAL · TIM BISMILLAH', d.judul_baris);
  // trilemma triangle, corners in the colours used again on slides 2 and 13
  const tri = { x: 1.8, y: 2.95, w: 3.7, h: 2.7 };
  R(s, { shape: SH.triangle, ...tri, fill: WHITE, alpha: 93, line: GREY, lw: 1.5, name: 'segitiga trilema' });
  const [emisi, jaringan, biaya] = item(d, 0).split(' · ');
  const corners = [
    { cx: tri.x + tri.w / 2, cy: tri.y, fill: AMBER, icon: 'FaLeaf', label: emisi, lx: tri.x + tri.w / 2 + 0.6, ly: tri.y - 0.22, lw: 2.4, la: 'left' },
    { cx: tri.x, cy: tri.y + tri.h, fill: VIOLET, icon: 'FaWallet', label: biaya, lx: tri.x - 1.2, ly: tri.y + tri.h + 0.55, lw: 2.4, la: 'center' },
    { cx: tri.x + tri.w, cy: tri.y + tri.h, fill: TEAL, icon: 'FaBolt', label: jaringan, lx: tri.x + tri.w - 1.4, ly: tri.y + tri.h + 0.55, lw: 2.8, la: 'center' },
  ];
  for (const c of corners) {
    await badge(s, c.icon, c.cx, c.cy, 0.9, c.fill, { ring: WHITE, ringW: 2, name: `sudut ${c.label}` });
    T(s, c.label, { x: c.lx, y: c.ly, w: c.lw, h: 0.44, size: 18, bold: true, color: WHITE, align: c.la, valign: 'middle' });
  }
  T(s, 'saling tarik', { x: tri.x + 0.8, y: tri.y + 1.75, w: tri.w - 1.6, h: 0.4, size: 15, italic: true, color: GREY,
    align: 'center', valign: 'middle' });

  // the number
  const [big, caption] = item(d, 1).split(': ');
  T(s, big, { x: 6.85, y: 2.55, w: 5.88, h: 1.35, size: 80, bold: true, color: AMBER, valign: 'middle' });
  T(s, caption, { x: 6.85, y: 3.95, w: 5.6, h: 1.0, size: 20, color: WHITE });
  chip(s, 'ASUMSI', 6.85, 5.2, { fill: WHITE, alpha: 85, color: WHITE });
  T(s, item(d, 2, 'ASUMSI'), { x: 6.85, y: 5.65, w: 5.6, h: 0.75, size: 15, color: GREY });
  footerCells(s, d, true);
  mainNotes(s, d);
}

async function slide2() {
  const d = MAIN[2];
  const s = newSlide('ISI', 0, KICKER.I, d.judul_baris);
  const recs = [0, 1, 2].map((i) => item(d, i).split(' · '));
  if (recs.some((r, i) => r.length !== 4 || r[0] !== String(i + 1))) throw new Error('slide 2: expected "n · text · tag · quadrant"');
  const top = 2.0, h = 4.0, gap = 0.3, cw = (W - 2 * gap) / 3;
  const quads = [['Q1'], ['Q1', 'Q3'], ['Q2']];          // the "Program" column of the slide 8 table
  const dots = [[AMBER, VIOLET], [TEAL], [AMBER, TEAL]];
  for (let i = 0; i < 3; i++) {
    const x = X0 + i * (cw + gap), dark = i === 1;
    const [, text, tag, quad] = recs[i];
    if (!quads[i].every((q) => quad.startsWith(q) || quad.includes(` ${q}`))) throw new Error(`slide 2: quadrant text ${quad}`);
    R(s, { x, y: top, w: cw, h, fill: dark ? NAVY : SOFT, r: 0.12, name: `kartu R${i + 1}` });
    numberBadge(s, i + 1, x + 0.55, top + 0.55, 0.62, dark ? WHITE : [AMBER, NAVY, TEAL][i], dark ? NAVY : WHITE);
    quadGlyph(s, x + cw - 1.2, top + 0.22, quads[i], dark);
    if (i === 0) {
      // "*Retrofit* selubung dulu di 118 zona Q1 (47% konsumsi gas)": accent on 118 and 47%
      const m = text.match(/^(.*) di (\d+) zona (Q1) \((\d+%) konsumsi gas\)$/);
      if (!m) throw new Error('slide 2: unexpected R1 text ' + text);
      T(s, m[1] + ' di zona ' + m[3], { x: x + 0.3, y: top + 1.2, w: cw - 0.5, h: 0.85, size: 20, bold: true });
      T(s, [{ text: `${m[2]} zona`, options: { fontSize: 40, bold: true, color: AMBER } }],
        { x: x + 0.3, y: top + 2.05, w: cw - 0.5, h: 0.75, valign: 'middle' });
      T(s, `${m[4]} konsumsi gas`, { x: x + 0.3, y: top + 2.8, w: cw - 0.5, h: 0.4, size: 18, bold: true, color: AMBER });
    } else {
      T(s, text, { x: x + 0.3, y: top + 1.2, w: cw - 0.55, h: 1.25, size: dark ? 22 : 20, bold: true, color: dark ? WHITE : INK });
      T(s, 'zona ' + quad, { x: x + 0.3, y: top + 2.5, w: cw - 0.55, h: 0.75, size: 15, italic: true, color: dark ? GREY : MUTED });
    }
    trilemmaTag(s, tag, dots[i], x + 0.3, top + h - 0.6, dark ? WHITE : INK);
  }
  const nxt = item(d, 3);
  const [lab, rest] = nxt.split(': ');
  T(s, [{ text: lab + ': ', options: { bold: true, color: NAVY } }, { text: rest, options: { color: MUTED } }],
    { x: X0, y: 6.3, w: W, h: 0.4, size: 15, valign: 'middle' });
  footerCells(s, d);
  mainNotes(s, d);
}

/** trilemma demand tag: coloured dots + text ("biaya dan emisi") */
function trilemmaTag(s, text, dots, x, y, color = INK) {
  dots.forEach((c, i) => R(s, { shape: SH.ellipse, x: x + i * 0.26, y: y + 0.07, w: 0.2, h: 0.2, fill: c,
    name: `titik trilema ${i + 1}` }));
  T(s, text, { x: x + dots.length * 0.26 + 0.08, y, w: 3.0, h: 0.34, size: 15, bold: true, color, valign: 'middle' });
}

async function slide3() {
  const d = MAIN[3];
  const s = newSlide('ISI', 1, KICKER.II, d.judul_baris);
  const cols = [{ x: X0, w: 3.25 }, { x: 4.1, w: 2.95 }, { x: 7.3, w: X1 - 7.3 }];
  const heads = [['DIBANGUN', NAVY], ['RISIKO DIKENDALIKAN', AMBER], ['KONTRIBUSI', TEAL]];
  const cws = heads.map(([t, c], i) => chip(s, t, cols[i].x, Y0, { fill: c }));
  L(s, cols[0].x + cws[0] + 0.12, Y0 + 0.17, cols[1].x - 0.12, Y0 + 0.17, { color: GREY });
  L(s, cols[1].x + cws[1] + 0.1, Y0 + 0.17, cols[2].x - 0.1, Y0 + 0.17, { color: GREY });

  // built: the formula and the two house sizes
  const built = item(d, 0, 'DIBANGUN'); // "`area_per_meter` = luas lantai ÷ jumlah meter ≈ luas satu rumah"
  const [lhs, rhs] = built.split(' = ');
  const [ratio, approx] = rhs.split(' ≈ ');
  R(s, { x: cols[0].x, y: 2.38, w: cols[0].w, h: 1.92, fill: SOFT, r: 0.1, name: 'kartu rumus' });
  T(s, paras([
    [runs(lhs, { bold: true, color: NAVY, fontSize: 20 })],
    [runs('= ' + ratio, { fontSize: 17 })],
    [runs('≈ ' + approx, { fontSize: 17, bold: true, color: AMBER })],
  ]), { x: cols[0].x + 0.2, y: 2.48, w: cols[0].w - 0.35, h: 1.35 });
  T(s, '(bld_dim_06 ÷ bld_dim_12)', { x: cols[0].x + 0.2, y: 3.88, w: cols[0].w - 0.35, h: 0.3, size: 14, color: MUTED });
  const hc = [cols[0].x + 0.8, cols[0].x + 2.45];   // centres of the small and the large house
  await icon(s, 'FaHouse', hc[0] - 0.275, 4.95, 0.55, HEX.dk2, 'rumah kecil');
  await icon(s, 'FaHouse', hc[1] - 0.5, 4.5, 1.0, HEX.accent5, 'rumah besar');
  [['±37 m²', NAVY], ['±173 m²', MUTED]].forEach(([t, c], i) => T(s, t, { x: hc[i] - 0.75, y: 5.6, w: 1.5, h: 0.32,
    size: 15, bold: true, color: c, align: 'center' }));
  T(s, 'pita 5% terkecil dan terbesar', { x: cols[0].x, y: 6.05, w: cols[0].w, h: 0.3, size: 14, color: MUTED,
    align: 'center' });

  // risk controlled: four checks
  const checks = item(d, 1, 'RISIKO DIKENDALIKAN').split(' · ');
  if (checks.length !== 4) throw new Error('slide 3: expected four checks');
  for (const [i, c] of checks.entries()) {
    const y = 2.45 + i * 1.0;
    await badge(s, 'FaCheck', cols[1].x + 0.27, y + 0.3, 0.5, TEAL);
    T(s, c, { x: cols[1].x + 0.68, y: y + 0.0, w: cols[1].w - 0.68, h: 0.62, size: 15, valign: 'middle' });
  }

  // contribution: share of the derived physical ratios, then the top five features of two explanation models
  const cx = cols[2].x, cwid = cols[2].w;
  const [share, rank] = item(d, 2, 'KONTRIBUSI').split(' · ');
  const [big, shareTxt] = share.split(': ');
  const r = Object.values(DATA.s3_rasio_fisik);
  if (big !== `${Math.min(...r)}–${Math.max(...r)}%`) throw new Error(`slide 3: ${big} differs from the SHAP shares ${r}`);
  const ranks = rank.match(/^peringkat 1: (\S+) fitur kedua \(EUI\) dan (\S+) \(listrik\)$/);
  if (!ranks) throw new Error('slide 3: unexpected rank text ' + rank);
  T(s, big, { x: cx, y: 2.3, w: 1.95, h: 0.58, size: 32, bold: true, color: AMBER, valign: 'middle' });
  T(s, shareTxt, { x: cx + 1.95, y: 2.3, w: cwid - 1.95, h: 0.58, size: 15, bold: true, color: NAVY, valign: 'middle' });
  const panels = [['log intensitas gas (EUI)', ranks[1], DATA.s3_shap_top.gas, 2.98],
    ['log listrik per meter', ranks[2], DATA.s3_shap_top.listrik, 4.66]];
  for (const [title, ratioTxt, rows, y] of panels) {
    T(s, [{ text: title, options: { bold: true, color: NAVY } },
      { text: ` · peringkat 1: ${ratioTxt}${y < 4 ? ' fitur kedua' : ''}`, options: {} }],
      { x: cx, y, w: cwid, h: 0.3, size: 14 });
    const frame = { x: cx, y: y + 0.32, w: cwid, h: 1.34 };
    const plot = { x: cx + 1.62, y: frame.y + 0.02, w: 2.75, h: 1.3 };
    const g = bars(s, { ...frame, plot, cats: rows.map((x) => x.fitur), vals: rows.map((x) => x.nilai), max: 0.1,
      colors: rows.map((x) => (x.fitur === 'area_per_meter' ? HEX.accent1 : x.turunan ? HEX.dk2 : HEX.accent4)), gap: 38,
      name: `grafik SHAP ${title}`, alt: `Rata-rata |SHAP| lima fitur teratas, ${title}; fitur turunan berwarna biru tua dan jingga` });
    g.forEach((b, i) => {
      T(s, rows[i].fitur, { x: cx, y: b.cy - 0.13, w: 1.55, h: 0.26, size: 14, bold: rows[i].turunan,
        color: rows[i].turunan ? NAVY : MUTED, align: 'right', valign: 'middle' });
      if (i === 0) T(s, idn(rows[i].nilai, 4), { x: b.x1 + 0.06, y: b.cy - 0.13, w: 0.8, h: 0.26, size: 14, bold: true,
        color: AMBER, valign: 'middle' });
    });
  }
  // legend: derived features vs raw columns
  const [derived, raw, caption] = item(d, 3).split(' · ');
  const ly = 6.36;
  R(s, { x: cx, y: ly + 0.05, w: 0.18, h: 0.18, fill: AMBER, name: 'legenda area_per_meter' });
  R(s, { x: cx + 0.22, y: ly + 0.05, w: 0.18, h: 0.18, fill: NAVY, name: 'legenda fitur turunan' });
  T(s, derived, { x: cx + 0.47, y: ly, w: 2.5, h: 0.28, size: 14, bold: true, color: NAVY, valign: 'middle' });
  R(s, { x: cx + 3.0, y: ly + 0.05, w: 0.18, h: 0.18, fill: GREY, name: 'legenda kolom asli' });
  T(s, raw, { x: cx + 3.25, y: ly, w: cwid - 3.25, h: 0.28, size: 14, color: MUTED, valign: 'middle' });
  T(s, caption, { x: cx, y: ly + 0.28, w: cwid, h: 0.28, size: 14, color: MUTED });
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide4() {
  const d = MAIN[4];
  const s = newSlide('ISI', 1, KICKER.II, d.judul_baris);
  const lw = 3.95, rx = 4.95, rw = X1 - rx;
  // left: the two text cards
  R(s, { x: X0, y: Y0, w: lw, h: 2.05, fill: SOFT, r: 0.1, name: 'kartu dibangun' });
  chip(s, 'DIBANGUN', X0 + 0.2, Y0 + 0.2, { fill: NAVY });
  T(s, item(d, 0, 'DIBANGUN'), { x: X0 + 0.2, y: Y0 + 0.66, w: lw - 0.4, h: 1.32, size: 16 });
  R(s, { x: X0, y: 4.1, w: lw, h: 2.65, fill: SOFT, r: 0.1, name: 'kartu risiko' });
  chip(s, 'RISIKO DIKENDALIKAN', X0 + 0.2, 4.3, { fill: AMBER });
  T(s, item(d, 2, 'RISIKO DIKENDALIKAN'), { x: X0 + 0.2, y: 4.78, w: lw - 0.4, h: 1.2, size: 16 });
  await badge(s, 'FaShieldHalved', X0 + 0.55, 6.27, 0.6, AMBER, { name: 'ikon pengaman' });

  // right top: subzone-number strip with three nested windows
  const n = 31, cw = 0.235, sx = rx + (rw - n * cw) / 2, sy = 2.85;
  for (let i = 0; i < n; i++) {
    const mid = i === 15;
    R(s, { x: sx + i * cw, y: sy, w: cw - 0.03, h: 0.38, fill: mid ? AMBER : NAVY, alpha: mid ? 0 : 82,
      name: mid ? 'kawasan dinilai' : `subzona ${i + 1}` });
  }
  T(s, 'kawasan dinilai', { x: sx + 15.5 * cw - 0.8, y: sy + 0.42, w: 1.6, h: 0.28, size: 14, bold: true, color: AMBER,
    align: 'center' });
  const wins = [['±30', 3, 2.5], ['±100', 8, 2.2], ['±300', 15, 1.9]];
  for (const [lab, k, y] of wins) {
    const xa = sx + (15 - k) * cw, xb = sx + (16 + k) * cw - 0.03;
    L(s, xa, y + 0.14, xb, y + 0.14, { color: TEAL, lw: 1.5, arrow: false });
    L(s, xa, y + 0.04, xa, y + 0.24, { color: TEAL, lw: 1.5, arrow: false });
    L(s, xb, y + 0.04, xb, y + 0.24, { color: TEAL, lw: 1.5, arrow: false });
    T(s, lab, { x: xa - 0.75, y, w: 0.7, h: 0.28, size: 14, bold: true, color: TEAL, align: 'right', valign: 'middle' });
  }
  T(s, 'nomor subzona →', { x: sx + n * cw - 1.7, y: sy + 0.42, w: 1.7, h: 0.28, size: 14, color: MUTED, align: 'right' });
  const [fact, use] = item(d, 1).split(' → '); // "96,5% pasangan ... sama", "nomor berdekatan = pendekatan lokasi"
  const pct = fact.split(' ')[0];
  T(s, [{ text: pct, options: { bold: true, color: TEAL } }, { text: fact.slice(pct.length), options: { breakLine: true } },
    ...runs('→ ' + use, { bold: true, color: NAVY })], { x: rx, y: 3.58, w: rw, h: 0.55, size: 14, valign: 'top', align: 'center' });

  // right bottom: two-layer cross-fit; only the scored part is labelled inside, the rest is one caption per row
  T(s, 'Lapis luar: 10 *fold*', { x: rx, y: 4.2, w: 4, h: 0.3, size: 14, bold: true, color: NAVY });
  const bw = (rw - 9 * 0.07) / 10;
  for (let i = 0; i < 10; i++) {
    const hit = i === 2;
    R(s, { x: rx + i * (bw + 0.07), y: 4.52, w: bw, h: 0.48, fill: hit ? AMBER : NAVY, alpha: hit ? 0 : 82, r: 0.05,
      name: hit ? 'fold dinilai' : `fold ${i + 1}` });
    if (hit) T(s, 'dinilai', { x: rx + i * (bw + 0.07), y: 4.52, w: bw, h: 0.48, size: 14, bold: true, color: WHITE,
      align: 'center', valign: 'middle' });
  }
  T(s, '9 *fold* lain: sumber tetangga', { x: rx, y: 5.03, w: rw, h: 0.28, size: 14, color: MUTED, align: 'right' });
  T(s, 'Lapis dalam: bagian latih dibagi 5', { x: rx, y: 5.33, w: 4.5, h: 0.3, size: 14, bold: true, color: NAVY });
  const pw = (rw - 4 * 0.07) / 5;
  for (let i = 0; i < 5; i++) {
    const hit = i === 0;
    R(s, { x: rx + i * (pw + 0.07), y: 5.65, w: pw, h: 0.44, fill: TEAL, alpha: hit ? 0 : 80, r: 0.05,
      name: hit ? 'bagian dihitung' : `bagian ${i + 1}` });
    if (hit) T(s, 'dihitung', { x: rx, y: 5.65, w: pw, h: 0.44, size: 14, bold: true, color: WHITE, align: 'center',
      valign: 'middle' });
  }
  T(s, '4 bagian lain: sumber tetangga', { x: rx, y: 6.12, w: rw, h: 0.28, size: 14, color: MUTED, align: 'right' });
  await icon(s, 'FaArrowRightLong', rx, 6.48, 0.3, HEX.accent5, 'panah');
  T(s, 'Data uji: tetangga dari bagian latih model yang sama', { x: rx + 0.42, y: 6.43, w: rw - 0.42, h: 0.4,
    size: 14, color: MUTED, valign: 'middle' });
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide5() {
  const d = MAIN[5];
  const s = newSlide('ISI', 1, KICKER.II, d.judul_baris);
  const rows = [
    { text: item(d, 0), icon: 'FaCheck', fill: TEAL, verdict: 'lulus' },
    { text: item(d, 1), icon: 'FaCheck', fill: TEAL, verdict: 'lulus' },
    { text: item(d, 2, 'KONTROL'), icon: 'FaTriangleExclamation', fill: RED, verdict: 'tertangkap', label: 'KONTROL' },
  ];
  for (const [i, r] of rows.entries()) {
    const y = Y0 + 0.05 + i * 1.04;
    R(s, { x: X0, y, w: W, h: 0.95, fill: i === 2 ? RED : SOFT, alpha: i === 2 ? 92 : 0, r: 0.08, name: `uji ${i + 1}` });
    await badge(s, r.icon, X0 + 0.5, y + 0.475, 0.6, r.fill);
    let tx = X0 + 1.0;
    if (r.label) {
      tx += chip(s, r.label, tx, y + 0.3, { fill: RED, h: 0.35 }) + 0.15;
    }
    T(s, r.text, { x: tx, y: y + 0.06, w: 10.55 - tx, h: 0.83, size: 17, valign: 'middle' });
    R(s, { x: 10.75, y: y + 0.24, w: 1.8, h: 0.47, fill: r.fill, r: 0.23, name: `hasil ${i + 1}` });
    T(s, r.verdict, { x: 10.75, y: y + 0.24, w: 1.8, h: 0.47, size: 15, bold: true, color: WHITE, align: 'center',
      valign: 'middle' });
  }
  const scope = item(d, 4); // "Cakupan: dua blok fitur, satu *fold*"
  R(s, { x: X0, y: 5.13, w: 4.4, h: 0.42, line: MUTED, lw: 1, r: 0.21, name: 'label cakupan' });
  T(s, scope, { x: X0, y: 5.13, w: 4.4, h: 0.42, size: 14, color: MUTED, align: 'center', valign: 'middle' });

  R(s, { x: X0, y: 5.8, w: W, h: 0.95, fill: TEAL, alpha: 88, r: 0.1, name: 'pita kontribusi' });
  const kw = chip(s, 'KONTRIBUSI', X0 + 0.25, 6.105, { fill: TEAL });
  T(s, item(d, 3, 'KONTRIBUSI').replace(/ (\S+)$/, '\u00a0$1').replace('peta zona', 'peta\u00a0zona'),
    { x: X0 + 0.45 + kw, y: 5.83, w: 10.4 - (X0 + 0.45 + kw), h: 0.89, size: 17, valign: 'middle' });
  // the feature ends up in the quadrant map
  L(s, 10.5, 6.27, 11.25, 6.27, { color: TEAL, lw: 2 });
  quadGlyph(s, 11.35, 5.83, ['Q1', 'Q2', 'Q3', 'Q4'], false, 0.42);
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide6() {
  const d = MAIN[6];
  const s = newSlide('ISI', 2, KICKER.III, d.judul_baris);
  const l1 = item(d, 0), l2 = item(d, 1), stages = item(d, 2), l3 = item(d, 3), total = item(d, 4), err = item(d, 5);
  const facts = item(d, 6, 'FAKTA DATA').split(' · ');

  // data box and the bus to the three lanes
  R(s, { x: X0, y: 2.45, w: 1.45, h: 1.45, fill: SOFT, line: NAVY, lw: 1, r: 0.08, name: 'data' });
  T(s, 'Data latih + uji, 201 kolom', { x: X0 + 0.08, y: 2.5, w: 1.29, h: 1.35, size: 14, bold: true, color: NAVY,
    align: 'center', valign: 'middle' });
  const bus = 2.25;
  L(s, X0 + 1.45, 3.18, bus, 3.18, { color: MUTED, lw: 1.5, arrow: false });
  L(s, bus, 2.3, bus, 4.42, { color: MUTED, lw: 1.5, arrow: false });
  for (const y of [2.3, 3.22, 4.42]) L(s, bus, y, 2.43, y, { color: MUTED, lw: 1.5 });

  // lanes 1 and 2 (blended)
  const lx = 2.45, lwid = 3.25;
  R(s, { x: lx, y: 1.85, w: lwid, h: 0.92, fill: AMBER, alpha: 85, line: AMBER, lw: 1, r: 0.08, name: 'jalur 1' });
  T(s, l1, { x: lx + 0.15, y: 1.88, w: lwid - 0.3, h: 0.86, size: 15, bold: true, valign: 'middle' });
  R(s, { x: lx, y: 2.92, w: lwid, h: 0.6, fill: TEAL, alpha: 85, line: TEAL, lw: 1, r: 0.08, name: 'jalur 2' });
  T(s, l2, { x: lx + 0.15, y: 2.92, w: lwid - 0.3, h: 0.6, size: 15, bold: true, valign: 'middle' });
  R(s, { shape: SH.rightBrace, x: lx + lwid + 0.06, y: 1.85, w: 0.2, h: 1.67, line: NAVY, lw: 1.5, name: 'kurung 41 model' });
  T(s, paras([['41 model', { align: 'center' }], ['di-*blend*', { align: 'center' }]]), { x: 5.95, y: 2.05, w: 1.15, h: 0.56,
    size: 14, bold: true, color: NAVY, valign: 'bottom' });
  L(s, 5.97, 2.68, 7.08, 2.68, { color: NAVY, lw: 2 });

  // post-processing of lanes 1 and 2
  const [, chainTxt] = stages.split(': ');
  const steps = chainTxt.split(' → ');
  R(s, { x: 7.12, y: 1.85, w: 2.75, h: 1.67, fill: NAVY, r: 0.08, name: 'tahap gabungan' });
  T(s, paras(steps.flatMap((t, i) => {
    const p = [[runs(t, { bold: true, color: WHITE, fontSize: 14 }), { align: 'center' }]];
    if (i < steps.length - 1) p.push([runs('↓', { color: GREY, fontSize: 14 }), { align: 'center' }]);
    return p;
  })), { x: 7.17, y: 1.88, w: 2.65, h: 1.61, valign: 'middle' });

  // lane 3: two targets without big models
  const p3 = l3.split(' · ');   // "Jalur 3 · Porsi bersih dan adopsi: tanpa model besar", "porsi bersih: rumus", ...
  if (p3.length !== 5 || p3[0] !== 'Jalur 3' || !p3[1].startsWith('Porsi bersih dan adopsi')) {
    throw new Error('slide 6: unexpected lane-3 text ' + l3);
  }
  const lgb = p3[4].match(/^(2 LightGBM terpisah) \(([^)]+)\)$/);
  if (!lgb) throw new Error('slide 6: unexpected LightGBM text ' + p3[4]);
  R(s, { x: lx, y: 3.7, w: 7.42, h: 1.47, fill: GREY, alpha: 80, line: GREY, lw: 1, r: 0.08, name: 'jalur 3' });
  T(s, `${p3[0]} · ${p3[1]}`, { x: lx + 0.15, y: 3.75, w: 7.1, h: 0.32, size: 15, bold: true });
  const iy = 4.12, ih = 0.95;
  R(s, { x: lx + 0.15, y: iy, w: 1.65, h: ih, fill: WHITE, r: 0.06, name: 'porsi bersih' });
  T(s, p3[2], { x: lx + 0.2, y: iy, w: 1.55, h: ih, size: 14, align: 'center', valign: 'middle' });
  R(s, { x: lx + 1.95, y: iy, w: 2.25, h: ih, fill: WHITE, r: 0.06, name: 'salin subzona' });
  T(s, p3[3], { x: lx + 2.0, y: iy, w: 2.15, h: ih, size: 14, align: 'center', valign: 'middle' });
  T(s, '+', { x: lx + 4.2, y: iy, w: 0.3, h: ih, size: 20, bold: true, color: MUTED, align: 'center', valign: 'middle' });
  R(s, { x: lx + 4.5, y: iy, w: 2.77, h: ih, fill: WHITE, line: RED, lw: 1.25, dash: 'dash', r: 0.06, name: '2 LightGBM terpisah' });
  T(s, paras([[runs(lgb[1], { bold: true, fontSize: 14, color: RED }), { align: 'center' }],
    [runs(lgb[2].replace('subzona baru', 'subzona\u00a0baru'), { fontSize: 14, color: MUTED }), { align: 'center' }]]),
    { x: lx + 4.55, y: iy, w: 2.67, h: ih, valign: 'middle' });

  // outputs with honest errors and the model count
  const [errHead, errVals] = err.split(': ');
  const vals = errVals.split(' · ');
  const ox = 10.12, ow = X1 - ox;
  T(s, errHead, { x: ox - 0.1, y: 1.5, w: ow + 0.1, h: 0.66, size: 14, color: MUTED, align: 'right', valign: 'bottom' });
  const outs = [['Gas', 2.2], ['EUI', 2.64], ['Listrik', 3.08], ['Porsi bersih', 3.82], ['Adopsi', 4.26]];
  outs.forEach(([name, y], i) => {
    R(s, { x: ox, y, w: ow, h: 0.38, fill: SOFT, r: 0.06, name: `keluaran ${name}` });
    T(s, name, { x: ox + 0.12, y, w: 1.45, h: 0.38, size: 14, valign: 'middle' });
    T(s, vals[i], { x: ox + 1.45, y, w: ow - 1.55, h: 0.38, size: 16, bold: true, color: i < 3 ? NAVY : MUTED, align: 'right',
      valign: 'middle' });
  });
  L(s, 9.87, 2.68, 10.08, 2.68, { color: NAVY, lw: 2 });
  L(s, 9.87, 4.25, 10.08, 4.25, { color: MUTED, lw: 2 });
  const m = total.match(/^(43 model) = (41) di-\*blend\* \+ (2) terpisah$/);
  if (!m) throw new Error('slide 6: unexpected total ' + total);
  T(s, [
    { text: m[1], options: { bold: true, color: NAVY, align: 'right' } }, { text: ' =', options: { color: MUTED, align: 'right', breakLine: true } },
    { text: m[2], options: { bold: true, color: AMBER, align: 'right' } }, { text: ' di-', options: { align: 'right' } },
    { text: 'blend', options: { italic: true, align: 'right' } }, { text: ' + ', options: { color: MUTED, align: 'right' } },
    { text: m[3], options: { bold: true, color: MUTED, align: 'right' } }, { text: ' terpisah', options: { align: 'right' } },
  ], { x: ox - 0.1, y: 4.7, w: ow + 0.1, h: 0.6, size: 15, align: 'right', valign: 'top', name: 'jumlah model' });

  // three facts of the data structure (deck v2, slide 3): which target needs a big model
  const fy = 5.38;
  chip(s, 'FAKTA DATA', X0, fy, { fill: NAVY });
  const fw = (W - 2 * 0.2) / 3, lanes = [AMBER, GREY, GREY];
  facts.forEach((f, i) => {
    const [head, proof] = f.split(': ');
    const x = X0 + i * (fw + 0.2), y = fy + 0.45;
    R(s, { x, y, w: fw, h: 0.98, fill: SOFT, r: 0.08, name: `fakta ${i + 1}` });
    R(s, { x, y: y + 0.12, w: 0.07, h: 0.74, fill: lanes[i], name: `penanda jalur fakta ${i + 1}` });
    T(s, paras([[runs(head, { bold: true, color: NAVY, fontSize: 16 })], [runs(proof, { fontSize: 14, color: INK })]]),
      { x: x + 0.22, y: y + 0.06, w: fw - 0.32, h: 0.86, valign: 'middle' });
  });
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide7() {
  const d = MAIN[7];
  const s = newSlide('ISI', 2, KICKER.III, d.judul_baris);
  const lb = DATA.s7_lb;
  const [caption, privTxt] = item(d, 0).split(' · privat ');   // "Skor *leaderboard* publik per langkah (...)", "0,24844 · peringkat 8"
  T(s, caption, { x: X0, y: 1.88, w: 7.95, h: 0.3, size: 14, bold: true, color: NAVY });
  const frame = { x: X0, y: 2.3, w: 7.95, h: 3.5 }, plot = { x: X0 + 0.05, y: 2.35, w: 7.85, h: 3.4 };
  const vals = lb.langkah.map((r) => r.skor);
  const ch = lineChart(s, { frame, plot, cats: lb.langkah.map((r) => r.label.join(' ')), vals, min: 0.248, max: 0.2504,
    color: HEX.dk2, name: 'grafik skor per langkah', alt: 'Skor leaderboard publik per langkah pengembangan, dari 0,25025 ke 0,24913, dan skor privat 0,24844' });
  ch.pts.forEach((p, i) => {
    const v = idn(vals[i], 5);
    if (i === 0) T(s, v, { x: p.cx + 0.1, y: p.cy - 0.33, w: 0.9, h: 0.28, size: 14, color: INK });
    else T(s, v, { x: p.cx - 0.5, y: p.cy + 0.1, w: 1.0, h: 0.28, size: 14, color: INK, align: 'center' });
    T(s, paras(lb.langkah[i].label.map((t) => [runs(t, {}), { align: 'center' }])), { x: p.cx - ch.band / 2, y: plot.y + plot.h + 0.1,
      w: ch.band, h: 0.5, size: 14, color: INK });
  });
  // private score at the last step
  const last = ch.pts[ch.pts.length - 1], py = ch.y(lb.privat);
  if (!privTxt.startsWith(idn(lb.privat, 5)) || !privTxt.endsWith(`peringkat ${lb.peringkat_privat}`)) {
    throw new Error('slide 7: private score text differs from the chart data');
  }
  L(s, last.cx, last.cy + 0.42, last.cx, py - 0.14, { color: AMBER, lw: 1.25, dash: 'dash', arrow: false, name: 'garis privat' });
  R(s, { shape: SH.ellipse, x: last.cx - 0.13, y: py - 0.13, w: 0.26, h: 0.26, fill: AMBER, line: WHITE, lw: 1.5, name: 'titik privat' });
  T(s, `privat ${privTxt}`, { x: last.cx - 3.35, y: py - 0.16, w: 3.1, h: 0.32, size: 14, bold: true, color: AMBER, align: 'right',
    valign: 'middle' });

  // two tiles on the right
  const tiles = [{ t: item(d, 1), icon: 'FaRankingStar', color: AMBER }, { t: item(d, 2), icon: 'FaScaleBalanced', color: TEAL }];
  const tx = 8.85, tw = X1 - tx;
  for (const [i, tile] of tiles.entries()) {
    const y = 1.88 + i * 2.5;
    const [head, ...rest] = tile.t.split(' · ');
    R(s, { x: tx, y, w: tw, h: 2.32, fill: SOFT, r: 0.12, name: `ubin ${i + 1}` });
    await badge(s, tile.icon, tx + 0.55, y + 0.55, 0.62, tile.color);
    T(s, head, { x: tx + 1.0, y: y + 0.15, w: tw - 1.1, h: 0.8, size: 44, bold: true, color: tile.color, valign: 'middle' });
    T(s, rest.join(' · '), { x: tx + 0.3, y: y + 1.05, w: tw - 0.5, h: 1.15, size: 16 });
  }
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide8() {
  const d = MAIN[8];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  s.addImage({ path: path.join(HERE, 'img', 'peta_kuadran.png'), x: X0, y: Y0, w: 6.55, h: 4.45,
    altText: 'Peta kuadran 330 zona: intensitas gas zona terhadap kenaikan konsumsi listrik tahunan, ukuran titik = konsumsi gas zona' });
  T(s, item(d, 0), { x: X0 + 0.05, y: 6.33, w: 6.5, h: 0.5,
    size: 14, color: MUTED });

  // quadrant table
  const rows = table(d.isi);
  const rx = 7.4, rw = X1 - rx;
  const colW = [0.86, 0.66, 0.78, 1.12, 1.0, 0.913];
  const head = rows[0].map((h) => ({ text: runs(h, { bold: true, color: HEX.lt1, fontSize: 14 }),
    options: { fill: { color: HEX.dk2 }, valign: 'middle', align: 'left' } }));
  const body = rows.slice(1).map((r) => r.map((c, j) => ({
    text: runs(c, { bold: j === 0 || j === 5 || r[0] === 'Q1', color: r[0] === 'Q1' ? HEX.accent1 : HEX.dk1, fontSize: 14 }),
    options: { fill: { color: r[0] === 'Q1' ? 'FBEEDF' : HEX.lt1 }, valign: 'middle', align: j === 0 || j === 5 ? 'left' : 'right' },
  })));
  s.addTable([head, ...body], { x: rx, y: Y0, w: rw, colW, rowH: [0.66, 0.38, 0.38, 0.38, 0.38], margin: [3, 5, 3, 5],
    border: { type: 'solid', pt: 0.75, color: LINE_HEX }, fontSize: 14, objectName: 'tabel kuadran' });

  // assumption and robustness boxes
  const boxes = [[item(d, 1, 'ASUMSI'), 'ASUMSI', VIOLET, 4.28], [item(d, 2, 'UJI KETAHANAN'), 'UJI KETAHANAN', TEAL, 5.56]];
  for (const [txt, lab, c, y] of boxes) {
    R(s, { x: rx, y, w: rw, h: 1.18, fill: c, alpha: 90, r: 0.08, name: `kotak ${lab}` });
    chip(s, lab, rx + 0.15, y + 0.12, { fill: c, h: 0.32 });
    T(s, txt, { x: rx + 0.15, y: y + 0.5, w: rw - 0.3, h: 0.64, size: 14 });
  }
  footerCells(s, d);
  mainNotes(s, d);
}

/** the policy chain TEMUAN -> INTERPRETASI -> REKOMENDASI -> LANGKAH on the right of slides 9 to 11 */
function chain(s, d, opts = {}) {
  const labels = [['TEMUAN', NAVY], ['INTERPRETASI', VIOLET], ['REKOMENDASI', AMBER], ['LANGKAH', TEAL]];
  // box heights follow the number of 14 pt text lines (0.22 in + lines x 0.235 in); the space left in the column
  // (Y0 to the implementer line) is shared out as equal gaps, each with a small arrow in its middle
  const x = 6.6, w = X1 - x, hs = opts.heights, bottom = 6.2;
  const gap = (bottom - Y0 - hs.reduce((a, b) => a + b, 0)) / 3;
  if (gap < 0.16) throw new Error(`slide ${d.n}: chain too tall (gap ${gap.toFixed(2)} in)`);
  let y = Y0;
  labels.forEach(([lab, c], i) => {
    const h = hs[i];
    const key = lab === 'REKOMENDASI';
    R(s, { x, y, w, h, fill: c, alpha: key ? 84 : 91, r: 0.08, name: `rantai ${lab}` });
    chip(s, lab, x + 0.13, y + 0.11, { fill: c, w: 1.72, h: 0.34 });
    const txt = opts.breaks && opts.breaks[i] ? item(d, i, lab).replace(opts.breaks[i], '\n' + opts.breaks[i].trimStart())
      : item(d, i, lab);
    T(s, txt, { x: x + 1.98, y: y + 0.1, w: w - 2.08, h: (opts.textH && opts.textH[i]) || h - 0.16, size: 14,
      bold: key, valign: 'top' });
    if (opts.extra && opts.extra[i]) opts.extra[i](x + 1.98, y, w - 2.08, h);
    if (i < 3) R(s, { shape: SH.triangle, x: x + 0.84, y: y + h + (gap - 0.12) / 2, w: 0.3, h: 0.12, fill: GREY, flipV: true,
      name: 'panah rantai' });
    y += h + gap;
  });
}

async function slide9() {
  const d = MAIN[9];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  s.addImage({ path: path.join(HERE, 'img', 'dep_gas_ukuran.png'), x: X0, y: Y0, w: 5.7, h: 3.05,
    altText: 'Dependensi SHAP luas lantai per meter terhadap gas per m2, model penjelas' });
  // retrofit indicator of the three candidate zones (native chart)
  const rows = DATA.s9_retrofit;
  T(s, 'Indikator *retrofit* tiga zona kandidat (GWh; % gas zona)', { x: X0, y: 5.0, w: 5.7, h: 0.3, size: 14, bold: true,
    color: NAVY });
  const frame = { x: X0, y: 5.34, w: 5.7, h: 1.4 };
  const plot = { x: X0 + 1.02, y: frame.y + 0.03, w: 2.2, h: 1.34 };
  const g = bars(s, { ...frame, plot, cats: rows.map((r) => r.zona), vals: rows.map((r) => r.gwh), max: 180,
    colors: rows.map(() => HEX.accent1), gap: 45, name: 'grafik indikator retrofit',
    alt: 'Indikator potensi retrofit tiga zona kandidat, GWh' });
  g.forEach((b, i) => {
    T(s, rows[i].zona, { x: X0, y: b.cy - 0.14, w: 0.96, h: 0.28, size: 14, align: 'right', valign: 'middle' });
    T(s, `≈${idn(rows[i].gwh, 0)} GWh (${idn(rows[i].pct, 2)}%)`, { x: b.x1 + 0.08, y: b.cy - 0.14, w: 2.45,
      h: 0.28, size: 14, color: MUTED, valign: 'middle' });
  });
  chain(s, d, { heights: [0.93, 0.93, 0.93, 0.93], breaks: [null, ' asosiasi', null, null] });
  await implementer(s, d, 4, 6.3);
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide10() {
  const d = MAIN[10];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  // SHAP basis of the recommendation: electricity per house against house size (deck v2, slide 8, redrawn)
  s.addImage({ path: path.join(HERE, 'img', 'dep_listrik_ukuran.png'), x: X0, y: Y0, w: 5.7, h: 2.75,
    altText: 'Dependensi SHAP luas lantai per meter terhadap listrik per meter, model penjelas: rumah besar sekitar +75% listrik per rumah' });
  const k = DATA.s10_kenaikan;
  T(s, [{ text: 'Kenaikan listrik tahunan, lima zona tertinggi (COP 3)', options: { bold: true, color: NAVY, breakLine: true } },
    { text: 'jingga = titik awal', options: { color: MUTED } }], { x: X0, y: 4.66, w: 4.35, h: 0.52, size: 14 });
  T(s, 'peringkat COP per iklim', { x: 4.95, y: 4.66, w: 1.35, h: 0.52, size: 14, color: MUTED, align: 'center' });
  const cats = [...k.lima.map((r) => r.zona), 'median 330 zona'];
  const vals = [...k.lima.map((r) => r.pct), k.median];
  const start = new Set(k.sepuluh_besar_dua_ukuran_dua_skenario);
  const frame = { x: X0, y: 5.2, w: 5.7, h: 1.62 };
  const plot = { x: X0 + 1.5, y: frame.y + 0.02, w: 1.85, h: 1.58 };
  const colors = [...k.lima.map((r) => (start.has(r.zona) ? HEX.accent1 : HEX.accent4)), MEDIAN_HEX];
  const g = bars(s, { ...frame, plot, cats, vals, max: 125, colors, gap: 35, name: 'grafik kenaikan tertinggi',
    alt: 'Kenaikan konsumsi listrik tahunan lima zona tertinggi dan median 330 zona' });
  g.forEach((b, i) => {
    const isMed = i === cats.length - 1;
    T(s, cats[i], { x: X0 - 0.05, y: b.cy - 0.13, w: 1.5, h: 0.26, size: 14, color: isMed ? MUTED : INK,
      align: 'right', valign: 'middle', bold: !isMed && start.has(cats[i]) });
    T(s, idn(vals[i], 2, true) + '%', { x: b.x1 + 0.05, y: b.cy - 0.13, w: 1.05, h: 0.26, size: 14, bold: !isMed,
      color: isMed ? MUTED : INK, valign: 'middle' });
    if (!isMed) T(s, String(k.lima[i].peringkat_cop_iklim), { x: 4.95, y: b.cy - 0.13, w: 1.35, h: 0.26, size: 14,
      bold: true, color: NAVY, align: 'center', valign: 'middle' });
  });
  chain(s, d, { heights: [0.93, 0.7, 0.7, 0.93] });
  await implementer(s, d, 4, 6.3);
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide11() {
  const d = MAIN[11];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  // main visual: SHAP of the explanation model for adoption against the gas-network share (deck v2, slide 12)
  s.addImage({ path: path.join(HERE, 'img', 'dep_adopsi_besar.png'), x: X0, y: Y0, w: 5.75, h: 4.75,
    altText: 'Dependensi SHAP adopsi terhadap porsi gas jaringan, model penjelas: positif di bawah 40%, negatif di atas 80%' });
  const reg = 'REG_033: gas jaringan 83,21%, adopsi 4,99%';
  chain(s, d, { heights: [0.93, 0.7, 1.0, 0.93], textH: [null, null, 0.58, null], extra: [null, null, (x, y, w, h) => {
    T(s, reg, { x, y: y + h - 0.38, w, h: 0.3, size: 14, color: MUTED, valign: 'middle' });
  }, null] });
  await implementer(s, d, 4, 6.3);
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide12() {
  const d = MAIN[12];
  const s = newSlide('ISI', 4, KICKER.V, d.judul_baris);
  const icons = ['FaGaugeHigh', 'FaCircleQuestion', 'FaFlask'];
  const tw = 3.83, gap = (W - 3 * tw) / 2;
  for (let i = 0; i < 3; i++) {
    const [unknown, handling] = item(d, i).split(' → ');
    const x = X0 + i * (tw + gap), y = 2.0;
    R(s, { x, y, w: tw, h: 2.3, fill: SOFT, r: 0.1, name: `belum diketahui ${i + 1}` });
    await badge(s, icons[i], x + 0.55, y + 0.55, 0.62, MUTED);
    chip(s, 'BELUM DIKETAHUI', x + 1.0, y + 0.38, { fill: MUTED });
    T(s, unknown, { x: x + 0.3, y: y + 1.05, w: tw - 0.6, h: 1.15, size: 17 });
    R(s, { shape: SH.downArrow, x: x + tw / 2 - 0.22, y: y + 2.4, w: 0.44, h: 0.45, fill: GREY, name: 'panah penanganan' });
    R(s, { x, y: y + 2.95, w: tw, h: 1.6, fill: TEAL, alpha: 88, r: 0.1, name: `penanganan ${i + 1}` });
    chip(s, 'PENANGANAN', x + 0.3, y + 3.12, { fill: TEAL });
    T(s, handling, { x: x + 0.3, y: y + 3.56, w: tw - 0.6, h: 0.9, size: 18, bold: true, color: NAVY });
  }
  footerCells(s, d);
  mainNotes(s, d);
}

async function slide13() {
  const d = MAIN[13];
  const s = newSlide('PENUTUP', 4, 'PENUTUP', d.judul_baris);
  const recap = item(d, 0).split(' · ').map((t) => t.match(/^(\d) (.+): (.+)$/));
  if (recap.some((m) => !m)) throw new Error('slide 13: unexpected recap ' + item(d, 0));
  const colors = [AMBER, WHITE, TEAL];
  const dots = [[AMBER, VIOLET], [TEAL], [AMBER, TEAL]];
  const quads = [['Q1'], ['Q1', 'Q3'], ['Q2']];
  const tw = 3.83, gap = (W - 3 * tw) / 2, y = 2.82;
  recap.forEach((m, i) => {
    const x = X0 + i * (tw + gap);
    if (!quads[i].every((q) => m[2].includes(q))) throw new Error(`slide 13: card ${i + 1} lacks ${quads[i]}`);
    R(s, { x, y, w: tw, h: 1.55, fill: WHITE, alpha: 90, r: 0.12, name: `kartu penutup ${i + 1}` });
    numberBadge(s, m[1], x + 0.5, y + 0.5, 0.56, colors[i], i === 1 ? NAVY : WHITE);
    T(s, runs(m[2], {}), { x: x + 0.92, y: y + 0.18, w: tw - 1.02, h: 0.66, size: 20, bold: true, color: WHITE, valign: 'middle' });
    dots[i].forEach((c, j) => R(s, { shape: SH.ellipse, x: x + 0.35 + j * 0.26, y: y + 1.07, w: 0.2, h: 0.2, fill: c,
      line: WHITE, lw: 1, name: 'titik trilema' }));
    T(s, m[3], { x: x + 0.4 + dots[i].length * 0.26, y: y + 0.95, w: 3.0, h: 0.44, size: 16, color: GREY, valign: 'middle' });
  });
  // next steps, one under each recommendation card
  const [lab, rest] = item(d, 1).split(': ');
  const steps = rest.split(' · ');
  if (steps.length !== 3) throw new Error('slide 13: expected three next steps');
  T(s, lab, { x: X0, y: 4.62, w: 4, h: 0.4, size: 16, bold: true, color: AMBER });
  steps.forEach((t, i) => {
    const x = X0 + i * (tw + gap);
    R(s, { x, y: 5.08, w: tw, h: 0.85, line: GREY, lw: 1, r: 0.1, name: `langkah berikutnya R${i + 1}` });
    T(s, t, { x: x + 0.2, y: 5.08, w: tw - 0.4, h: 0.85, size: 16, color: WHITE, valign: 'middle' });
  });
  T(s, item(d, 2), { x: X0, y: 6.2, w: 6, h: 0.6, size: 28, bold: true, color: WHITE, valign: 'middle' });
  footerCells(s, d, true);
  mainNotes(s, d);
}

// ======================================================================================================================
// Backup slides (C1 to C11): evidence on screen at 14 pt or more; the jury questions and the answer guide are in the
// notes only (v3 also printed the questions on the slide)
// ======================================================================================================================
function backupSlide(id) {
  const b = BACK[id];
  const s = newSlide('CADANGAN', 5, `CADANGAN · ${id}`, b.judul_baris);
  backupNotes(s, b);
  return [s, b];
}

/** native table from a markdown table; o: x, y, w, colW, size, head (hex), align per column, rowH, highlight(row) */
function mdTable(s, rows, o) {
  const size = o.size || 14;
  const head = rows[0].map((h, j) => ({ text: runs(h, { bold: true, color: HEX.lt1, fontSize: size }),
    options: { fill: { color: o.head || HEX.dk2 }, valign: 'middle', align: (o.align && o.align[j]) || 'left' } }));
  const body = rows.slice(1).map((r, i) => r.map((c, j) => {
    const hi = o.highlight && o.highlight(r, i);
    return {
      text: runs(c, { fontSize: size, color: hi && hi.color ? hi.color : HEX.dk1, bold: !!(hi && hi.bold) || (o.boldCol === j) }),
      options: { fill: { color: hi && hi.fill ? hi.fill : (i % 2 ? 'F6F8FA' : HEX.lt1) }, valign: 'middle',
        align: (o.align && o.align[j]) || 'left' },
    };
  }));
  s.addTable([head, ...body], { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, margin: o.margin || [3, 6, 3, 6],
    border: { type: 'solid', pt: 0.75, color: LINE_HEX }, fontSize: size, objectName: o.name || 'tabel' });
}

/** bullet list (markup strings) in a text box */
function bullets(s, list, o) {
  T(s, paras(list.map((t) => [runs(t, { fontSize: o.size || 14, color: o.color || INK }),
    { bullet: { indent: 15 }, paraSpaceAfter: o.after === undefined ? 6 : o.after }])), o);
}

/** numbers on a backup slide must come from its markdown section: build-time guard */
function fromDoc(b, ...texts) {
  const doc = JSON.stringify(b);
  for (const t of texts) {
    for (const n of plain(t).match(/\d+(?:[.,]\d+)*/g) || []) {
      if (!doc.includes(n)) throw new Error(`${b.id}: number ${n} not in the markdown section`);
    }
  }
  return texts;
}

const subhead = (s, text, x, y, w, h = 0.3) => T(s, text, { x, y, w, h, size: 14, bold: true, color: NAVY });

async function c1() {
  const [s, b] = backupSlide('C1');
  bullets(s, fromDoc(b,
    'Bobot berubah halus menurut peringkat luas (A), jumlah meter (M), dan prediksi global (p)',
    '2 × 2 × 2 = 8 bobot pojok per model, NNLS (bobot tak negatif)',
    '224 bobot gas, 224 EUI, 104 listrik untuk 429.303 baris'), { x: X0, y: Y0, w: 6.1, h: 1.95, size: 15 });
  subhead(s, 'Skor jujur: bobot di-*fit* ulang per *fold* luar (RMSE/σ)', X0, 3.95, 6.1);
  mdTable(s, table(b.bukti), { x: X0, y: 4.3, w: 6.1, colW: [1.6, 1.5, 1.5, 1.5], size: 15, rowH: 0.42,
    align: ['left', 'right', 'right', 'right'], name: 'tabel skor jujur', highlight: (r) => (r[0] === 'EUI' ? { bold: true } : null) });
  s.addImage({ path: path.join(HERE, 'img', 'basis_pojok.png'), x: 7.0, y: Y0, w: 5.75, h: 3.0,
    altText: 'Basis dua simpul bobot adaptif' });
  bullets(s, fromDoc(b, 'Rata-rata lima target −0,000125; tidak ada kolom memburuk',
    '*Leaderboard* publik 0,24934 menjadi 0,24925; tiga simpul tidak lebih baik'), { x: 7.0, y: 5.05, w: 5.73,
    h: 1.5, size: 15 });
}

async function c2() {
  const [s, b] = backupSlide('C2');
  const steps = fromDoc(b, 'Galat *out-of-fold* gabungan (*notebook*: koreksi galat wilayah)',
    'Dirata-rata per zona, petak iklim, ±30 subzona, kawasan termirip',
    'Dikecilkan n/(n + 30), atau n/(n + 5) bila tetangga sedikit',
    'Dikali λ dari *holdout set*, dibekukan di resid_knn.json');
  steps.forEach((t, i) => {
    const y = Y0 + i * 0.94;
    R(s, { x: X0, y, w: 6.0, h: 0.8, fill: i === 3 ? NAVY : SOFT, r: 0.08, name: `langkah koreksi ${i + 1}` });
    numberBadge(s, i + 1, X0 + 0.38, y + 0.4, 0.44, i === 3 ? WHITE : TEAL, i === 3 ? NAVY : WHITE);
    T(s, t, { x: X0 + 0.75, y: y + 0.03, w: 5.12, h: 0.74, size: 14, color: i === 3 ? WHITE : INK, valign: 'middle' });
    if (i < 3) R(s, { shape: SH.triangle, x: X0 + 0.25, y: y + 0.83, w: 0.26, h: 0.1, fill: GREY, flipV: true, name: 'panah' });
  });
  subhead(s, 'λ untuk gas (W = jendela, k = tetangga)', 6.95, Y0, 5.78);
  mdTable(s, table(b.bukti), { x: 6.95, y: 2.22, w: 5.78, colW: [4.58, 1.2], size: 14, rowH: 0.36, align: ['left', 'right'],
    name: 'tabel lambda' });
  bullets(s, fromDoc(b, 'Koreksi rms/σ: gas 0,02047; EUI 0,04344; listrik 0,01195'), { x: X0, y: 5.65, w: 6.0, h: 0.6,
    size: 14, after: 4 });
}

async function c3() {
  const [s, b] = backupSlide('C3');
  const rows = table(b.bukti);
  mdTable(s, rows, { x: X0, y: 2.0, w: 7.75, colW: [3.4, 0.85, 1.25, 2.25], size: 14,
    rowH: rows.map((r, i) => (i === 0 ? 0.58 : r[3].length > 12 || r[0].length > 34 ? 0.62 : 0.42)),
    align: ['left', 'right', 'right', 'left'], name: 'tabel uji label acak',
    highlight: (r) => (r[0].startsWith('KONTROL') ? { fill: 'F6E1E1', bold: true } : null) });
  bullets(s, fromDoc(b, 'Rasio fisik: per baris, tanpa target',
    'Agregat kawasan: fitur latih dan uji, tanpa target',
    'Target wilayah dan kawasan termirip:\n*out-of-fold* dua lapis',
    'Belum diuji label acak: tetangga model adopsi, koreksi galat, bobot gabungan, penyesuaian akhir'),
  { x: 8.6, y: 2.0, w: 4.13, h: 4.45, size: 15, after: 10 });
}

async function c4() {
  const [s, b] = backupSlide('C4');
  mdTable(s, table(b.bukti), { x: X0, y: Y0, w: 6.4, colW: [3.8, 1.3, 1.3], size: 15, rowH: 0.44,
    align: ['left', 'right', 'right'], name: 'tabel keluarga model',
    highlight: (r) => (r[0] === 'Jumlah' ? { bold: true, fill: 'E3EAF2' } : null) });
  bullets(s, fromDoc(b, 'Jalur 3: 2 LightGBM adopsi (q_xent, r_reg) untuk 8,14% kawasan uji, tidak ikut *blending*',
    '*Notebook* melatih 42 model di jalur 1 dan 2; mk1_gpm tidak terpilih',
    '41 per keluaran: MLP mk2 menyumbang ke dua jalur',
    '550 model terlatih di balik 41 anggota (per *fold*, sebagian tiga *seed*)'), { x: 7.35, y: Y0, w: 5.38, h: 3.45, size: 15 });
  R(s, { x: 7.35, y: 5.45, w: 5.38, h: 0.8, fill: SOFT, r: 0.1, name: 'pita 43' });
  T(s, [{ text: '43', options: { bold: true, color: NAVY } }, { text: ' = ', options: { color: MUTED } },
    { text: '41', options: { bold: true, color: AMBER } }, { text: ' + ', options: { color: MUTED } },
    { text: '2', options: { bold: true, color: MUTED } }], { x: 7.35, y: 5.45, w: 5.38, h: 0.8, size: 26, align: 'center',
    valign: 'middle' });
}

async function c5() {
  const [s, b] = backupSlide('C5');
  const r2 = fromDoc(b, 'log EUI|0,563', 'log gas per meter|0,747', 'log listrik per meter|0,491', 'adopsi|0,991')
    .map((t) => t.split('|'));
  subhead(s, 'R² model penjelas (data uji internal)', X0, Y0, 4.4, 0.55);
  mdTable(s, [['Model penjelas', 'R²'], ...r2], { x: X0, y: 2.48, w: 4.4, colW: [3.1, 1.3], size: 15, rowH: 0.42,
    align: ['left', 'right'], name: 'tabel R2' });
  const fam = DATA.c5_kelompok;
  subhead(s, 'Porsi SHAP per kelompok fitur (*writeup* B Tabel 3)', 5.3, Y0, 7.43, 0.55);
  const short = { 'EUI (intensitas gas)': 'EUI', 'Gas per meter': 'Gas per meter', 'Listrik per meter': 'Listrik per meter',
    'Adopsi pemanas listrik': 'Adopsi' };
  const famRows = [['Target', ...fam.kelompok], ...fam.target.map((t) => [short[t.nama] || t.nama, ...t.porsi.map((p) => `${p}%`)])];
  mdTable(s, famRows, { x: 5.3, y: 2.48, w: 7.43, colW: [1.3, 1.47, 0.95, 1.05, 1.05, 0.6, 1.01], size: 14,
    rowH: [0.64, 0.42, 0.56, 0.56, 0.42], align: ['left', 'right', 'right', 'right', 'right', 'right', 'right'],
    name: 'tabel porsi kelompok', highlight: (r) => (r[0] === 'Adopsi' ? { color: HEX.accent3, bold: true } : null) });
  T(s, 'Sisa porsi tiap baris: kelompok lokasi dan agregat kawasan', { x: 5.3, y: 5.12, w: 7.43, h: 0.3, size: 14, color: MUTED });
  R(s, { x: X0, y: 5.55, w: W, h: 1.1, fill: VIOLET, alpha: 90, r: 0.08, name: 'kotak penegas' });
  T(s, 'SHAP = sumbangan pada prediksi model penjelas, bukan sebab-akibat',
    { x: X0 + 0.25, y: 5.6, w: W - 0.5, h: 1.0, size: 17, color: INK, valign: 'middle' });
}

async function c6() {
  const [s, b] = backupSlide('C6');
  subhead(s, 'Celah data', X0, Y0, 3);
  const gaps = ['beban puncak (kW)', 'kapasitas gardu', 'COP lokal', 'koordinat geografis', 'biaya *retrofit*', 'tarif',
    'intensitas karbon listrik'];
  let gx = X0, gy = Y0 + 0.38;
  for (const g of gaps) {
    const w = 0.3 + plain(g).length * 0.098;
    if (gx + w > X1) { gx = X0; gy += 0.48; }
    R(s, { x: gx, y: gy, w, h: 0.4, fill: SOFT, line: GREY, lw: 0.75, r: 0.2, name: `celah ${plain(g)}` });
    T(s, g, { x: gx, y: gy, w, h: 0.4, size: 14, align: 'center', valign: 'middle' });
    gx += w + 0.12;
  }
  subhead(s, 'Rencana validasi', X0, gy + 0.65, 3);
  mdTable(s, table(b.bukti), { x: X0, y: gy + 1.0, w: W, colW: [2.05, 2.95, 3.85, 3.283], size: 14,
    rowH: [0.42, 0.66, 0.66, 0.66], name: 'tabel rencana validasi' });
}

async function c7() {
  const [s, b] = backupSlide('C7');
  subhead(s, '*Writeup* B Tabel 1: LightGBM, RMSE/σ', X0, Y0, 7);
  mdTable(s, table(b.bukti), { x: X0, y: 2.22, w: 7.2, colW: [1.7, 1.75, 1.25, 1.25, 1.25], size: 16, rowH: [0.68, 0.55, 0.55, 0.55, 0.55],
    align: ['left', 'left', 'right', 'right', 'right'], name: 'tabel skema validasi',
    highlight: (r) => (r[0] === 'per zona' && r[1] === 'dengan' ? { color: HEX.accent6, bold: true } : null) });
  bullets(s, fromDoc(b, 'Pemisahan data: AUC adversarial 0,4997; kawasan uji di subzona baru 8,14% vs harapan 8,17%',
    'Konsisten dengan tidak *overfit*: privat 0,24844 vs publik 0,24913',
    '*Leaderboard* hanya konfirmasi arah; satu perubahan batal karena tidak terkonfirmasi (0,24938 vs 0,24934)'), { x: 8.2, y: Y0, w: 4.53, h: 4.4,
    size: 16, after: 12 });
}

async function c8() {
  const [s, b] = backupSlide('C8');
  mdTable(s, table(b.bukti), { x: X0, y: Y0, w: 7.0, colW: [2.95, 1.35, 1.35, 1.35], size: 16, rowH: 0.6,
    align: ['left', 'right', 'right', 'right'], name: 'tabel penyesuaian akhir' });
  bullets(s, fromDoc(b, 'S3: tebakan gas dekat batas 999.765 kWh ditarik turun; S4: dipotong ke batas data latih',
    'S1, S3, dan S4: *leaderboard* 0,24920 menjadi 0,24913', '77 kawasan uji identik dengan kawasan latih disalin',
    '*Seed* dikunci, mode deterministik, sha256 aec745ab…d66e',
    'Pelatihan penuh sekitar 20 jam; tanpa pelatihan sekitar 5 jam'), { x: 7.95, y: Y0, w: 4.78, h: 4.6, size: 16, after: 12 });
}

async function c9() {
  const [s, b] = backupSlide('C9');
  // households N of a subzone: recorded heating categories vs K
  const x = X0, y = 2.3, w = 6.6;
  subhead(s, 'N rumah tangga di satu subzona', x, Y0, w);
  R(s, { x, y, w: w * 0.62, h: 0.75, fill: GREY, alpha: 40, name: 'kategori tercatat' });
  T(s, 'kategori pemanas tercatat (pecahan N)', { x, y, w: w * 0.62, h: 0.75, size: 14, align: 'center', valign: 'middle' });
  R(s, { x: x + w * 0.62, y, w: w * 0.38, h: 0.75, fill: TEAL, name: 'K' });
  T(s, 'K', { x: x + w * 0.62, y, w: w * 0.38, h: 0.75, size: 22, bold: true, color: WHITE, align: 'center', valign: 'middle' });
  L(s, x + w * 0.81, y + 0.78, x + w * 0.55, y + 1.37, { color: TEAL, lw: 1.5 });
  L(s, x + w * 0.81, y + 0.78, x + w * 0.88, y + 1.37, { color: TEAL, lw: 1.5 });
  R(s, { x: x + w * 0.36, y: y + 1.4, w: w * 0.38, h: 0.75, fill: TEAL, alpha: 75, name: 'pemanas listrik' });
  T(s, 'pemanas listrik (adopsi)', { x: x + w * 0.36, y: y + 1.4, w: w * 0.38, h: 0.75, size: 14, bold: true, color: NAVY,
    align: 'center', valign: 'middle' });
  R(s, { x: x + w * 0.76, y: y + 1.4, w: w * 0.24, h: 0.75, fill: VIOLET, alpha: 75, name: 'kategori lain' });
  T(s, 'kategori lain', { x: x + w * 0.76, y: y + 1.4, w: w * 0.24, h: 0.75, size: 14, bold: true, color: VIOLET,
    align: 'center', valign: 'middle' });
  bullets(s, fromDoc(b, 'N: bilangan bulat terkecil yang membuat semua pecahan komposisi bulat',
    'K: rumah tangga di luar kategori tercatat',
    'q_xent menebak peluang K di kategori lain; r_reg menebak porsinya; adopsi = sisa'),
  { x, y: 4.7, w, h: 2.0, size: 14 });
  subhead(s, 'Skor validasi, dibobot ke subzona baru', 7.6, Y0, 5.13, 0.55);
  mdTable(s, [['Model', 'Skor (poin)'], ...fromDoc(b, 'q_xent|1,0329', 'r_reg|1,0351', 'rata-rata keduanya|1,0303').map((t) => t.split('|'))],
    { x: 7.6, y: 2.5, w: 5.13, colW: [3.33, 1.8], size: 15, rowH: 0.44, align: ['left', 'right'], name: 'tabel skor adopsi',
      highlight: (r) => (r[0].startsWith('rata-rata') ? { bold: true, fill: 'E3F1F4' } : null) });
  bullets(s, fromDoc(b, '1,0303 poin = 0,143 setelah dibagi σ adopsi', '8,14% adalah porsi kawasan uji (harapan 8,17%), bukan porsi subzona'),
    { x: 7.6, y: 4.5, w: 5.13, h: 1.5, size: 15 });
}

async function c10() {
  const [s, b] = backupSlide('C10');
  const cards = [
    ['Derau prediksi acak (200 simulasi)', 'Spearman 0,998 · median 8 zona pindah kuadran · 5 zona teratas tetap', TEAL],
    ['COP 2,5 sampai 3,5 menurut iklim zona', 'Spearman 0,81 · 68 zona pindah kuadran · 1 dari 5 zona teratas tetap', AMBER],
  ];
  const cw = (W - 0.15) / 2;
  cards.forEach(([h, t, c], i) => {
    fromDoc(b, h, t);
    const x = X0 + i * (cw + 0.15);
    R(s, { x, y: Y0, w: cw, h: 0.95, fill: c, alpha: 88, r: 0.08, name: `uji ketahanan ${i + 1}` });
    T(s, h, { x: x + 0.15, y: Y0 + 0.06, w: cw - 0.3, h: 0.3, size: 14, bold: true, color: c });
    T(s, t, { x: x + 0.15, y: Y0 + 0.36, w: cw - 0.3, h: 0.56, size: 14 });
  });
  const rows = table(b.bukti);
  const startRows = ['REG_273', 'REG_270', 'REG_045', 'REG_263', 'REG_033'];
  mdTable(s, rows, { x: X0, y: 3.1, w: W, colW: [2.25, 1.8, 2.6, 2.95, 2.533], size: 14, margin: [2, 6, 2, 6],
    align: ['left', 'left', 'center', 'center', 'center'], rowH: rows.map((r, i) => (i === 0 ? 0.62 : 0.36)),
    name: 'tabel zona contoh', highlight: (r) => (startRows.includes(r[0]) ? { bold: true } : null) });
}

async function c11() {
  const [s, b] = backupSlide('C11');
  s.addImage({ path: path.join(HERE, 'img', 'dep_ukuran_dua.png'), x: X0, y: Y0, w: 7.3, h: 3.45,
    altText: 'Dependensi SHAP ukuran rumah: gas per m2 dan listrik per rumah' });
  bullets(s, fromDoc(b, 'Rumah kecil: sumbangan gas per m² sekitar +65%', 'Rumah besar: sumbangan listrik per rumah sekitar +75%',
    'Pita 5% ekstrem; antarkuartil +34% dan +35%'), { x: X0, y: 5.45, w: 7.3, h: 1.3, size: 15, after: 3 });
  subhead(s, 'Target lomba dan aspek trilema (*writeup* bab 1)', 8.3, Y0, 4.43, 0.55);
  const pairs = fromDoc(b, 'gas total|emisi', 'listrik total|beban jaringan', 'EUI|emisi dan biaya', 'porsi bangunan bersih|emisi',
    'adopsi|transisi').map((t) => t.split('|'));
  mdTable(s, [['Target', 'Aspek trilema'], ...pairs], { x: 8.3, y: 2.48, w: 4.43, colW: [2.35, 2.08], size: 15, rowH: 0.46,
    name: 'tabel target dan trilema' });
}

// ======================================================================================================================
async function main() {
  defineLayouts();
  for (const f of [slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8, slide9, slide10, slide11, slide12, slide13]) {
    await f();
  }
  for (const f of [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]) await f();
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(`wrote ${path.relative(path.join(HERE, '..', '..'), OUT)}: ${ISI.utama.length} main + ${ISI.cadangan.length} backup slides`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
