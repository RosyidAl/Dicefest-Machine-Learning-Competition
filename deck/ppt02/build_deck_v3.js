/*
 * Builds deck/ppt02/BISMILLAH_Final_v3.pptx with pptxgenjs 4.0.1 (structured deck: theme, layouts with named
 * placeholders, sections, speaker notes).
 *
 * Sources (nothing is typed twice where the markdown already has it):
 *   build/isi_v3.json   titles, on-screen text ("Isi Teks"), narration, timing, sources and backup evidence,
 *                       exported from 01_struktur_deck_minto.md by isi_deck.py
 *   img/data_grafik.json numbers behind the native bar charts, checked against the notebook by make_visuals_v3.py
 *   img/*.png           quadrant map and SHAP dependence plots (make_visuals_v3.py)
 *
 * Native charts carry no axis text: every label is a text box placed from the chart's fixed inner plot area
 * (manual layout), so numbers keep the Indonesian decimal comma whatever the locale of the presenting computer.
 *
 * usage:
 *   python3 deck/ppt02/isi_deck.py
 *   NODE_PATH=<dir>/node_modules PPTX_SKILL_DIR=<pptx skill dir> node deck/ppt02/build_deck_v3.js
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
const ISI = JSON.parse(fs.readFileSync(path.join(HERE, 'build', 'isi_v3.json'), 'utf8'));
const DATA = JSON.parse(fs.readFileSync(path.join(HERE, 'img', 'data_grafik.json'), 'utf8'));
const OUT = path.join(HERE, 'BISMILLAH_Final_v3.pptx');
const SKILL = process.env.PPTX_SKILL_DIR;
if (!SKILL || !fs.existsSync(path.join(SKILL, 'scripts', 'apply_theme.js'))) {
  throw new Error('set PPTX_SKILL_DIR to the pptx skill directory (it holds scripts/apply_theme.js)');
}
const { applyTheme } = require(path.join(SKILL, 'scripts', 'apply_theme.js'));

// ---- theme: the v2 palette (design system B4 of the structure document) ---------------------------------------
const THEME = {
  name: 'BISMILLAH Final v3',
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
pres.subject = 'Presentasi final v3 (top-down, 9 menit 30 detik)';

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

/** inline markup of the markdown (*italic*, **bold**, `code`) -> pptxgenjs runs */
function runs(text, base = {}) {
  text = typo(text);
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), options: { ...base } });
    const t = m[0];
    if (t.startsWith('**')) out.push({ text: t.slice(2, -2), options: { ...base, bold: true } });
    else if (t.startsWith('*')) out.push({ text: t.slice(1, -1), options: { ...base, italic: true } });
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

const plain = (t) => t.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1').replace(/`([^`]+)`/g, '$1');

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
  const size = o.size || 11;
  const w = o.w || 0.16 + text.length * size * 0.0092;
  const h = o.h || 0.3;
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
        { placeholder: { options: { name: 'kicker', type: 'body', x: X0, y: 0.3, w: W, h: 0.3, fontSize: 12, bold: true,
          color: TEAL, align: 'left', valign: 'top', margin: 0 }, text: '' } },
        { placeholder: { options: { name: 'title', type: 'title', x: X0, y: 0.6, w: W, h: 1.05,
          fontSize: title === 'ISI' ? 28 : 26, bold: true, color: NAVY, align: 'left', valign: 'top', margin: 0 }, text: '' } },
        { text: { text: footer, options: { x: X0, y: 7.02, w: 9, h: 0.28, fontSize: 10, color: MUTED, margin: 0,
          valign: 'top', isTextBox: true } } },
      ],
      slideNumber: { x: 12.233, y: 7.02, w: 0.5, h: 0.28, fontSize: 10, color: MUTED, align: 'right', margin: 0 },
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
      { placeholder: { options: { name: 'kicker', type: 'body', x: X0, y: 0.45, w: W, h: 0.34, fontSize: 13, bold: true,
        color: AMBER, align: 'left', valign: 'top', margin: 0 }, text: '' } },
      { placeholder: { options: { name: 'title', type: 'title', x: X0, y: 0.88, w: W, h: titleH, fontSize: 32, bold: true,
        color: WHITE, align: 'left', valign: 'top', margin: 0 }, text: '' } },
      { text: { text: FOOTER, options: { x: X0, y: 7.02, w: 8, h: 0.28, fontSize: 10, color: GREY, margin: 0,
        valign: 'top', isTextBox: true } } },
    ],
    slideNumber: { x: 12.233, y: 7.02, w: 0.5, h: 0.28, fontSize: 10, color: GREY, align: 'right', margin: 0 },
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
    T(s, c.label, { x: c.lx, y: c.ly, w: c.lw, h: 0.44, size: 18, bold: true, color: WHITE,
      align: c.la, valign: 'middle' });
  }
  T(s, 'saling tarik', { x: tri.x + 0.8, y: tri.y + 1.75, w: tri.w - 1.6, h: 0.4, size: 14, italic: true, color: GREY,
    align: 'center', valign: 'middle' });

  // the number
  const [big, caption] = item(d, 1).split(': ');
  T(s, big, { x: 6.85, y: 2.55, w: 5.88, h: 1.35, size: 80, bold: true, color: AMBER, valign: 'middle' });
  T(s, caption, { x: 6.85, y: 3.95, w: 5.6, h: 1.0, size: 20, color: WHITE });
  chip(s, 'ASUMSI', 6.85, 5.2, { fill: WHITE, alpha: 85, color: WHITE, w: 1.05 });
  T(s, item(d, 2, 'ASUMSI'), { x: 6.85, y: 5.6, w: 5.6, h: 0.75, size: 15, color: GREY });
  mainNotes(s, d);
}

async function slide2() {
  const d = MAIN[2];
  const s = newSlide('ISI', 0, KICKER.I, d.judul_baris);
  const r1 = item(d, 0).split(' · '), r2 = item(d, 1).split(' · '), r3 = item(d, 2).split(' · ');
  if ([r1, r2, r3].some((r, i) => r.length !== 3 || r[0] !== String(i + 1))) throw new Error('slide 2: expected "n · text · tag"');
  const top = 2.1, h = 3.75;
  // R1 card (left)
  const cw = 3.35, gx = 4.9, gw = 3.533, x3 = 9.383;   // R1 card, gate, R3 card; 0.95 in gaps for the arrows
  R(s, { x: X0, y: top, w: cw, h, fill: SOFT, r: 0.12, name: 'kartu R1' });
  numberBadge(s, 1, X0 + 0.55, top + 0.55, 0.62, AMBER);
  // "*Retrofit* selubung dulu di 118 zona Q1 (47% konsumsi gas)": accent on 118 and 47%
  const m1 = r1[1].match(/^(.*) di (\d+) zona (Q1) \((\d+%) konsumsi gas\)$/);
  if (!m1) throw new Error('slide 2: unexpected R1 text ' + r1[1]);
  T(s, m1[1] + ' di zona ' + m1[3], { x: X0 + 0.3, y: top + 1.1, w: cw - 0.4, h: 0.85, size: 20, bold: true });
  T(s, [{ text: `${m1[2]} zona`, options: { fontSize: 40, bold: true, color: AMBER } }],
    { x: X0 + 0.3, y: top + 1.95, w: cw - 0.5, h: 0.75, valign: 'middle' });
  T(s, `${m1[4]} konsumsi gas`, { x: X0 + 0.3, y: top + 2.68, w: cw - 0.5, h: 0.4, size: 18, bold: true, color: AMBER });
  trilemmaTag(s, r1[2], [AMBER, VIOLET], X0 + 0.3, top + h - 0.6);

  // R2 gate (middle, taller, dark)
  R(s, { x: gx, y: top - 0.25, w: gw, h: h + 0.5, fill: NAVY, r: 0.12, name: 'gerbang R2' });
  numberBadge(s, 2, gx + 0.55, top + 0.35, 0.62, WHITE, NAVY);
  await icon(s, 'FaPlugCircleCheck', gx + gw - 0.95, top + 0.05, 0.62, HEX.lt1, 'ikon gardu');
  T(s, r2[1], { x: gx + 0.3, y: top + 1.0, w: gw - 0.6, h: 1.65, size: 24, bold: true, color: WHITE });
  T(s, 'gerbang untuk pompa panas di Q1 dan Q2', { x: gx + 0.3, y: top + 2.6, w: gw - 0.6, h: 0.6, size: 14, italic: true,
    color: GREY });
  trilemmaTag(s, r2[2], [TEAL], gx + 0.3, top + h - 0.35, WHITE);

  // R3 card (right)
  R(s, { x: x3, y: top, w: cw, h, fill: SOFT, r: 0.12, name: 'kartu R3' });
  numberBadge(s, 3, x3 + 0.55, top + 0.55, 0.62, TEAL);
  T(s, r3[1], { x: x3 + 0.3, y: top + 1.1, w: cw - 0.5, h: 1.0, size: 20, bold: true });
  await badge(s, 'FaFlask', x3 + 0.62, top + 2.55, 0.75, TEAL, { name: 'ikon pilot' });
  trilemmaTag(s, r3[2], [AMBER, TEAL], x3 + 0.3, top + h - 0.6);

  // heat pumps from both sides pass the gate
  for (const [x1, x2] of [[X0 + cw + 0.08, gx - 0.08], [x3 - 0.08, gx + gw + 0.08]]) {
    L(s, x1, top + 2.2, x2, top + 2.2, { color: MUTED, lw: 2 });
  }
  for (const x of [X0 + cw, gx + gw]) {
    T(s, paras([['pompa', { align: 'center' }], ['panas', { align: 'center' }]]), { x: x + 0.1, y: top + 1.52, w: 0.75,
      h: 0.6, size: 12, color: MUTED, valign: 'bottom' });
  }

  // where the talk goes next
  const nxt = item(d, 3);
  const [lab, rest] = nxt.split(': ');
  T(s, [{ text: lab + ': ', options: { bold: true, color: NAVY } }, { text: rest, options: { color: MUTED } }],
    { x: X0, y: 6.32, w: W, h: 0.4, size: 15, valign: 'middle' });
  mainNotes(s, d);
}

/** trilemma demand tag: coloured dots + text ("biaya dan emisi") */
function trilemmaTag(s, text, dots, x, y, color = INK) {
  dots.forEach((c, i) => R(s, { shape: SH.ellipse, x: x + i * 0.26, y: y + 0.06, w: 0.2, h: 0.2, fill: c,
    name: `titik trilema ${i + 1}` }));
  T(s, text, { x: x + dots.length * 0.26 + 0.08, y, w: 2.6, h: 0.32, size: 14, bold: true, color, valign: 'middle' });
}

async function slide3() {
  const d = MAIN[3];
  const s = newSlide('ISI', 1, KICKER.II, d.judul_baris);
  const cols = [{ x: X0, w: 3.55 }, { x: 4.5, w: 3.3 }, { x: 8.15, w: 4.583 }];
  const heads = [['DIBANGUN', NAVY], ['RISIKO DIKENDALIKAN', AMBER], ['KONTRIBUSI', TEAL]];
  heads.forEach(([t, c], i) => chip(s, t, cols[i].x, Y0, { fill: c, size: 12, h: 0.34 }));
  L(s, cols[0].x + cols[0].w - 0.35, Y0 + 0.17, cols[1].x - 0.12, Y0 + 0.17, { color: GREY });
  L(s, cols[1].x + cols[1].w - 0.12, Y0 + 0.17, cols[2].x - 0.12, Y0 + 0.17, { color: GREY });

  // built: the formula and the two house sizes
  const built = item(d, 0, 'DIBANGUN'); // "`area_per_meter` = luas lantai ÷ jumlah meter ≈ luas satu rumah"
  const [lhs, rhs] = built.split(' = ');
  const [ratio, approx] = rhs.split(' ≈ ');
  R(s, { x: cols[0].x, y: 2.4, w: cols[0].w, h: 1.75, fill: SOFT, r: 0.1, name: 'kartu rumus' });
  T(s, paras([
    [runs(lhs, { bold: true, color: NAVY, fontSize: 20 })],
    [runs('= ' + ratio, { fontSize: 17 })],
    [runs('≈ ' + approx, { fontSize: 17, bold: true, color: AMBER })],
  ]), { x: cols[0].x + 0.2, y: 2.52, w: cols[0].w - 0.35, h: 1.2 });
  T(s, '(bld_dim_06 ÷ bld_dim_12)', { x: cols[0].x + 0.2, y: 3.72, w: cols[0].w - 0.35, h: 0.3, size: 12, color: MUTED });
  const hc = [cols[0].x + 0.85, cols[0].x + 2.65];   // centres of the small and the large house
  await icon(s, 'FaHouse', hc[0] - 0.275, 4.75, 0.55, HEX.dk2, 'rumah kecil');
  await icon(s, 'FaHouse', hc[1] - 0.5, 4.3, 1.0, HEX.accent5, 'rumah besar');
  [['±37 m²', NAVY], ['±173 m²', MUTED]].forEach(([t, c], i) => T(s, paras([[t, { align: 'center' }],
    ['per meter', { align: 'center' }]]), { x: hc[i] - 0.75, y: 5.38, w: 1.5, h: 0.55, size: 14, bold: true, color: c }));
  T(s, 'rata-rata 5% kawasan terkecil dan terbesar', { x: cols[0].x, y: 6.05, w: cols[0].w, h: 0.3, size: 11, color: MUTED,
    align: 'center' });

  // risk controlled: four checks
  const checks = item(d, 1, 'RISIKO DIKENDALIKAN').split(' · ');
  if (checks.length !== 4) throw new Error('slide 3: expected four checks');
  for (const [i, c] of checks.entries()) {
    const y = 2.45 + i * 0.98;
    await badge(s, 'FaCheck', cols[1].x + 0.27, y + 0.3, 0.5, TEAL);
    T(s, c, { x: cols[1].x + 0.68, y: y + 0.02, w: cols[1].w - 0.7, h: 0.6, size: 16, valign: 'middle' });
  }

  // contribution: top five features of two explanation models (native charts)
  T(s, item(d, 2, 'KONTRIBUSI'), { x: cols[2].x, y: 2.38, w: cols[2].w, h: 0.62, size: 14 });
  const panels = [['log intensitas gas (EUI)', DATA.s3_shap_top.gas, 3.08], ['log listrik per meter', DATA.s3_shap_top.listrik, 4.78]];
  for (const [title, rows, y] of panels) {
    T(s, title, { x: cols[2].x, y, w: cols[2].w, h: 0.28, size: 12, bold: true, color: NAVY });
    const frame = { x: cols[2].x, y: y + 0.3, w: cols[2].w, h: 1.32 };
    const plot = { x: cols[2].x + 1.62, y: frame.y + 0.04, w: 2.15, h: 1.24 };
    const g = bars(s, { ...frame, plot, cats: rows.map((r) => r.fitur), vals: rows.map((r) => r.nilai), max: 0.1,
      colors: rows.map((r) => (r.fitur === 'area_per_meter' ? HEX.accent1 : HEX.accent4)), gap: 45,
      name: `grafik SHAP ${title}`, alt: `Rata-rata |SHAP| lima fitur teratas, ${title}` });
    g.forEach((b, i) => {
      T(s, rows[i].fitur, { x: cols[2].x, y: b.cy - 0.12, w: 1.55, h: 0.24, size: 11, color: INK, align: 'right', valign: 'middle' });
      T(s, idn(rows[i].nilai, 4), { x: b.x1 + 0.06, y: b.cy - 0.12, w: 0.7, h: 0.24, size: 11, bold: i === 0,
        color: i === 0 ? AMBER : MUTED, valign: 'middle' });
    });
  }
  T(s, 'rata-rata |SHAP| pada sampel 20.000 baris latih', { x: cols[2].x, y: 6.45, w: cols[2].w, h: 0.28, size: 11, color: MUTED });
  mainNotes(s, d);
}

async function slide4() {
  const d = MAIN[4];
  const s = newSlide('ISI', 1, KICKER.II, d.judul_baris);
  const lw = 3.95, rx = 4.95, rw = X1 - rx;
  // left: the two text cards
  R(s, { x: X0, y: Y0, w: lw, h: 2.0, fill: SOFT, r: 0.1, name: 'kartu dibangun' });
  chip(s, 'DIBANGUN', X0 + 0.2, Y0 + 0.2, { fill: NAVY, size: 12, h: 0.32 });
  T(s, item(d, 0, 'DIBANGUN'), { x: X0 + 0.2, y: Y0 + 0.65, w: lw - 0.4, h: 1.25, size: 16 });
  R(s, { x: X0, y: 4.15, w: lw, h: 2.55, fill: SOFT, r: 0.1, name: 'kartu risiko' });
  chip(s, 'RISIKO DIKENDALIKAN', X0 + 0.2, 4.35, { fill: AMBER, size: 12, h: 0.32 });
  T(s, item(d, 2, 'RISIKO DIKENDALIKAN'), { x: X0 + 0.2, y: 4.8, w: lw - 0.4, h: 1.15, size: 16 });
  await badge(s, 'FaShieldHalved', X0 + 0.55, 6.15, 0.6, AMBER, { name: 'ikon pengaman' });

  // right top: subzone-number strip with three nested windows
  const n = 31, cw = 0.235, sx = rx + (rw - n * cw) / 2, sy = 2.82;
  for (let i = 0; i < n; i++) {
    const mid = i === 15;
    R(s, { x: sx + i * cw, y: sy, w: cw - 0.03, h: 0.38, fill: mid ? AMBER : NAVY, alpha: mid ? 0 : 82,
      name: mid ? 'kawasan dinilai' : `subzona ${i + 1}` });
  }
  T(s, 'kawasan dinilai', { x: sx + 15.5 * cw - 0.8, y: sy + 0.42, w: 1.6, h: 0.28, size: 11, bold: true, color: AMBER,
    align: 'center' });
  const wins = [['±30', 3, 2.46], ['±100', 8, 2.17], ['±300', 15, 1.88]];
  for (const [lab, k, y] of wins) {
    const xa = sx + (15 - k) * cw, xb = sx + (16 + k) * cw - 0.03;
    L(s, xa, y + 0.14, xb, y + 0.14, { color: TEAL, lw: 1.5, arrow: false });
    L(s, xa, y + 0.04, xa, y + 0.24, { color: TEAL, lw: 1.5, arrow: false });
    L(s, xb, y + 0.04, xb, y + 0.24, { color: TEAL, lw: 1.5, arrow: false });
    T(s, lab, { x: xa - 0.75, y: y, w: 0.7, h: 0.28, size: 12, bold: true, color: TEAL, align: 'right', valign: 'middle' });
  }
  T(s, 'nomor subzona →', { x: sx + n * cw - 1.6, y: sy + 0.42, w: 1.6, h: 0.28, size: 11, color: MUTED, align: 'right' });
  const [fact, use] = item(d, 1).split(' → '); // "96,5% pasangan ... sama", "nomor berdekatan = pendekatan lokasi"
  const pct = fact.split(' ')[0];
  T(s, [{ text: pct, options: { bold: true, color: TEAL } }, { text: fact.slice(pct.length), options: { breakLine: true } },
    ...runs('→ ' + use, { bold: true, color: NAVY })], { x: rx, y: 3.53, w: rw, h: 0.55, size: 14, valign: 'top', align: 'center' });

  // right bottom: two-layer cross-fit
  T(s, 'Lapis luar: 10 fold', { x: rx, y: 4.15, w: 3, h: 0.3, size: 13, bold: true, color: NAVY });
  const bw = (rw - 9 * 0.07) / 10;
  for (let i = 0; i < 10; i++) {
    const hit = i === 2;
    R(s, { x: rx + i * (bw + 0.07), y: 4.48, w: bw, h: 0.6, fill: hit ? AMBER : NAVY, alpha: hit ? 0 : 85, r: 0.05,
      name: hit ? 'fold dinilai' : `fold ${i + 1}` });
    T(s, hit ? 'dinilai' : 'sumber tetangga', { x: rx + i * (bw + 0.07), y: 4.48, w: bw, h: 0.6, size: hit ? 12 : 10,
      bold: hit, color: hit ? WHITE : NAVY, align: 'center', valign: 'middle' });
  }
  T(s, 'Lapis dalam: bagian latih dibagi 5', { x: rx, y: 5.25, w: 4, h: 0.3, size: 13, bold: true, color: NAVY });
  const pw = (rw - 4 * 0.07) / 5;
  for (let i = 0; i < 5; i++) {
    const hit = i === 0;
    R(s, { x: rx + i * (pw + 0.07), y: 5.58, w: pw, h: 0.5, fill: TEAL, alpha: hit ? 0 : 82, r: 0.05,
      name: hit ? 'bagian dihitung' : `bagian ${i + 1}` });
    T(s, hit ? 'dihitung' : 'sumber tetangga', { x: rx + i * (pw + 0.07), y: 5.58, w: pw, h: 0.5, size: 12, bold: hit,
      color: hit ? WHITE : TEAL, align: 'center', valign: 'middle' });
  }
  await icon(s, 'FaArrowRightLong', rx, 6.3, 0.32, HEX.accent5, 'panah');
  T(s, 'Data uji mengambil tetangga dari bagian latih model yang sama', { x: rx + 0.45, y: 6.26, w: rw - 0.45, h: 0.4,
    size: 13, color: MUTED, valign: 'middle' });
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
      chip(s, r.label, tx, y + 0.31, { fill: RED, size: 12, h: 0.33 });
      tx += 1.15;
    }
    T(s, r.text, { x: tx, y: y + 0.06, w: 10.55 - tx, h: 0.83, size: 17, valign: 'middle' });
    R(s, { x: 10.75, y: y + 0.24, w: 1.8, h: 0.47, fill: r.fill, r: 0.23, name: `hasil ${i + 1}` });
    T(s, r.verdict, { x: 10.75, y: y + 0.24, w: 1.8, h: 0.47, size: 15, bold: true, color: WHITE, align: 'center',
      valign: 'middle' });
  }
  const scope = item(d, 4); // "Cakupan: dua blok fitur, satu *fold*"
  R(s, { x: X0, y: 5.16, w: 4.2, h: 0.38, line: MUTED, lw: 1, r: 0.19, name: 'label cakupan' });
  T(s, scope, { x: X0, y: 5.16, w: 4.2, h: 0.38, size: 13, color: MUTED, align: 'center', valign: 'middle' });

  R(s, { x: X0, y: 5.8, w: W, h: 0.95, fill: TEAL, alpha: 88, r: 0.1, name: 'pita kontribusi' });
  chip(s, 'KONTRIBUSI', X0 + 0.25, 6.105, { fill: TEAL, size: 12, h: 0.34 });
  T(s, item(d, 3, 'KONTRIBUSI'), { x: X0 + 1.85, y: 5.83, w: 8.6, h: 0.89, size: 17, valign: 'middle' });
  // small quadrant-map glyph: where the feature ends up
  L(s, 10.75, 6.15, 11.45, 6.15, { color: TEAL, lw: 2 });
  const q = [[AMBER, 1, 0], [TEAL, 1, 1], [NAVY, 0, 0], [GREY, 0, 1]];   // [colour, right?, bottom?]
  for (const [c, i, j] of q) R(s, { x: 11.62 + i * 0.28, y: 5.88 + j * 0.28, w: 0.25, h: 0.25, fill: c, name: 'ikon peta kuadran' });
  T(s, 'peta zona', { x: 11.42, y: 6.43, w: 0.9, h: 0.2, size: 10, color: TEAL, align: 'center' });
  mainNotes(s, d);
}

async function slide6() {
  const d = MAIN[6];
  const s = newSlide('ISI', 2, KICKER.III, d.judul_baris);
  const l1 = item(d, 0), l2 = item(d, 1), stages = item(d, 2), l3 = item(d, 3), total = item(d, 4), err = item(d, 5);

  // data box and the bus to the three lanes
  R(s, { x: X0, y: 2.85, w: 1.45, h: 1.5, fill: SOFT, line: NAVY, lw: 1, r: 0.08, name: 'data' });
  T(s, 'Data latih + uji, 201 kolom', { x: X0 + 0.1, y: 2.9, w: 1.25, h: 1.4, size: 14, bold: true, color: NAVY,
    align: 'center', valign: 'middle' });
  const bus = 2.3;
  L(s, X0 + 1.45, 3.6, bus, 3.6, { color: MUTED, lw: 1.5, arrow: false });
  L(s, bus, 2.35, bus, 4.95, { color: MUTED, lw: 1.5, arrow: false });
  for (const y of [2.35, 3.38, 4.95]) L(s, bus, y, 2.48, y, { color: MUTED, lw: 1.5 });

  // lanes 1 and 2 (blended)
  const lx = 2.5, lwid = 3.0;
  R(s, { x: lx, y: 1.9, w: lwid, h: 0.92, fill: AMBER, alpha: 85, line: AMBER, lw: 1, r: 0.08, name: 'jalur 1' });
  T(s, paras([[runs(l1, { bold: true, fontSize: 15 })], [runs('XGBoost, LightGBM, MLP, NCA', { fontSize: 12, color: MUTED })]]),
    { x: lx + 0.15, y: 1.93, w: lwid - 0.3, h: 0.86, valign: 'middle' });
  R(s, { x: lx, y: 3.0, w: lwid, h: 0.76, fill: TEAL, alpha: 85, line: TEAL, lw: 1, r: 0.08, name: 'jalur 2' });
  T(s, l2, { x: lx + 0.15, y: 3.0, w: lwid - 0.3, h: 0.76, size: 15, bold: true, valign: 'middle' });
  R(s, { shape: SH.rightBrace, x: lx + lwid + 0.08, y: 1.9, w: 0.22, h: 1.86, line: NAVY, lw: 1.5, name: 'kurung 41 model' });
  T(s, paras([['41 model', { align: 'center' }], ['di-*blend*', { align: 'center' }]]), { x: 5.85, y: 2.2, w: 1.25, h: 0.56,
    size: 14, bold: true, color: NAVY, valign: 'bottom' });
  L(s, 5.85, 2.83, 7.08, 2.83, { color: NAVY, lw: 2 });

  // post-processing of lanes 1 and 2
  const [lab12, chainTxt] = stages.split(': ');
  const steps = chainTxt.split(' → ');
  R(s, { x: 7.12, y: 1.9, w: 2.7, h: 1.86, fill: NAVY, r: 0.08, name: 'tahap gabungan' });
  T(s, paras(steps.flatMap((t, i) => {
    const p = [[runs(t, { bold: true, color: WHITE, fontSize: 14 }), { align: 'center' }]];
    if (i < steps.length - 1) p.push([runs('↓', { color: GREY, fontSize: 12 }), { align: 'center' }]);
    return p;
  })), { x: 7.17, y: 1.95, w: 2.6, h: 1.76, valign: 'middle' });
  T(s, lab12, { x: 7.12, y: 3.8, w: 2.7, h: 0.28, size: 11, color: MUTED, align: 'center' });

  // lane 3 (no blending)
  const m3 = l3.match(/^Jalur 3 · Adopsi: (salin subzona \([^)]+\)) \+ (2 LightGBM terpisah) \(([^)]+)\) · (Porsi bersih: rumus)$/);
  if (!m3) throw new Error('slide 6: unexpected lane-3 text ' + l3);
  R(s, { x: lx, y: 4.3, w: 7.32, h: 1.35, fill: GREY, alpha: 80, line: GREY, lw: 1, r: 0.08, name: 'jalur 3' });
  T(s, 'Jalur 3 · Adopsi:', { x: lx + 0.15, y: 4.36, w: 2.2, h: 0.3, size: 15, bold: true });
  R(s, { x: lx + 0.15, y: 4.75, w: 1.95, h: 0.75, fill: WHITE, r: 0.06, name: 'salin subzona' });
  T(s, m3[1], { x: lx + 0.2, y: 4.75, w: 1.85, h: 0.75, size: 14, align: 'center', valign: 'middle' });
  T(s, '+', { x: lx + 2.12, y: 4.85, w: 0.3, h: 0.55, size: 20, bold: true, color: MUTED, align: 'center', valign: 'middle' });
  R(s, { x: lx + 2.45, y: 4.75, w: 2.75, h: 0.75, fill: WHITE, line: RED, lw: 1.25, dash: 'dash', r: 0.06,
    name: '2 LightGBM terpisah' });
  T(s, paras([[runs(m3[2], { bold: true, fontSize: 14 })], [runs(m3[3], { fontSize: 12, color: MUTED })]]),
    { x: lx + 2.5, y: 4.75, w: 2.65, h: 0.75, align: 'center', valign: 'middle' });
  T(s, 'terpisah, tidak ikut *blending*', { x: lx + 2.45, y: 4.36, w: 2.75, h: 0.3, size: 12, bold: true, color: RED,
    align: 'center', valign: 'middle' });
  R(s, { x: lx + 5.45, y: 4.75, w: 1.72, h: 0.75, fill: WHITE, r: 0.06, name: 'porsi bersih' });
  T(s, m3[4], { x: lx + 5.5, y: 4.75, w: 1.62, h: 0.75, size: 14, align: 'center', valign: 'middle' });

  // outputs with honest errors
  const [errHead, errVals] = err.split(': ');
  const vals = errVals.split(' · ');
  const outs = [['Gas', 2.33], ['EUI', 2.81], ['Listrik', 3.29], ['Porsi bersih', 4.55], ['Adopsi', 5.05]];
  T(s, paras(errHead.split('; ').map((t, i, a) => [i < a.length - 1 ? t + ';' : t, { align: 'right' }])),
    { x: 10.2, y: 1.7, w: 2.53, h: 0.5, size: 11, color: MUTED, valign: 'bottom' });
  outs.forEach(([name, y], i) => {
    R(s, { x: 10.2, y, w: 2.53, h: 0.42, fill: SOFT, r: 0.06, name: `keluaran ${name}` });
    T(s, name, { x: 10.32, y, w: 1.4, h: 0.42, size: 14, valign: 'middle' });
    T(s, vals[i], { x: 11.55, y, w: 1.08, h: 0.42, size: 16, bold: true, color: i < 3 ? NAVY : MUTED, align: 'right',
      valign: 'middle' });
  });
  L(s, 9.82, 2.83, 10.15, 2.83, { color: NAVY, lw: 2 });
  L(s, 9.82, 4.98, 10.15, 4.98, { color: MUTED, lw: 2 });

  // the count
  R(s, { x: X0, y: 5.95, w: W, h: 0.8, fill: SOFT, r: 0.1, name: 'pita 43 model' });
  const m = total.match(/^(43 model) = (41) di-\*blend\* \+ (2) terpisah$/);
  if (!m) throw new Error('slide 6: unexpected total ' + total);
  T(s, [
    { text: m[1], options: { bold: true, color: NAVY } }, { text: ' = ', options: { color: MUTED } },
    { text: m[2], options: { bold: true, color: AMBER } }, { text: ' di-', options: {} },
    { text: 'blend', options: { italic: true } }, { text: ' + ', options: { color: MUTED } },
    { text: m[3], options: { bold: true, color: MUTED } }, { text: ' terpisah', options: {} },
  ], { x: X0, y: 5.95, w: W, h: 0.8, size: 26, align: 'center', valign: 'middle' });
  mainNotes(s, d);
}

async function slide7() {
  const d = MAIN[7];
  const s = newSlide('ISI', 2, KICKER.III, d.judul_baris);
  const tiles = [
    { t: item(d, 0), icon: 'FaRankingStar', color: AMBER, size: 54 },
    { t: item(d, 1), icon: 'FaScaleBalanced', color: TEAL, size: 54 },
    { t: item(d, 2), icon: 'FaFingerprint', color: NAVY, size: 34 },
  ];
  const tw = 3.83, gap = (W - 3 * tw) / 2;
  for (const [i, tile] of tiles.entries()) {
    const x = X0 + i * (tw + gap), y = 2.05;
    R(s, { x, y, w: tw, h: 4.05, fill: SOFT, r: 0.12, name: `ubin ${i + 1}` });
    await badge(s, tile.icon, x + 0.6, y + 0.6, 0.7, tile.color);
    const parts = tile.t.split(' · ');
    const head = parts[0];
    const rest = (i === 2 ? 'sidik sha256 berkas submisi · ' : '') + parts.slice(1).join(' · ');
    T(s, head, { x: x + 0.3, y: y + 1.15, w: tw - 0.5, h: 1.0, size: tile.size, bold: true, color: tile.color, valign: 'middle' });
    T(s, rest, { x: x + 0.3, y: y + 2.25, w: tw - 0.55, h: 1.6, size: 17 });
  }
  mainNotes(s, d);
}

async function slide8() {
  const d = MAIN[8];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  s.addImage({ path: path.join(HERE, 'img', 'peta_kuadran.png'), x: X0, y: Y0, w: 6.55, h: 4.515,
    altText: 'Peta kuadran 330 zona: intensitas gas zona terhadap kenaikan konsumsi listrik tahunan, ukuran titik = konsumsi gas zona' });
  T(s, 'Peta kuadran, bukan peta geografis; data tidak memuat koordinat. ' + item(d, 0), { x: X0 + 0.07, y: 6.4, w: 6.48, h: 0.42,
    size: 11, color: MUTED });

  // quadrant table
  const rows = table(d.isi);
  const rx = 7.45, rw = X1 - rx;
  const colW = [0.78, 0.6, 0.7, 1.14, 1.0, 1.063];
  const head = rows[0].map((h) => ({ text: runs(h, { bold: true, color: HEX.lt1, fontSize: 11 }),
    options: { fill: { color: HEX.dk2 }, valign: 'middle', align: 'left' } }));
  const body = rows.slice(1).map((r) => r.map((c, j) => ({
    text: runs(c, { bold: j === 0 || j === 5 || r[0] === 'Q1', color: r[0] === 'Q1' ? HEX.accent1 : HEX.dk1, fontSize: 14 }),
    options: { fill: { color: r[0] === 'Q1' ? 'FBEEDF' : HEX.lt1 }, valign: 'middle', align: j === 0 || j === 5 ? 'left' : 'right' },
  })));
  s.addTable([head, ...body], { x: rx, y: Y0, w: rw, colW, rowH: [0.62, 0.36, 0.36, 0.36, 0.36], margin: [3, 5, 3, 5],
    border: { type: 'solid', pt: 0.75, color: LINE_HEX }, fontSize: 14, objectName: 'tabel kuadran' });

  // assumption and robustness boxes
  const boxes = [[item(d, 1, 'ASUMSI'), 'ASUMSI', VIOLET, 4.15], [item(d, 2, 'UJI KETAHANAN'), 'UJI KETAHANAN', TEAL, 5.47]];
  for (const [txt, lab, c, y] of boxes) {
    R(s, { x: rx, y, w: rw, h: 1.15, fill: c, alpha: 90, r: 0.08, name: `kotak ${lab}` });
    chip(s, lab, rx + 0.15, y + 0.12, { fill: c, size: 11, h: 0.28 });
    T(s, txt, { x: rx + 0.15, y: y + 0.43, w: rw - 0.3, h: 0.74, size: 14 });
  }
  mainNotes(s, d);
}

/** the policy chain TEMUAN -> INTERPRETASI -> REKOMENDASI -> LANGKAH on the right of slides 9 to 11 */
function chain(s, d, opts = {}) {
  const labels = [['TEMUAN', NAVY], ['INTERPRETASI', VIOLET], ['REKOMENDASI', AMBER], ['LANGKAH', TEAL]];
  // box heights follow the number of text lines (0.1 in + lines x 0.254 in + 0.16 in at 15 pt); the space left in the
  // column (Y0 to 6.75 in) is shared out as equal gaps, each with a small arrow in its middle
  const x = 6.6, w = X1 - x, hs = opts.heights, gap = (6.75 - Y0 - hs.reduce((a, b) => a + b, 0)) / 3;
  let y = Y0;
  labels.forEach(([lab, c], i) => {
    const h = hs[i];
    const key = lab === 'REKOMENDASI';
    R(s, { x, y, w, h, fill: c, alpha: key ? 84 : 91, r: 0.08, name: `rantai ${lab}` });
    chip(s, lab, x + 0.15, y + 0.12, { fill: c, size: 11, w: 1.42, h: 0.3 });
    const txt = item(d, i, lab);
    T(s, txt, { x: x + 1.72, y: y + 0.1, w: w - 1.85, h: (opts.textH && opts.textH[i]) || h - 0.18, size: opts.size || 15,
      bold: key, valign: 'top' });
    if (opts.extra && opts.extra[i]) opts.extra[i](x + 1.72, y, w - 1.85, h);
    if (i < 3) R(s, { shape: SH.triangle, x: x + 0.71, y: y + h + (gap - 0.12) / 2, w: 0.3, h: 0.12, fill: GREY, flipV: true,
      name: 'panah rantai' });
    y += h + gap;
  });
}

async function slide9() {
  const d = MAIN[9];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  s.addImage({ path: path.join(HERE, 'img', 'dep_gas_ukuran.png'), x: X0, y: Y0, w: 5.7, h: 3.35,
    altText: 'Dependensi SHAP luas lantai per meter terhadap gas per m2, model penjelas' });
  // retrofit indicator of the three candidate zones (native chart)
  const rows = DATA.s9_retrofit;
  T(s, 'Indikator potensi *retrofit*, tiga zona kandidat', { x: X0, y: 5.32, w: 5.7, h: 0.3, size: 12, bold: true, color: NAVY });
  const frame = { x: X0, y: 5.62, w: 5.7, h: 1.12 };
  const plot = { x: X0 + 1.0, y: frame.y + 0.03, w: 2.85, h: 1.06 };
  const g = bars(s, { ...frame, plot, cats: rows.map((r) => r.zona), vals: rows.map((r) => r.gwh), max: 180,
    colors: rows.map(() => HEX.accent1), gap: 45, name: 'grafik indikator retrofit',
    alt: 'Indikator potensi retrofit tiga zona kandidat, GWh' });
  g.forEach((b, i) => {
    T(s, rows[i].zona, { x: X0, y: b.cy - 0.13, w: 0.92, h: 0.26, size: 12, align: 'right', valign: 'middle' });
    T(s, `≈${idn(rows[i].gwh, 0)} GWh · ${idn(rows[i].pct, 2)}% gas zona`, { x: b.x1 + 0.08, y: b.cy - 0.13, w: 2.0,
      h: 0.26, size: 12, color: MUTED, valign: 'middle' });
  });
  chain(s, d, { heights: [1.02, 0.77, 1.02, 0.77] });
  mainNotes(s, d);
}

async function slide10() {
  const d = MAIN[10];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  const k = DATA.s10_kenaikan;
  T(s, 'Kenaikan konsumsi listrik tahunan, lima zona tertinggi (COP 3)', { x: X0, y: Y0, w: 5.7, h: 0.3, size: 12,
    bold: true, color: NAVY });
  T(s, 'peringkat pada COP per iklim', { x: 5.05, y: Y0 + 0.28, w: 1.25, h: 0.42, size: 10, color: MUTED, align: 'center' });
  T(s, 'jingga = titik awal penilaian · merah = keluar dari sepuluh besar', { x: X0, y: Y0 + 0.3, w: 4.3, h: 0.36, size: 10,
    color: MUTED });
  const cats = [...k.lima.map((r) => r.zona), 'median 330 zona'];
  const vals = [...k.lima.map((r) => r.pct), k.median];
  const start = new Set(k.sepuluh_besar_dua_ukuran_dua_skenario);
  const frame = { x: X0, y: Y0 + 0.7, w: 5.7, h: 2.0 };
  const plot = { x: X0 + 1.42, y: frame.y + 0.03, w: 2.15, h: 1.94 };
  const colors = [...k.lima.map((r) => (start.has(r.zona) ? HEX.accent1 : HEX.accent4)), MEDIAN_HEX];
  const g = bars(s, { ...frame, plot, cats, vals, max: 125, colors, gap: 40, name: 'grafik kenaikan tertinggi',
    alt: 'Kenaikan konsumsi listrik tahunan lima zona tertinggi dan median 330 zona' });
  g.forEach((b, i) => {
    const isMed = i === cats.length - 1;
    T(s, cats[i], { x: X0, y: b.cy - 0.14, w: 1.35, h: 0.28, size: 12, color: isMed ? MUTED : INK, align: 'right', valign: 'middle',
      bold: !isMed && start.has(cats[i]) });
    T(s, idn(vals[i], 2, true) + '%', { x: b.x1 + 0.06, y: b.cy - 0.14, w: 0.95, h: 0.28, size: 12, bold: !isMed,
      color: isMed ? MUTED : INK, valign: 'middle' });
    if (!isMed) T(s, String(k.lima[i].peringkat_cop_iklim), { x: 5.05, y: b.cy - 0.14, w: 1.25, h: 0.28, size: 13,
      bold: true, color: k.lima[i].peringkat_cop_iklim <= 10 ? TEAL : RED, align: 'center', valign: 'middle' });
  });
  const ta = k.tambahan_absolut_terbesar;
  T(s, [{ text: 'Tambahan absolut terbesar: ', options: { color: MUTED } },
    { text: `${ta.zona}, ${idn(ta.TWh, 2)}\u00a0TWh`, options: { bold: true, color: AMBER, breakLine: true } },
    { text: 'peringkat 1 pada kedua skenario COP', options: { color: MUTED } }],
  { x: X0, y: 4.6, w: 5.7, h: 0.44, size: 12, valign: 'top' });

  // what the scenario computes and what nobody knows yet
  const mx = X0, my = 5.13, mw = 2.78;
  const cells = [['DIHITUNG DARI SKENARIO', 'data dan prediksi; 80% gas untuk pemanasan, COP 3', TEAL],
    ['BELUM DIKETAHUI', 'kapasitas gardu terukur, beban puncak dalam kW, COP lokal', MUTED]];
  cells.forEach(([lab, txt, c], i) => {
    const x = mx + i * (mw + 0.14);
    R(s, { x, y: my, w: mw, h: 1.04, fill: c, alpha: 90, r: 0.08, name: `matriks ${lab}` });
    T(s, lab, { x: x + 0.12, y: my + 0.06, w: mw - 0.2, h: 0.26, size: 11, bold: true, color: c });
    T(s, txt, { x: x + 0.12, y: my + 0.33, w: mw - 0.2, h: 0.68, size: 13 });
  });
  R(s, { x: mx, y: 6.3, w: 5.7, h: 0.42, fill: NAVY, r: 0.08, name: 'skrining' });
  T(s, 'Skrining awal → penilaian teknis operator', { x: mx, y: 6.3, w: 5.7, h: 0.42, size: 14, bold: true, color: WHITE,
    align: 'center', valign: 'middle' });
  chain(s, d, { heights: [1.53, 0.77, 0.77, 1.02] });
  mainNotes(s, d);
}

async function slide11() {
  const d = MAIN[11];
  const s = newSlide('ISI', 3, KICKER.IV, d.judul_baris);
  const rows = DATA.s11_adopsi;
  T(s, 'Tingkat adopsi pemanas listrik menurut porsi gas jaringan kawasan', { x: X0, y: Y0, w: 5.7, h: 0.3, size: 12,
    bold: true, color: NAVY });
  const frame = { x: X0, y: Y0 + 0.35, w: 5.7, h: 2.35 };
  const plot = { x: X0 + 0.2, y: frame.y + 0.35, w: 5.3, h: 1.8 };
  const g = bars(s, { ...frame, plot, dir: 'col', cats: rows.map((r) => r.pita), vals: rows.map((r) => r.adopsi), max: 42,
    colors: rows.map((r) => (r.pita === '>80%' ? HEX.accent1 : HEX.accent2)), gap: 55, name: 'grafik adopsi per porsi gas',
    alt: 'Tingkat adopsi pemanas listrik per pita porsi gas jaringan' });
  g.forEach((b, i) => {
    T(s, idn(rows[i].adopsi, 2) + '%', { x: b.cx - 0.6, y: b.top - 0.32, w: 1.2, h: 0.3, size: 14, bold: true,
      color: rows[i].pita === '>80%' ? AMBER : INK, align: 'center', valign: 'bottom' });
    T(s, rows[i].pita, { x: b.cx - 0.65, y: b.bottom + 0.04, w: 1.3, h: 0.28, size: 13, bold: true, align: 'center' });
    T(s, `${idn(rows[i].kawasan, 2)}% kawasan`, { x: b.cx - 0.68, y: b.bottom + 0.3, w: 1.36, h: 0.26, size: 11,
      color: MUTED, align: 'center' });
  });
  T(s, 'porsi gas jaringan kawasan', { x: X0, y: 4.93, w: 5.7, h: 0.24, size: 11, color: MUTED, align: 'center' });
  s.addImage({ path: path.join(HERE, 'img', 'dep_adopsi_kecil.png'), x: X0, y: 5.2, w: 3.0, h: 1.6,
    altText: 'Dependensi SHAP adopsi terhadap porsi gas jaringan, model penjelas' });
  T(s, 'Model penjelas menunjukkan pola yang sama: sumbangan SHAP adopsi turun saat porsi gas jaringan naik',
    { x: X0 + 3.15, y: 5.25, w: 2.55, h: 1.5, size: 13, color: MUTED, valign: 'middle' });
  const reg = 'REG_033: gas jaringan 83,21%, adopsi 4,99%';
  chain(s, d, { heights: [0.77, 0.77, 1.1, 0.77], textH: [null, null, 0.6, null], extra: [null, null, (x, y, w, h) => {
    T(s, reg, { x, y: y + h - 0.42, w, h: 0.32, size: 12, color: MUTED, valign: 'middle' });
  }, null] });
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
    chip(s, 'BELUM DIKETAHUI', x + 1.0, y + 0.4, { fill: MUTED, size: 11, h: 0.3 });
    T(s, unknown, { x: x + 0.3, y: y + 1.05, w: tw - 0.6, h: 1.15, size: 17 });
    R(s, { shape: SH.downArrow, x: x + tw / 2 - 0.22, y: y + 2.4, w: 0.44, h: 0.45, fill: GREY, name: 'panah penanganan' });
    R(s, { x, y: y + 2.95, w: tw, h: 1.6, fill: TEAL, alpha: 88, r: 0.1, name: `penanganan ${i + 1}` });
    chip(s, 'PENANGANAN', x + 0.3, y + 3.15, { fill: TEAL, size: 11, h: 0.3 });
    T(s, handling, { x: x + 0.3, y: y + 3.55, w: tw - 0.6, h: 0.9, size: 18, bold: true, color: NAVY });
  }
  mainNotes(s, d);
}

async function slide13() {
  const d = MAIN[13];
  const s = newSlide('PENUTUP', 4, 'PENUTUP', d.judul_baris);
  const recap = item(d, 0).split(' · ').map((t) => t.match(/^(\d) (.+): (.+)$/));
  if (recap.some((m) => !m)) throw new Error('slide 13: unexpected recap ' + item(d, 0));
  const colors = [AMBER, WHITE, TEAL];
  const dots = [[AMBER, VIOLET], [TEAL], [AMBER, TEAL]];
  const tw = 3.83, gap = (W - 3 * tw) / 2, y = 2.82;
  recap.forEach((m, i) => {
    const x = X0 + i * (tw + gap);
    R(s, { x, y, w: tw, h: 1.55, fill: WHITE, alpha: 90, r: 0.12, name: `kartu penutup ${i + 1}` });
    numberBadge(s, m[1], x + 0.5, y + 0.5, 0.56, colors[i], i === 1 ? NAVY : WHITE);
    T(s, runs(m[2], {}), { x: x + 0.95, y: y + 0.22, w: tw - 1.1, h: 0.6, size: 22, bold: true, color: WHITE, valign: 'middle' });
    dots[i].forEach((c, j) => R(s, { shape: SH.ellipse, x: x + 0.35 + j * 0.26, y: y + 1.07, w: 0.2, h: 0.2, fill: c,
      line: WHITE, lw: 1, name: 'titik trilema' }));
    T(s, m[3], { x: x + 0.4 + dots[i].length * 0.26, y: y + 0.95, w: 2.8, h: 0.44, size: 16, color: GREY, valign: 'middle' });
  });
  const [lab, rest] = item(d, 1).split(': ');
  T(s, lab, { x: X0, y: 4.65, w: 4, h: 0.4, size: 16, bold: true, color: AMBER });
  rest.split(' · ').forEach((t, i) => {
    const x = X0 + i * (tw + gap);
    R(s, { x, y: 5.1, w: tw, h: 0.8, line: GREY, lw: 1, r: 0.1, name: `langkah berikutnya ${i + 1}` });
    T(s, t, { x: x + 0.2, y: 5.1, w: tw - 0.4, h: 0.8, size: 16, color: WHITE, valign: 'middle' });
  });
  T(s, item(d, 2), { x: X0, y: 6.2, w: 6, h: 0.6, size: 28, bold: true, color: WHITE, valign: 'middle' });
  mainNotes(s, d);
}

// ======================================================================================================================
// Backup slides (C1 to C11): evidence on screen, questions and the answer guide in the notes
// ======================================================================================================================
function backupSlide(id) {
  const b = BACK[id];
  const s = newSlide('CADANGAN', 5, `CADANGAN TANYA JAWAB · ${id}`, b.judul_baris);
  backupNotes(s, b);
  return [s, b];
}

/** "Pertanyaan juri" strip at the bottom of a backup slide */
function questions(s, b) {
  T(s, [{ text: 'Menjawab: ', options: { bold: true, color: TEAL } },
    { text: b.pertanyaan.map((q) => `"${plain(q)}"`).join(' · '), options: { italic: true, color: MUTED } }],
  { x: X0, y: 6.4, w: W, h: 0.5, size: 12, valign: 'top' });
}

/** native table from a markdown table; o: x, y, w, colW, size, head (hex), align per column, rowH, highlight(row) */
function mdTable(s, rows, o) {
  const size = o.size || 13;
  const head = rows[0].map((h, j) => ({ text: runs(h, { bold: true, color: HEX.lt1, fontSize: size - 1 }),
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
    { bullet: { indent: 14 }, paraSpaceAfter: o.after === undefined ? 6 : o.after }])), o);
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

async function c1() {
  const [s, b] = backupSlide('C1');
  const pts = fromDoc(b,
    'Bobot setiap model berubah halus menurut peringkat luas (A), jumlah meter (M), dan prediksi gabungan global (p) di data latih',
    '2 × 2 × 2 = 8 bobot pojok per model, digabung dengan NNLS (*non-negative least squares*, bobot tak negatif)',
    '224 bobot gas, 224 bobot EUI, dan 104 bobot listrik untuk 429.303 baris');
  bullets(s, pts, { x: X0, y: Y0, w: 6.1, h: 2.2, size: 14 });
  T(s, 'Skor jujur: bobot di-*fit* ulang di setiap *fold* luar (RMSE/σ)', { x: X0, y: 4.1, w: 6.1, h: 0.3, size: 12,
    bold: true, color: NAVY });
  mdTable(s, table(b.bukti), { x: X0, y: 4.42, w: 6.1, colW: [1.6, 1.5, 1.5, 1.5], size: 14, rowH: 0.34,
    align: ['left', 'right', 'right', 'right'],
    name: 'tabel skor jujur', highlight: (r) => (r[0] === 'EUI' ? { bold: true } : null) });
  s.addImage({ path: path.join(HERE, 'img', 'basis_pojok.png'), x: 7.0, y: Y0, w: 5.73, h: 3.05,
    altText: 'Basis dua simpul bobot adaptif' });
  bullets(s, fromDoc(b, 'Rata-rata lima target −0,000125; tidak ada kolom yang memburuk',
    '*Leaderboard* publik 0,24934 menjadi 0,24925; tiga simpul tidak lebih baik daripada dua'), { x: 7.0, y: 5.05, w: 5.73,
    h: 1.1, size: 14 });
  questions(s, b);
}

async function c2() {
  const [s, b] = backupSlide('C2');
  const steps = fromDoc(b, 'Galat *out-of-fold* gabungan (di *notebook*: koreksi galat wilayah)',
    'Rata-rata galat per zona, petak iklim, ±30 nomor subzona, dan k kawasan termirip',
    'Dikecilkan n/(n + 30), atau n/(n + 5) bila tetangganya sedikit',
    'Dikali bobot λ yang dipilih di *holdout set* (30% data latih), lalu dibekukan di configs/resid_knn.json');
  steps.forEach((t, i) => {
    const y = Y0 + i * 0.9;
    R(s, { x: X0, y, w: 6.0, h: 0.74, fill: i === 3 ? NAVY : SOFT, r: 0.08, name: `langkah koreksi ${i + 1}` });
    numberBadge(s, i + 1, X0 + 0.38, y + 0.37, 0.44, i === 3 ? WHITE : TEAL, i === 3 ? NAVY : WHITE);
    T(s, t, { x: X0 + 0.75, y: y + 0.03, w: 5.1, h: 0.68, size: 14, color: i === 3 ? WHITE : INK, valign: 'middle' });
    if (i < 3) R(s, { shape: SH.triangle, x: X0 + 0.25, y: y + 0.77, w: 0.26, h: 0.1, fill: GREY, flipV: true, name: 'panah' });
  });
  T(s, 'λ untuk gas', { x: 6.95, y: Y0, w: 3, h: 0.3, size: 12, bold: true, color: NAVY });
  mdTable(s, table(b.bukti), { x: 6.95, y: Y0 + 0.32, w: 5.78, colW: [4.6, 1.18], size: 13, rowH: 0.3, align: ['left', 'right'],
    name: 'tabel lambda' });
  bullets(s, fromDoc(b, 'Besar koreksi rms/σ: gas 0,02047; EUI 0,04344; listrik 0,01195',
    'Bagian kNN menurunkan *leaderboard* publik sekitar 0,0004 (0,2499 menjadi 0,24954)'), { x: X0, y: 5.5, w: 6.0, h: 0.8, size: 14, after: 2 });
  questions(s, b);
}

async function c3() {
  const [s, b] = backupSlide('C3');
  T(s, fromDoc(b, 'Yang diacak: kolom turunan tiga target konsumsi (gas, listrik, EUI) · modul *notebook* yang sama, *fold* 0 dari 10')[0],
    { x: X0, y: Y0, w: W, h: 0.32, size: 14, color: MUTED });
  mdTable(s, table(b.bukti), { x: X0, y: 2.25, w: 7.6, colW: [3.5, 0.75, 1.3, 2.05], size: 13,
    rowH: [0.36, 0.36, 0.36, 0.36, 0.36, 0.7, 0.7, 0.36],
    align: ['left', 'right', 'right', 'left'], name: 'tabel uji label acak',
    highlight: (r) => (r[0].startsWith('KONTROL') ? { fill: 'F6E1E1', bold: true } : null) });
  bullets(s, fromDoc(b, 'Rasio fisik seperti `area_per_meter`: per baris, tanpa target',
    'Agregat kawasan (jumlah kawasan, total luas, jumlah meter): dari fitur data latih dan uji, tanpa target',
    'Rata-rata target wilayah dan kawasan termirip:\n*out-of-fold* dua lapis',
    'Di luar cakupan uji: fitur tetangga model adopsi, koreksi galat tetangga, bobot gabungan, faktor penyesuaian akhir (dirancang dengan prediksi *out-of-fold* atau *holdout set*)'),
  { x: 8.45, y: 2.25, w: 4.28, h: 3.9, size: 13 });
  questions(s, b);
}

async function c4() {
  const [s, b] = backupSlide('C4');
  mdTable(s, table(b.bukti), { x: X0, y: Y0, w: 6.4, colW: [3.8, 1.3, 1.3], size: 14, rowH: 0.42,
    align: ['left', 'right', 'right'], name: 'tabel keluarga model',
    highlight: (r) => (r[0] === 'Jumlah' ? { bold: true, fill: 'E3EAF2' } : null) });
  bullets(s, fromDoc(b, 'Jalur 3: 2 LightGBM adopsi (q_xent dan r_reg) dirata-rata, hanya untuk 8,14% kawasan uji; tidak ikut *blending*',
    '*Notebook* melatih 42 model di jalur 1 dan 2; satu keluaran (mk1_gpm) tidak terpilih',
    '41 dihitung per keluaran: MLP dua keluaran mk2 menyumbang ke kedua jalur',
    'Di balik 41 anggota ada 550 model terlatih (per *fold*, sebagian tiga *seed*)'), { x: 7.35, y: Y0, w: 5.38, h: 3.4, size: 14 });
  R(s, { x: 7.35, y: 5.35, w: 5.38, h: 0.75, fill: SOFT, r: 0.1, name: 'pita 43' });
  T(s, [{ text: '43', options: { bold: true, color: NAVY } }, { text: ' = ', options: { color: MUTED } },
    { text: '41', options: { bold: true, color: AMBER } }, { text: ' + ', options: { color: MUTED } },
    { text: '2', options: { bold: true, color: MUTED } }], { x: 7.35, y: 5.35, w: 5.38, h: 0.75, size: 26, align: 'center',
    valign: 'middle' });
  questions(s, b);
}

async function c5() {
  const [s, b] = backupSlide('C5');
  const r2 = fromDoc(b, 'log EUI|0,563', 'log gas per meter|0,747', 'log listrik per meter|0,491', 'adopsi|0,991')
    .map((t) => t.split('|'));
  T(s, 'Kecocokan model penjelas (R² di data yang tidak dipakai melatih)', { x: X0, y: Y0, w: 4.4, h: 0.5, size: 12,
    bold: true, color: NAVY });
  mdTable(s, [['Model penjelas', 'R²'], ...r2], { x: X0, y: 2.38, w: 4.4, colW: [3.2, 1.2], size: 12, rowH: 0.32,
    align: ['left', 'right'], name: 'tabel R2' });
  const fam = DATA.c5_kelompok;
  T(s, 'Porsi kontribusi per kelompok fitur (*writeup* B Tabel 3)', { x: 5.3, y: Y0, w: 7.43, h: 0.5, size: 12, bold: true,
    color: NAVY });
  const short = { 'EUI (intensitas gas)': 'EUI', 'Gas per meter': 'Gas per meter', 'Listrik per meter': 'Listrik per meter',
    'Adopsi pemanas listrik': 'Adopsi' };
  const famRows = [['Target', ...fam.kelompok], ...fam.target.map((t) => [short[t.nama] || t.nama, ...t.porsi.map((p) => `${p}%`)])];
  mdTable(s, famRows, { x: 5.3, y: 2.38, w: 7.43, colW: [1.43, 1.3, 0.95, 1.0, 1.0, 0.65, 1.1], size: 12,
    rowH: [0.46, 0.32, 0.32, 0.32, 0.32],
    align: ['left', 'right', 'right', 'right', 'right', 'right', 'right'], name: 'tabel porsi kelompok',
    highlight: (r) => (r[0] === 'Adopsi' ? { color: HEX.accent3, bold: true } : null) });
  T(s, 'Kalimat aman', { x: X0, y: 4.42, w: 4, h: 0.3, size: 12, bold: true, color: NAVY });
  mdTable(s, table(b.bukti), { x: X0, y: 4.74, w: W, colW: [4.6, 7.533], size: 13, rowH: 0.34, head: HEX.accent3,
    name: 'tabel kalimat aman' });
  questions(s, b);
}

async function c6() {
  const [s, b] = backupSlide('C6');
  T(s, 'Celah data', { x: X0, y: Y0, w: 3, h: 0.3, size: 12, bold: true, color: NAVY });
  const gaps = ['beban puncak (kW)', 'kapasitas gardu', 'COP lokal', 'koordinat geografis', 'biaya *retrofit*', 'tarif',
    'intensitas karbon listrik'];
  let gx = X0;
  for (const g of gaps) {
    const w = 0.24 + plain(g).length * 0.088;
    R(s, { x: gx, y: Y0 + 0.36, w, h: 0.38, fill: SOFT, line: GREY, lw: 0.75, r: 0.19, name: `celah ${plain(g)}` });
    T(s, g, { x: gx, y: Y0 + 0.36, w, h: 0.38, size: 13, align: 'center', valign: 'middle' });
    gx += w + 0.12;
  }
  T(s, 'Makna kolom yang disamarkan berupa dugaan; ambang kuadran berupa median.', { x: X0, y: Y0 + 0.82, w: W, h: 0.3,
    size: 13, color: MUTED });
  T(s, 'Rencana validasi', { x: X0, y: 3.12, w: 3, h: 0.3, size: 12, bold: true, color: NAVY });
  mdTable(s, table(b.bukti), { x: X0, y: 3.44, w: W, colW: [1.75, 2.45, 4.1, 3.833], size: 13, rowH: [0.36, 0.56, 0.78, 0.56],
    name: 'tabel rencana validasi' });
  questions(s, b);
}

async function c7() {
  const [s, b] = backupSlide('C7');
  T(s, '*Writeup* B Tabel 1: LightGBM, RMSE/σ', { x: X0, y: Y0, w: 7, h: 0.3, size: 12, bold: true, color: NAVY });
  mdTable(s, table(b.bukti), { x: X0, y: 2.18, w: 7.2, colW: [1.7, 1.75, 1.25, 1.25, 1.25], size: 16, rowH: 0.55,
    align: ['left', 'left', 'right', 'right', 'right'], name: 'tabel skema validasi',
    highlight: (r) => (r[0] === 'per zona' && r[1] === 'dengan' ? { color: HEX.accent6, bold: true } : null) });
  bullets(s, fromDoc(b, 'Pemisahan data: AUC adversarial 0,4997; kawasan uji di subzona baru 8,14% vs harapan 8,17%',
    'Tidak *overfit* ke *leaderboard*: privat 0,24844 vs publik 0,24913',
    'Satu perubahan ditolak karena tidak dikonfirmasi *leaderboard* (0,24938 vs 0,24934)'), { x: 8.2, y: Y0, w: 4.53, h: 4.3, size: 16, after: 12 });
  questions(s, b);
}

async function c8() {
  const [s, b] = backupSlide('C8');
  mdTable(s, table(b.bukti), { x: X0, y: Y0, w: 7.0, colW: [2.95, 1.35, 1.35, 1.35], size: 16, rowH: 0.6,
    align: ['left', 'right', 'right', 'right'], name: 'tabel penyesuaian akhir' });
  bullets(s, fromDoc(b, 'S3 menarik turun tebakan gas di dekat batas 999.765 kWh; S4 memotong prediksi ke batas data latih',
    '77 kawasan uji yang identik dengan kawasan latih disalin; *leaderboard* publik 0,24920 menjadi 0,24913',
    '*Seed* dikunci, mode deterministik, sidik sha256 aec745ab…d66e',
    'Pelatihan penuh sekitar 20 jam; tanpa pelatihan sekitar 5 jam'), { x: 7.95, y: Y0, w: 4.78, h: 4.4, size: 16, after: 12 });
  questions(s, b);
}

async function c9() {
  const [s, b] = backupSlide('C9');
  // households N of a subzone: recorded heating categories vs K
  const x = X0, y = 2.3, w = 6.6;
  T(s, 'N rumah tangga di satu kawasan', { x, y: Y0, w, h: 0.3, size: 13, bold: true, color: NAVY });
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
  T(s, 'kategori lain', { x: x + w * 0.76, y: y + 1.4, w: w * 0.24, h: 0.75, size: 13, bold: true, color: VIOLET,
    align: 'center', valign: 'middle' });
  bullets(s, fromDoc(b, 'N: bilangan bulat terkecil yang membuat semua pecahan komposisi pemanas bulat',
    'K: rumah tangga di luar kategori tercatat = pengguna pemanas listrik + satu kategori lain',
    'q_xent menebak peluang K berada di kategori lain (*cross-entropy* dengan bobot K); r_reg menebak porsinya langsung; adopsi = sisa'),
  { x, y: 4.7, w, h: 1.6, size: 14 });
  T(s, 'Skor validasi lima *fold* per subzona, dibobot ulang ke komposisi subzona baru', { x: 7.6, y: Y0, w: 5.13, h: 0.55,
    size: 12, bold: true, color: NAVY });
  mdTable(s, [['Model', 'Skor (poin)'], ...fromDoc(b, 'q_xent|1,0329', 'r_reg|1,0351', 'rata-rata keduanya|1,0303').map((t) => t.split('|'))],
    { x: 7.6, y: 2.45, w: 5.13, colW: [3.33, 1.8], size: 15, rowH: 0.42, align: ['left', 'right'], name: 'tabel skor adopsi',
      highlight: (r) => (r[0].startsWith('rata-rata') ? { bold: true, fill: 'E3F1F4' } : null) });
  bullets(s, fromDoc(b, '1,0303 poin = 0,143 setelah dibagi σ adopsi', '8,14% adalah porsi kawasan uji (harapan 8,17%), bukan porsi subzona'),
    { x: 7.6, y: 4.35, w: 5.13, h: 1.4, size: 14 });
  questions(s, b);
}

async function c10() {
  const [s, b] = backupSlide('C10');
  const cards = [
    ['Derau acak per kawasan (200 simulasi)', 'Spearman 0,998 · median 8 zona pindah kuadran · 5 zona teratas tetap', TEAL],
    ['COP 2,5 sampai 3,5 menurut iklim zona', 'Spearman 0,81 · 68 zona pindah kuadran · 1 dari 5 zona teratas tetap', AMBER],
  ];
  cards.forEach(([h, t, c], i) => {
    fromDoc(b, h, t);
    const x = X0 + i * 4.1;
    R(s, { x, y: Y0, w: 3.95, h: 1.1, fill: c, alpha: 88, r: 0.08, name: `uji ketahanan ${i + 1}` });
    T(s, h, { x: x + 0.15, y: Y0 + 0.08, w: 3.65, h: 0.3, size: 13, bold: true, color: c });
    T(s, t, { x: x + 0.15, y: Y0 + 0.38, w: 3.65, h: 0.68, size: 14 });
  });
  bullets(s, fromDoc(b, 'Galat berkelompok antartetangga belum diuji',
    'Asumsi seragam hanya menskalakan angka (median +71,1% sampai +128,0%)',
    '33 dari 47 zona Q2 dan 98 dari 118 zona Q1 tetap di kuadrannya; indikator potensi *retrofit* relatif: Q2 6,6%, Q1 5,5%'),
  { x: 8.85, y: Y0 - 0.02, w: 3.88, h: 1.3, size: 12, after: 2 });
  mdTable(s, table(b.bukti), { x: X0, y: 3.22, w: W, colW: [1.6, 2.55, 4.6, 3.383], size: 12, margin: [2, 6, 2, 6],
    rowH: [0.27, 0.27, 0.27, 0.27, 0.27, 0.46, 0.46, 0.27, 0.27],
    name: 'tabel zona contoh', highlight: (r) => (['REG_263', 'REG_270', 'REG_045', 'REG_033'].includes(r[0]) ? { bold: true } : null) });
  questions(s, b);
}

async function c11() {
  const [s, b] = backupSlide('C11');
  s.addImage({ path: path.join(HERE, 'img', 'dep_ukuran_dua.png'), x: X0, y: Y0, w: 7.3, h: 3.376,
    altText: 'Dependensi SHAP ukuran rumah: gas per m2 dan listrik per rumah' });
  bullets(s, fromDoc(b, 'Rumah kecil: sumbangan gas per m² sekitar +65%', 'Rumah besar: sumbangan listrik per rumah sekitar +75%',
    'Pita 5% ekstrem; antarkuartil +34% dan +35%'), { x: X0, y: 5.35, w: 7.3, h: 0.9, size: 13, after: 2 });
  T(s, 'Target lomba dan aspek trilema (*writeup* bab 1)', { x: 8.3, y: Y0, w: 4.43, h: 0.3, size: 12, bold: true, color: NAVY });
  const pairs = fromDoc(b, 'gas total|emisi', 'listrik total|beban jaringan', 'EUI|emisi dan biaya', 'porsi bangunan bersih|emisi',
    'adopsi|transisi').map((t) => t.split('|'));
  mdTable(s, [['Target', 'Aspek trilema'], ...pairs], { x: 8.3, y: 2.2, w: 4.43, colW: [2.35, 2.08], size: 14, rowH: 0.42,
    name: 'tabel target dan trilema' });
  questions(s, b);
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
