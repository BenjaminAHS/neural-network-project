// Builds the 10-minute defence deck (museum_guide_defence.pptx). Every number comes from reports/results/*.json.
// Run: NODE_PATH=<folder with node_modules> node presentation/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME);

const ROOT = path.resolve(__dirname, "..");
const ASSETS = path.join(__dirname, "assets");
const FIG = path.join(ROOT, "reports", "figures");
// PUBLIC=1: a version for the Git repository, with colour squares instead of the paintings (WikiArt licence).
const PUBLIC = process.env.PUBLIC === "1";
const OUT = process.env.OUT || path.join(__dirname, "museum_guide_defence.pptx");
const STYLE_COLORS = { Baroque: "2A78D6", Cubism: "EB6834", Impressionism: "1BAF7A", Pop_Art: "EDA100", Ukiyo_e: "E87BA4" };
const R = (name) => JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "results", `${name}.json`), "utf8"));

const THEME = {
  name: "Museum Guide",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1B1F", lt1: "FFFFFF", dk2: "1C2433", lt2: "F2F3F5",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100", accent5: "E87BA4", accent6: "6250D6",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { neural: "2A78D6", classical: "EB6834", hybrid: "1BAF7A", baseline: "A9AEB8", quiet: "A9AEB8", axis: "3A3F4B" };
const MUTED = "5F6473";      // captions on white
const MUTED_DARK = "C9CED8"; // captions on the dark slides
const CARD_DARK = "2A3448";

// ---------------------------------------------------------------- numbers (read, not typed)
const d01 = R("01_data"), d02 = R("02_mlp"), d03 = R("03_monitoring"), d05 = R("05_resnet"), d06 = R("06_final");
const d08 = R("08_second_test"); // the second test: 220 new paintings, used once
const pct = (x, digits = 1) => `${(100 * x).toFixed(digits)} %`;
const test = Object.fromEntries(d06.test.map((t) => [t.model, t]));
const speed = Object.fromEntries(d06.speed.map((s) => [s.model, s]));
const tools = Object.fromEntries(d03.tools.map((t) => [t.tool, t]));
const heads = Object.fromEntries(d05.heads.map((h) => [h.head, h]));
const ftMean = d05.finetune_runs.reduce((a, r) => a + r.val_acc, 0) / d05.finetune_runs.length;
const leakMean = (folds, key) => {
  const rows = d05.leakage.filter((r) => r.folds === folds);
  return rows.reduce((a, r) => a + r[key], 0) / rows.length;
};
const paired = d06.paired.find((p) => p.A === "ResNet18 frozen + linear head" && p.B === "random forest on hand-made features");
const FROZEN = "ResNet18 frozen + linear head", FT = "ResNet18 fine-tuned (layer4)", RF = "random forest on hand-made features";
const MLP = "regularised MLP on pixels";
const t2 = Object.fromEntries(d08.table.map((r) => [r.model, r]));
const both = d08.both_tests_acc; // every model on the 440 test paintings (mean of the seeds)
const pairOf = (a, b, testName, seed = 0) => d08.paired.find((r) => r.A === a && r.B === b && r.test === testName && r.seed === seed);
const ftOverFrozen = pairOf(FROZEN, FT, "test 1 + test 2 (440)"); // diff = frozen - fine-tuned

// ---------------------------------------------------------------- deck, theme, layouts
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Is a neural network worth it on a museum kiosk?";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// A number never leaves its unit or its thousands at a line break: non-breaking spaces ("4 %", "1 061").
const NB = " ";
const nb = (t) => t.replace(/(\d) (?=\d{3}\b)/g, `$1${NB}`).replace(/(\d) (?=[%A-Za-z(])/g, `$1${NB}`);
const nbText = (t) => (typeof t === "string" ? nb(t) : t.map((r) => ({ ...r, text: nb(r.text) })));
// Speaker notes of the main slides come from the oral script (speaker_notes.json, written by the script builder).
const NOTES_FILE = path.join(__dirname, "speaker_notes.json");
const SCRIPT_NOTES = fs.existsSync(NOTES_FILE) ? JSON.parse(fs.readFileSync(NOTES_FILE, "utf8")) : [];
let slideCount = 0;
const addSlide = (opts) => {
  const slide = pres.addSlide(opts);
  const index = slideCount++;
  const addText = slide.addText.bind(slide), addTable = slide.addTable.bind(slide), addNotes = slide.addNotes.bind(slide);
  slide.addNotes = (t) => addNotes(SCRIPT_NOTES[index] || t);
  slide.addText = (t, o) => addText(nbText(t), o);
  slide.addTable = (rows, o) => addTable(rows.map((row) => row.map((c) => (typeof c === "string" ? nb(c) : { ...c, text: nb(c.text) }))), o);
  return slide;
};

// Chart labels must read "45.5 %" whatever the language of PowerPoint: values are stored in tenths of a percent
// and printed with a literal dot (a locale decimal separator would give "45,5 %" on a French PC).
const T10 = (x) => Math.round(1000 * x);
const FMT = '0"."0" %";;';   // zero values hidden (used by the highlight trick of stacked bars)

const footer = (color) => ({ text: { text: "Museum guide · MSA-DATB09-03 final project",
  options: { x: 0.5, y: 5.2, w: 6, h: 0.3, fontSize: 10, color, margin: 0 } } });
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 8.8, h: 1.3, fontSize: 34, bold: true,
      color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.8, w: 8.8, h: 0.45, fontSize: 17,
      color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.7, fontSize: 30, bold: true,
      color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    footer(MUTED),
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.7, fontSize: 30, bold: true,
      color: C.background1, valign: "middle", margin: 0 }, text: "" } },
    footer(MUTED_DARK),
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: MUTED_DARK, align: "right" },
});

// ---------------------------------------------------------------- small building blocks
const card = (slide, x, y, w, h, name, fill = C.background2) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 },
    rectRadius: 0.08, objectName: name });

// A museum label: a card with a big number and a caption under it.
const statCard = (slide, x, y, w, h, value, caption, opts = {}) => {
  const strong = opts.strong || false;
  card(slide, x, y, w, h, `card ${value}`, strong ? C.accent1 : C.background2);
  slide.addText(value, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.8, fontSize: opts.size || 40, bold: true,
    fontFace: THEME.headFontFace, color: strong ? C.background1 : (opts.color || C.text2), margin: 0, valign: "middle",
    isTextBox: true, objectName: `value ${value}` });
  slide.addText(caption, { x: x + 0.2, y: y + 1.0, w: w - 0.4, h: h - 1.15, fontSize: opts.captionSize || 14,
    color: strong ? C.background1 : C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `caption ${value}` });
};

// A titled note: bold heading line + body lines, on a light card.
const noteCard = (slide, x, y, w, h, heading, body, opts = {}) => {
  card(slide, x, y, w, h, `note ${heading}`, opts.fill || C.background2);
  slide.addText([
    { text: heading, options: { bold: true, fontSize: opts.headSize || 15, color: opts.headColor || C.text2, breakLine: true } },
    { text: body, options: { fontSize: opts.size || 14, color: opts.color || C.text1 } },
  ], { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: h - 0.24, valign: "top", margin: 0, paraSpaceAfter: 4,
    isTextBox: true, objectName: `text ${heading}` });
};

// The six steps of the talk: [tag label, roadmap title, roadmap line]. Each content slide shows its step above the title.
const STEPS = [
  ["Data", "Data and a fair split", "1 500 WikiArt paintings, 5 styles, split by painter"],
  ["Our MLP", "Our own network (MLP)", "built from nn.Linear, trained on the pixels"],
  ["Monitoring", "Monitoring and diagnosis", "look inside the training, diagnose it, fix it"],
  ["ResNet18", "A pretrained ResNet18", "reuse a network trained on 1.2 million photos"],
  ["Final test", "The final test", "every model once on 220 new paintings, classical models too"],
  ["Kiosk cost", "The cost on a kiosk", "time per photo, size, training time"],
];
const stepTag = (slide, n) => slide.addText(`STEP ${n} OF ${STEPS.length} · ${STEPS[n - 1][0].toUpperCase()}`, {
  x: 0.5, y: 0.06, w: 9, h: 0.26, fontSize: 11, bold: true, color: C.accent1, align: "center", charSpacing: 2,
  margin: 0, isTextBox: true, objectName: `step tag ${n}` });

const bullets = (items, size = 15) => items.map((t, i) => ({ text: t,
  options: { bullet: true, fontSize: size, breakLine: i < items.length - 1, paraSpaceAfter: 8 } }));
const head = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });

// Diagram pieces: a box with a heading and a short line, an arrow between boxes, a painting thumbnail
// (a colour square in the public copy, WikiArt licence).
const diagramBox = (slide, x, y, w, h, heading, body, opts = {}) => {
  card(slide, x, y, w, h, `box ${heading}`, opts.fill || C.background2);
  slide.addText([
    { text: heading, options: { bold: true, fontSize: opts.headSize || 14, color: opts.headColor || C.text2, breakLine: true } },
    { text: body, options: { fontSize: opts.size || 11, color: opts.color || C.text1 } },
  ], { x: x + 0.1, y: y + 0.08, w: w - 0.2, h: h - 0.16, align: "center", valign: "middle", margin: 0, paraSpaceAfter: 3,
    isTextBox: true, objectName: `box text ${heading}` });
};
const arrow = (slide, x, yMid, name) => slide.addShape(pres.shapes.RIGHT_ARROW, { x: x + 0.05, y: yMid - 0.11, w: 0.2, h: 0.22,
  fill: { color: C.text2 }, line: { color: C.text2, width: 0 }, objectName: name });
const thumb = (slide, file, x, y, size) => (PUBLIC
  ? slide.addShape(pres.shapes.RECTANGLE, { x, y, w: size, h: size, fill: { color: STYLE_COLORS[file] },
    line: { color: STYLE_COLORS[file], width: 0 }, objectName: `colour ${file}` })
  : slide.addImage({ path: path.join(ASSETS, `${file}.jpg`), x, y, w: size, h: size, altText: `${file} painting from the dataset`,
    objectName: `painting ${file}` }));
const paintingBox = (slide, x, y, w, h, line) => {
  card(slide, x, y, w, h, "box painting");
  const size = Math.min(w - 0.4, h - 0.65);
  thumb(slide, "Impressionism", x + (w - size) / 2, y + 0.12, size);
  slide.addText([{ text: "1 painting", options: { bold: true, fontSize: 13, color: C.text2, breakLine: true } },
    { text: line, options: { fontSize: 11, color: C.text1 } }],
  { x: x + 0.05, y: y + size + 0.17, w: w - 0.1, h: h - size - 0.22, align: "center", valign: "top", margin: 0,
    isTextBox: true, objectName: "painting caption" });
};

const pngSize = (file) => { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelFontSize: 12, catAxisLabelColor: HEX.axis,
  dataLabelFontSize: 12, legendFontSize: 12, titleFontSize: 14, titleColor: HEX.axis };

// ================================================================= 1. title
pres.addSection({ title: "The question" });
let s = addSlide({ masterName: "TITLE", sectionTitle: "The question" });
s.addText("Is a neural network worth it on a museum kiosk?", { placeholder: "title" });
s.addText("Recognising painting styles with an MLP and a pretrained ResNet18", { placeholder: "body" });
const styles = [["Baroque", "Baroque"], ["Cubism", "Cubism"], ["Impressionism", "Impressionism"], ["Pop_Art", "Pop Art"], ["Ukiyo_e", "Ukiyo-e"]];
const tw = 1.45, tgap = 0.3, tx0 = (10 - (5 * tw + 4 * tgap)) / 2, ty = 2.6;
styles.forEach(([file, label], i) => {
  const x = tx0 + i * (tw + tgap);
  s.addShape(pres.shapes.RECTANGLE, { x: x - 0.05, y: ty - 0.05, w: tw + 0.1, h: tw + 0.1, fill: { color: C.background1 },
    line: { color: C.background1, width: 0 }, objectName: `frame ${label}` });
  if (PUBLIC) {
    s.addShape(pres.shapes.RECTANGLE, { x, y: ty, w: tw, h: tw, fill: { color: STYLE_COLORS[file] },
      line: { color: STYLE_COLORS[file], width: 0 }, objectName: `colour ${label}` });
  } else {
    s.addImage({ path: path.join(ASSETS, `${file}.jpg`), x, y: ty, w: tw, h: tw, altText: `${label} painting from the dataset`,
      objectName: `painting ${label}` });
  }
  s.addText(label, { x: x - 0.1, y: ty + tw + 0.12, w: tw + 0.2, h: 0.3, fontSize: 13, color: MUTED_DARK, align: "center",
    margin: 0, isTextBox: true, objectName: `label ${label}` });
});
s.addText(PUBLIC ? "MSA-DATB09-03 · Final project · Benjamin Caujolle · paintings not shown in this public copy (WikiArt licence)"
  : "MSA-DATB09-03 · Final project · Benjamin Caujolle", { x: 0.6, y: 4.85, w: 8.8, h: 0.35, fontSize: 13,
  color: MUTED_DARK, margin: 0, isTextBox: true, objectName: "byline" });
s.addNotes("Hello. Our museum wants a guide on visitors' phones and on kiosks in the galleries. A visitor points the camera at a painting, and the guide tells them its style and the story of that movement. Here are the five styles we chose: Baroque, Cubism, Impressionism, Pop Art and Ukiyo-e. The kiosks have small CPUs, the team is small and the budget is limited. So the director asks one question: is a neural network worth it, or would a simpler classical model do the job? In the next ten minutes we answer with measured numbers.");

// ================================================================= 2. the question and our six steps (no answer yet)
s = addSlide({ masterName: "CONTENT", sectionTitle: "The question" });
s.addText("Six steps to answer the director", { placeholder: "title" });
noteCard(s, 0.5, 1.05, 9, 1.0, "The question",
  "Is a neural network worth it on a kiosk with a small CPU and a small team, or would a simpler classical model do the job?",
  { headSize: 14, size: 14 });
STEPS.forEach(([label, title, line], i) => {
  const x = 0.5 + (i % 3) * 3.1, y = 2.2 + Math.floor(i / 3) * 1.47;
  card(s, x, y, 2.8, 1.3, `step ${i + 1}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: y + 0.2, w: 0.5, h: 0.5, fill: { color: C.accent1 },
    line: { color: C.accent1, width: 0 }, objectName: `step circle ${i + 1}` });
  s.addText(String(i + 1), { x: x + 0.18, y: y + 0.2, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: C.background1,
    align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `step number ${i + 1}` });
  s.addText([{ text: title, options: { bold: true, fontSize: 14, color: C.text2, breakLine: true } },
    { text: line, options: { fontSize: 12, color: C.text1 } }],
  { x: x + 0.82, y: y + 0.15, w: 1.85, h: 1.12, valign: "top", margin: 0, isTextBox: true, objectName: `step text ${i + 1}` });
});
s.addNotes("The question, and our six steps.");

// ================================================================= 3. a fair split
s = addSlide({ masterName: "CONTENT", sectionTitle: "The question" });
s.addText("A fair test: one painter, one part", { placeholder: "title" });
stepTag(s, 1);
s.addText(bullets([
  `${d01.n_images.toLocaleString("en-US").replace(",", " ")} WikiArt paintings: 5 styles, 300 per style`,
  "Same share of each style in every part",
  "All the paintings of a known painter go to the same part",
  "Test set locked until the very end, then used once",
]), { x: 0.5, y: 1.25, w: 4.4, h: 3.7, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "split bullets" });
card(s, 5.2, 1.2, 4.3, 3.8, "split panel");
s.addText("How the paintings are split", { x: 5.45, y: 1.35, w: 3.8, h: 0.3, fontSize: 13, bold: true, color: C.text2, margin: 0,
  isTextBox: true, objectName: "split caption" });
const parts = [["train", d01.split_counts.train, C.accent1], ["validation", d01.split_counts.val, C.accent4], ["test", d01.split_counts.test, C.text2]];
let bx = 5.45;
parts.forEach(([name, n, color]) => {
  const bw = 3.8 * n / d01.n_images;
  s.addShape(pres.shapes.RECTANGLE, { x: bx, y: 1.8, w: bw, h: 0.45, fill: { color }, line: { color: C.background2, width: 1.5 },
    objectName: `bar ${name}` });
  s.addText(`${n.toLocaleString("en-US").replace(",", " ")}\n${name}`, { x: name === "train" ? bx : bx - 0.1, y: 2.3,
    w: name === "train" ? bw : bw + 0.2, h: 0.5, fontSize: 12, color: C.text1, align: "center", margin: 0, isTextBox: true,
    objectName: `label ${name}` });
  bx += bw;
});
s.addText(String(d01.leakage.shared_painters_artist_split), { x: 5.45, y: 3.0, w: 0.9, h: 1.0, fontSize: 60, bold: true,
  fontFace: THEME.headFontFace, color: C.accent2, margin: 0, valign: "middle", isTextBox: true, objectName: "zero shared" });
s.addText([
  { text: "known painters in two parts", options: { fontSize: 15, bold: true, color: C.text2, breakLine: true } },
  { text: `a random split would share ${d01.leakage.shared_painters_random_split} of ${d01.leakage.known_painters}`, options: { fontSize: 14, color: C.text1 } },
], { x: 6.4, y: 3.05, w: 2.95, h: 0.95, valign: "middle", margin: 0, isTextBox: true, objectName: "zero caption" });
s.addText("Otherwise a model can recognise the painter instead of the style (we measured it: +11 points).",
  { x: 5.45, y: 4.15, w: 3.85, h: 0.7, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true, objectName: "split note" });
s.addNotes("We used 1 500 WikiArt paintings, 300 per style. We split them into 1 061 to train, 219 to validate and 220 to test, with the same share of each style. One more rule: all the paintings of a known painter go to the same part. With a random split, 23 of our 32 known painters would be in two parts, and a model could recognise the painter instead of the style. With our split, zero. The test set stayed locked until the very end: we used it once. Later I will show that this rule changes the score by 11 points.");

// ================================================================= 4. MLP overfits
pres.addSection({ title: "Our MLP" });
s = addSlide({ masterName: "CONTENT", sectionTitle: "Our MLP" });
s.addText("Our MLP learns the paintings by heart", { placeholder: "title" });
stepTag(s, 2);
const curves = path.join(FIG, "02_baseline_curves.png"), cs = pngSize(curves);
const ch = 2.8, cw = ch * cs.w / cs.h; // fits between the title and the three cards
s.addImage({ path: curves, x: (10 - cw) / 2, y: 1.05, w: cw, h: ch, altText: "Training and validation loss and accuracy of the MLP baseline: training accuracy reaches 99.7 % while validation stays near 40 %",
  objectName: "baseline curves" });
const b = d02.baseline_64px;
noteCard(s, 0.5, 4.0, 2.85, 1.05, "The network", `nn.Linear layers, 12 288 → 512 → 256 → 5, ${(b.parameters / 1e6).toFixed(1)} M parameters`);
noteCard(s, 3.575, 4.0, 2.85, 1.05, "Two checks first", `first loss ${d02.first_loss["last_scale_0.1"].toFixed(2)} ≈ ln 5; one batch learned ${pct(d02.overfit_one_batch.accuracy, 0)}`);
noteCard(s, 6.65, 4.0, 2.85, 1.05, "The problem", `${pct(b.final_train_acc)} on train, ${pct(b.val_acc_at_best)} on validation`, { headColor: C.accent2 });
s.addNotes("Our first model is the network of the course: a multi-layer perceptron built from nn.Linear layers, 12 288 pixel inputs, two hidden layers, 6.4 million parameters. Before trusting it, we ran two checks. The first loss must be close to ln 5, the loss of a model that knows nothing: we had to shrink the last layer to get 1.59. And the model must learn one small batch by heart: it does, 100 percent. Then the real training: 99.7 percent on the training images, but only 40.6 percent on validation, and the validation loss rises after epoch 3. This is overfitting: 6 000 parameters per training image.");

// ================================================================= 5. monitoring + regularisation
s = addSlide({ masterName: "CONTENT", sectionTitle: "Our MLP" });
s.addText("Healthy training, but the MLP overfits", { placeholder: "title" });
stepTag(s, 3);
const toolNames = ["baseline (lr 0.0001)", "weight decay", "batch norm", "dropout", "early stopping", "flips", "all together"];
const toolLabels = ["Baseline (lr 0.0001)", "Weight decay", "Batch norm", "Dropout", "Early stopping", "Flips", "All together"];
s.addChart(pres.charts.BAR, [
  { name: "one tool", labels: toolLabels, values: toolNames.map((t) => (t === "all together" ? 0 : T10(tools[t].val_acc))) },
  { name: "all together", labels: toolLabels, values: toolNames.map((t) => (t === "all together" ? T10(tools[t].val_acc) : 0)) },
], { x: 0.4, y: 1.1, w: 5.4, h: 3.95, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.quiet, HEX.neural],
  showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: FMT, dataLabelColor: "FFFFFF",
  valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 500, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: false, showTitle: true, title: "Validation accuracy, mean of 3 seeds", barGapWidthPct: 45, ...chartText,
  objectName: "chart regularisation" });
s.addText(bullets([
  "Gradients reach every layer and only 4 % of neurons are dead: the training is healthy",
  "Learning-rate sweep: 0.0001 is best; 0.001 overfits after 2 epochs",
  `All tools together: gap train - validation ${tools["baseline (lr 0.0001)"].gap.toFixed(2)} → ${tools["all together"].gap.toFixed(2)}`,
  "But only 7 % of Cubist paintings are found: the MLP sees colours, not shapes",
], 14), { x: 6.05, y: 1.2, w: 3.45, h: 3.85, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "monitoring bullets" });
s.addNotes("We then looked inside the network during training: the gradient of each layer and the share of dead neurons. The gradient reaches every layer and only 4 percent of neurons are dead, so the training is healthy: the problem is overfitting. A learning-rate sweep chose 0.0001. Then we tested the tools of session 6, each with 3 seeds. Alone, each one helps a little. All together, they take the MLP from 38.8 to 45.5 percent, and the gap between train and validation from 0.61 to 0.12. But the MLP finds only 7 percent of the Cubist paintings: it sees colours, not shapes.");

// ================================================================= 6. ResNet18, explained
pres.addSection({ title: "Pretrained ResNet" });
s = addSlide({ masterName: "CONTENT", sectionTitle: "Pretrained ResNet" });
s.addText("ResNet18: a network made for images", { placeholder: "title" });
stepTag(s, 4);
{
  const y = 1.15, h = 1.8, mid = y + h / 2;
  paintingBox(s, 0.5, y, 1.3, h, "224 x 224 pixels");
  arrow(s, 1.8, mid, "arrow 1");
  card(s, 2.1, y, 3.5, h, "box resnet", C.text2);
  s.addText([
    { text: "ResNet18, pretrained on ImageNet", options: { bold: true, fontSize: 14, color: C.background1, breakLine: true } },
    { text: `18 layers, ${Math.round(d05.parameters["ResNet18 + linear head (total)"] / 1e6)} million weights learned on 1.2 million photos. We never retrain all of them.`,
      options: { fontSize: 11, color: MUTED_DARK } },
  ], { x: 2.25, y: y + 0.1, w: 3.2, h: 0.85, valign: "top", margin: 0, isTextBox: true, objectName: "resnet text" });
  // What each part sees, and what happens to it: always frozen, or learning again when fine-tuned.
  [["first layers", "edges, colours", "always frozen"], ["middle layers", "textures", "always frozen"],
    ["last block", "shapes", "fine-tuned: learns"]].forEach(([when, what, status], i) => {
    const x = 2.25 + i * 1.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.95, w: 0.98, h: 0.75, fill: { color: CARD_DARK },
      line: { color: i === 2 ? C.accent4 : CARD_DARK, width: i === 2 ? 1.5 : 0 }, rectRadius: 0.06, objectName: `level ${i + 1}` });
    s.addText([{ text: when, options: { fontSize: 9, color: MUTED_DARK, breakLine: true } },
      { text: what, options: { fontSize: 11, bold: true, color: C.background1, breakLine: true } },
      { text: status, options: { fontSize: 9, color: i === 2 ? C.accent4 : MUTED_DARK } }],
    { x, y: y + 0.95, w: 0.98, h: 0.75, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `level text ${i + 1}` });
  });
  arrow(s, 5.6, mid, "arrow 2");
  diagramBox(s, 5.9, y, 1.3, h, "512 numbers", "a summary of the painting");
  arrow(s, 7.2, mid, "arrow 3");
  diagramBox(s, 7.5, y, 2.0, h, "Our new last layer", `replaces ResNet's own: 512 → 5 styles, ${heads["linear probe"].trained_parameters.toLocaleString("en-US").replace(",", " ")} weights, trained in both ways`,
    { fill: C.accent1, headColor: C.background1, color: C.background1, size: 12 });
  const why = [
    ["Made for images", "Small filters slide over the picture and look at neighbouring pixels, wherever they are. Our MLP saw 12 288 separate pixels."],
    ["Why ResNet18", "ResNet, a famous CNN family (ImageNet winner in 2015), is asked by the brief. ResNet18 is its smallest member: 45 MB, light enough for a kiosk CPU."],
    ["Two ways to reuse it", `Frozen: only our new layer learns (2 565 weights, ${heads["linear probe"].seconds.toFixed(1)} s). Fine-tuned: the last block learns too (8.4 M weights, minutes), the first layers stay frozen.`],
  ];
  why.forEach(([hd, body], i) => noteCard(s, 0.5 + i * 3.05, 3.15, 2.9, 1.9, hd, body, { headSize: 14, size: 12 }));
}
s.addNotes("ResNet18 is a convolutional network, made for images. Its layers slide small filters over the picture: edges and colours first, then textures, then shapes. It was trained on 1.2 million ImageNet photos. We keep it frozen: each painting becomes 512 numbers, and we only train one linear layer from these 512 numbers to our 5 styles.");

// ================================================================= 7. ResNet results
s = addSlide({ masterName: "CONTENT", sectionTitle: "Pretrained ResNet" });
s.addText("A pretrained ResNet18 sees shapes", { placeholder: "title" });
stepTag(s, 4);
const rnLabels = ["Our best MLP (pixels)", "ResNet18, layer4 fine-tuned", "ResNet18 frozen + linear head"];
s.addChart(pres.charts.BAR, [
  { name: "other", labels: rnLabels, values: [T10(tools["all together"].val_acc), T10(ftMean), 0] },
  { name: "chosen", labels: rnLabels, values: [0, 0, T10(heads["linear probe"].val_acc)] },
], { x: 0.4, y: 1.1, w: 4.9, h: 3.95, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.quiet, HEX.neural],
  showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: FMT, dataLabelColor: "FFFFFF",
  valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1000, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: false, showTitle: true, title: "Validation accuracy, mean of 3 seeds", barGapWidthPct: 45, ...chartText,
  objectName: "chart resnet" });
const cubism = (rows) => rows.find((r) => r.style === "Cubism").recall;
noteCard(s, 5.55, 1.15, 3.95, 1.2, "It sees shapes",
  `Cubist paintings found (validation): ${pct(cubism(d03.regularised_seed0_per_class), 0)} with our MLP, ${pct(cubism(d05.probe_seed0_per_class), 0)} with the ResNet.`);
noteCard(s, 5.55, 2.5, 3.95, 1.2, "Fine-tuned: lower on validation",
  `Last block learning again (8.4 M weights), first layers still frozen: ${pct(ftMean)} on validation. The tests will say otherwise.`);
noteCard(s, 5.55, 3.85, 3.95, 1.2, "Monitored like the MLP",
  `Train and validation curves for every run. Frozen + linear head: gap of only ${heads["linear probe"].gap.toFixed(2)} (our first MLP: ${tools["baseline (lr 0.0001)"].gap.toFixed(2)}).`);
s.addNotes("A network that can see shapes is a convolutional network. With 1 061 images we cannot train one from zero, so we reuse ResNet18, trained on 1.2 million ImageNet photos. We freeze it: each painting becomes 512 numbers, and we train only one linear layer, 2 565 weights, in 0.4 seconds. Result: 88.1 percent on validation, almost twice our best MLP. We monitored it like the MLP: the gap between train and validation is only 0.09. Fine-tuning the last block, 8.4 million weights, does not help here: it overfits.");

// ================================================================= 8. the classical challenger, explained
pres.addSection({ title: "The answer" });
s = addSlide({ masterName: "CONTENT", sectionTitle: "The answer" });
s.addText("Our classical challenger: a random forest", { placeholder: "title" });
stepTag(s, 5);
{
  const y = 1.15, h = 1.45, mid = y + h / 2;
  paintingBox(s, 0.5, y, 1.3, h, "the same photo");
  arrow(s, 1.8, mid, "arrow 1");
  card(s, 2.1, y, 3.1, h, "box features");
  s.addText("102 hand-made numbers", { x: 2.2, y: y + 0.08, w: 2.9, h: 0.32, fontSize: 14, bold: true, color: C.text2, align: "center",
    margin: 0, isTextBox: true, objectName: "features title" });
  [["Colours: 82", "hue, saturation, brightness"], ["Brush strokes: 20", "direction and strength of the edges"]].forEach(([hd, body], i) => {
    const x = 2.22 + i * 1.46;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.45, w: 1.38, h: 0.88, fill: { color: C.background1 },
      line: { color: C.background1, width: 0 }, rectRadius: 0.06, objectName: `feature group ${i + 1}` });
    s.addText([{ text: hd, options: { bold: true, fontSize: 11, color: C.accent2, breakLine: true } },
      { text: body, options: { fontSize: 10, color: C.text1 } }],
    { x: x + 0.05, y: y + 0.45, w: 1.28, h: 0.88, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `feature text ${i + 1}` });
  });
  arrow(s, 5.2, mid, "arrow 2");
  diagramBox(s, 5.5, y, 2.1, h, "500 decision trees", "each one: a random draw of the paintings, 10 of the 102 numbers per question");
  arrow(s, 7.6, mid, "arrow 3");
  diagramBox(s, 7.9, y, 1.6, h, "They vote", "the style with the most votes wins", { fill: C.accent2, headColor: C.background1, color: C.background1, size: 12 });

  // One tree, as an example: yes/no questions down to a style.
  card(s, 0.5, 2.8, 5.3, 2.25, "tree card");
  s.addText("One tree, for example (the real questions use our 102 numbers)", { x: 0.65, y: 2.88, w: 5.0, h: 0.28, fontSize: 11,
    color: MUTED, margin: 0, isTextBox: true, objectName: "tree caption" });
  const node = (cx, cy, w, text, fill, color, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx - w / 2, y: cy, w, h: 0.36, fill: { color: fill }, line: { color: fill, width: 0 },
      rectRadius: 0.06, objectName: name });
    s.addText(text, { x: cx - w / 2, y: cy, w, h: 0.36, fontSize: 11, bold: true, color, align: "center", valign: "middle", margin: 0,
      isTextBox: true, objectName: `${name} text` });
  };
  const link = (x1, y1, x2, y2, label, name) => {
    s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: y1, w: Math.abs(x2 - x1), h: y2 - y1, flipH: x2 < x1,
      line: { color: MUTED, width: 1.25 }, objectName: name });
    s.addText(label, { x: (x1 + x2) / 2 + (x2 < x1 ? -0.42 : 0.07), y: (y1 + y2) / 2 - 0.13, w: 0.35, h: 0.22, fontSize: 9,
      color: MUTED, align: x2 < x1 ? "right" : "left", margin: 0, isTextBox: true, objectName: `${name} label` });
  };
  const rootX = 3.15, l2 = [1.85, 4.45], leaves = [[1.25, "Cubism"], [2.45, "Baroque"], [3.85, "Pop_Art"], [5.05, "Impressionism"]];
  link(rootX, 3.58, l2[0], 3.95, "yes", "edge root yes"); link(rootX, 3.58, l2[1], 3.95, "no", "edge root no");
  link(l2[0], 4.31, leaves[0][0], 4.6, "yes", "edge left yes"); link(l2[0], 4.31, leaves[1][0], 4.6, "no", "edge left no");
  link(l2[1], 4.31, leaves[2][0], 4.6, "yes", "edge right yes"); link(l2[1], 4.31, leaves[3][0], 4.6, "no", "edge right no");
  node(rootX, 3.22, 2.0, "Many dark colours?", C.text2, C.background1, "node root");
  node(l2[0], 3.95, 1.9, "Many straight edges?", C.text2, C.background1, "node left");
  node(l2[1], 3.95, 1.9, "Vivid flat colours?", C.text2, C.background1, "node right");
  leaves.forEach(([cx, style]) => node(cx, 4.6, 1.1, style.replace("_", " "), STYLE_COLORS[style], C.background1, `leaf ${style}`));

  const rfBest = Math.max(...Object.values(d06.classical_tuning[RF]));
  card(s, 6.0, 2.8, 3.5, 2.25, "training card");
  s.addText("How we trained it", { x: 6.18, y: 2.9, w: 3.15, h: 0.32, fontSize: 14, bold: true, color: C.text2, margin: 0,
    isTextBox: true, objectName: "training title" });
  s.addText(bullets([
    `scikit-learn, 500 trees, on the ${d01.split_counts.train.toLocaleString("en-US").replace(",", " ")} training paintings`,
    `4 settings tried, the best chosen on validation: ${pct(rfBest)}`,
    `Training ${speed[RF].training_seconds.toFixed(0)} s; ${Math.round(speed[RF].ms_1_thread)} ms per photo`,
  ], 12), { x: 6.18, y: 3.28, w: 3.15, h: 1.7, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "training bullets" });
}
s.addNotes("The classical challenger: 102 hand-made numbers per painting (colours and brush strokes), then a random forest of 500 decision trees that vote. Settings chosen on validation.");

// ================================================================= 9. the two final tests (notebooks 06 and 08)
s = addSlide({ masterName: "CONTENT", sectionTitle: "The answer" });
s.addText("Final tests: the fine-tuned ResNet18 wins", { placeholder: "title" });
stepTag(s, 5);
{
  const models = [[MLP, "Our MLP, pixels"], [RF, "Random forest, 102 numbers"], [FROZEN, "ResNet18 frozen + linear head"], [FT, "ResNet18, layer4 fine-tuned"]];
  s.addChart(pres.charts.BAR, [
    { name: `Test 1 (${d01.split_counts.test} paintings)`, labels: models.map((m) => m[1]), values: models.map((m) => T10(t2[m[0]]["test 1 acc"])) },
    { name: `Test 2 (${d08.data.n} new paintings, new painters)`, labels: models.map((m) => m[1]), values: models.map((m) => T10(t2[m[0]]["test 2 acc"])) },
  ], { x: 0.3, y: 1.05, w: 6.35, h: 4.05, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.quiet, HEX.neural],
    showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: FMT, dataLabelColor: "FFFFFF",
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1000, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", showTitle: true, title: "Test accuracy, mean of 3 seeds, each test used once", barGapWidthPct: 40, ...chartText,
    objectName: "chart final tests" });
  const nnOverRf = pairOf(FROZEN, RF, "test 1 + test 2 (440)");
  statCard(s, 6.85, 1.15, 2.65, 1.85, `+${(100 * nnOverRf.diff).toFixed(1)}`,
    `points for even the frozen ResNet over the random forest, 440 paintings (${(100 * nnOverRf.ci_low).toFixed(1)} to ${(100 * nnOverRf.ci_high).toFixed(1)})`,
    { strong: true, size: 36, captionSize: 11 });
  statCard(s, 6.85, 3.2, 2.65, 1.85, `+${(-100 * ftOverFrozen.diff).toFixed(1)}`,
    `points for fine-tuning over frozen, 440 paintings (${(-100 * ftOverFrozen.ci_high).toFixed(1)} to ${(-100 * ftOverFrozen.ci_low).toFixed(1)}): not luck`,
    { size: 36, captionSize: 11 });
}
s.addNotes("The two final tests, each used once. On the first 220 paintings, the pretrained models were at the top and the fine-tuned one was first, while it was second on validation. To check whether this was luck, we drew 220 new paintings by painters our models never saw: same ranking. On the 440 paintings, even the frozen model is 18 points above the random forest, and fine-tuning beats frozen by 5 points, between 2 and 8.");

// ================================================================= 10. cost on a kiosk
s = addSlide({ masterName: "CONTENT", sectionTitle: "The answer" });
s.addText("Fast and cheap enough for a kiosk", { placeholder: "title" });
stepTag(s, 6);
const sp = speed[FT]; // the model we recommend (same speed and size as the frozen one)
[[`${Math.round(sp.ms_1_thread)} ms`, "per photo on one CPU thread"], [`${Math.round(sp.ms_4_thread)} ms`, "per photo with 4 threads"],
  [`${Math.round(sp.size_mb)} MB`, "model file (11 M numbers)"], [`${(sp.training_seconds / 60).toFixed(1)} min`, "training on a laptop, no GPU"]]
  .forEach(([v, c], i) => statCard(s, 0.5 + i * 2.3, 1.2, 2.1, 2.0, v, c, { strong: i === 0, size: 32 }));
noteCard(s, 0.5, 3.45, 9, 1.25, "In practice",
  `A kiosk 4 times slower still answers in about 0.2 s. Random forest: ${Math.round(speed[RF].ms_1_thread)} ms and ${Math.round(speed[RF].size_mb)} MB, but ${pct(test[RF].test_acc)}. When the collection changes: retrain the last block and the head, in a few minutes.`);
s.addNotes("Is it affordable on a kiosk? We timed one photo at a time, from the image to the answer. The fine-tuned ResNet takes 53 milliseconds on one CPU thread and 27 with four threads, like the frozen one. Even on a kiosk four times slower, that is a fifth of a second. The model file is 45 megabytes, and training takes about 2.5 minutes on a laptop, without a GPU.");

// ================================================================= 11. recommendation
s = addSlide({ masterName: "CLOSING", sectionTitle: "The answer" });
s.addText("Our choice: a fine-tuned ResNet18", { placeholder: "title" });
s.addText(`Both tests, ${d01.split_counts.test + d08.data.n} new paintings:  our MLP ${pct(both[MLP])}  →  random forest ${pct(both[RF])}  →  fine-tuned ResNet18 ${pct(both[FT])}`,
  { x: 0.5, y: 1.15, w: 9, h: 0.4, fontSize: 14, color: MUTED_DARK, align: "center", margin: 0, isTextBox: true, objectName: "journey" });
const cols = [
  ["Why", C.accent3, [`1 visitor in ${Math.round(1 / (1 - both[FT]))} gets a wrong style, against 1 in ${Math.round(1 / (1 - both[RF]))} with the best classical model`, "53 ms per photo on a modest CPU", "Retrained in a few minutes when the collection changes"]],
  ["Before launch", C.accent4, ["Test a few hundred real kiosk photos", "Add more styles and paintings", "Measure the real kiosk hardware"]],
];
cols.forEach(([headText, color, items], i) => {
  const x = 0.5 + i * 4.65;
  card(s, x, 1.75, 4.35, 3.15, `column ${headText}`, CARD_DARK);
  s.addText(headText, { x: x + 0.25, y: 1.9, w: 3.85, h: 0.45, fontSize: 22, bold: true, fontFace: THEME.headFontFace, color,
    margin: 0, isTextBox: true, objectName: `head ${headText}` });
  s.addText(bullets(items, 16), { x: x + 0.25, y: 2.5, w: 3.85, h: 2.3, valign: "top", margin: 0, color: C.background1,
    isTextBox: true, objectName: `items ${headText}` });
});
s.addNotes("So our recommendation: deploy a ResNet18 pretrained on ImageNet, with its last block fine-tuned on our paintings. It is the accurate choice: one visitor in eight gets a wrong style, against one in three with the best classical model. It is fast enough, and cheap to maintain for a small team. Before the launch, we must test real photos taken at the kiosks, add more styles, and measure the real hardware. Thank you, we are happy to take your questions.");

// ================================================================= backup slides (for the questions)
pres.addSection({ title: "Backup" });
// The classical challenger of the final test, explained (notebook 06, src/museum_guide/baselines.py).
s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: classical models, all the scores", { placeholder: "title" });
noteCard(s, 0.5, 1.15, 4.4, 1.75, "Why hand-made numbers?",
  `A question on one pixel means nothing: move the painting a little and it changes. Logistic regression: ${pct(test["logistic regression on pixels"].test_acc)} on pixels, ${pct(test["logistic regression on hand-made features"].test_acc)} on our 102 numbers.`);
noteCard(s, 5.1, 1.15, 4.4, 1.75, "The numbers matter more than the model",
  `The same logistic regression: ${pct(test["logistic regression on hand-made features"].test_acc)} on our 102 numbers, ${pct(test["logistic regression on ResNet features"].test_acc)} on the 512 numbers of the ResNet.`);
[["logistic regression on pixels", "Logistic regression on pixels"],
  ["logistic regression on hand-made features", "Logistic regression on the 102 numbers"],
  [RF, "Random forest on the 102 numbers"],
  [FT, "ResNet18 fine-tuned (our choice)"]].forEach(([model, caption], i) =>
  statCard(s, 0.5 + i * 2.3167, 3.1, 2.05, 1.6, pct(test[model].test_acc), caption,
    { strong: model === FT, color: model === RF ? HEX.classical : undefined, size: 30, captionSize: 12 }));
const rfBestVal = Math.max(...Object.values(d06.classical_tuning[RF]));
s.addText(`Test accuracy, ${d01.split_counts.test} paintings. Random forest: best of 4 settings on validation (${pct(rfBestVal)}), training ${speed[RF].training_seconds.toFixed(0)} s, ${Math.round(speed[RF].ms_1_thread)} ms per photo, ${Math.round(speed[RF].size_mb)} MB.`,
  { x: 0.5, y: 4.78, w: 9, h: 0.35, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true, objectName: "forest caption" });
s.addNotes(`If asked about the random forest: it is our best model without any neural network, the "classical model" of the director's question. A classical model cannot read raw pixels well, so we describe each painting with 102 hand-made numbers: its colours and the directions of its brush strokes. A random forest is 500 decision trees: each one learns from a random draw of the paintings and sees a random part of the numbers at each question, and they vote. Settings chosen on validation. On the first test: ${pct(test[RF].test_acc)}, ${(100 * (test[FT].test_acc - test[RF].test_acc)).toFixed(1)} points below the fine-tuned ResNet18. The same logistic regression goes from ${pct(test["logistic regression on hand-made features"].test_acc)} on our numbers to ${pct(test["logistic regression on ResNet features"].test_acc)} on the ResNet features: the features matter more than the model on top.`);

// Painter shortcut (notebook 05): out of the talk, kept for the questions on the split by painter.
s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: seeing the painter inflates the score", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Random folds: the painter was seen in training", labels: ["Known painters (536 images)", "Unknown Artist (744 images, control)"],
    values: [T10(leakMean("random folds", "known painters")), T10(leakMean("random folds", "unknown painters"))] },
  { name: "Painter folds: the painter was never seen", labels: ["Known painters (536 images)", "Unknown Artist (744 images, control)"],
    values: [T10(leakMean("painter folds", "known painters")), T10(leakMean("painter folds", "unknown painters"))] },
], { x: 0.4, y: 1.1, w: 5.6, h: 3.95, barDir: "col", barGrouping: "clustered", chartColors: [HEX.neural, HEX.classical],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: FMT, dataLabelColor: HEX.axis,
  valAxisHidden: true, valAxisMinVal: 500, valAxisMaxVal: 950, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "t", showTitle: false, barGapWidthPct: 60, ...chartText, objectName: "chart leakage" });
statCard(s, 6.3, 1.15, 3.2, 1.75, `+${(100 * (leakMean("random folds", "known painters") - leakMean("painter folds", "known painters"))).toFixed(1)}`,
  "points when the painter was seen in training", { strong: true, size: 40 });
s.addText(bullets([
  "Same images, same model: 5-fold cross-validation on train + validation",
  "Control group (unknown painters): no change",
  "So we split by painter: a random split would flatter the score",
], 14), { x: 6.3, y: 3.1, w: 3.2, h: 1.95, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "leakage bullets" });
s.addNotes("If asked why we split by painter: we scored the same images twice with cross-validation, changing only one thing: whether the model saw other paintings by the same painter. On images by known painters, seeing the painter gives 83.7 percent instead of 72.3: 11 points of flattery. On the control group, unknown painters, nothing changes. So a random split would have given the director a score that is too optimistic.");

s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: validation and the two tests", { placeholder: "title" });
s.addTable([
  [head("Model"), head("Validation (219)"), head("Test 1 (220)"), head("Test 2 (220)"), head("Both tests (440)")],
  ["ResNet18 frozen + linear head", pct(heads["linear probe"].val_acc), pct(t2[FROZEN]["test 1 acc"]), pct(t2[FROZEN]["test 2 acc"]), pct(both[FROZEN])],
  ["ResNet18, layer4 fine-tuned (our choice)", pct(ftMean), pct(t2[FT]["test 1 acc"]), pct(t2[FT]["test 2 acc"]), pct(both[FT])],
], { x: 0.5, y: 1.2, w: 9, colW: [3.3, 1.45, 1.4, 1.4, 1.45], fontSize: 14, color: C.text1, valign: "middle", margin: 0.08,
  border: { type: "solid", pt: 0.75, color: "D9DCE2" }, rowH: [0.45, 0.55, 0.55], objectName: "disagreement table" });
s.addText(bullets([
  `Validation favoured frozen; two independent tests favour fine-tuning: +${(-100 * ftOverFrozen.diff).toFixed(1)} points on 440 paintings (interval ${(-100 * ftOverFrozen.ci_high).toFixed(1)} to ${(-100 * ftOverFrozen.ci_low).toFixed(1)})`,
  "219 validation paintings were too few to separate them: next time, choose with cross-validation on train + validation",
  "The tests helped this choice, so the fine-tuned scores are slightly optimistic; a third set of new paintings would give the clean score",
], 15), { x: 0.5, y: 2.95, w: 9, h: 2.1, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "disagreement bullets" });
s.addNotes("If asked why we changed our choice after the test: validation chose the frozen model, the first test said the opposite, so we drew a second test of 220 new paintings to check, used once. It confirms the fine-tuned model, by 5 points on the 440 test paintings, so it was not luck. Because both tests helped this choice, the fine-tuned scores are slightly optimistic; with more data, we would choose with cross-validation and keep a third set for the final score.");

s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: why you can trust the numbers", { placeholder: "title" });
[["Reproducible", "Fixed seeds and a pinned dataset revision. Notebooks 06 and 08 train the models again and get the same scores, to 4 decimals."],
  ["Tested", "33 automatic tests: split without painter leakage, parameter counts, training loop, metrics, benchmark helpers."],
  ["Versioned", "Small commits, one step at a time, on GitHub; milestone tags m1, m2, m3 and v1.0. Every number is saved in reports/results/*.json."],
  ["Honest", "Two test sets, each used once; 3 seeds for noisy runs; 95 % bootstrap intervals; no image of the paintings in Git (licence)."]]
  .forEach(([h, t], i) => noteCard(s, 0.5 + (i % 2) * 4.6, 1.2 + Math.floor(i / 2) * 1.95, 4.4, 1.75, h, t, { size: 14 }));
s.addNotes("If asked how the work can be checked: everything runs again from the repository with two commands, and the final notebook re-trains every neural model and finds exactly the same validation scores.");

s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: the training recipes", { placeholder: "title" });
s.addTable([
  [head("Model"), head("Input"), head("Trained weights"), head("Settings")],
  ["Regularised MLP", "32 x 32 pixels", "1.7 M", "512-256 hidden, batch norm, dropout 0.5 (0.2 input), AdamW lr 0.0001, wd 5, flips, cosine, 60 epochs, early stopping 8"],
  ["Linear probe", "512 ResNet18 features", "2 565", "AdamW lr 0.001, wd 0.01, early stopping 20"],
  ["Fine-tuned ResNet18", "224 x 224 pixels", "8.4 M", "layer4 lr 0.0001, head lr 0.001, wd 0.01, random crop + flip, cosine, 10 epochs, early stopping 3"],
  ["Random forest", "102 hand-made features", "500 trees", "chosen on validation: min 1 sample per leaf, sqrt features"],
], { x: 0.5, y: 1.15, w: 9, colW: [1.9, 1.9, 1.4, 3.8], fontSize: 12, color: C.text1, valign: "middle", margin: 0.07,
  border: { type: "solid", pt: 0.75, color: "D9DCE2" }, objectName: "recipes table" });
s.addNotes("If asked about hyperparameters: all of them were chosen on the validation set (learning-rate sweep, 3 seeds per setting), and they are written once in src/museum_guide/recipes.py.");

// Training doctor (notebook 04): out of the talk, kept for the questions on monitoring.
s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: four broken runs, four signatures", { placeholder: "title" });
s.addTable([
  [head("Bug we caused"), head("Symptom we measured"), head("Cause"), head("Cure → result")],
  ["Learning rate too big", "loss stuck at ln 5; 99.6 % of neurons dead in layer 2", "steps push every ReLU below 0", "lr 0.0001 → 43.8 %"],
  ["zero_grad() forgotten", "loss jumps to 4.9, then one class only", "gradients pile up: x26 in 2 epochs", "zero_grad() → 41.1 %"],
  ["Raw pixels (0 to 255)", "first loss 16.3 instead of 1.6", "inputs dozens of times too big", "standardise → 43.8 %"],
  ["8 sigmoid layers", "loss stuck at ln 5", "vanishing gradient: layer 1 gets 6e-7 of layer 9", "ReLU + He init → 39.3 %"],
], { x: 0.5, y: 1.15, w: 9, colW: [1.85, 2.75, 2.45, 1.95], fontSize: 13, color: C.text1, valign: "middle", margin: 0.07,
  border: { type: "solid", pt: 0.75, color: "D9DCE2" }, rowH: [0.42, 0.62, 0.62, 0.62, 0.62], objectName: "doctor table" });
noteCard(s, 0.5, 4.15, 9, 0.9, "Lesson", "Same symptom (loss stuck at ln 5), two causes: only the monitor (dead neurons, gradient per layer) tells them apart.",
  { headSize: 14, size: 14 });
s.addNotes(`If asked whether our monitoring tools really detect problems: in notebook 04 we broke the MLP training four times on purpose, and each bug left its own signature. Two of them give the same loss stuck at ln 5; only the dead neurons or the gradient per layer tell them apart. For the ResNet, two silent bugs: forgetting model.eval() gives ${pct(d05.extraction_bugs["forgot model.eval()"])}, forgetting the ImageNet normalisation ${pct(d05.extraction_bugs["forgot the ImageNet normalisation"])}.`);

s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: where the chosen model fails", { placeholder: "title" });
const pc = [...d08.finetuned_seed0_per_class_both_tests].sort((a, b) => a.recall - b.recall); // best style at the top
const styleName = (r) => r.style.replace("_", " ").replace("Ukiyo e", "Ukiyo-e");
s.addChart(pres.charts.BAR, [{ name: "recall", labels: pc.map(styleName), values: pc.map((r) => T10(r.recall)) }],
  { x: 0.4, y: 1.1, w: 5.4, h: 3.95, barDir: "bar", chartColors: [HEX.neural], showValue: true, dataLabelPosition: "inEnd",
    dataLabelFormatCode: FMT, dataLabelColor: "FFFFFF", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1000,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, showTitle: true,
    title: "Fine-tuned ResNet18: share of each style found, both tests (seed 0)", barGapWidthPct: 45, ...chartText, objectName: "chart per class" });
s.addText(bullets([
  `Easiest: ${styleName(pc[4])} and ${styleName(pc[3])}`,
  `Hardest: ${styleName(pc[0])} (${pct(pc[0].recall, 0)}) and ${styleName(pc[1])}`,
  "On validation, Cubism and Pop Art were the pair most often confused: flat strong colours, hard edges",
], 14), { x: 6.05, y: 1.2, w: 3.45, h: 3.85, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "per class bullets" });
s.addNotes(`If asked about errors: on the 440 test paintings, the fine-tuned ResNet finds every Ukiyo-e painting and at least ${pct(pc[0].recall, 0)} of every style; the hardest are ${styleName(pc[0])} and ${styleName(pc[1])}.`);

// Bonus backup slide: transformers (notebook 07), only when its results exist.
const T_FILE = path.join(ROOT, "reports", "results", "07_transformers.json");
if (fs.existsSync(T_FILE)) {
  const d07 = JSON.parse(fs.readFileSync(T_FILE, "utf8"));
  const pt = Object.fromEntries(d07.probe_table.map((r) => [r.model, r]));
  const sp7 = Object.fromEntries(d07.speed.map((r) => [r.model, r]));
  s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: what about transformers (BERT)?", { placeholder: "title" });
  // Times as a ratio to ResNet18 measured in the same run (absolute times move by about 30 % between runs).
  const ratio = (model) => sp7[model].ms_1_thread / sp7["ResNet18 + linear head"].ms_1_thread;
  const row = (name, acc, model) => [name, acc, `x ${ratio(model).toFixed(1)}`, `${Math.round(sp7[model].parameters_millions)} M`];
  s.addTable([
    [head("Frozen network (validation, 219)"), head("Validation"), head("Time per photo vs ResNet18"), head("Parameters")],
    row("ResNet18 + linear head (our choice)", pct(pt["ResNet18 frozen + linear head"].val_acc), "ResNet18 + linear head"),
    row("ViT-B/16 (BERT on image patches) + linear head", pct(pt["ViT-B/16 frozen + linear head"].val_acc), "ViT-B/16 + linear head"),
    row("CLIP ViT-B/32, zero-shot (no training at all)", pct(d07.clip_zero_shot.val_acc), "CLIP ViT-B/32 + linear head"),
    row("CLIP ViT-B/32 + linear head", pct(pt["CLIP ViT-B/32 frozen + linear head"].val_acc), "CLIP ViT-B/32 + linear head"),
  ], { x: 0.5, y: 1.15, w: 9, colW: [4.3, 1.4, 2.0, 1.3], fontSize: 13, color: C.text1, valign: "middle", margin: 0.07,
    border: { type: "solid", pt: 0.75, color: "D9DCE2" }, objectName: "transformers table" });
  s.addText(bullets([
    "BERT itself reads words, not pixels; its image cousins are ViT and CLIP",
    `CLIP is the most accurate (+6.8 points over ResNet18, paired interval 2.3 to 11.4) but ${ratio("CLIP ViT-B/32 + linear head").toFixed(1)} times slower; trained on Internet images, it has probably seen WikiArt`,
    "Validation only, the test stays closed; CLIP is our first next step, and it can also answer questions in words",
  ], 14), { x: 0.5, y: 3.55, w: 9, h: 1.55, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "transformers bullets" });
  s.addNotes("If asked about BERT or transformers: BERT reads text, so it cannot classify paintings alone. Its image cousin ViT gives the same accuracy as ResNet18 but is about 9 times slower. CLIP, trained on images with captions, reaches about 89 percent with no training at all and about 95 percent with our linear head, on validation. Two cautions: it probably saw WikiArt during its training, and we did not reopen the test set. It is our first next step, and it would also answer visitors' questions in words.");
}

// Inside ResNet18 + our linear head: the stages with their output size and weights (torchvision resnet18).
s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: inside ResNet18 + our linear head", { placeholder: "title" });
{
  const stages = [ // [name, output shape, weights, output side in pixels, role]
    ["1 painting", "224 x 224 x 3", "", 224, "input"],
    ["conv1 + pool", "56 x 56 x 64", "9.5 k weights", 56, "body"],
    ["layer1", "56 x 56 x 64", "148 k weights", 56, "body"],
    ["layer2", "28 x 28 x 128", "526 k weights", 28, "body"],
    ["layer3", "14 x 14 x 256", "2.1 M weights", 14, "body"],
    ["layer4", "7 x 7 x 512", "8.4 M weights", 7, "body"],
    ["average", "512 numbers", "no weights", 1, "body"],
    ["linear head", "512 → 5", `${heads["linear probe"].trained_parameters.toLocaleString("en-US").replace(",", " ")} weights`, 1, "head"],
    ["5 styles", "highest score wins", "", 1, "output"],
  ];
  const bw = 0.82, gap = 0.2, x0 = 0.51, mid = 2.05;
  const xs = stages.map((_, i) => x0 + i * (bw + gap));
  stages.forEach(([name, shape, weights, side, role], i) => {
    const h = 0.45 + 1.35 * Math.log(side) / Math.log(224), y = mid - h / 2, x = xs[i];
    const fill = role === "body" ? C.text2 : role === "head" ? C.accent1 : C.background2;
    const color = role === "body" || role === "head" ? C.background1 : C.text2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h, fill: { color: fill }, line: { color: fill, width: 0 },
      rectRadius: 0.06, objectName: `stage ${name}` });
    s.addText(name, { x, y, w: bw, h, fontSize: 10, bold: true, color, align: "center", valign: "middle", margin: 0.02,
      isTextBox: true, objectName: `stage name ${name}` });
    s.addText([{ text: shape, options: { breakLine: weights !== "" } }, ...(weights ? [{ text: weights }] : [])],
      { x: x - 0.08, y: 3.03, w: bw + 0.16, h: 0.42, fontSize: 9, color: C.text1, align: "center", valign: "top", margin: 0,
        isTextBox: true, objectName: `stage caption ${name}` });
    if (i < stages.length - 1) {
      s.addShape(pres.shapes.LINE, { x: x + bw + 0.03, y: mid, w: gap - 0.06, h: 0, line: { color: HEX.axis, width: 1.5, endArrowType: "triangle" },
        objectName: `stage arrow ${i + 1}` });
    }
  });
  // What the stages learned to see on ImageNet, from small details to shapes.
  [[1, 2, "edges, colours"], [3, 3, "textures"], [4, 4, "patterns"], [5, 5, "shapes"]].forEach(([a, b, what]) =>
    s.addText(what, { x: xs[a] - 0.05, y: 1.08, w: xs[b] + bw - xs[a] + 0.1, h: 0.2, fontSize: 9, italic: true, color: C.accent1,
      align: "center", margin: 0, isTextBox: true, objectName: `sees ${what}` }));
  // Frozen body and trained head.
  const bar = (from, to, color, label, name) => {
    s.addShape(pres.shapes.RECTANGLE, { x: xs[from], y: 3.52, w: xs[to] + bw - xs[from], h: 0.05, fill: { color }, line: { color, width: 0 },
      objectName: `${name} bar` });
    s.addText(label, { x: xs[from], y: 3.6, w: xs[to] + bw - xs[from], h: 0.22, fontSize: 10, bold: true, color, align: "center",
      margin: 0, isTextBox: true, objectName: `${name} label` });
  };
  bar(1, 4, C.text2, "Frozen: 2.8 million weights learned on ImageNet", "frozen");
  bar(5, 5, C.accent4, "Fine-tuned", "fine-tuned");
  bar(7, 7, C.accent1, "Trained by us", "trained");

  // One residual block: two 3 x 3 convolutions, and a shortcut that adds the input back.
  s.addText("One residual block (2 per layer, 8 in all)", { x: 0.5, y: 3.93, w: 3.6, h: 0.2, fontSize: 11, bold: true,
    color: C.text2, margin: 0, isTextBox: true, objectName: "block title" });
  const flow = [["x", 0.3, "label"], ["conv 3x3", 0.8, "op"], ["BN + ReLU", 0.9, "op"], ["conv 3x3", 0.8, "op"], ["BN", 0.5, "op"],
    ["+", 0.34, "add"], ["ReLU", 0.6, "op"], ["out", 0.35, "label"]];
  const fy = 4.62, fh = 0.34, fgap = 0.17;
  let fx = 0.5;
  const centres = [];
  flow.forEach(([label, w, kind], i) => {
    centres.push(fx + w / 2);
    if (kind === "op") {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: fx, y: fy, w, h: fh, fill: { color: C.text2 }, line: { color: C.text2, width: 0 },
        rectRadius: 0.05, objectName: `block ${i} ${label}` });
    } else if (kind === "add") {
      s.addShape(pres.shapes.OVAL, { x: fx, y: fy, w, h: fh, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 },
        objectName: "block add" });
    }
    s.addText(label, { x: fx, y: fy, w, h: fh, fontSize: kind === "add" ? 14 : 10, bold: true,
      color: kind === "label" ? C.text2 : C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true,
      objectName: `block text ${i} ${label}` });
    if (i < flow.length - 1) {
      s.addShape(pres.shapes.LINE, { x: fx + w + 0.02, y: fy + fh / 2, w: fgap - 0.04, h: 0,
        line: { color: HEX.axis, width: 1.25, endArrowType: "triangle" }, objectName: `block arrow ${i + 1}` });
    }
    fx += w + fgap;
  });
  const [cx, cplus] = [centres[0], centres[5]], top = 4.33;
  s.addShape(pres.shapes.LINE, { x: cx, y: top, w: 0, h: fy - top, line: { color: C.accent1, width: 1.5 }, objectName: "shortcut up" });
  s.addShape(pres.shapes.LINE, { x: cx, y: top, w: cplus - cx, h: 0, line: { color: C.accent1, width: 1.5 }, objectName: "shortcut across" });
  s.addShape(pres.shapes.LINE, { x: cplus, y: top, w: 0, h: fy - top, line: { color: C.accent1, width: 1.5, endArrowType: "triangle" },
    objectName: "shortcut down" });
  s.addText("shortcut: the input x is added back", { x: cx, y: top - 0.22, w: cplus - cx, h: 0.2, fontSize: 9, italic: true,
    color: C.accent1, align: "center", margin: 0, isTextBox: true, objectName: "shortcut label" });
  s.addText([
    { text: "Why \"Res\" (residual): ", options: { bold: true, color: C.text2 } },
    { text: "each block only learns a correction added to its input. The gradient flows back through the shortcuts, so all the layers learn. 18 layers = conv1 + 16 convolutions in the blocks + the last layer." },
  ], { x: 6.45, y: 3.93, w: 3.05, h: 1.15, fontSize: 11, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "residual text" });
}
s.addNotes("If asked how ResNet18 is built: the painting (224 by 224 pixels, 3 colours) goes through a first convolution and four layers of residual blocks. At each layer the picture gets smaller (56, 28, 14, 7 pixels) but the number of channels grows (64 to 512): from edges and colours to textures, patterns and shapes. The average of the last 512 maps gives 512 numbers. In our final model, conv1 to layer3 stay frozen (2.8 million weights learned on ImageNet), layer4 is fine-tuned on our paintings (8.4 million), and our linear head turns the 512 numbers into 5 scores, one per style (2 565 weights). In the frozen version, layer4 stays frozen too. Residual means each block adds its input back through a shortcut, so the gradient reaches every layer.");

// Why pretrained? The same ResNet18 trained from zero (scripts/resnet_from_zero.py), validation only.
const ZERO_FILE = path.join(ROOT, "reports", "results", "resnet_from_zero.json");
if (fs.existsSync(ZERO_FILE)) {
  const zero = JSON.parse(fs.readFileSync(ZERO_FILE, "utf8"));
  s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: why pretrained? ResNet18 from zero", { placeholder: "title" });
  const zLabels = ["Our best MLP (pixels)", "ResNet18 from zero", "Random forest, 102 numbers", "ResNet18, layer4 fine-tuned", "ResNet18 frozen + linear head"];
  const zValues = [tools["all together"].val_acc, zero.best_val_acc, Math.max(...Object.values(d06.classical_tuning[RF])), ftMean, heads["linear probe"].val_acc];
  const pick = (k) => zValues.map((v, i) => (i === k ? T10(v) : 0));
  s.addChart(pres.charts.BAR, [
    { name: "other", labels: zLabels, values: zValues.map((v, i) => (i === 1 || i === 4 ? 0 : T10(v))) },
    { name: "from zero", labels: zLabels, values: pick(1) },
    { name: "chosen", labels: zLabels, values: pick(4) },
  ], { x: 0.4, y: 1.1, w: 5.4, h: 3.95, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.quiet, "EDA100", HEX.neural],
    showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: FMT, dataLabelColor: "FFFFFF",
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1000, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: false, showTitle: true, title: "Validation accuracy (219 paintings)", barGapWidthPct: 45, ...chartText,
    objectName: "chart from zero" });
  statCard(s, 6.05, 1.15, 3.45, 1.55, `-${(100 * (heads["linear probe"].val_acc - zero.best_val_acc)).toFixed(1)}`,
    "points without the pretraining: same network, same paintings", { strong: true, size: 36, captionSize: 12 });
  noteCard(s, 6.05, 2.85, 3.45, 1.05, "How we trained it",
    `Random weights, every layer learns (11.2 M). Crop + flip, AdamW, ${zero.settings.epochs} epochs, ${Math.round(zero.minutes)} min. One seed.`,
    { headSize: 13, size: 11 });
  noteCard(s, 6.05, 4.0, 3.45, 1.05, "What it shows",
    "Better than our MLP, but only like the random forest: the 1.2 million ImageNet photos make the difference.",
    { headSize: 13, size: 11 });
  s.addNotes(`If asked why we did not train ResNet18 from zero: we measured it. With random weights and every layer learning, the same network reaches only ${pct(zero.best_val_acc)} on validation at best, against ${pct(heads["linear probe"].val_acc)} for the frozen pretrained one: about the level of the random forest. It is better than our MLP, because convolutions see shapes, but 1 061 paintings are far too few to learn 11 million weights. It also took ${Math.round(zero.minutes)} minutes of CPU, against less than a second for our linear head. One seed, validation only.`);
}

// pptxgenjs writes the notes of a slide as one run of text. The transition sentence of the oral script
// comes after a blank line: make it a paragraph of its own, in bold, so it stands out while presenting.
async function splitTransitionNotes(file) {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    const split = xml.replace(/(?:\r?\n){2}(TRANSITION: )/g,
      '</a:t></a:r></a:p><a:p><a:pPr><a:spcBef><a:spcPts val="900"/></a:spcBef></a:pPr><a:r><a:rPr lang="en-US" b="1" dirty="0"/><a:t>$1');
    if (split !== xml) zip.file(name, split);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  await splitTransitionNotes(OUT);
  console.log(`written ${OUT}`);
});
