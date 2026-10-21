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
s.addText("Otherwise a model can recognise the painter instead of the style (measured later: +11 points).",
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

// ================================================================= 6. ResNet
pres.addSection({ title: "Pretrained ResNet" });
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
noteCard(s, 5.55, 1.15, 3.95, 1.2, "Frozen features",
  `Each painting → 512 numbers (${d05.feature_ms_per_image_batch64.toFixed(1)} ms). Only one linear layer is trained: 2 565 weights, ${heads["linear probe"].seconds.toFixed(1)} s.`);
noteCard(s, 5.55, 2.5, 3.95, 1.2, "Fine-tuning layer4",
  `8.4 M weights for 1 061 images: 99.9 % on train, ${pct(ftMean)} on validation. It overfits.`);
noteCard(s, 5.55, 3.85, 3.95, 1.2, "Monitored like the MLP",
  `Train and validation curves for every run. Frozen + linear head: gap of only ${heads["linear probe"].gap.toFixed(2)} (our first MLP: ${tools["baseline (lr 0.0001)"].gap.toFixed(2)}).`);
s.addNotes("A network that can see shapes is a convolutional network. With 1 061 images we cannot train one from zero, so we reuse ResNet18, trained on 1.2 million ImageNet photos. We freeze it: each painting becomes 512 numbers, and we train only one linear layer, 2 565 weights, in 0.4 seconds. Result: 88.1 percent on validation, almost twice our best MLP. We monitored it like the MLP: the gap between train and validation is only 0.09. Fine-tuning the last block, 8.4 million weights, does not help here: it overfits.");

// ================================================================= 7. painter leakage
s = addSlide({ masterName: "CONTENT", sectionTitle: "Pretrained ResNet" });
s.addText("Seeing the painter inflates the score", { placeholder: "title" });
stepTag(s, 4);
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
s.addNotes("Now the painter rule from the beginning. To measure it fairly, we scored the same images twice with cross-validation, changing only one thing: whether the model saw other paintings by the same painter. On images by known painters, seeing the painter gives 83.7 percent instead of 72.3: 11 points of pure flattery. On the control group, unknown painters, nothing changes. So the model partly recognises painters, and a random split would have given the director a score that is too optimistic. Our split by painter avoids it.");

// ================================================================= 8. final test
pres.addSection({ title: "The answer" });
s = addSlide({ masterName: "CONTENT", sectionTitle: "The answer" });
s.addText("Final test: the pretrained network wins", { placeholder: "title" });
stepTag(s, 5);
const order = [["always the most frequent style", "Always the same style", "baseline"],
  ["logistic regression on pixels", "Logistic regression, pixels", "classical"],
  [MLP, "Our MLP, pixels", "neural"],
  ["logistic regression on hand-made features", "Logistic regression, hand-made", "classical"],
  [RF, "Random forest, hand-made", "classical"],
  [FROZEN, "ResNet18 frozen + linear head (ours)", "neural"],
  ["logistic regression on ResNet features", "ResNet18 frozen + logistic regression", "hybrid"],
  [FT, "ResNet18, layer4 fine-tuned", "neural"]];
const fams = [["baseline", "Baseline", HEX.baseline], ["classical", "Classical", HEX.classical], ["hybrid", "Hybrid", HEX.hybrid], ["neural", "Neural", HEX.neural]];
s.addChart(pres.charts.BAR, fams.map(([fam, name]) => ({ name, labels: order.map((o) => o[1]),
  values: order.map((o) => (o[2] === fam ? T10(test[o[0]].test_acc) : 0)) })),
{ x: 0.3, y: 1.05, w: 6.35, h: 4.05, barDir: "bar", barGrouping: "stacked", chartColors: fams.map((f) => f[2]),
  showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: FMT, dataLabelColor: "FFFFFF",
  valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1000, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", showTitle: true, title: `Test accuracy, ${d01.split_counts.test} new paintings`, barGapWidthPct: 35,
  ...chartText, objectName: "chart final test" });
statCard(s, 6.85, 1.15, 2.65, 2.15, `+${(100 * paired.diff).toFixed(1)}`,
  `points over the best classical model (95 % interval ${(100 * paired.ci_low).toFixed(1)} to ${(100 * paired.ci_high).toFixed(1)})`, { strong: true, size: 40, captionSize: 13 });
noteCard(s, 6.85, 3.45, 2.65, 1.6, "Honest note",
  `Fine-tuned: 1st on test (${pct(test[FT].test_acc)}), 2nd on validation (${pct(ftMean)}). We keep the model chosen before the test.`,
  { headSize: 14, size: 12 });
s.addNotes("Finally, the test, measured once on 220 new paintings. The pretrained models are at the top, from 83 to 88 percent. The best classical model reaches 67 percent, and our MLP 41 percent. The gap between our chosen model and the random forest is 16 points, and a paired bootstrap on the same paintings gives an interval from 9 to 24 points: it is not luck. One honest note: the fine-tuned model is first on test, but it was second on validation. With 220 images a score moves by about 5 points, so we keep the model we chose before the test.");

// ================================================================= 9. cost on a kiosk
s = addSlide({ masterName: "CONTENT", sectionTitle: "The answer" });
s.addText("Fast and cheap enough for a kiosk", { placeholder: "title" });
stepTag(s, 6);
const sp = speed[FROZEN];
[[`${Math.round(sp.ms_1_thread)} ms`, "per photo on one CPU thread"], [`${Math.round(sp.ms_4_thread)} ms`, "per photo with 4 threads"],
  [`${Math.round(sp.size_mb)} MB`, "model file (11 M numbers)"], [`${Math.round(sp.training_seconds)} s`, "training on a laptop, no GPU"]]
  .forEach(([v, c], i) => statCard(s, 0.5 + i * 2.3, 1.2, 2.1, 2.0, v, c, { strong: i === 0, size: 36 }));
noteCard(s, 0.5, 3.45, 9, 1.25, "In practice",
  `A kiosk 4 times slower still answers in about 0.2 s. Random forest: ${Math.round(speed[RF].ms_1_thread)} ms and ${Math.round(speed[RF].size_mb)} MB, but ${pct(test[RF].test_acc)}. When the collection changes: compute the new features, retrain one layer, in seconds.`);
s.addNotes("Is it affordable on a kiosk? We timed one photo at a time, from the image to the answer. The ResNet takes 53 milliseconds on one CPU thread and 26 with four threads. Even on a kiosk four times slower, that is a fifth of a second. The model file is 45 megabytes, and training takes 17 seconds on a laptop, without a GPU. When the collection changes, we only compute the new features and retrain the last layer, in seconds.");

// ================================================================= 10. recommendation
s = addSlide({ masterName: "CLOSING", sectionTitle: "The answer" });
s.addText("Our choice: frozen ResNet18 + linear head", { placeholder: "title" });
s.addText(`Test, ${d01.split_counts.test} new paintings:  our MLP ${pct(test[MLP].test_acc)}  →  best classical ${pct(test[RF].test_acc)}  →  frozen ResNet18 ${pct(test[FROZEN].test_acc)}`,
  { x: 0.5, y: 1.15, w: 9, h: 0.4, fontSize: 14, color: MUTED_DARK, align: "center", margin: 0, isTextBox: true, objectName: "journey" });
const cols = [
  ["Why", C.accent3, ["1 visitor in 6 gets a wrong style, against 1 in 3 with the best classical model", "53 ms per photo on a modest CPU", "Retrained in seconds when the collection changes"]],
  ["Not now", C.accent2, ["Our MLP on pixels: 41 %", "A classical model alone: 61 to 67 %", "Fine-tuning: slower, not proven better"]],
  ["Before launch", C.accent4, ["Test a few hundred real kiosk photos", "Add more styles and paintings", "Measure the real kiosk hardware"]],
];
cols.forEach(([headText, color, items], i) => {
  const x = 0.5 + i * 3.1;
  card(s, x, 1.75, 2.8, 3.15, `column ${headText}`, CARD_DARK);
  s.addText(headText, { x: x + 0.2, y: 1.88, w: 2.4, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color,
    margin: 0, isTextBox: true, objectName: `head ${headText}` });
  s.addText(bullets(items, 14), { x: x + 0.2, y: 2.42, w: 2.45, h: 2.4, valign: "top", margin: 0, color: C.background1,
    isTextBox: true, objectName: `items ${headText}` });
});
s.addNotes("So our recommendation: deploy a frozen ResNet18 with a linear head. It is the accurate choice: one visitor in six gets a wrong style, against one in three with the best classical model. It is fast enough, and cheap to maintain for a small team. We do not recommend our MLP on pixels, a classical model alone, or fine-tuning for now. Before the launch, we must test real photos taken at the kiosks, add more styles, and measure the real hardware. Everything is reproducible in our Git repository. Thank you, we are happy to take your questions.");

// ================================================================= backup slides (for the questions)
pres.addSection({ title: "Backup" });
s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: validation and test disagree", { placeholder: "title" });
s.addTable([
  [head("Model"), head("Validation (219)"), head("Test (220)"), head("Test, 95 % interval")],
  ["ResNet18 frozen + linear head (chosen)", pct(heads["linear probe"].val_acc), pct(test[FROZEN].test_acc), `${pct(test[FROZEN].ci_low)} to ${pct(test[FROZEN].ci_high)}`],
  ["ResNet18, layer4 fine-tuned", pct(ftMean), pct(test[FT].test_acc), `${pct(test[FT].ci_low)} to ${pct(test[FT].ci_high)}`],
], { x: 0.5, y: 1.2, w: 9, colW: [3.6, 1.7, 1.5, 2.2], fontSize: 14, color: C.text1, valign: "middle", margin: 0.08,
  border: { type: "solid", pt: 0.75, color: "D9DCE2" }, rowH: [0.45, 0.55, 0.55], objectName: "disagreement table" });
s.addText(bullets([
  "With about 220 paintings per set, a score moves by about 5 points just by chance",
  "We chose the model on validation, before the test: choosing on the test would make its score too optimistic",
  "Both models have the same speed and size at inference; fine-tuning is the first thing to retry with more data",
], 15), { x: 0.5, y: 2.95, w: 9, h: 2.1, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "disagreement bullets" });
s.addNotes("If asked why we did not pick the fine-tuned model: the test set is used once to report, not to choose. Validation and test point in opposite directions, and their intervals overlap, so our data cannot separate the two variants.");

s = addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: why you can trust the numbers", { placeholder: "title" });
[["Reproducible", "Fixed seeds and a pinned dataset revision. Notebook 06 trains the 9 neural models again and gets the same validation scores, to 4 decimals."],
  ["Tested", "33 automatic tests: split without painter leakage, parameter counts, training loop, metrics, benchmark helpers."],
  ["Versioned", "Small commits, one step at a time, on GitHub; milestone tags m1, m2, m3 and v1.0. Every number is saved in reports/results/*.json."],
  ["Honest", "Test set used once; 3 seeds for noisy runs; 95 % bootstrap intervals; no image of the paintings in Git (licence)."]]
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
const pc = [...d06.best_model_test_per_class].sort((a, b) => a.recall - b.recall); // best style at the top
s.addChart(pres.charts.BAR, [{ name: "recall", labels: pc.map((r) => r.style.replace("_", " ").replace("Ukiyo e", "Ukiyo-e")), values: pc.map((r) => T10(r.recall)) }],
  { x: 0.4, y: 1.1, w: 5.4, h: 3.95, barDir: "bar", chartColors: [HEX.neural], showValue: true, dataLabelPosition: "inEnd",
    dataLabelFormatCode: FMT, dataLabelColor: "FFFFFF", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1000,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, showTitle: true,
    title: "Share of each style found, test set (seed 0)", barGapWidthPct: 45, ...chartText, objectName: "chart per class" });
s.addText(bullets([
  "Easiest: Ukiyo-e and Impressionism",
  "Hardest: Pop Art and Cubism; on validation they were mostly confused with each other (flat strong colours, hard edges)",
  "The MLP found only 7 % of Cubist paintings; the ResNet finds most of them",
], 14), { x: 6.05, y: 1.2, w: 3.45, h: 3.85, valign: "top", margin: 0, color: C.text1, isTextBox: true, objectName: "per class bullets" });
s.addNotes("If asked about errors: the ResNet's main confusion is Cubism versus Pop Art, which share flat areas of strong colour and hard edges.");

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
