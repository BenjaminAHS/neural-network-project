// Builds the oral script of the 10-minute defence: presentation/oral_script.docx (to read aloud) and
// presentation/speaker_notes.json (the notes of the 10 main slides, read by build_deck.js). Every number comes
// from reports/results/*.json. Run it before build_deck.js:
//   NODE_PATH=<folder with node_modules> node presentation/build_script.js
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, LevelFormat, Packer, PageNumber, Paragraph, ShadingType, TextRun,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const R = (name) => JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "results", `${name}.json`), "utf8"));
const WORDS_PER_MINUTE = 130;

// ---------------------------------------------------------------- numbers (read, not typed)
const d01 = R("01_data"), d02 = R("02_mlp"), d03 = R("03_monitoring"), d05 = R("05_resnet"), d06 = R("06_final");
const d08 = R("08_second_test"), d09 = R("09_big_model");
const RF = "random forest on hand-made features", MLP = "regularised MLP on pixels";
const FROZEN = "ResNet18 frozen + linear head", FT = "ResNet18 fine-tuned (layer4)", BIG = "CLIP ViT-L/14 + linear head";
const tools = Object.fromEntries(d03.tools.map((t) => [t.tool, t]));
const probe = d05.heads.find((h) => h.head === "linear probe");
const ftVal = d05.finetune_runs.reduce((a, r) => a + r.val_acc, 0) / d05.finetune_runs.length;
const rfVal = Math.max(...Object.values(d06.classical_tuning[RF]));
const bigVal = d09.head_runs.reduce((a, r) => a + r.val_acc, 0) / d09.head_runs.length;
const both = { ...d08.both_tests_acc, [BIG]: d09.tests["both acc"] };
const kiosk = Object.fromEntries(d09.kiosk.map((r) => [r.model, r]));
const cubism = (rows) => rows.find((r) => r.style === "Cubism").recall;
const ftOverFrozen = d08.paired.find((r) => r.A === FROZEN && r.B === FT && r.test === "test 1 + test 2 (440)" && r.seed === 0);
const nTests = d01.split_counts.test + d08.data.n;

// Spoken forms of the numbers.
const n = (x) => x.toLocaleString("en-US").replace(",", " ");
const p1 = (x) => (100 * x).toFixed(1);
const p0 = (x) => Math.round(100 * x).toString();
const WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
const oneIn = (acc) => WORD[Math.round(1 / (1 - acc))];
const seconds = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} seconds` : `${Math.round(ms)} milliseconds`);
const megabytes = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} gigabytes` : `${Math.round(mb)} megabytes`);
const k1 = kiosk[FT], k2 = kiosk[BIG];

// ---------------------------------------------------------------- the script
// A paragraph is a list of pieces: a string is read as it is, {b} is the sentence to remember (bold),
// {cue} is a stage cue (not read), {opt} can be skipped when late.
const SLIDES = [
  {
    title: "Is a neural network worth it on a museum kiosk?",
    paragraphs: [
      ["Good morning. Our museum wants kiosks that recognise the style of a painting from a photo."],
      ["We worked on five styles: Baroque, Cubism, Impressionism, Pop Art and Ukiyo-e. ", { cue: "point at the five paintings" }],
      ["The kiosks have small CPUs, the team is small and the budget is limited. So the director asks one question: ",
        { b: "is a neural network worth it in production, or would a simpler classical model do the job?" }],
    ],
    transition: ["To answer it, we did not jump straight to a big model.", "So now, here is our path: six steps."],
  },
  {
    title: "Six steps to answer the director",
    paragraphs: [
      [{ cue: "point at the six steps, one by one" }, " Step one: the data, and a fair test. Step two: a baseline, a classical model without any neural network. Step three: our own neural network, an MLP. Step four: a pretrained ResNet18. Step five: a much bigger model, CLIP. Step six: all of them, criterion by criterion."],
      [{ b: "Each model must beat the baseline, and fit on a kiosk." }],
    ],
    transition: ["", "So now, let's start with step one: the data, and how we made the test fair."],
  },
  {
    title: "A fair test : one painter, one part",
    paragraphs: [
      [`We took ${n(d01.n_images)} paintings from WikiArt, 300 per style.`],
      [`We cut them into three piles: ${n(d01.split_counts.train)} to train, ${d01.split_counts.val} to validate, that is, to choose our settings, and ${d01.split_counts.test} to test, only once, at the end.`],
      ["One rule matters a lot: all the paintings of a known painter go to the same pile. ", { cue: "point at the zero" },
        ` With a random split, ${d01.leakage.shared_painters_random_split} of our ${d01.leakage.known_painters} known painters would be in two piles, and a model could learn to recognise Monet instead of Impressionism. `,
        { b: "With our split: zero." }],
    ],
    transition: ["Our data is ready and our test is fair.", "So now, step two: the baseline, the score every model must beat."],
  },
  {
    title: "Our baseline : a classical random forest",
    paragraphs: [
      ["Our baseline uses no neural network. It cannot read raw pixels well, so we first describe each painting with 102 hand-made numbers: ",
        { cue: "point at the second box" }, " 82 for its colours, and 20 for the direction and strength of its brush strokes."],
      [{ cue: "point at the tree" }, " Then a random forest: 500 decision trees. Each tree asks yes or no questions on these numbers, like \"many dark colours?\", until it names a style. They vote, and the vote cancels many mistakes."],
      [{ cue: "point at the orange card" }, ` We chose its settings on validation: ${p1(rfVal)} percent. `, { b: "This is the score to beat." }],
    ],
    transition: [`We saw that the baseline reaches ${p1(rfVal)} percent.`, "So now, step three: can our own neural network beat it?"],
  },
  {
    title: "Our MLP : bad to generalize",
    paragraphs: [
      ["Our first neural network is the simplest one of the course: a multi-layer perceptron built from nn.Linear layers and ReLU."],
      ["Before trusting it, we ran two checks. First, a model that knows nothing should have a loss of ln 5, about 1.61. ",
        { opt: `Ours started at ${d02.first_loss["last_scale_1.0"].toFixed(2)}, too confident, so we made its last layer smaller and got ${d02.first_loss["last_scale_0.1"].toFixed(2)}.` },
        ` Second, it must be able to learn one small batch by heart: it does, ${p0(d02.overfit_one_batch.accuracy)} percent.`],
      [{ cue: "point at the curves" }, ` The training accuracy goes to ${p1(d02.baseline_64px.final_train_acc)} percent, but the validation accuracy stays around ${Math.round(100 * d02.baseline_64px.val_acc_at_best / 10) * 10} percent. `,
        { b: "This is overfitting: the network learns the paintings by heart." }],
    ],
    transition: ["We saw that our MLP is overfitting.", "So now, we look inside its training and try to fix it."],
  },
  {
    title: "Healthy training, but still below the baseline",
    paragraphs: [
      ["Before fixing it, we looked inside the network: the gradient reaches every layer and only 4 percent of the neurons are dead. The training is healthy: the problem really is overfitting."],
      ["Then we attacked it with the tools of session 6. ", { cue: "point at the chart" },
        ` Together, weight decay, dropout, batch norm, flips and early stopping take the MLP from ${p1(tools["baseline (lr 0.0001)"].val_acc)} to ${p1(tools["all together"].val_acc)} percent, and the gap from ${tools["baseline (lr 0.0001)"].gap.toFixed(2)} to ${tools["all together"].gap.toFixed(2)}.`],
      [{ cue: "point at the orange bar" }, ` But it stays far below the baseline: ${p1(tools["all together"].val_acc)} against ${p1(rfVal)}. `,
        { b: `It finds only ${p0(cubism(d03.regularised_seed0_per_class))} percent of the Cubist paintings: it sees colours, not shapes.` }],
    ],
    transition: ["Our MLP stays below the baseline.", "So now, step four: a model that sees shapes, and that is fast."],
  },
  {
    title: "ResNet18: a network made for images",
    paragraphs: [
      ["To see shapes, we need a network made for images: a convolutional network, or CNN. The brief asks for a ResNet, a family of CNNs that won the ImageNet competition in 2015. ResNet18 is its smallest member: light enough for a kiosk, and PyTorch gives it already trained."],
      [{ cue: "point at the dark box" }, " Its 18 layers slide small filters over the picture: the first layers find edges and colours, the next ones textures, the last ones shapes. ",
        { opt: "\"Res\" means residual: shortcuts between its layers let a deep network learn well." }],
      [`ResNet18 already learned this on 1.2 million ImageNet photos, and we only have ${n(d01.split_counts.train)} paintings, so we never retrain all of it. `, { cue: "point at the 512 numbers" },
        " It turns each painting into 512 numbers, and we replace its last layer by our own, for our 5 styles."],
      [{ cue: "point at the bottom right card" }, ` We tried two ways. Frozen: only our new layer learns, ${n(probe.trained_parameters)} weights. Fine-tuned: the last block learns again too, ${(d05.parameters["trained by B (fine-tuning of layer4 + head)"] / 1e6).toFixed(1)} million weights. `,
        { b: "Like hiring an experienced art expert: frozen, we only teach them our five style names; fine-tuned, we also retrain their eye." }],
    ],
    transition: ["We saw how ResNet18 works.", "So now, let's see what it gives on our paintings."],
  },
  {
    title: "A pretrained ResNet18 sees shapes",
    paragraphs: [
      [{ cue: "point at the top bar" }, ` These are the final tests, run once at the very end: ${nTests} paintings never seen before, our test pile and ${d08.data.n} new paintings from another part of WikiArt. The fine-tuned ResNet18 gets ${p1(both[FT])} percent: `,
        { b: `${p0(both[FT] - both[RF])} points above the baseline, and twice our MLP.` }],
      [{ cue: "point at the middle card" }, ` On validation, the frozen and the fine-tuned versions were too close to choose, on only ${d01.split_counts.val} paintings. On the tests, the fine-tuned one wins by ${p0(-ftOverFrozen.diff)} points, and it is not luck. `,
        { b: "That is why we keep the fine-tuned one." }],
      [{ cue: "point at the top card" }, ` And it does see shapes: on validation, the ResNet finds ${p0(cubism(d05.probe_seed0_per_class))} percent of the Cubist paintings, against ${p0(cubism(d03.regularised_seed0_per_class))} for our MLP.`],
      [{ cue: "point at the bottom card" }, " ", { opt: `We monitored it like the MLP: the gap between train and validation is only ${probe.gap.toFixed(2)}, against ${tools["baseline (lr 0.0001)"].gap.toFixed(2)} for our first MLP.` }],
    ],
    transition: ["We saw that ResNet18 beats the baseline by far.", "So now, step five: would a much bigger model do even better?"],
  },
  {
    title: "CLIP ViT-L/14: more accurate, but too big",
    paragraphs: [
      ["ResNet18 is one of the smallest image networks, so we also tried one of the biggest: CLIP ViT-L/14, by OpenAI. A transformer, like BERT, that learned from 400 million Internet images and their captions. We used it like ResNet18: frozen, with our linear head."],
      [{ cue: "point at the big number" }, ` On validation, it reaches ${p1(bigVal)} percent: `, { b: "the best of all our models." }],
      [{ cue: "point at the three rows" }, ` But look at the price. One photo takes ${seconds(k2.ms_1_thread)} on one CPU thread, ${Math.round(k2.ms_1_thread / k1.ms_1_thread)} times ResNet18, and its file weighs ${megabytes(k2.size_mb)}, ${Math.round(k2.size_mb / k1.size_mb)} times more. `,
        { b: `On a kiosk, that is about ${Math.round(k2.ms_1_thread * d09.budget.kiosk_slowdown / 1000)} seconds per photo: too slow and too big. It would need a GPU.` }],
      [{ opt: "And it has probably seen WikiArt paintings during its training." }],
    ],
    transition: ["We saw that CLIP is more accurate, but does not fit on a kiosk.", "So now, step six: all our models side by side, criterion by criterion."],
  },
  {
    title: "Every model, by criteria: ResNet18 stands out",
    paragraphs: [
      [{ cue: "point at the accuracy column" }, ` On the same ${nTests} new paintings, the baseline gets ${p0(both[RF])} percent, our MLP ${p0(both[MLP])}, the fine-tuned ResNet18 ${p0(both[FT])}, and CLIP ${p0(both[BIG])}.`],
      [{ cue: "point at the other columns" }, ` Then the kiosk criteria, fixed before the tests: one second per photo on the kiosk, and a file under ${d09.budget.size_mb} megabytes. `,
        { cue: "point at the blue row" }, ` ResNet18 answers in ${seconds(k1.ms_1_thread)} on our laptop, weighs ${megabytes(k1.size_mb)} and retrains in ${(k1.retrain_seconds / 60).toFixed(1)} minutes. `,
        { cue: "point at the CLIP row" }, " CLIP breaks both limits."],
      [{ b: "ResNet18 is the only model that is both accurate and fits the kiosk." }],
      [{ cue: "point at the blue banner" }, " ", { b: "So our final choice is a fine-tuned ResNet18: it answers the director's question. Yes, a neural network is worth it." }],
      [` One visitor in ${oneIn(both[FT])} gets a wrong style, against one in ${oneIn(both[RF])} with the baseline, and it runs on a modest kiosk. Before launch, we test it on real kiosk photos.`],
      ["Thank you. We are happy to take your questions."],
    ],
  },
];

// ---------------------------------------------------------------- words, timing, notes
const text = (piece) => (typeof piece === "string" ? piece : piece.b || piece.opt || "");
const count = (s) => (s.trim() ? s.trim().split(/\s+/).length : 0);
const words = (slide) => slide.paragraphs.flat().reduce((a, p) => a + count(text(p)), 0)
  + (slide.transition ? count(slide.transition.join(" ")) : 0);
const optionalWords = SLIDES.reduce((a, sl) => a + sl.paragraphs.flat().reduce((b, p) => b + count(p.opt || ""), 0), 0);
const clock = (sec) => `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
const minutes = (w) => { const sec = Math.round(w / WORDS_PER_MINUTE * 60); return `${Math.floor(sec / 60)} min ${String(sec % 60).padStart(2, "0")} s`; };
let t = 0;
SLIDES.forEach((sl) => {
  sl.words = words(sl);
  sl.seconds = Math.round(sl.words / WORDS_PER_MINUTE * 60 / 5) * 5;
  sl.start = t;
  t += sl.seconds;
});
const totalWords = SLIDES.reduce((a, sl) => a + sl.words, 0);

const notes = SLIDES.map((sl) => {
  const body = sl.paragraphs.map((p) => p.map(text).join("")).join(" ").replace(/\s+/g, " ").trim();
  return sl.transition ? `${body}\n\nTRANSITION: ${sl.transition.join(" ").trim()}` : body;
});
fs.writeFileSync(path.join(__dirname, "speaker_notes.json"), `${JSON.stringify(notes, null, 2)}\n`, "utf8");

// ---------------------------------------------------------------- the Word document
const INK = "1C2433", MUTED = "5F6473", CUE = "1C5FB0", OPTIONAL = "8A8F9C", GREEN = "1E6B45", GREEN_BG = "EAF5EE";
const BODY = 25, SMALL = 21; // half-points
// A number never leaves its thousands at a line break ("1 061"): a non-breaking space, as in the deck.
const nb = (t) => t.replace(/(\d) (?=\d{3}\b)/g, "$1 ");
const run = (piece, size = BODY) => {
  if (typeof piece === "string") return new TextRun({ text: nb(piece), size });
  if (piece.b) return new TextRun({ text: nb(piece.b), bold: true, size });
  if (piece.opt) return new TextRun({ text: nb(`(optional) ${piece.opt}`), italics: true, color: OPTIONAL, size });
  return new TextRun({ text: `[${piece.cue}]`, italics: true, color: CUE, size: SMALL });
};
const box = { left: { style: BorderStyle.SINGLE, size: 18, color: GREEN, space: 8 } };
const boxShade = { type: ShadingType.CLEAR, color: "auto", fill: GREEN_BG };

const children = [
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Oral script: the 10-minute defence", font: "Cambria", bold: true, color: INK, size: 44 })] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "Museum Guide project, MSA-DATB09-03. One section per slide, to read aloud.", color: MUTED })] }),
  new Paragraph({ style: "Heading2", children: [new TextRun("How to use it")] }),
  ...[
    `About ${totalWords} words: at a calm pace (about ${WORDS_PER_MINUTE} words per minute), this is about ${minutes(totalWords)}. Without the ${optionalWords} optional words, about ${minutes(totalWords - optionalWords)}. The rest of the 10 minutes is a safety margin for pauses, clicks and nerves.`,
    "The story: a classical baseline first, then each model has to beat it (our MLP, ResNet18, a much bigger CLIP), and a last table compares them all, criterion by criterion.",
    "Every slide ends with a green TRANSITION box, always built the same way: \"We saw that... So now...\". Say it while you click to the next slide, and never skip it, even when you are late: it turns the six steps into one story.",
    "Read it aloud 3 times, with the slides, and time yourself once. Speak slowly; pause after each bold sentence: it is the one the jury must remember.",
    "Blue text in brackets is a stage cue (where to point); do not read it. Grey sentences marked (optional) can be skipped if you are late.",
    "If you present with a teammate, a natural split is slides 1 to 6 (data, baseline, MLP), then slides 7 to 11 (ResNet18, CLIP, recap, choice).",
  ].map((line) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: line, size: SMALL })] })),
  new Paragraph({ style: "Heading2", children: [new TextRun("The story, transition by transition")] }),
  ...SLIDES.slice(0, -1).map((sl, i) => new Paragraph({ spacing: { after: 60 }, children: [
    new TextRun({ text: `${i + 1} to ${i + 2}:  `, bold: true, color: GREEN, size: 20 }),
    new TextRun({ text: sl.transition.join(" ").trim(), size: 20 }),
  ] })),
];

SLIDES.forEach((sl, i) => {
  // keepNext on every paragraph of a slide but the last: a slide never breaks across two pages.
  const section = [
    { style: "Heading1", children: [new TextRun(`Slide ${i + 1} · ${sl.title}`)] },
    { spacing: { after: 160 }, children: [new TextRun({ text: `${clock(sl.start)} to ${clock(sl.start + sl.seconds)} · ${sl.seconds} s · ${sl.words} words`, color: MUTED, size: 19 })] },
    ...sl.paragraphs.map((p) => ({ spacing: { after: 160, line: 320 }, children: p.map((piece) => run(piece)) })),
  ];
  if (sl.transition) {
    const [saw, now] = sl.transition;
    section.push({ border: box, shading: boxShade, indent: { left: 170, right: 170 }, spacing: { before: 60 },
      children: [new TextRun({ text: `TRANSITION TO SLIDE ${i + 2}`, bold: true, color: GREEN, size: 17 })] });
    section.push({ border: box, shading: boxShade, indent: { left: 170, right: 170 }, spacing: { after: 200, line: 320 },
      children: [...(saw ? [new TextRun({ text: `${saw} `, color: GREEN, size: BODY })] : []), new TextRun({ text: now, bold: true, color: GREEN, size: BODY })] });
  }
  children.push(...section.map((options, k) => new Paragraph({ ...options, keepNext: k < section.length - 1 })));
});

const doc = new Document({
  creator: "Benjamin Caujolle",
  title: "Oral script: the 10-minute defence",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Cambria", bold: true, size: 30, color: INK }, paragraph: { spacing: { before: 360, after: 60 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Cambria", bold: true, size: 24, color: INK }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, right: 1247, bottom: 1134, left: 1247 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "Oral script    ", color: MUTED, size: 16 }), new TextRun({ children: [PageNumber.CURRENT], color: MUTED, size: 16 }),
    ] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "oral_script.docx"), buf);
  console.log(`written oral_script.docx and speaker_notes.json: ${totalWords} words (${optionalWords} optional), about ${minutes(totalWords)}`);
  SLIDES.forEach((sl, i) => console.log(`  slide ${String(i + 1).padStart(2)}  ${clock(sl.start)}  ${String(sl.seconds).padStart(3)} s  ${String(sl.words).padStart(3)} words  ${sl.title}`));
});
