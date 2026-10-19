# Museum Guide: recognising painting styles with neural networks

Final project of the course *Neural Networks & Deep Learning Foundations* (MSA-DATB09-03, Albert School).

## The story

We work in the AI team of a (fictional) French fine-arts museum. The museum wants a guide on visitors'
phones and on kiosks in the galleries. A visitor points the camera at a painting, and the guide tells
them its **style** (Impressionism, Baroque, Cubism...) and the story of that movement.

The kiosks have small CPUs, the team is small and the budget is limited. So the director asks one clear
question: **is a neural network worth it in production, or would a simpler classical model do the job?**

Our rules: networks built from PyTorch primitives (no "magic" training library), every number measured,
every claim backed by evidence.

## What we build

| Part | What it is | Where |
|---|---|---|
| 1 | A feed-forward network (MLP) in PyTorch that classifies paintings from their pixels | `notebooks/02_mlp` |
| 2 | Training monitoring (curves, gradients, activations) and failure diagnosis | `notebooks/03_monitoring_diagnosis` |
| 3 | A pretrained CNN (ResNet18) used as a feature extractor and fine-tuned | `notebooks/04_resnet` |
| 4 | Classical models, CPU speed tests and our recommendation to the director | `notebooks/05_final_comparison` |

## Data

We use a small subset of [WikiArt](https://huggingface.co/datasets/huggan/wikiart): **5 styles x 300
images** (Impressionism, Baroque, Cubism, Ukiyo-e, Pop Art), resized to 224 x 224 pixels.

The images are **not** in this repository: their licence is unknown, so we only share the script and
the exact command. To rebuild the same data:

```bash
pip install -r requirements.txt
bash scripts/get_data.sh          # about 8.4 GB streamed once, about 40 MB written to data/
```

`scripts/get_data.sh` explains each option. In short: the dataset revision and the random seed are
fixed, so everybody gets the same 1 500 images, and one painter gives at most 50 images, so that the
model learns the style and not the painter.

## Repository structure

```
make_wikiart_subset.py   builds the data subset (script given by the course)
scripts/get_data.sh      the exact command we used
src/museum_guide/        our Python code (data, models, training loop, diagnostics, benchmarks)
notebooks/               the experiments, run in order (01 to 05)
tests/                   small automatic checks (data split, model shapes)
reports/                 figures and measured results (JSON/CSV), no artwork images
```

## Project status

- [x] Repository and data subset
- [ ] **m1**: data split and a first MLP baseline
- [ ] **m2**: monitoring curves and failure diagnosis
- [ ] Pretrained ResNet18
- [ ] **v1.0**: classical models, CPU benchmark, final test and recommendation
