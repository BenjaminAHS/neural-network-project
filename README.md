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
| 1 | A feed-forward network (MLP) in PyTorch that classifies paintings from their pixels | [02_mlp](notebooks/02_mlp.ipynb) |
| 2 | Training monitoring (curves, gradients, dead neurons, learning rate) and regularisation | [03_monitoring](notebooks/03_monitoring.ipynb) |
| 2 | Failure diagnosis: four broken runs, diagnosed and fixed | [04_training_doctor](notebooks/04_training_doctor.ipynb) |
| 3 | A pretrained CNN (ResNet18) used as a feature extractor and fine-tuned | `notebooks/05_resnet` |
| 4 | Classical models, CPU speed tests and our recommendation to the director | `notebooks/06_final_comparison` |

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

**Our split** ([notebook 01](notebooks/01_data.ipynb)): 1 061 images to train, 219 to validate and
220 to test, with the same share of each style in each part. All the images of a known painter go to
the same part (0 known painter in two parts, against 23 of 32 with a random split), so the scores
measure the style and not the memory of a painter. The test set is used only once, at the very end.
The split is saved in [reports/split.csv](reports/split.csv).

## Results so far

| Model | Validation accuracy | Notes |
|---|---|---|
| Always the most frequent style | 22.8 % | the chance level |
| MLP on 64 x 64 pixels (6.4 M parameters) | 40.6 % (best epoch: 3) | 99.7 % on train at the end: strong overfitting |
| MLP on 16 / 32 / 64 pixels, mean of 3 seeds | 39.4 / 39.0 / 37.1 % | more pixels do not help an MLP |
| Regularised MLP, 32 x 32 pixels, mean of 3 seeds | **45.5 %** (std 0.7) | train/validation gap 0.12 instead of 0.61 |

Sanity checks ([notebook 02](notebooks/02_mlp.ipynb)): the first loss is 1.591, close to ln 5 = 1.609
(after shrinking the last layer, see the notebook), and the model learns one batch of 32 images by
heart (100 %). Every number comes from [reports/results/](reports/results/).

![MLP baseline curves](reports/figures/02_baseline_curves.png)

## Training monitoring

[Notebook 03](notebooks/03_monitoring.ipynb) looks inside the network during training: the gradient of
each layer, the share of dead neurons and the size of the weights, not only the loss.

- **The baseline is healthy but overfits.** The gradient reaches every layer and only about 4 % of the
  neurons are dead, but the training loss falls to 0.008 while the validation loss rises from 1.54 to 4.06.
- **Learning rate.** A sweep of 8 values gives a U-shaped validation loss. The best value is 0.0001; Adam's
  usual 0.001 overfits after 2 epochs, and 0.01 or more breaks the network.
- **Regularisation.** Weight decay, dropout, batch norm, flips and early stopping each help a little.
  All together they lift the MLP from 38.8 % to 45.5 % and shrink the train/validation gap from 0.61 to 0.12.
- **Errors.** The MLP recognises colours (dark Baroque: recall 69 %), not shapes (Cubism: recall 7 %).

![Learning-rate sweep](reports/figures/03_lr_sweep.png)

## Failure diagnosis

[Notebook 04](notebooks/04_training_doctor.ipynb) breaks the training on purpose four times, then
diagnoses and fixes each run. Every number below was measured there.

| Patient | Symptom (measured) | Cause | Cure | After the cure |
|---|---|---|---|---|
| Learning rate too big (Adam 0.01) | loss stuck near ln 5, 212 of 219 images predicted Baroque, over 99 % of layer-2 neurons dead; with SGD at lr 1, `nan` at mini-batch 8 | steps so big that the ReLU neurons die or the weights blow up | learning rate 0.0001 (from the sweep) | 43.8 % validation, 0.4 % dead neurons |
| `zero_grad()` forgotten | loss jumps to 4.94, then one class only (19.2 %) | gradients add up: the norm is 26 times bigger after 2 epochs | `optimizer.zero_grad()` before `backward()` | smooth loss, 41.1 % validation |
| Raw pixels (0 to 255) | first loss 16.3 instead of 1.6; SGD loss up to 3 x 10^8; Adam noisy (35.2 %) | inputs dozens of times too big | standardise with the training mean and std | first loss 1.610, 43.8 % validation |
| 8 sigmoid layers | loss stuck at 1.609 = ln 5, 19.2 % | vanishing gradient: layer 1 gets 6 x 10^-7 of the gradient of layer 9 | ReLU and He initialisation | ratio 0.17, 39.3 % validation |

A loss stuck at ln 5 can mean dead neurons (patient 1) or a vanishing gradient (patient 4): only the
monitor tells them apart. A fifth, silent failure (the same painter in training and test) cannot be
seen in any curve; we measure it with the ResNet in notebook 05.

![Vanishing gradient](reports/figures/04_p4_grad_per_layer.png)

## How to run

```bash
pip install -r requirements.txt
bash scripts/get_data.sh                  # once: builds data/wikiart_style5
python -m pytest                          # automatic checks
python scripts/run_notebooks.py           # rebuilds and runs every notebook (FAST_RUN=1 for a quick check)
```

The notebooks are stored twice: the `.py` file is the source (easy to read in a Git diff) and the
`.ipynb` file is the same notebook with its outputs. Everything runs on a laptop CPU.

## Repository structure

```
make_wikiart_subset.py   builds the data subset (script given by the course)
scripts/                 the exact data command, and a script that runs all the notebooks
src/museum_guide/        our Python code (data, models, training loop, diagnostics)
notebooks/               the experiments, run in order (01 to 06)
tests/                   small automatic checks (data split, model, training loop)
reports/                 figures, the split and every measured result (JSON), no artwork images
```

## Project status

- [x] Repository and data subset
- [x] **m1**: data split and a first MLP baseline
- [x] **m2**: monitoring curves and failure diagnosis
- [ ] Pretrained ResNet18
- [ ] **v1.0**: classical models, CPU benchmark, final test and recommendation
