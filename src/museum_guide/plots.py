"""Plots of the project. Figures are saved to reports/figures (they never show artworks)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import FIGURES_DIR

# Categorical colours in a fixed order (checked for colour-blind readers); a colour follows its
# entity everywhere: the same style always has the same colour.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = dict(zip(["Baroque", "Cubism", "Impressionism", "Pop_Art", "Ukiyo_e"], PALETTE))
TRAIN_COLOR, VAL_COLOR = PALETTE[0], PALETTE[1]
INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
LABELS = {"train_loss": "training loss", "val_loss": "validation loss",
          "train_acc": "training accuracy", "val_acc": "validation accuracy"}

mpl.rcParams.update({
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    "font.size": 10, "axes.titlesize": 11,
})


def save_figure(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def plot_history(history, title="", save_as=None):
    """Loss and accuracy curves, train and validation, one point per epoch."""
    e = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, key in zip(axes, ("loss", "acc")):
        ax.plot(e, history[f"train_{key}"], color=TRAIN_COLOR, label="train")
        ax.plot(e, history[f"val_{key}"], color=VAL_COLOR, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy" if key == "acc" else "loss")
        ax.legend()
    fig.suptitle(title)
    if save_as:
        save_figure(fig, save_as)
    plt.show()


def plot_layers(history, key, title="", ylabel="", log=False, save_as=None):
    """One curve per layer over the epochs, for a per-layer quantity recorded by the monitor."""
    values = list(zip(*history[key]))  # one tuple of epochs per layer
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k, v in enumerate(values, 1):
        ax.plot(range(1, len(v) + 1), v, color=PALETTE[k - 1], label=f"layer {k}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel or key)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    if save_as:
        save_figure(fig, save_as)
    plt.show()


def plot_confusion(cm, classes, title="", save_as=None):
    """Confusion matrix with counts; rows = true style, columns = predicted style."""
    blues = mpl.colors.LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm.numpy(), cmap=blues)
    vmax = cm.max().item()
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j].item()
            ax.text(j, i, v, ha="center", va="center", color="#fcfcfb" if v > 0.55 * vmax else "#0b0b0b")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted style")
    ax.set_ylabel("true style")
    ax.grid(False)
    ax.set_title(title)
    if save_as:
        save_figure(fig, save_as)
    plt.show()


def plot_compare(histories, key="val_loss", title="", save_as=None):
    """One curve per run, for the same metric."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, h in histories.items():
        ax.plot(range(1, len(h[key]) + 1), h[key], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(LABELS.get(key, key))
    ax.set_title(title)
    ax.legend()
    if save_as:
        save_figure(fig, save_as)
    plt.show()
