"""Check: ResNet18 trained from zero (random weights, nothing frozen) on our split, validation only.

Same data, augmentation and optimiser family as the fine-tuning of notebook 05; the only change is that the
network starts from random weights instead of the ImageNet ones. One seed, about 25 minutes on a laptop CPU.
The test sets stay closed: this run is only there to answer "why not train the CNN from zero?".

Run from the repository root:  python scripts/resnet_from_zero.py [--epochs 30]
Writes reports/results/resnet_from_zero.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from museum_guide.config import RESULTS_DIR, set_seed  # noqa: E402
from museum_guide.data import load_split_images  # noqa: E402
from museum_guide.models import ResNetClassifier, count_trainable  # noqa: E402
from museum_guide.training import crop_and_flip, fit, make_loaders  # noqa: E402

SETTINGS = dict(lr=1e-3, weight_decay=1e-2, batch_size=32, augment="random crop + flip", schedule="cosine", seed=0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epochs", type=int, default=30)
    epochs = p.parse_args().epochs
    set_seed()
    images, _ = load_split_images(224)  # train and validation only
    loaders = make_loaders(*images["train"], *images["val"], batch_size=SETTINGS["batch_size"], eval_batch_size=128)
    torch.manual_seed(SETTINGS["seed"])
    model = ResNetClassifier(train_from="conv1", pretrained=False)  # random weights, every block learns
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"], weight_decay=SETTINGS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start = time.perf_counter()

    def report(model, history):
        print(f"epoch {len(history['val_acc']):2d}/{epochs}  train acc {history['train_acc'][-1]:.3f}  "
              f"val acc {history['val_acc'][-1]:.3f}  ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)

    history = fit(model, optimizer, loaders, epochs, scheduler=scheduler, augment=crop_and_flip, on_epoch_end=report, seed=1)
    best = max(range(len(history["val_acc"])), key=lambda i: history["val_acc"][i])
    out = {"settings": {**SETTINGS, "epochs": epochs}, "trainable_parameters": count_trainable(model),
           "best_val_acc": history["val_acc"][best], "best_epoch": best + 1, "last_val_acc": history["val_acc"][-1],
           "train_acc_at_best": history["train_acc"][best], "minutes": (time.perf_counter() - start) / 60,
           "history": {k: history[k] for k in ("train_acc", "val_acc", "train_loss", "val_loss")}}
    path = RESULTS_DIR / "resnet_from_zero.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"best val acc {out['best_val_acc']:.3f} at epoch {out['best_epoch']} -> {path}")


if __name__ == "__main__":
    main()
