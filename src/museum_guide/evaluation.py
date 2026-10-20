"""Metrics, and saving every measured number so that the README and the slides quote files, not memory."""
import json

import numpy as np
import pandas as pd
import torch

from .config import RESULTS_DIR


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i, j] = number of images of true class i predicted as class j."""
    y_true, y_pred = torch.as_tensor(y_true), torch.as_tensor(y_pred)
    return torch.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def per_class_report(cm, classes):
    """Recall (share of the class that is found), precision (share of the predictions that are right)
    and F1 (their harmonic mean) for each class."""
    cm = cm.double()
    recall = cm.diag() / cm.sum(dim=1)
    precision = cm.diag() / cm.sum(dim=0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return pd.DataFrame({"recall": recall.numpy(), "precision": precision.numpy(), "f1": f1.numpy(),
                         "images": cm.sum(dim=1).long().numpy()}, index=classes)


def macro_f1(y_true, y_pred, n_classes):
    """Mean of the F1 scores of the classes (each style counts the same)."""
    return float(per_class_report(confusion_matrix(y_true, y_pred, n_classes), list(range(n_classes)))["f1"].mean())


def bootstrap_ci(correct, n_boot=5000, level=0.95, seed=0):
    """95 % interval of an accuracy by bootstrap: we redraw the test images with replacement many times
    and look at how much the accuracy moves. correct: one True/False per test image."""
    c = np.asarray(correct, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(c), (n_boot, len(c)))
    lo, hi = np.percentile(c[idx].mean(axis=1), [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def paired_bootstrap(correct_a, correct_b, n_boot=5000, level=0.95, seed=0):
    """Difference of accuracy A - B on the same test images, with its bootstrap interval, and the share
    of redraws where A is not better than B (small = the difference is not luck)."""
    d = np.asarray(correct_a, dtype=float) - np.asarray(correct_b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    diffs = d[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [50 * (1 - level), 50 * (1 + level)])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "share_not_better": float((diffs <= 0).mean())}


def most_confused(cm, classes, k=3):
    """The k most frequent mistakes, as (true class, predicted class, count)."""
    off = cm.clone()
    off.fill_diagonal_(0)
    values, idx = off.flatten().topk(k)
    n = len(classes)
    return [(classes[i // n], classes[i % n], int(v)) for v, i in zip(values.tolist(), idx.tolist())]


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    return x


def save_results(name, results):
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(_clean(results), indent=2), encoding="utf-8")
    return path


def load_results(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
