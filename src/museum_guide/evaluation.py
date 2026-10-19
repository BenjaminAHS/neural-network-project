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
