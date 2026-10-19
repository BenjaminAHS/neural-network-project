"""Load the WikiArt subset, split it without artist leakage and turn the images into tensors."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .config import DATA_DIR, SEED

UNKNOWN_ARTIST = "Unknown Artist"
SPLITS = ("train", "val", "test")


def load_metadata(data_dir=DATA_DIR):
    """One row per image: path, label, label_idx, id, style, genre, artist, original size."""
    meta = pd.read_csv(Path(data_dir) / "metadata.csv")
    return meta.sort_values(["label_idx", "id"]).reset_index(drop=True)


def class_names(meta):
    """Class names in the order of label_idx (sorted folder names, like ImageFolder)."""
    return list(meta.drop_duplicates("label_idx").sort_values("label_idx")["label"])


def artist_groups(meta):
    """One group per known painter. "Unknown Artist" mixes many painters, so each of its images is
    its own group (we cannot keep together painters we do not know)."""
    unknown = meta["artist"] == UNKNOWN_ARTIST
    return np.where(unknown, "unknown_" + meta["id"], meta["artist"]).astype(str)


def make_split(meta, test_frac=0.15, val_frac=0.15, by_artist=True, seed=SEED):
    """Return an array with 'train', 'val' or 'test' for each row of meta.

    Stratified: each part keeps the same share of each style.
    by_artist=True: all the images of a known painter go to the same part, so the validation and
    test scores measure the style, not the memory of a painter seen during training.
    by_artist=False: a plain stratified random split (used to measure the leakage).
    """
    y = meta["label_idx"].to_numpy()
    groups = artist_groups(meta) if by_artist else meta["id"].to_numpy().astype(str)

    def one_fold(idx, frac, seed):
        cv = StratifiedGroupKFold(n_splits=round(1 / frac), shuffle=True, random_state=seed)
        _, fold = next(cv.split(np.zeros(len(idx)), y[idx], groups[idx]))
        return idx[fold]

    split = np.full(len(meta), "train", dtype=object)
    all_idx = np.arange(len(meta))
    test_idx = one_fold(all_idx, test_frac, seed)
    split[test_idx] = "test"
    rest = np.setdiff1d(all_idx, test_idx)
    split[one_fold(rest, val_frac / (1 - test_frac), seed + 1)] = "val"
    return split.astype(str)


def load_images(meta, size, data_dir=DATA_DIR):
    """uint8 tensor of shape (N, 3, size, size). The files are 224 x 224; PIL shrinks them with
    anti-aliasing when a smaller size is asked."""
    out = torch.empty(len(meta), 3, size, size, dtype=torch.uint8)
    for i, rel in enumerate(meta["path"]):
        with Image.open(Path(data_dir) / rel) as im:
            im = im.convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.Resampling.BILINEAR)
            out[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    return out


def channel_stats(images):
    """Mean and standard deviation of each colour channel (pixels in [0, 1]), shape (1, 3, 1, 1)."""
    x = images.float() / 255
    return x.mean(dim=(0, 2, 3), keepdim=True), x.std(dim=(0, 2, 3), keepdim=True)


def standardise(images, mean, std):
    """Pixels in [0, 1], then (x - mean) / std with the statistics of the TRAINING images only."""
    return (images.float() / 255 - mean) / std


def load_split_tensors(size, by_artist=True, include_test=False, seed=SEED, data_dir=DATA_DIR):
    """Standardised tensors for the MLP: {'train': (X, y), 'val': (X, y)} and the metadata with
    a 'split' column. The test set stays locked (include_test=False) until the final comparison."""
    meta = load_metadata(data_dir)
    split = make_split(meta, by_artist=by_artist, seed=seed)
    images = load_images(meta, size, data_dir)
    y = torch.tensor(meta["label_idx"].to_numpy())
    parts = SPLITS if include_test else SPLITS[:2]
    masks = {s: torch.from_numpy(split == s) for s in parts}
    mean, std = channel_stats(images[masks["train"]])
    data = {s: (standardise(images[m], mean, std), y[m]) for s, m in masks.items()}
    return data, meta.assign(split=split)
