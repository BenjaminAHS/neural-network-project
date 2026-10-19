"""Checks of the data split: complete, stratified, reproducible and without painter leakage."""
import numpy as np
import pandas as pd
import pytest
import torch

from museum_guide.config import DATA_DIR
from museum_guide.data import UNKNOWN_ARTIST, channel_stats, load_metadata, make_split, standardise


def fake_meta(n_per_class=60, n_classes=5):
    """Half of the images by unknown painters, the other half by 7 known painters per class."""
    rows = []
    for c in range(n_classes):
        for i in range(n_per_class):
            artist = UNKNOWN_ARTIST if i % 2 == 0 else f"painter_{c}_{i % 7}"
            rows.append({"id": f"c{c}_{i:03d}", "label_idx": c, "label": f"class{c}", "artist": artist})
    return pd.DataFrame(rows)


def test_every_image_gets_exactly_one_part():
    meta = fake_meta()
    split = make_split(meta)
    assert len(split) == len(meta)
    assert set(split) == {"train", "val", "test"}


def test_known_painters_stay_in_one_part():
    meta = fake_meta()
    named = meta.assign(split=make_split(meta)).query("artist != @UNKNOWN_ARTIST")
    assert (named.groupby("artist")["split"].nunique() == 1).all()


def test_random_split_mixes_painters():
    """The leakage we want to avoid: without groups, a painter appears in several parts."""
    meta = fake_meta()
    named = meta.assign(split=make_split(meta, by_artist=False)).query("artist != @UNKNOWN_ARTIST")
    assert (named.groupby("artist")["split"].nunique() > 1).any()


def test_split_is_stratified_with_the_right_sizes():
    meta = fake_meta()
    split = make_split(meta)
    for part, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        share_per_class = pd.Series(split == part).groupby(meta["label_idx"]).mean()
        assert np.allclose(share_per_class, target, atol=0.08), (part, share_per_class.to_dict())


def test_split_is_reproducible():
    meta = fake_meta()
    assert (make_split(meta, seed=0) == make_split(meta, seed=0)).all()
    assert (make_split(meta, seed=0) != make_split(meta, seed=1)).any()


def test_standardise_with_train_statistics():
    torch.manual_seed(0)
    images = torch.randint(0, 256, (20, 3, 8, 8), dtype=torch.uint8)
    mean, std = channel_stats(images)
    x = standardise(images, mean, std)
    assert torch.allclose(x.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(x.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-4)


@pytest.mark.skipif(not (DATA_DIR / "metadata.csv").exists(), reason="WikiArt subset not downloaded")
def test_real_split_has_no_painter_leakage():
    meta = load_metadata()
    meta = meta.assign(split=make_split(meta))
    named = meta[meta["artist"] != UNKNOWN_ARTIST]
    assert (named.groupby("artist")["split"].nunique() == 1).all()
    assert meta.groupby("split").size().min() > 150
