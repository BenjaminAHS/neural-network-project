"""ResNet18 features: 512 numbers per painting, computed once and cached (data/cache, ignored by Git)."""
import time

import torch

from .config import CACHE_DIR
from .data import load_images
from .models import ResNetClassifier


@torch.no_grad()
def extract_features(model, images, batch_size=64):
    """Features of all the images, in batches. The caller chooses the mode (model.eval() normally)."""
    return torch.cat([model.features(images[i:i + batch_size]) for i in range(0, len(images), batch_size)])


def cached_features(meta, name="resnet18", normalise=True, refresh=False):
    """Features of every image of meta (frozen pretrained ResNet18, eval mode, 224 x 224 pixels).

    Returns (features, seconds per image). The test images get features too, but no notebook before the
    final comparison looks at their labels or their predictions.
    """
    path = CACHE_DIR / f"{name}.pt"
    ids = list(meta["id"])
    if path.exists() and not refresh:
        cache = torch.load(path)
        if cache["ids"] == ids:
            return cache["features"], cache["seconds_per_image"]
    images = load_images(meta, 224)
    model = ResNetClassifier(train_from=None, normalise=normalise).eval()
    t0 = time.perf_counter()
    feats = extract_features(model, images)
    seconds = (time.perf_counter() - t0) / len(images)
    torch.save({"ids": ids, "features": feats, "seconds_per_image": seconds}, path)
    return feats, seconds
