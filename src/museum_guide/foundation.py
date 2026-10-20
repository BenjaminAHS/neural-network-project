"""Bonus: large pretrained transformers (the BERT family) as frozen feature extractors.

- ViT-B/16 (torchvision): the transformer architecture of BERT, applied to 16 x 16 patches of the image.
- CLIP ViT-B/32 (OpenAI, through Hugging Face transformers): an image encoder trained together with a text
  encoder on 400 million image-caption pairs, so images and sentences live in the same space. It can classify
  "zero-shot": no training, only one sentence per class.
"""
import time

import torch
import torch.nn as nn

from .config import CACHE_DIR
from .models import ImageNetNormalise

CLIP_NAME = "openai/clip-vit-base-patch32"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)  # the pixel statistics CLIP was trained with
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@torch.no_grad()
def batched(fn, images, batch_size=32):
    """fn applied to all the images in batches; returns (features, seconds per image)."""
    t0 = time.perf_counter()
    feats = torch.cat([fn(images[i:i + batch_size]) for i in range(0, len(images), batch_size)])
    return feats, (time.perf_counter() - t0) / len(images)


class ViTEncoder(nn.Module):
    """Frozen ViT-B/16 trained on ImageNet: 768 numbers per image (its CLS token, as in BERT)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        self.normalise = ImageNetNormalise()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.eval().requires_grad_(False)

    def forward(self, x):
        return self.vit(self.normalise(x))


class CLIPEncoder(nn.Module):
    """Frozen CLIP ViT-B/32: 512 numbers per image, and 512 numbers per sentence, in the same space."""

    def __init__(self, name=CLIP_NAME):
        super().__init__()
        from huggingface_hub.utils import logging as hub_logging
        from transformers import CLIPModel, CLIPTokenizer
        from transformers.utils import logging as hf_logging
        hub_logging.set_verbosity_error()  # quiet notebooks: no download bars or hub notices
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.clip = CLIPModel.from_pretrained(name)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    @staticmethod
    def _tensor(out):
        return out if torch.is_tensor(out) else out.pooler_output  # depends on the transformers version

    def forward(self, x):
        """Image features from uint8 images of 224 x 224 pixels."""
        return self._tensor(self.clip.get_image_features(pixel_values=(x.float() / 255 - self.mean) / self.std))

    @torch.no_grad()
    def text(self, sentences):
        tokens = self.tokenizer(sentences, padding=True, return_tensors="pt")
        return self._tensor(self.clip.get_text_features(**tokens))


def class_text_embeddings(encoder, prompts, classes):
    """One unit vector per class: the mean of the unit vectors of its sentences (prompt ensembling)."""
    rows = []
    for c in classes:
        e = encoder.text(prompts[c])
        e = e / e.norm(dim=1, keepdim=True)
        rows.append(e.mean(dim=0))
    T = torch.stack(rows)
    return T / T.norm(dim=1, keepdim=True)


def zero_shot_predict(image_features, class_text):
    """Each image gets the class whose sentence vector has the highest cosine similarity with it."""
    F = image_features / image_features.norm(dim=1, keepdim=True)
    return (F @ class_text.T).argmax(dim=1)


def cached(name, meta, build_encoder, images):
    """Features of the images (rows of meta), computed once with the encoder and saved in data/cache."""
    path = CACHE_DIR / f"{name}.pt"
    ids = list(meta["id"])
    if path.exists():
        cache = torch.load(path)
        if cache["ids"] == ids:
            return cache["features"], cache["seconds_per_image"]
    feats, seconds = batched(build_encoder(), images)
    torch.save({"ids": ids, "features": feats, "seconds_per_image": seconds}, path)
    return feats, seconds
