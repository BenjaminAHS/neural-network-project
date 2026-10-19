"""Paths, flags and seeds shared by the whole project."""
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "wikiart_style5"))
CACHE_DIR = ROOT / "data" / "cache"            # rebuilt features, ignored by Git
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"          # figures without artworks (committed)
RESULTS_DIR = REPORTS_DIR / "results"          # every measured number, as JSON or CSV (committed)
PRIVATE_DIR = REPORTS_DIR / "private"          # figures that show artworks, ignored by Git

FAST_RUN = os.environ.get("FAST_RUN", "0") == "1"            # True: short trainings for a quick check
SHOW_ARTWORKS = os.environ.get("SHOW_ARTWORKS", "0") == "1"  # True: display paintings in the notebooks
SEED = 0


def set_seed(seed=SEED):
    """Same random numbers for everybody: Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


for _d in (CACHE_DIR, FIGURES_DIR, RESULTS_DIR, PRIVATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
