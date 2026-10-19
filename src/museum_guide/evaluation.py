"""Save every measured number, so that the README and the slides quote files, not memory."""
import json

import numpy as np

from .config import RESULTS_DIR


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
