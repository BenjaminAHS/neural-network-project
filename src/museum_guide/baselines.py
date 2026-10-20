"""Classical models (no neural network) for the comparison: hand-made features, logistic regression and
random forest from scikit-learn."""
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _orientation_features(grey, bins=8):
    """Histogram of edge directions (0 to 180 degrees), weighted by edge strength, plus the mean edge
    strength and the share of strong edges: a simple description of the texture and the brush strokes."""
    gy, gx = np.gradient(grey)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(angle, bins=bins, range=(0, np.pi), weights=magnitude)
    return list(hist / (magnitude.sum() + 1e-8)) + [magnitude.mean(), (magnitude > 0.1).mean()]


def hand_features(image):
    """102 hand-made numbers for one uint8 image of shape (3, H, W):

    - colour: histogram of hue (8 bins) x saturation (3) x value (3) in HSV space, 72 numbers;
    - colour statistics: mean and std of R, G, B, of the brightness and of the saturation, 10 numbers;
    - texture: histogram of edge directions and edge strength, at full and at half resolution, 20 numbers.
    """
    rgb = np.ascontiguousarray(np.asarray(image).transpose(1, 2, 0))
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255
    hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 3, 3), range=((0, 1), (0, 1), (0, 1)))
    x = rgb.astype(np.float32) / 255
    stats = [x[..., c].mean() for c in range(3)] + [x[..., c].std() for c in range(3)]
    stats += [hsv[..., 2].mean(), hsv[..., 2].std(), hsv[..., 1].mean(), hsv[..., 1].std()]
    grey = x.mean(axis=-1)
    h, w = grey.shape[0] // 2 * 2, grey.shape[1] // 2 * 2
    half = grey[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    texture = _orientation_features(grey) + _orientation_features(half)
    return np.concatenate([hist.ravel() / grey.size, stats, texture]).astype(np.float32)


def hand_features_batch(images):
    return np.stack([hand_features(img) for img in images])


def make_logreg(C=1.0):
    """Standardisation then multinomial logistic regression (a linear model, like our linear probe)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def make_forest(seed=0, max_features="sqrt", min_samples_leaf=1):
    return RandomForestClassifier(n_estimators=500, max_features=max_features, min_samples_leaf=min_samples_leaf,
                                  random_state=seed, n_jobs=-1)


def tune_on_validation(build, grid, X_train, y_train, X_val, y_val):
    """Fit one model per setting of the grid on the training set and keep the best on validation.
    Returns (best fitted model, best setting, {setting: validation accuracy})."""
    scores, models = {}, {}
    for setting in grid:
        model = build(**setting).fit(X_train, y_train)
        key = ", ".join(f"{k}={v}" for k, v in setting.items())
        scores[key], models[key] = model.score(X_val, y_val), (model, setting)
    best = max(scores, key=scores.get)
    return models[best][0], models[best][1], scores
