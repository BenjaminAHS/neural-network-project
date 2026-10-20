"""Checks of the classical baselines, the bootstrap intervals and the benchmark helpers."""
import numpy as np
import torch
import torch.nn as nn

from museum_guide.baselines import hand_features, hand_features_batch, make_logreg, tune_on_validation
from museum_guide.benchmark import cpu_threads, latency_ms, size_mb
from museum_guide.evaluation import bootstrap_ci, macro_f1, paired_bootstrap


def test_hand_features_have_102_finite_numbers_and_a_normalised_histogram():
    img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    f = hand_features(img)
    assert f.shape == (102,) and np.isfinite(f).all()
    assert abs(f[:72].sum() - 1) < 1e-5


def test_a_flat_image_has_no_edges():
    f = hand_features(torch.full((3, 32, 32), 128, dtype=torch.uint8))
    assert np.allclose(f[82:], 0)


def test_tune_on_validation_keeps_the_best_setting():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    model, best, scores = tune_on_validation(make_logreg, [{"C": 1e-6}, {"C": 1.0}], X[:150], y[:150], X[150:], y[150:])
    assert best == {"C": 1.0} and scores["C=1.0"] > 0.9
    assert hand_features_batch(torch.zeros(2, 3, 8, 8, dtype=torch.uint8)).shape == (2, 102)


def test_bootstrap_interval_contains_the_accuracy():
    correct = np.array([1] * 80 + [0] * 20)
    lo, hi = bootstrap_ci(correct)
    assert lo < 0.8 < hi and hi - lo < 0.2


def test_paired_bootstrap_sees_a_clear_difference():
    a, b = np.ones(100), np.r_[np.ones(60), np.zeros(40)]
    out = paired_bootstrap(a, b)
    assert abs(out["diff"] - 0.4) < 1e-9 and out["ci_low"] > 0.25 and out["share_not_better"] == 0


def test_macro_f1_is_one_for_perfect_predictions():
    y = torch.tensor([0, 1, 2, 2])
    assert macro_f1(y, y, 3) == 1.0


def test_benchmark_helpers():
    old = torch.get_num_threads()
    with cpu_threads(1):
        assert torch.get_num_threads() == 1
        timing = latency_ms(lambda x: x * 2, [torch.ones(3)], warmup=1, runs=5)
    assert torch.get_num_threads() == old
    assert set(timing) == {"median_ms", "p95_ms"}
    assert size_mb(nn.Linear(1000, 100)) > 0.4  # 100 100 float32 numbers = 0.4 MB
