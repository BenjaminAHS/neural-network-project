"""Checks of the MLP and of the training loop on tiny synthetic data."""
import math

import torch
import torch.nn as nn

from museum_guide.diagnostics import initial_loss, overfit_one_batch
from museum_guide.models import count_parameters, init_weights, make_mlp, mlp_parameters
from museum_guide.training import evaluate, fit, make_loaders, random_hflip, summary


def test_mlp_output_shape_and_parameter_count():
    model = make_mlp(3 * 8 * 8, hidden=(16, 8), n_out=5)
    assert model(torch.zeros(4, 3, 8, 8)).shape == (4, 5)
    assert count_parameters(model) == mlp_parameters(192, (16, 8), 5) == 16 * 193 + 8 * 17 + 5 * 9


def test_baseline_mlp_has_6_4_million_parameters():
    assert count_parameters(make_mlp(3 * 64 * 64)) == 6_424_581


def test_init_gives_zero_biases_and_a_first_loss_near_ln_c():
    torch.manual_seed(0)
    model = init_weights(make_mlp(3 * 16 * 16, hidden=(64, 32)), last_scale=0.01)
    assert all((m.bias == 0).all() for m in model.modules() if isinstance(m, nn.Linear))
    X, y = torch.randn(200, 3, 16, 16), torch.arange(200) % 5
    assert abs(initial_loss(model, X, y) - math.log(5)) < 0.05


def test_overfit_one_batch_reaches_100_percent():
    torch.manual_seed(0)
    X, y = torch.randn(16, 3, 4, 4), torch.arange(16) % 5
    model = init_weights(make_mlp(48, hidden=(64,)))
    losses, acc = overfit_one_batch(model, X, y, torch.optim.Adam(model.parameters(), lr=1e-2), steps=150)
    assert acc == 1.0 and losses[-1] < 0.05


def test_early_stopping_reloads_the_best_epoch():
    torch.manual_seed(0)
    X, y = torch.randn(120, 3, 4, 4), torch.randint(0, 5, (120,))  # random labels: val loss rises fast
    loaders = make_loaders(X[:80], y[:80], X[80:], y[80:], batch_size=16)
    model = init_weights(make_mlp(48, hidden=(128,)))
    history = fit(model, torch.optim.Adam(model.parameters(), lr=1e-2), loaders, n_epochs=30, patience=3)
    s = summary(history)
    assert s["epochs_run"] == s["best_epoch"] + 3
    val_loss, _ = evaluate(model, loaders.val, nn.CrossEntropyLoss())
    assert abs(val_loss - s["best_val_loss"]) < 1e-5


def test_random_hflip_flips_about_half_the_images():
    torch.manual_seed(0)
    xb = torch.arange(1000 * 4, dtype=torch.float).reshape(1000, 1, 1, 4)
    flipped = (random_hflip(xb) != xb).any(dim=(1, 2, 3)).float().mean().item()
    assert 0.4 < flipped < 0.6
