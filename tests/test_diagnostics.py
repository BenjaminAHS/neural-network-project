"""Checks of the monitoring tools and of the metrics."""
import torch
import torch.nn as nn

from museum_guide.diagnostics import activation_stats, layer_grad_norms, make_monitor, weight_norm
from museum_guide.evaluation import confusion_matrix, most_confused, per_class_report
from museum_guide.models import init_weights, make_mlp
from museum_guide.training import fit, make_loaders


def test_layer_grad_norms_one_value_per_linear_layer():
    torch.manual_seed(0)
    model = init_weights(make_mlp(12, hidden=(8, 8), n_out=3))
    norms = layer_grad_norms(model, torch.randn(10, 12), torch.arange(10) % 3)
    assert len(norms) == 3 and all(n > 0 for n in norms)
    assert all(p.grad is None for p in model.parameters()), "the gradients must be cleaned afterwards"


def test_dead_units_are_detected():
    torch.manual_seed(0)
    model = init_weights(make_mlp(12, hidden=(8, 8), n_out=3))
    with torch.no_grad():
        model[1].bias.fill_(-100.0)  # first hidden layer: every pre-activation is negative
    stats = activation_stats(model, torch.randn(50, 12))
    assert stats[0]["dead"] == 1.0 and stats[0]["std"] == 0.0


def test_monitor_records_one_entry_per_epoch():
    torch.manual_seed(0)
    X, y = torch.randn(64, 12), torch.arange(64) % 3
    model = init_weights(make_mlp(12, hidden=(8,), n_out=3))
    history = fit(model, torch.optim.Adam(model.parameters()), make_loaders(X, y, X, y, batch_size=16), 3,
                  on_epoch_end=make_monitor(X[:32], y[:32]))
    assert len(history["grad_norms"]) == len(history["dead_units"]) == len(history["weight_norm"]) == 3
    assert len(history["grad_norms"][0]) == 2


def test_weight_norm_counts_weights_not_biases():
    layer = nn.Linear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 4.0]]))
        layer.bias.fill_(100.0)
    assert weight_norm(nn.Sequential(layer)) == 5.0


def test_confusion_matrix_and_report():
    y_true = torch.tensor([0, 0, 0, 1, 1, 2])
    y_pred = torch.tensor([0, 0, 1, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 2, 0], [0, 1, 0]]
    report = per_class_report(cm, ["a", "b", "c"])
    assert abs(report.loc["a", "recall"] - 2 / 3) < 1e-9 and report.loc["b", "precision"] == 0.5
    assert most_confused(cm, ["a", "b", "c"], k=1) == [("a", "b", 1)]
