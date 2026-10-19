"""Checks that tell us whether a training run is healthy (course, session 6, slide "diagnosing a run")."""
import math

import torch
import torch.nn as nn

from .training import fit

ACTIVATIONS = (nn.ReLU, nn.Sigmoid, nn.Tanh)


@torch.no_grad()
def initial_loss(model, X, y, loss_fn=None):
    """Loss of the untrained model. A model that knows nothing predicts 1/C for each class: ln(C)."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()
    return loss_fn(model(X), y).item()


def chance_loss(n_classes):
    return math.log(n_classes)


def overfit_one_batch(model, xb, yb, optimizer, steps=200, loss_fn=None):
    """Train on ONE small batch only. A healthy model and loop must learn it by heart (loss near 0,
    accuracy 100 %). If it cannot, there is a bug before any question of generalisation."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    losses = []
    for _ in range(steps):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        acc = (model(xb).argmax(dim=1) == yb).float().mean().item()
    return losses, acc


def layer_grad_norms(model, X, y, loss_fn=None):
    """Norm of the gradient of each nn.Linear weight matrix, on one fixed batch (course, session 5).

    Measured in eval mode so that batch norm does not update its running statistics. Very small
    norms in the first layers mean a vanishing gradient: those layers almost stop learning.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    model.eval()
    model.zero_grad()
    loss_fn(model(X), y).backward()
    norms = [layer.weight.grad.norm().item() for layer in linears]
    model.zero_grad()
    return norms


@torch.no_grad()
def activation_stats(model, X):
    """For each hidden activation layer: standard deviation of its outputs and share of dead units
    (a dead ReLU unit gives 0 for every image, so it never learns again)."""
    outputs = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: outputs.append(out.flatten(1)))
             for m in model.modules() if isinstance(m, ACTIVATIONS)]
    model.eval()
    model(X)
    for h in hooks:
        h.remove()
    return [{"std": a.std().item(), "dead": (a == 0).all(dim=0).float().mean().item()} for a in outputs]


def weight_norm(model):
    """Square root of the sum of all squared weights of the nn.Linear layers (what weight decay shrinks)."""
    return math.sqrt(sum((m.weight ** 2).sum().item() for m in model.modules() if isinstance(m, nn.Linear)))


def make_monitor(X, y, loss_fn=None):
    """A function for fit(on_epoch_end=...) that stores, after each epoch, the gradient norm of each
    layer, the activation statistics and the weight norm, measured on the fixed batch (X, y)."""
    def monitor(model, history):
        stats = activation_stats(model, X)
        history.setdefault("grad_norms", []).append(layer_grad_norms(model, X, y, loss_fn))
        history.setdefault("act_std", []).append([s["std"] for s in stats])
        history.setdefault("dead_units", []).append([s["dead"] for s in stats])
        history.setdefault("weight_norm", []).append(weight_norm(model))
    return monitor


def lr_sweep(build_model, lrs, loaders, n_epochs, optimizer=torch.optim.Adam, seed=0):
    """Short trainings of the same model with different learning rates; returns {lr: history}."""
    histories = {}
    for lr in lrs:
        torch.manual_seed(seed)
        model = build_model()
        histories[lr] = fit(model, optimizer(model.parameters(), lr=lr), loaders, n_epochs, seed=seed + 1)
    return histories
