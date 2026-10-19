"""Checks that tell us whether a training run is healthy (course, session 6, slide "diagnosing a run")."""
import math

import torch
import torch.nn as nn


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
