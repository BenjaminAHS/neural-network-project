"""The training loop of the course (sessions 4 to 6), with the data loaders passed as arguments."""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Loaders:
    train: DataLoader       # shuffled mini-batches, used for the updates
    train_eval: DataLoader  # the whole training set in order, to measure train metrics like val ones
    val: DataLoader


def make_loaders(X_train, y_train, X_val, y_val, batch_size=64, eval_batch_size=512):
    train_ds = TensorDataset(X_train, y_train)
    return Loaders(train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                   train_eval=DataLoader(train_ds, batch_size=eval_batch_size),
                   val=DataLoader(TensorDataset(X_val, y_val), batch_size=eval_batch_size))


def random_hflip(xb):
    """Data augmentation: flip half of the images left-right (the style does not depend on the side)."""
    flip = torch.rand(len(xb)) < 0.5
    xb = xb.clone()
    xb[flip] = xb[flip].flip(-1)
    return xb


def crop_and_flip(xb, size=224, scale=(0.6, 1.0)):
    """Data augmentation for the ResNet: a random crop (60 to 100 % of the area) resized back to
    size x size, then a random left-right flip. Works on uint8 images."""
    from torchvision.transforms import v2
    crop = v2.RandomResizedCrop(size, scale=scale, antialias=True)
    return random_hflip(torch.stack([crop(x) for x in xb]))


def train_one_epoch(model, loader, loss_fn, optimizer, augment=None, clip_norm=None):
    """One pass over the training data with the 5 lines of the training loop; returns the mean loss."""
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        if augment is not None:
            xb = augment(xb)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / n


@torch.no_grad()
def predict(model, loader):
    """Logits and labels of a whole loader, in evaluation mode."""
    model.eval()
    logits, labels = [], []
    for xb, yb in loader:
        logits.append(model(xb))
        labels.append(yb)
    return torch.cat(logits), torch.cat(labels)


def evaluate(model, loader, loss_fn):
    """(mean loss, accuracy) in evaluation mode, without gradients."""
    logits, y = predict(model, loader)
    return loss_fn(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item()


def new_history():
    return {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}


def record(history, model, optimizer, loaders, loss_fn, epoch_time):
    """Measure the model on the training and validation sets (same way, eval mode) and store it."""
    tl, ta = evaluate(model, loaders.train_eval, loss_fn)
    vl, va = evaluate(model, loaders.val, loss_fn)
    history["train_loss"].append(tl)
    history["train_acc"].append(ta)
    history["val_loss"].append(vl)
    history["val_acc"].append(va)
    history["lr"].append(optimizer.param_groups[0]["lr"])
    history["epoch_time"].append(epoch_time)


def fit(model, optimizer, loaders, n_epochs, loss_fn=None, scheduler=None, patience=None, augment=None,
        clip_norm=None, train_epoch_fn=train_one_epoch, on_epoch_end=None, verbose=False, seed=1):
    """Train for n_epochs and return the history (one value per epoch in each list).

    patience=k turns on early stopping: training stops k epochs after the best validation loss,
    and the weights of the best epoch are loaded back into the model.
    on_epoch_end(model, history) is called after each epoch, e.g. to record gradient norms.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    torch.manual_seed(seed)  # same order of the mini-batches for every run
    history = new_history()
    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        train_epoch_fn(model, loaders.train, loss_fn, optimizer, augment, clip_norm)
        record(history, model, optimizer, loaders, loss_fn, time.perf_counter() - t0)
        if on_epoch_end is not None:
            on_epoch_end(model, history)
        if scheduler is not None:
            scheduler.step()
        if verbose:
            print(f"epoch {epoch:3d}  train loss {history['train_loss'][-1]:.3f}  "
                  f"val loss {history['val_loss'][-1]:.3f}  val acc {history['val_acc'][-1]:.3f}")
        if patience is not None:
            if history["val_loss"][-1] < best_loss:
                best_loss, best_state, wait = history["val_loss"][-1], copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def best_epoch(history):
    """Epoch (counted from 1) with the smallest validation loss, or None if the loss was never finite."""
    losses = np.array(history["val_loss"], dtype=float)
    losses[~np.isfinite(losses)] = np.inf  # a nan epoch can never be the best one
    return None if np.isinf(losses).all() else int(np.argmin(losses)) + 1


def summary(history):
    """The numbers we report for one run: at the best epoch and at the last epoch."""
    b = best_epoch(history)
    out = {"epochs_run": len(history["val_loss"]),
           "final_train_acc": history["train_acc"][-1], "final_val_acc": history["val_acc"][-1],
           "final_gap": history["train_acc"][-1] - history["val_acc"][-1],
           "seconds_per_epoch": float(np.mean(history["epoch_time"]))}
    if b is not None:
        i = b - 1
        out.update(best_epoch=b, best_val_loss=history["val_loss"][i], val_acc_at_best=history["val_acc"][i],
                   gap_at_best=history["train_acc"][i] - history["val_acc"][i])
    return out
