"""The training recipes we chose, written once so that every notebook uses exactly the same settings."""
import copy

import torch

from .models import ResNetClassifier, init_weights, make_head, make_mlp
from .training import crop_and_flip, fit, make_loaders, random_hflip

# Notebook 02: plain MLP, Adam at its usual learning rate, no regularisation.
MLP_BASELINE = dict(size=32, hidden=(512, 256), lr=1e-3, epochs=40)

# Notebook 03: learning rate from the sweep, and all the tools of session 6 together.
MLP_REGULARISED = dict(size=32, hidden=(512, 256), batchnorm=True, dropout=0.5, input_dropout=0.2,
                       lr=1e-4, weight_decay=5.0, cosine=True, flip=True, epochs=60, patience=8)

# Notebook 05: fine-tuning of the last block of ResNet18 (with crop and flip augmentation).
FINETUNE = dict(train_from="layer4", lr_body=1e-4, lr_head=1e-3, weight_decay=1e-2, epochs=10, patience=3)


def train_mlp(loaders, seed=0, size=32, hidden=(512, 256), batchnorm=False, dropout=0.0, input_dropout=0.0,
              lr=1e-4, weight_decay=0.0, cosine=False, flip=False, epochs=40, patience=None, on_epoch_end=None):
    """Build an MLP (He init, small last layer) and train it with AdamW; returns (model, history).

    AdamW with weight_decay=0 is plain Adam. With weight decay, AdamW multiplies every weight by
    (1 - lr * weight_decay) at each step.
    """
    torch.manual_seed(seed)  # same initial weights for the same seed
    model = init_weights(make_mlp(3 * size * size, hidden, 5, batchnorm=batchnorm, dropout=dropout,
                                  input_dropout=input_dropout), last_scale=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None
    history = fit(model, optimizer, loaders, epochs, scheduler=scheduler, patience=patience,
                  augment=random_hflip if flip else None, on_epoch_end=on_epoch_end, seed=seed + 1)
    return model, history


def train_head(F_train, y_train, F_val, y_val, seed=0, hidden=(), dropout=0.0, lr=1e-3, weight_decay=1e-2,
               epochs=200, patience=20, batch_size=64):
    """Train a head on frozen ResNet features (linear probe if hidden=()); returns (model, history)."""
    torch.manual_seed(seed)
    model = make_head(F_train, hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loaders = make_loaders(F_train, y_train, F_val, y_val, batch_size=batch_size)
    return model, fit(model, optimizer, loaders, epochs, patience=patience, seed=seed + 1)


def train_finetune(loaders, seed=0, train_from="layer4", head=None, lr_body=1e-4, lr_head=1e-3, weight_decay=1e-2,
                   epochs=10, patience=3, augment=crop_and_flip):
    """Fine-tune a pretrained ResNet18 from the block train_from (the blocks before stay frozen).

    head: a head already trained on the frozen features (a copy is used), or None for a new linear head.
    The pretrained block gets a small learning rate: its weights are already good, we only want to adapt
    them to paintings. Cosine schedule and early stopping on the validation loss.
    """
    torch.manual_seed(seed)
    model = ResNetClassifier(train_from=train_from, head=copy.deepcopy(head) if head is not None else None)
    if head is None:
        init_weights(model.head, last_scale=0.1)
    body = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": body, "lr": lr_body}, {"params": model.head.parameters(), "lr": lr_head}],
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = fit(model, optimizer, loaders, epochs, scheduler=scheduler, patience=patience, augment=augment,
                  seed=seed + 1)
    return model, history
