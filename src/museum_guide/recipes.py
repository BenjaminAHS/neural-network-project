"""The training recipes we chose, written once so that every notebook uses exactly the same settings."""
import torch

from .models import init_weights, make_mlp
from .training import fit, random_hflip

# Notebook 02: plain MLP, Adam at its usual learning rate, no regularisation.
MLP_BASELINE = dict(size=32, hidden=(512, 256), lr=1e-3, epochs=40)

# Notebook 03: learning rate from the sweep, and all the tools of session 6 together.
MLP_REGULARISED = dict(size=32, hidden=(512, 256), batchnorm=True, dropout=0.5, input_dropout=0.2,
                       lr=1e-4, weight_decay=5.0, cosine=True, flip=True, epochs=60, patience=8)


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
