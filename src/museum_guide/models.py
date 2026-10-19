"""Feed-forward networks built from PyTorch primitives (nn.Linear and activations), as in the course."""
import torch
import torch.nn as nn


def make_mlp(n_in, hidden=(512, 256), n_out=5, activation=nn.ReLU, batchnorm=False, dropout=0.0,
             input_dropout=0.0):
    """Flatten, then [Linear, (BatchNorm), activation, (Dropout)] for each hidden layer, then Linear.

    No softmax at the end: nn.CrossEntropyLoss takes the raw scores (logits).
    """
    layers = [nn.Flatten()]
    if input_dropout:
        layers.append(nn.Dropout(input_dropout))
    for width in hidden:
        layers.append(nn.Linear(n_in, width))
        if batchnorm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(activation())
        if dropout:
            layers.append(nn.Dropout(dropout))
        n_in = width
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


def init_weights(model, kind="he", last_scale=1.0):
    """He (Kaiming) for ReLU networks or Xavier for sigmoid/tanh, biases at 0.

    last_scale multiplies the weights of the output layer. A small value (0.1) gives first logits
    close to 0, so the first loss is close to ln(C): the loss of a model that knows nothing yet.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for layer in linears:
        if kind == "he":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif kind == "xavier":
            nn.init.xavier_normal_(layer.weight)
        else:
            raise ValueError(f"unknown init {kind!r}")
        nn.init.zeros_(layer.bias)
    with torch.no_grad():
        linears[-1].weight.mul_(last_scale)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def mlp_parameters(n_in, hidden, n_out):
    """Formula of session 3: a layer with n_out neurons and n_in inputs has n_out (n_in + 1) parameters."""
    sizes = [n_in, *hidden, n_out]
    return sum(b * (a + 1) for a, b in zip(sizes[:-1], sizes[1:]))
