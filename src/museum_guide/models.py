"""Our networks: feed-forward MLPs built from nn.Linear (as in the course), and a pretrained ResNet18
with our own head for transfer learning."""
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # the pixel statistics the pretrained ResNet was trained with
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET_BLOCKS = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


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


class Standardise(nn.Module):
    """(x - mean) / std with fixed statistics (of the training features), stored inside the model."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone().clamp_min(1e-6))

    def forward(self, x):
        return (x - self.mean) / self.std


def make_head(train_features, hidden=(), n_classes=5, dropout=0.0):
    """A head for frozen ResNet features: standardisation with the training statistics, then one
    nn.Linear (hidden=(), a "linear probe") or a small MLP."""
    body = init_weights(make_mlp(train_features.shape[1], hidden, n_classes, dropout=dropout), last_scale=0.1)
    return nn.Sequential(Standardise(train_features.mean(dim=0), train_features.std(dim=0)), body)


class ImageNetNormalise(nn.Module):
    """Turn uint8 images (0 to 255) into the input scale of the pretrained ResNet: pixels in [0, 1], then
    (x - mean) / std with the ImageNet statistics. enabled=False skips the second step (to show the bug)."""

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255
        return (x - self.mean) / self.std if self.enabled else x


def load_resnet18(pretrained=True):
    """ResNet18 from torchvision, with the weights learned on ImageNet (1.2 million photos, 1 000 classes)."""
    from torchvision.models import ResNet18_Weights, resnet18
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)


class ResNetClassifier(nn.Module):
    """Pretrained ResNet18 body + our own head (one nn.Linear, or a small MLP).

    train_from: the first block that we train ("layer4" = fine-tune the last block only). Every block
    before it is frozen: its weights do not change and its batch norm layers stay in eval mode, so they
    keep the statistics learned on ImageNet. train_from=None freezes the whole body (feature extractor).
    """

    def __init__(self, n_classes=5, train_from="layer4", head=None, normalise=True, pretrained=True):
        super().__init__()
        self.normalise = ImageNetNormalise(normalise)
        self.backbone = load_resnet18(pretrained)
        self.backbone.fc = nn.Identity()  # the body now returns 512 numbers per image
        self.head = head if head is not None else nn.Linear(512, n_classes)
        k = RESNET_BLOCKS.index(train_from) if train_from else len(RESNET_BLOCKS)
        self.frozen_blocks = RESNET_BLOCKS[:k]
        for name in self.frozen_blocks:
            for p in getattr(self.backbone, name).parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for name in self.frozen_blocks:  # frozen blocks always behave as at test time
            getattr(self.backbone, name).eval()
        return self

    def features(self, x):
        """512 numbers per image: the average of the last feature maps of the ResNet."""
        return self.backbone(self.normalise(x))

    def forward(self, x):
        return self.head(self.features(x))


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
