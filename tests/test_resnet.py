"""Checks of the ResNet wrapper (built without pretrained weights, so no download is needed)."""
import torch
import torch.nn as nn

from museum_guide.models import ImageNetNormalise, ResNetClassifier, count_trainable, make_head
from museum_guide.training import crop_and_flip


def test_frozen_blocks_have_no_trainable_weights_and_stay_in_eval_mode():
    model = ResNetClassifier(train_from="layer4", pretrained=False)
    trainable_body = sum(p.numel() for p in model.backbone.layer4.parameters())
    assert count_trainable(model) == trainable_body + 512 * 5 + 5
    model.train()
    assert not model.backbone.layer3.training and not model.backbone.bn1.training
    assert model.backbone.layer4.training and model.head.training


def test_feature_extractor_freezes_the_whole_body():
    model = ResNetClassifier(train_from=None, pretrained=False)
    assert count_trainable(model) == 512 * 5 + 5


def test_output_shape_from_uint8_images():
    model = ResNetClassifier(pretrained=False).eval()
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    assert model(x).shape == (2, 5)
    assert model.features(x).shape == (2, 512)


def test_imagenet_normalisation():
    x = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)
    out = ImageNetNormalise()(x)
    assert torch.allclose(out[0, :, 0, 0], torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]))
    assert torch.allclose(ImageNetNormalise(enabled=False)(x), torch.ones(1, 3, 2, 2))


def test_head_standardises_with_training_statistics():
    torch.manual_seed(0)
    F = torch.randn(100, 512) * 3 + 7
    head = make_head(F)
    assert torch.allclose(head[0](F).mean(dim=0), torch.zeros(512), atol=1e-4)
    assert sum(isinstance(m, nn.Linear) for m in head.modules()) == 1


def test_crop_and_flip_keeps_shape_and_type():
    xb = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    out = crop_and_flip(xb, size=64)
    assert out.shape == xb.shape and out.dtype == torch.uint8
