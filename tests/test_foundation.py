"""Checks of the zero-shot logic (no model download needed)."""
import torch

from museum_guide.foundation import zero_shot_predict


def test_zero_shot_picks_the_closest_sentence_by_cosine():
    class_text = torch.eye(3)  # three classes along the three axes
    images = torch.tensor([[5.0, 0.1, 0.0],   # long vector, closest to class 0
                           [0.1, 0.2, 0.9],   # closest to class 2
                           [0.0, 2.0, 1.0]])  # closest to class 1
    assert zero_shot_predict(images, class_text).tolist() == [0, 2, 1]


def test_zero_shot_ignores_the_length_of_the_image_vector():
    class_text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    a = torch.tensor([[0.9, 0.1]])
    assert zero_shot_predict(a, class_text).item() == zero_shot_predict(100 * a, class_text).item() == 0
