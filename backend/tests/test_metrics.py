from app.ml.explain import cam_to_mask, dice_coefficient
import numpy as np


def test_dice_perfect():
    m = np.ones((8, 8), dtype=np.uint8)
    assert dice_coefficient(m, m) == 1.0


def test_cam_threshold():
    cam = np.array([[0.1, 0.9], [0.6, 0.2]])
    mask = cam_to_mask(cam, 0.55)
    assert mask[0, 1] == 1
    assert mask[0, 0] == 0
