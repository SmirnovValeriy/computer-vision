from numpy import array, dsplit
from bayer import get_bayer_masks

def test_masks():
    masks = get_bayer_masks(2, 2)
    assert masks.shape == (2, 2, 3)
    assert (masks[..., 0] == array([[0, 1], [0, 0]])).all()
    assert (masks[..., 1] == array([[1, 0], [0, 1]])).all()
    assert (masks[..., 2] == array([[0, 0], [1, 0]])).all()

def test_masks_2():
    masks = get_bayer_masks(3, 3)
    assert masks.shape == (3, 3, 3)
    assert (masks[..., 0] == array([[0, 1, 0],
                                    [0, 0, 0],
                                    [0, 1, 0]])).all()
    assert (masks[..., 1] == array([[1, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 1]])).all()
    assert (masks[..., 2] == array([[0, 0, 0],
                                    [1, 0, 1],
                                    [0, 0, 0]])).all()

def test_masks_3():
    masks = get_bayer_masks(4, 4)
    assert masks.shape == (4, 4, 3)
    assert (masks[..., 0] == array([[0, 1, 0, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 0, 0]])).all()
    assert (masks[..., 1] == array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [1, 0, 1, 0],
                                    [0, 1, 0, 1]])).all()
    assert (masks[..., 2] == array([[0, 0, 0, 0],
                                    [1, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [1, 0, 1, 0]])).all()
