from bayer import get_colored_img
from numpy import array

def test_colored_img():
    raw_img = array([[1, 2], [3, 4]])
    res = array([[[0, 1, 0], [2, 0, 0]],
                 [[0, 0, 3], [0, 4, 0]]])
    output = get_colored_img(raw_img)
    assert output.shape == (2, 2, 3)
    assert (output == res).all()

def test_colored_img_2():
    raw_img = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    res = array([[[0, 1, 0], [2, 0, 0], [0, 3, 0]],
                 [[0, 0, 4], [0, 5, 0], [0, 0, 6]],
                 [[0, 7, 0], [8, 0, 0], [0, 9, 0]]])
    output = get_colored_img(raw_img)
    assert output.shape == (3, 3, 3)
    assert (output == res).all()
