from bayer import improved_interpolation
from numpy import array

def test_improved_interpolation():
    raw_img = array([[8, 5, 3, 7, 1, 3],
                     [5, 2, 6, 8, 8, 1],
                     [9, 9, 8, 1, 6, 4],
                     [9, 4, 2, 3, 6, 8],
                     [5, 4, 3, 2, 8, 7],
                     [7, 3, 3, 6, 9, 3]], dtype='uint8')

    img = improved_interpolation(raw_img)
    assert abs(img[2:-2, 2:-2, 0] - array([[6, 1], [1, 0]])).max() <= 1
    assert abs(img[2:-2, 2:-2, 1] - array([[8, 4], [2, 3]])).max() <= 1
    assert abs(img[2:-2, 2:-2, 2] - array([[7, 2], [2, 2]])).max() <= 1
