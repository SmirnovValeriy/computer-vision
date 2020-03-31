from bayer import compute_psnr
from numpy import array
from pytest import raises

def test_exception():
    img_pred = array([[1, 2, 3]])
    img_gt = img_pred.copy()

    with raises(ValueError):
        compute_psnr(img_pred, img_gt)

def test_psnr():
    img_pred = array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    img_gt = array([[1, 2, 3],
                    [4, 0, 6],
                    [7, 8, 9]])
    assert compute_psnr(img_pred, img_gt) == 14.64787519645937
