from bayer import bilinear_interpolation, get_colored_img
from numpy import array
from skimage import img_as_ubyte

def test_bilinear_interpolation():
    raw_img = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='uint8')
    colored_img = get_colored_img(raw_img)

    img = img_as_ubyte(bilinear_interpolation(colored_img))
    assert (img[1, 1] == [5, 5, 5]).all()
