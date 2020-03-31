from bayer import bilinear_interpolation, get_colored_img
from glob import glob
from numpy import abs
from os.path import abspath, basename, dirname, join
from skimage import img_as_ubyte
from skimage.io import imread, imsave

def test():
    test_dir = dirname(abspath(__file__))
    for img_filename in sorted(glob(join(test_dir, '[0-9][0-9].png'))):
        raw_img = img_as_ubyte(imread(join(test_dir, img_filename)))
        img = img_as_ubyte(bilinear_interpolation(get_colored_img(raw_img)))
        out_filename = join(test_dir, 'gt_' + basename(img_filename))
        gt_img = img_as_ubyte(imread(out_filename))
        assert abs(img[1:-1, 1:-1] - gt_img[1:-1, 1:-1]).max() <= 1, f'Testing on img {img_filename} failed'
