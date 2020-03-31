import numpy as np 
import scipy.signal as sc 
from glob import glob
from os.path import abspath, basename, dirname, join
from skimage import img_as_ubyte
from skimage.io import imread, imsave, imshow
from pytest import raises


def get_bayer_masks(n_rows, n_cols):
    red = np.zeros((n_rows, n_cols))
    green = np.zeros((n_rows, n_cols))
    blue = np.zeros((n_rows, n_cols))
    red_row = np.tile([0, 1], n_cols // 2)
    green_row1 = np.tile([1, 0], n_cols // 2)
    green_row2 = np.tile([0, 1], n_cols // 2)
    blue_row = np.tile([1,0], n_cols // 2)
    if n_cols % 2:
        red_row = np.append(red_row, [0])
        blue_row = np.append(blue_row, [1])
        green_row1 = np.append(green_row1, [1])
        green_row2 = np.append(green_row2, [0])
    for i in range(0, n_rows, 1):
        if i % 2:
            np.copyto(blue[i], blue_row)
            np.copyto(green[i],green_row2)
        else:
            np.copyto(red[i], red_row)
            np.copyto(green[i], green_row1)
    return np.stack([red, green, blue], axis = 2)

def get_colored_img(raw_img):
    rows = raw_img.shape[0]
    cols = raw_img.shape[1]
    masks = get_bayer_masks(rows, cols)
    red = masks[..., 0]
    green = masks[..., 1]
    blue = masks[..., 2]
    x = raw_img*red
    x = x.reshape(-1, 1)
    x = np.hstack((x,(raw_img*green).reshape(-1,1)))
    x = np.hstack((x,(raw_img*blue).reshape(-1,1)))
    x = x.reshape(rows, cols, 3)
    return x

def bilinear_interpolation(colored_img):
    length, width = colored_img.shape[:2]
    kernel = np.ones((3, 3))
    masks = get_bayer_masks(length, width)
    red_mask1 = masks[..., 0]
    green_mask1 = masks[..., 1]
    blue_mask1 = masks[..., 2]
    red_mask2 = red_mask1[1 : length - 1, 1 : width - 1]
    green_mask2 = green_mask1[1 : length - 1, 1 : width - 1]
    blue_mask2 = blue_mask1[1 : length - 1, 1 : width - 1]
    red = sc.convolve2d(colored_img[..., 0], kernel, mode = 'valid') * (green_mask2 + blue_mask2)
    red = red / (sc.convolve2d(red_mask1, kernel, mode = 'valid'))
    green = sc.convolve2d(colored_img[..., 1], kernel, mode = 'valid') * (red_mask2 + blue_mask2)
    green = green / (sc.convolve2d(green_mask1, kernel, mode = 'valid'))
    blue = sc.convolve2d(colored_img[..., 2], kernel, mode = 'valid') * (green_mask2 + red_mask2)
    blue = blue / (sc.convolve2d(blue_mask1, kernel, mode = 'valid'))
                     
    colored_img[...,0][1 : length - 1, 1 : width - 1] = colored_img[...,0][1 : length - 1, 1 : width - 1] + red
    colored_img[...,1][1 : length - 1, 1 : width - 1] = colored_img[...,1][1 : length - 1, 1 : width - 1] + green
    colored_img[...,2][1 : length - 1, 1 : width - 1] = colored_img[...,2][1 : length - 1, 1 : width - 1] + blue
    colored_img = colored_img.astype(np.uint16)
    return colored_img

def improved_interpolation(raw_img):
    
    raw_img = get_colored_img(raw_img)
    
    raw_img = raw_img / 1.0
    two_r = np.array(([[0, 0, 0, 0, 0],
             [0, 0, 0.25, 0, 0],
             [0, 0.25, 0, 0.25, 0],
             [0, 0, 0.25, 0, 0],
             [0, 0, 0, 0, 0]]))
    two_b = np.array(([[0, 0, 0, 0, 0],
             [0, 0.25, 0, 0.25, 0],
             [0, 0, 0, 0, 0],
             [0, 0.25, 0, 0.25, 0],
             [0, 0, 0, 0, 0]]))
    onefour_g = np.array(([[0, 0, -0.125, 0, 0],
             [0, 0, 0, 0, 0],
             [-0.125, 0, 0.5, 0, -0.125],
             [0, 0, 0, 0, 0],
             [0, 0, -0.125, 0, 0]]))
    r_green_1 = np.array(([[0, 0, 0.0625, 0, 0],
             [0, -0.125, 0, -0.125, 0],
             [-0.125, 0, 0.625, 0, -0.125],
             [0, -0.125, 0, -0.125, 0],
             [0, 0, 0.0625, 0, 0]]))
    r_green_2 = np.array(([[0, 0, -0.125, 0, 0],
             [0, -0.125, 0, -0.125, 0],
             [0.0625, 0, 0.625, 0, 0.0625],
             [0, -0.125, 0, -0.125, 0],
             [0, 0, -0.125, 0, 0]]))
    r_blue_1 = np.array(([[0, 0, -0.1875, 0, 0],
             [0, 0, 0, 0, 0],
             [-0.1875, 0, 0.75, 0, -0.1875],
             [0, 0, 0, 0, 0],
             [0, 0, -0.1875, 0, 0]]))
    four_2 = np.array(([[0, 0, 0, 0, 0],
             [0, 0, 0.5, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0.5, 0, 0],
             [0, 0, 0, 0, 0]]))
    four_1 = np.array(([[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0.5, 0, 0.5, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]])) 
    
    length, width = raw_img.shape[:2]
    masks = get_bayer_masks(length, width)
    red_mask = masks[..., 0][2 : length - 2, 2 : width - 2]
    blue_mask = masks[..., 2][2 : length - 2, 2 : width - 2]
    gr_mask = get_bayer_masks(length, width + 1)
    gr_mask_1 = gr_mask[..., 0][:, 1 : width]
    gr_mask_1 = gr_mask_1[2 : length - 2, 2 : width - 2]
    gr_mask_2 = masks[...,1][2 : length - 2, 2 : width - 2] - gr_mask_1
    
    red = sc.convolve2d(raw_img[..., 2], r_blue_1, mode = 'valid') * blue_mask + \
    sc.convolve2d(raw_img[..., 0], two_b, mode = 'valid') * blue_mask + \
    sc.convolve2d(raw_img[..., 1], r_green_1, mode = 'valid') * gr_mask_1 + \
    sc.convolve2d(raw_img[..., 0], four_1, mode = 'valid') * gr_mask_1 + \
    sc.convolve2d(raw_img[..., 1], r_green_2, mode = 'valid') * gr_mask_2 + \
    sc.convolve2d(raw_img[..., 0], four_2, mode = 'valid') * gr_mask_2 
    
    green = sc.convolve2d(raw_img[..., 0], onefour_g, mode = 'valid') * red_mask + \
    sc.convolve2d(raw_img[..., 1], two_r, mode = 'valid') * red_mask + \
    sc.convolve2d(raw_img[..., 2], onefour_g, mode = 'valid') * blue_mask + \
    sc.convolve2d(raw_img[..., 1], two_r, mode = 'valid') * blue_mask
    
    
    blue = sc.convolve2d(raw_img[..., 0], r_blue_1, mode = 'valid') * red_mask + \
    sc.convolve2d(raw_img[..., 2], two_b, mode = 'valid') * red_mask + \
    sc.convolve2d(raw_img[..., 1], r_green_1, mode = 'valid') * gr_mask_2 + \
    sc.convolve2d(raw_img[..., 2], four_1, mode = 'valid') * gr_mask_2 + \
    sc.convolve2d(raw_img[..., 1], r_green_2, mode = 'valid') * gr_mask_1 +\
    sc.convolve2d(raw_img[..., 2], four_2, mode = 'valid') * gr_mask_1 
    
    red = np.clip(red, 0, 255)
    green = np.clip(green, 0, 255)
    blue = np.clip(blue, 0, 255)
    
    raw_img[..., 0][2 : length - 2, 2 : width - 2] = red + raw_img[..., 0][2 : length - 2, 2 : width - 2]
    raw_img[..., 1][2 : length - 2, 2 : width - 2] = green + raw_img[..., 1][2 : length - 2, 2 : width - 2]
    raw_img[..., 2][2 : length - 2, 2 : width - 2] = blue + raw_img[..., 2][2 : length - 2, 2 : width - 2]
    raw_img = raw_img.astype(np.uint8)
    return raw_img

def mse(img1, img2):
    if(img1.ndim == 3):
        length, width, heigth = img1.shape[:3]
        temp = img1 - img2 
        temp = temp ** 2 
        temp = (1 / (length * width * heigth)) * np.sum(temp)
        return temp
    elif(img1.ndim == 2):
        length, width = img1.shape[:2]
        temp = img1 - img2 
        temp = temp ** 2 
        temp = (1 / (length * width)) * np.sum(temp)
        return temp
    else:
        length = img1.shape
        temp = img1 - img2 
        temp = temp ** 2 
        temp = (1 / length) * np.sum(temp)
        return temp

def compute_psnr(img_pred, img_gt):
    img_pred = img_pred / 1.0
    img_gt = img_gt / 1.0
    M = mse(img_pred, img_gt)
    if(M == 0):
        raise(ValueError)
    psnr = img_gt.max()
    psnr = psnr ** 2
    psnr = psnr / M
    psnr = np.log10(psnr)
    psnr = psnr * 10
    return psnr

