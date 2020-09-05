import numpy as np 
import scipy.signal as sc 
import time
from glob import glob
from os.path import join
from skimage import img_as_ubyte
from skimage.io import imread, imsave, imshow
from copy import deepcopy

def seam_carve(img, mode, mask = None):
    heigth, width = img.shape[:2]
    if mask is None:
        mask = np.zeros((heigth, width), dtype = np.int64)
    img1 = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

    vertical_flag = 0 
    if 'vertical' in mode:
        img = np.transpose(img, (1, 0, 2))
        img1 = np.transpose(img1)
        mask = np.transpose(mask)
        heigth, width = img.shape[:2]
        vertical_flag = 1
    
    expand_flag = 0
    if 'expand' in mode: 
        expand_flag = 1
        
    #create partial derivatives masks
    mask_x = np.array(([[0, 0, 0],
                       [1, 0, -1],
                       [0, 0, 0]]))
    mask_y = np.array(([[0, 1, 0],
                       [0, 0, 0],
                       [0, -1, 0]]))
    tic = time.time()
    I_x = sc.convolve2d(img1, mask_x, mode = 'same')
    I_x[:, 0] -= img1[:, 0]
    I_x[:, width - 1] += img1[:, width - 1]
    
    I_y = sc.convolve2d(img1, mask_y, mode = 'same')
    I_y[0, :] -= img1[0, :]
    I_y[heigth - 1, :] += img1[heigth - 1, :]
    
    norm = (I_x ** 2 + I_y ** 2) ** (0.5)
    norm += mask.astype(np.int64) * heigth * width * 256
    
    for i in range(1, heigth, 1):
        for j in range(0, width, 1):
            if j == 0:
                norm[i][j] += min(norm[i - 1][0], norm[i - 1][1])
            elif j == width - 1:
                norm[i][j] += min(norm[i - 1][width - 2], norm[i - 1][width - 1])
            else:
                norm[i][j] += min(norm[i - 1][j - 1], norm[i - 1][j], norm[i - 1][j + 1])          
    curve = np.zeros((heigth, width), dtype = np.int64)
    
    helping = np.zeros(heigth, dtype = np.int64)
    min_j = 0
    for i in range(heigth - 1, -1, -1):
        if i == heigth - 1:
            min_j = np.where(norm[heigth - 1] == min(norm[heigth - 1]))[0][0]
        elif min_j != 0 and min_j != (width - 1):
            temp_arr = norm[i, min_j - 1 : min_j + 2]
            min_j = min_j + np.where(temp_arr == min(temp_arr))[0][0] - 1
        elif min_j == 0:
            temp_arr = norm[i, 0 : 2]
            min_j = min_j + np.where(temp_arr == min(temp_arr))[0][0]
        elif min_j == width - 1:
            temp_arr = norm[i, width - 2 : width]
            min_j = min_j + np.where(temp_arr == min(temp_arr))[0][0] - 1
        else:
            print("ERROR while collecting the curve mask")
        helping[i] = min_j
        curve[i][min_j] += 1
    
    curve = curve.astype(np.int64)
    if not expand_flag:
        my_mask = np.ones((heigth, width), dtype = np.int64)
        my_mask -= curve
        new_img = np.zeros((heigth, width - 1, 3))

        new_img[:,:,0] = img[..., 0][my_mask > 0].reshape((heigth, width - 1))
        new_img[:,:,1] = img[..., 1][my_mask > 0].reshape((heigth, width - 1))
        new_img[:,:,2] = img[..., 2][my_mask > 0].reshape((heigth, width - 1))
        
        mask = mask[my_mask > 0].reshape((heigth, width - 1))
    else:
        new_img = np.zeros((heigth, width + 1, 3))
        mask = mask + curve
        new_mask = np.zeros((heigth, width + 1), dtype = np.int64)
        for i in range(0, heigth, 1):
            if helping[i] == width - 1: 
                new_mask[i] = np.concatenate((mask[i, :], np.array([0])), axis = 0)
                for j in [0, 1, 2]:
                    new_img[i, :, j] = np.concatenate((img[i, :, j], np.array([img[i, width - 1, j]])), axis = 0)
            else:
                new_mask[i] = np.concatenate((mask[i, 0 : helping[i] + 1], np.array([0]), mask[i, helping[i] + 1 :]), axis = 0)
                for j in [0, 1, 2]:
                    new_elem = (int(img[i, helping[i], j]) + int(img[i, helping[i] + 1, j])) // 2
                    new_img[i, :, j] = np.concatenate((img[i, 0 : helping[i] + 1, j], np.array([new_elem], dtype = np.uint8), img[i, helping[i] + 1 :, j]), axis = 0)
                    new_img = new_img.astype(np.uint8)
        mask = new_mask
    
    if(vertical_flag):
        new_img = np.transpose(new_img, (1, 0, 2))
        curve = np.transpose(curve)
        mask = np.transpose(mask)

    return new_img, mask, curve
