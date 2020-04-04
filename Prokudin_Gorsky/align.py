import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.io import imread, imsave, imshow
from json import dumps, load
from os import environ
from os.path import join
from sys import argv, exit
from PIL import Image

def mse_min(img1, img2, n1, n2, m1, m2):
    length, width = img1.shape[:2]
    min = 100000
    
    if (n1 < 0) and (n2 < 0) and (m1 < 0) and (m2 < 0):
        for i in range(n1, n2 + 1, 1):
            for j in range(m1, m2 + 1, 1):
                t_im = img1[0 : length + i, 0 : width + j] - img2[-i :, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width + j))) * np.sum(t_im) 
          
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [0, width + j]
                    z = [-i, length]
                    w = [-j, width]
    elif (n1 < 0) and (n2 < 0) and (m1 < 0) and (m2 >= 0): 
        for i in range(n1, n2 + 1, 1):
            for j in range(m1, 0, 1):
                t_im = img1[0 : length + i, 0 : width + j] - img2[-i :, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width + j))) * np.sum(t_im) 
          
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [0, width + j]
                    z = [-i, length]
                    w = [-j, width]
            for j in range(0, m2 + 1, 1):
                t_im = img1[0 : length + i, j :] - img2[-i :, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 
                
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [j, width]
                    z = [-i, length]
                    w = [0, width - j]
    elif (n1 < 0) and (n2 < 0) and (m1 >= 0) and (m2 >= 0): 
        for i in range(n1, n2 + 1, 1):      
            for j in range(m1, m2 + 1, 1):
                t_im = img1[0 : length + i, j :] - img2[-i :, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 

                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [j, width]
                    z = [-i, length]
                    w = [0, width - j]
    elif (n1 < 0) and (n2 >= 0) and (m1 < 0) and (m2 < 0):
        for i in range(n1, 0, 1):
            for j in range(m1, m2 + 1, 1):
                t_im = img1[0 : length + i, 0 : width + j] - img2[-i :, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width + j))) * np.sum(t_im) 
       
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [0, width + j]
                    z = [-i, length]
                    w = [-j, width]
        for i in range(0, n2 + 1, 1):
            for j in range(m1, m2 + 1, 1):
                t_im = img1[i :, 0 : width + j] - img2[0 : length - i, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [0, width + j]
                    z = [0, length - i]
                    w = [-j, width]
                   
    elif (n1 < 0) and (n2 >= 0) and (m1 < 0) and (m2 >= 0):
        for i in range(n1, 0, 1):
            for j in range(m1, 0, 1):
                t_im = img1[0 : length + i, 0 : width + j] - img2[-i :, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width + j))) * np.sum(t_im) 
            
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [0, width + j]
                    z = [-i, length]
                    w = [-j, width]
            for j in range(0, m2 + 1, 1):
                t_im = img1[0 : length + i, j :] - img2[-i :, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [j, width]
                    z = [-i, length]
                    w = [0, width - j]
                    
        for i in range(0, n2 + 1, 1):
            for j in range(m1, 0, 1):
                t_im = img1[i :, 0 : width + j] - img2[0 : length - i, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [0, width + j]
                    z = [0, length - i]
                    w = [-j, width]
            for j in range(0, m2 + 1, 1):
                t_im = img1[i :, j :] - img2[0 : length - i, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length - i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [j, width]
                    z = [0, length - i]
                    w = [0, width - j]

    elif (n1 < 0) and (n2 >= 0) and (m1 >= 0) and (m2 >= 0):
        for i in range(n1, 0, 1):     
            for j in range(m1, m2 + 1, 1):
                t_im = img1[0 : length + i, j :] - img2[-i :, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [0, length + i]
                    y = [j, width]
                    z = [-i, length]
                    w = [0, width - j]
        for i in range(0, m2 + 1, 1):
            for j in range(m1, m2 + 1, 1):
                t_im = img1[i :, j :] - img2[0 : length - i, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length - i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [j, width]
                    z = [0, length - i]
                    w = [0, width - j]
    elif (n1 >= 0) and (n2 >= 0) and (m1 < 0) and (m2 < 0):
        for i in range(n1, n2 + 1, 1):
            for j in range(m1, m2 + 1, 1):
                t_im = img1[i :, 0 : width + j] - img2[0 : length - i, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [0, width + j]
                    z = [0, length - i]
                    w = [-j, width]
    elif (n1 >= 0) and (n2 >= 0) and (m1 < 0) and (m2 >= 0):
        for i in range(n1, n2 + 1, 1):
            for j in range(m1, 0, 1):
                t_im = img1[i :, 0 : width + j] - img2[0 : length - i, -j :]
                t_im = t_im ** 2
                t_im = (1 / ((length + i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [0, width + j]
                    z = [0, length - i]
                    w = [-j, width]
            for j in range(0, m2 + 1, 1):
                t_im = img1[i :, j :] - img2[0 : length - i, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length - i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [j, width]
                    z = [0, length - i]
                    w = [0, width - j]
    elif (n1 >= 0) and (n2 >= 0) and (m1 >= 0) and (m2 >= 0):
        for i in range(n1, n2 + 1, 1):
            for j in range(m1, m2 + 1, 1):
                t_im = img1[i :, j :] - img2[0 : length - i, 0 : width - j]
                t_im = t_im ** 2
                t_im = (1 / ((length - i) * (width - j))) * np.sum(t_im) 
                if t_im < min: 
                    n_min_1 = i
                    n_min_2 = j
                    min = t_im 
                    x = [i, length]
                    y = [j, width]
                    z = [0, length - i]
                    w = [0, width - j]
   
    return n_min_1, n_min_2, x, y, z, w

def align(img, g_coord):
    length, width = img.shape[:2]
    # new height:
    newlength = length // 3
    
    # getting three channels:
    img1 = deepcopy(img[0 : newlength,])
    img2 = deepcopy(img[newlength : newlength * 2,])
    img3 = deepcopy(img[newlength * 2 : newlength * 3,])

    # cutting edges:
    newwidth1 = int(width * 0.1)
    newwidth2 = int(width * 0.9)
    nlength1 = int(newlength * 0.1)
    nlength2 = int(newlength * 0.9)
    
    img1 = img1[nlength1 : nlength2, newwidth1 : newwidth2]
    img2 = img2[nlength1 : nlength2, newwidth1 : newwidth2]
    img3 = img3[nlength1 : nlength2, newwidth1 : newwidth2]
    
    offset11 = 0
    offset12 = 0
    offset21 = 0
    offset22 = 0
    g1 = g2 = g3 = g4 = r1 = r2 = b1 = b2 = (0, 0) 
    
    i = 0
    temp1 = deepcopy(img1)
    temp2 = deepcopy(img2)
    while (temp1.shape[0] >= 500 or temp1.shape[1] >= 500 or temp2.shape[0] >= 500 or temp2.shape[1] >= 500):
        temp1 = temp1[ ::2, ::2]
        temp2 = temp2[ ::2, ::2]
        i = i + 1
    
    num = i
    
    for j in range(num + 1):
        if j == 0:
            offset11, offset12, b1, b2, g1, g2 = mse_min(temp1, temp2, -15, 15, -15, 15)
        else:
            offset11, offset12, b1, b2, g1, g2 = mse_min(temp1, temp2, 2 * offset11 - 1, 2 * offset11 + 1, 2 * offset12 - 1, 2 * offset12 + 1)
        temp1 = deepcopy(img1)
        temp2 = deepcopy(img2)
        i = i - 1
        for k in range(i):
            temp1 = temp1[ ::2, ::2]
            temp2 = temp2[ ::2, ::2]
    
    i = 0
    temp1 = deepcopy(img2)
    temp2 = deepcopy(img3)
    while (temp1.shape[0] >= 500 or temp1.shape[1] >= 500 or temp2.shape[0] >= 500 or temp2.shape[1] >= 500):
        temp1 = temp1[ ::2, ::2]
        temp2 = temp2[ ::2, ::2]
        i = i + 1
     
    num = i
    
    for j in range(num + 1):
        if j == 0:
            offset21, offset22, g3, g4, r1, r2 = mse_min(temp1, temp2, -15, 15, -15, 15)
        else:
            offset21, offset22, g3, g4, r1, r2 = mse_min(temp1, temp2, 2 * offset21 - 1, 2 * offset21 + 1, 2 * offset22 - 1, 2 * offset22 + 1)
        temp1 = deepcopy(img2)
        temp2 = deepcopy(img3)
        i = i - 1
        for k in range(i):
            temp1 = temp1[ ::2, ::2]
            temp2 = temp2[ ::2, ::2]
    
    b_row = g_coord[0] - newlength + offset11
    b_col = g_coord[1] + offset12
    
    r_row = g_coord[0] + newlength - offset21
    r_col = g_coord[1] - offset22
    m1 = np.zeros(4, dtype = np.uint16)
    m2 = np.zeros(4, dtype = np.uint16)
    m3 = np.zeros(4, dtype = np.uint16)
    if g1[0] <= g3[0]: 
        m1[0] = g3[0] - g1[0]
        m3[0] = g3[0]
    else: 
        m2[0] = g1[0] - g3[0]
        m3[0] = g1[0]
        
    if g1[1] <= g3[1]: 
        m1[1] = b1[1] - b1[0]
        m2[1] = r1[1] - r1[0] - (g3[1] - g1[1])
        m3[1] = g1[1]
    else:
        m1[1] = b1[1] - b1[0] - (g1[1] - g3[1])
        m2[1] = r1[1] - r1[0]
        m3[1] = g3[1]
        
    if g2[0] <= g4[0]: 
        m1[2] = g4[0] - g2[0]
        m3[2] = g4[0]
    else: 
        m2[2] = g2[0] - g4[0]
        m3[2] = g2[0]
        
    if g2[1] <= g4[1]: 
        m1[3] = b2[1] - b2[0]
        m2[3] = r2[1] - r2[0] - (g4[1] - g2[1])
        m3[3] = g2[1]
    else: 
        m1[3] = b2[1] - b2[0] - (g2[1] - g4[1])
        m2[3] = r2[1] - r2[0]
        m3[3] = g4[1]

    i1 = deepcopy(img1[b1[0] : b1[1], b2[0] : b2[1]])
    i3 = deepcopy(img3[r1[0] : r1[1], r2[0] : r2[1]])   
    res = np.stack([i3[m2[0] : m2[1], m2[2] : m2[3]], img2[m3[0] : m3[1], m3[2] : m3[3]], i1[m1[0] : m1[1], m1[2] : m1[3]]], axis = 2) 
    imshow(res)
    return img1, (b_row, b_col), (r_row, r_col)
