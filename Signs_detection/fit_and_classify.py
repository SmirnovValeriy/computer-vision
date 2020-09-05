import numpy as np 
import scipy.signal as sc 
import math
from skimage.io import imshow, imread
from time import time
from skimage.color import rgb2gray
from skimage.transform import resize

def extract_hog(img1): 

    #convert to grayscale
    img = rgb2gray(img1)
    
    #common size
    heigth, width = (64, 64)
    
    #resize image to common size 
    img = resize(img, (heigth, width))
    
    #derivatives calculating
    I_x = np.zeros((heigth, width))
    I_y = np.zeros((heigth, width))

    I_x[:, 0] = img[:, 1] - img[:, 0]
    I_x[:, -1] = img[:, -1] - img[:, -2]
    I_x[:, 1 : -1] = img[:, 2 : ] - img[:, : -2]

    I_y[0, :] = img[1, :] - img[0, :]
    I_y[-1, :] = img[-1, :] - img[-2, :]
    I_y[1 : -1, :] = img[2 : , :] - img[: -2, :]

    #gradient module calculating
    grad = np.sqrt(I_x ** 2 + I_y ** 2)

    #gradient direction calculating
    teta = (np.arctan2(I_y, I_x) + np.pi) / 2


    #number of pixels in each cell
    pix_rows = 8
    pix_cols = 8
    
    #number of cells  
    cell_rows = heigth // pix_rows #пvertically
    cell_cols = width // pix_cols #horizontally
    
    #bins number
    bin_count = 9
    
    #cell histograms matrix
    gist_cells = np.zeros((cell_rows, cell_cols, bin_count))

    #fill bins in the cell histograms
    for i in range(cell_rows):
        for j in range(cell_cols):
            temp_grad = grad[i * pix_rows : (i + 1) * pix_rows, j * pix_cols : (j + 1) * pix_cols]
            temp_teta = teta[i * pix_rows : (i + 1) * pix_rows, j * pix_cols : (j + 1) * pix_cols]
            cell_mask = (temp_teta * bin_count / np.pi).astype(int) % bin_count
            for bin_pos in range(bin_count):
                gist_cells[i, j, bin_pos] = np.sum(temp_grad[cell_mask == bin_pos])

    #number of cells in each block
    n_block_row = 4 #vertically
    n_block_col = 4 #horizontally
    
    #step
    step = 2
    
    #number of blocks
    block_row = 1 + ((cell_rows - n_block_row) // step) #vertically 
    block_col = 1 + ((cell_cols - n_block_col) // step) #horizontally
    
    #histogram length
    length = n_block_row * n_block_col * bin_count
    
    #block histograms matrix
    gist_blocks = np.zeros((block_row, block_col, length))
    
    #descriptors calculating
    for i in range(block_row):
        for j in range(block_col):
            gist_blocks[i, j] = gist_cells[step * i : (step * i + n_block_row), step * j : (step * j + n_block_col)].reshape(length)
            gist_blocks[i, j] = gist_blocks[i, j] / np.sqrt(np.sum(gist_blocks[i, j] ** 2) + 1e-16)
    return gist_blocks.reshape(block_row * block_col * length)

def fit_and_classify(train_features, train_labels, test_features):
    from sklearn.svm import SVC, LinearSVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.utils import shuffle

    #sample shuffling 
    sh_train_features, sh_train_labels = shuffle(train_features, train_labels)

    #choose training method; value 0.3247 was chosen as optimal for C
    svc = LinearSVC(C = 0.3247)

    #train model on the train sample
    svc.fit(sh_train_features, sh_train_labels)
    #return prediction on the test sample
    return svc.predict(test_features)
