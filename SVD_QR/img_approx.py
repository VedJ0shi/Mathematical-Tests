import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def image_approximation(fname, rank=100, dir='../test_data'):
    '''
    plots a low rank compression of the original image;
    returns % contribution of captured singular values and % data storage
    '''

    segments = dir.split('/')
    rgb = imread(os.path.join(*segments, fname)) #returns an m*n*3 numpy array
    greyscale = np.mean(rgb,-1)
    #print(greyscale.shape)
    U, s, V_ = np.linalg.svd(greyscale, full_matrices=False) #returns the economy SVD
    S = np.diag(s)
    #print(U.shape, S.shape, V_.shape)
    Approx = U[:,:rank]@S[:rank,:rank]@V_[:rank,:]
    original_storage = greyscale.shape[0] * greyscale.shape[1]
    compressed_storage = U.shape[0]*rank + rank + rank*V_.shape[1]
    img = plt.imshow(Approx)
    plt.show()
    return np.sum(s[:rank])/np.sum(s), compressed_storage/original_storage
    





    