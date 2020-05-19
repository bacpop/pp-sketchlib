# Copyright 2019-2020 John Lees

'''Python wrappers for complex c++ returns'''

import os, sys
import numpy as np
from scipy.sparse import coo_matrix

import pp_sketchlib

def sparsify(distMat, cutoff, kNN, threads):
    sparse_coordinates = pp_sketchlib.sparsifyDists(distMat, 
                                                    distCutoff=cutoff, 
                                                    kNN=kNN, 
                                                    num_threads=threads)
    sparse_scipy = coo_matrix((sparse_coordinates[2], 
                               (sparse_coordinates[0], sparse_coordinates[1])), 
                              shape=distMat.shape, 
                              dtype=np.float32)
    
    # Mirror to fill in lower triangle
    if cutoff > 0:
        sparse_scipy = sparse_scipy + sparse_scipy.transpose() 
    
    return(sparse_scipy)