# Copyright 2019-2020 John Lees

'''Python wrappers for complex c++ returns'''

import os, sys
import numpy as np
from scipy.sparse import coo_matrix

import pp_sketchlib

def sparsify(distMat, cutoff, kNN, threads):
    sparse_coordinates = pp_sketchlib.sparsifyDists(distMat,
                                                    distCutoff=cutoff,
                                                    kNN=kNN)
    sparse_scipy = ijv_to_coo(sparse_coordinates, distMat.shape, np.float32)

    # Mirror to fill in lower triangle
    if cutoff > 0:
        sparse_scipy = sparse_scipy + sparse_scipy.transpose()

    return(sparse_scipy)

def ijv_to_coo(ijv, shape, dtype):
    sparse_scipy = coo_matrix((ijv[2], (ijv[0], ijv[1])),
                              shape=shape,
                              dtype=dtype)
    return(sparse_scipy)
