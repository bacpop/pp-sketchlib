import os, sys
import numpy as np
from math import sqrt

# testing without install
#sys.path.insert(0, '../build/lib.macosx-10.9-x86_64-3.10')
import pp_sketchlib
try:
    from pp_sketch.matrix import sparsify
except ImportError as e:
    from scipy.sparse import coo_matrix
    def sparsify(distMat, cutoff, threads):
        sparse_coordinates = pp_sketchlib.sparsifyDistsByThreshold(distMat=distMat,
                                                                  distCutoff=cutoff,
                                                                  num_threads=threads)
        sparse_scipy = coo_matrix((sparse_coordinates[2],
                                (sparse_coordinates[0], sparse_coordinates[1])),
                                shape=distMat.shape,
                                dtype=np.float32)

        # Mirror to fill in lower triangle
        if cutoff > 0:
            sparse_scipy = sparse_scipy + sparse_scipy.transpose()

        return(sparse_scipy)

def check_res(res, expected):
    if (not np.all(res == expected)):
        print(res)
        print(expected)
        raise RuntimeError("Results don't match")

# Square to long
rr_mat = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
qq_mat = np.array([8], dtype=np.float32)
qr_mat = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float32)
# NB r_idx is inner/fastest: q0r0, q0r1, q0r2, q0r3, q1r0, q1r1, q1r2, q1r3

square1 = pp_sketchlib.longToSquare(distVec=rr_mat, num_threads=2)
square2 = pp_sketchlib.longToSquareMulti(distVec=rr_mat,
                                         query_ref_distVec=qr_mat,
                                         query_query_distVec=qq_mat)

square1_res = np.array([[0, 1, 2, 3],
                        [1, 0, 4, 5],
                        [2, 4, 0, 6],
                        [3, 5, 6, 0]], dtype=np.float32)


square2_res = np.array([[0, 1, 2, 3, 10, 14],
                        [1, 0, 4, 5, 11, 15],
                        [2, 4, 0, 6, 12, 16],
                        [3, 5, 6, 0, 13, 17],
                        [10, 11, 12, 13, 0, 8],
                        [14, 15, 16, 17, 8, 0]], dtype=np.float32)

check_res(square1_res, square1)
check_res(square2_res, square2)

check_res(pp_sketchlib.squareToLong(distMat=square1_res, num_threads=2), rr_mat)

# sparsification
sparse1 = sparsify(square2_res, cutoff=5, threads=2)

sparse1_res = square2_res.copy()
sparse1_res[sparse1_res >= 5] = 0
check_res(sparse1.todense(), sparse1_res)
