import os, sys
import numpy as np
from math import sqrt

# testing without install
#sys.path.insert(0, '../build/lib.macosx-10.9-x86_64-3.8')
import pp_sketchlib
try:
    from pp_sketch.matrix import sparsify
except ImportError as e:
    from scipy.sparse import coo_matrix
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

# Original PopPUNK function (with some improvements)
def withinBoundary(dists, x_max, y_max, slope=2):
    boundary_test = np.ones((dists.shape[0]))
    for row in range(boundary_test.size):
        if slope == 2:
            in_tri = dists[row, 1] * x_max + dists[row, 0] * y_max - x_max * y_max
        elif slope == 0:
            in_tri = dists[row, 0] - x_max
        elif slope == 1:
            in_tri = dists[row, 1] - y_max
        if abs(in_tri) < np.finfo(np.float32).eps:
          boundary_test[row] = 0
        elif in_tri < 0:
            boundary_test[row] = -1
    return(boundary_test)

def check_res(res, expected):
    if (not np.all(res == expected)):
        print(res)
        print(expected)
        raise RuntimeError("Results don't match")

# Square to long
rr_mat = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
qq_mat = np.array([8], dtype=np.float32)
qr_mat = np.array([10, 20, 10, 20, 10, 20, 10, 20], dtype=np.float32)

square1 = pp_sketchlib.longToSquare(rr_mat, 2)
square2 = pp_sketchlib.longToSquareMulti(rr_mat, qr_mat, qq_mat)

square1_res = np.array([[0, 1, 2, 3],
                        [1, 0, 4, 5],
                        [2, 4, 0, 6],
                        [3, 5, 6, 0]], dtype=np.float32)


square2_res = np.array([[0, 1, 2, 3, 10, 20],
                        [1, 0, 4, 5, 10, 20],
                        [2, 4, 0, 6, 10, 20],
                        [3, 5, 6, 0, 10, 20],
                        [10, 10, 10, 10, 0, 8],
                        [20, 20, 20, 20, 8, 0]], dtype=np.float32)

check_res(square1_res, square1)
check_res(square2_res, square2)

check_res(pp_sketchlib.squareToLong(square1_res, 2), rr_mat)

# assigning
x = np.arange(0, 1, 0.1, dtype=np.float32)
y = np.arange(0, 1, 0.1, dtype=np.float32)
xv, yv = np.meshgrid(x, y)
distMat = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
assign0 = pp_sketchlib.assignThreshold(distMat, 0, 0.5, 0.5, 2)
assign1 = pp_sketchlib.assignThreshold(distMat, 1, 0.5, 0.5, 2)
assign2 = pp_sketchlib.assignThreshold(distMat, 2, 0.5, 0.5, 2)

assign0_res = withinBoundary(distMat, 0.5, 0.5, 0)
assign1_res = withinBoundary(distMat, 0.5, 0.5, 1)
assign2_res = withinBoundary(distMat, 0.5, 0.5, 2)

check_res(assign0, assign0_res)
check_res(assign1, assign1_res)
check_res(assign2, assign2_res)

# move boundary
# example is symmetrical at points (0.1, 0.1); (0.2, 0.2); (0.3, 0.3)
samples = 100
distMat = np.random.rand(int(0.5 * samples * (samples - 1)), 2)
distMat = np.array(distMat, dtype = np.float32)
offsets = [x * sqrt(2) for x in [-0.1, 0.0, 0.1]]
i_vec, j_vec, idx_vec = pp_sketchlib.thresholdIterate(distMat, offsets, 2, 0.2, 0.2, 0.3, 0.3)
sketchlib_i = []
sketchlib_j = []
for offset_idx, offset in enumerate(offsets):
  for i, j, idx in zip(i_vec, j_vec, idx_vec):
    if idx > offset_idx:
      break
    elif idx == offset_idx:
      sketchlib_i.append(i)
      sketchlib_j.append(j)

  py_i = []
  py_j = []
  xmax = 0.4 + (2 * (offset/sqrt(2)))
  assign = pp_sketchlib.assignThreshold(distMat, 2, xmax, xmax, 1)
  dist_idx = 0
  for i in range(samples):
    for j in range(i + 1, samples):
      if assign[dist_idx] <= 0:
        py_i.append(i)
        py_j.append(j)
      dist_idx += 1
  if set(zip(py_i, py_j)) != set(zip(sketchlib_i, sketchlib_j)):
    raise RuntimeError("Threshold iterate mismatch at offset " + str(offset))

# sparsification
sparse1 = sparsify(square2_res, cutoff=5, kNN=0, threads=2)

sparse1_res = square2_res.copy()
sparse1_res[sparse1_res >= 5] = 0
check_res(sparse1.todense(), sparse1_res)

kNN = 2
sparse2 = sparsify(square2_res, cutoff=0, kNN=kNN, threads=2)

sparse2_res = square2_res.copy()
row_sort = np.argsort(sparse2_res, axis=1)
for i, row in enumerate(sparse2_res):
    neighbours = 0
    prev_val = 0
    for j in row_sort[i, :]:
        if i == j or row[j] == prev_val:
            continue
        else:
            prev_val = row[j]
            if neighbours >= kNN:
                sparse2_res[i, j] = 0
            else:
                neighbours += 1

check_res(sparse2.todense(), sparse2_res)
