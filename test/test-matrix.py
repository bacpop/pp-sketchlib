import os, sys

sys.path.insert(0, '../build/lib.linux-x86_64-3.7')
import pp_sketchlib
import numpy as np

# Original PopPUNK function
def withinBoundary(dists, x_max, y_max, slope=2):
    boundary_test = np.ones((dists.shape[0]))
    for row in range(boundary_test.size):
        if slope == 2:
            in_tri = dists[row, 0]*dists[row, 1] - (x_max-dists[row, 0])*(y_max-dists[row, 1])
        elif slope == 0:
            in_tri = dists[row, 0] - x_max
        elif slope == 1:
            in_tri = dists[row, 1] - y_max

        if in_tri < 0:
            boundary_test[row] = -1
        elif in_tri == 0:
            boundary_test[row] = 0
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
