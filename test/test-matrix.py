import pp_sketchlib
import numpy as np

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
assign1 = pp_sketchlib.assignThreshold(distMat, 0, 0.5, 0, 2)