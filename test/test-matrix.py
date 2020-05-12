import pp_sketchlib
import numpy as np

rr_mat = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
qq_mat = np.array([8], dtype=np.float32)
qr_mat = np.array([0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1], dtype=np.float32)

square1 = pp_sketchlib.longToSquare(rr_mat, 2)
square2 = pp_sketchlib.longToSquareMulti(rr_mat, qr_mat, qq_mat)

square1_res = np.array([[0, 1, 2, 3], 
                        [1, 0, 4, 5], 
                        [2, 4, 0, 6], 
                        [3, 5, 6, 0]], dtype=np.float32)


square2_res = np.array([[0, 1, 2, 3, 0.5, 1], 
                        [1, 0, 4, 5, 0.5, 1], 
                        [2, 4, 0, 6, 0.5, 1], 
                        [3, 5, 6, 0, 0.5, 1],
                        [0.5, 1, 0.5, 1, 0, 8],
                        [1, 0.5, 1, 0.5, 8, 0]], dtype=np.float32)

assert(square1 == square1_res)
assert(square2 == square2_res)
