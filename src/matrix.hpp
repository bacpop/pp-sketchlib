/*
 *
 * matrix.hpp
 * functions in matrix_ops.cpp
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

#ifdef GPU_AVAILABLE
#include "gpu.hpp"
#endif

#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NumpyMatrix;
typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<float>> sparse_coo;

NumpyMatrix long_to_square(const Eigen::VectorXf& rrDists, 
                            const Eigen::VectorXf& qrDists,
                            const Eigen::VectorXf& qqDists,
                            unsigned int num_threads = 1);

Eigen::VectorXf square_to_long(const NumpyMatrix& squareDists, 
                               const unsigned int num_threads);

sparse_coo sparsify_dists(const NumpyMatrix& denseDists,
                          const float distCutoff,
                          const unsigned long int kNN,
                          const unsigned int num_threads = 1);

Eigen::VectorXf assign_threshold(const NumpyMatrix& distMat,
                                 int slope,
                                 float x_max,
                                 float y_max,
                                 unsigned int num_threads = 1);

// GPU functions for gpu_api.cpp
#ifdef GPU_AVAILABLE
void longToSquareBlock(NumpyMatrix& coreSquare,
                       NumpyMatrix& accessorySquare,
                       const SketchSlice& sketch_subsample,
                       std::vector<float>& block_results,
                       const unsigned int num_threads);

NumpyMatrix twoColumnSquareToLong(const NumpyMatrix& coreSquare,
                       const NumpyMatrix& accessorySquare,
                       const unsigned int num_threads);
#endif