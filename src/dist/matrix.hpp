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

#include <Eigen/Dense>

#include "matrix_idx.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NumpyMatrix;
typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<float>> sparse_coo;

NumpyMatrix long_to_square(const Eigen::VectorXf &rrDists,
                           const Eigen::VectorXf &qrDists,
                           const Eigen::VectorXf &qqDists,
                           unsigned int num_threads = 1);

Eigen::VectorXf square_to_long(const NumpyMatrix &squareDists,
                               const unsigned int num_threads);

sparse_coo sparsify_dists(const NumpyMatrix &denseDists,
                          const float distCutoff,
                          const unsigned long int kNN,
                          const unsigned int num_threads = 1);

Eigen::VectorXf assign_threshold(const NumpyMatrix &distMat,
                                 const int slope,
                                 const float x_max,
                                 const float y_max,
                                 unsigned int num_threads);

std::tuple<std::vector<long>, std::vector<long>, std::vector<long>>
threshold_iterate(const NumpyMatrix &distMat,
                  const std::vector<double> &offsets,
                  const int slope,
                  const float x0,
                  const float y0,
                  const float x1,
                  const float y1,
                  const int num_threads = 1);
