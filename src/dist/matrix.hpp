/*
 *
 * matrix.hpp
 * functions in matrix_ops.cpp
 *
 */
#pragma once

#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

#include <Eigen/Dense>

#include "matrix_types.hpp"
#include "matrix_idx.hpp"

// This type not used in any nvcc code
using NumpyMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//https://stackoverflow.com/a/12399290
template <typename T>
std::vector<long> sort_indexes(const T &v)
{

  // initialize original index locations
  std::vector<long> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(idx.begin(), idx.end(),
                   [&v](long i1, long i2) { return v[i1] < v[i2]; });

  return idx;
}

NumpyMatrix long_to_square(const Eigen::VectorXf &rrDists,
                           const Eigen::VectorXf &qrDists,
                           const Eigen::VectorXf &qqDists,
                           unsigned int num_threads = 1);

Eigen::VectorXf square_to_long(const NumpyMatrix &squareDists,
                               const unsigned int num_threads);

sparse_coo sparsify_dists(const NumpyMatrix &denseDists,
                          const float distCutoff,
                          const unsigned long int kNN);
