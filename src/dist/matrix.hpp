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

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NumpyMatrix;
typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<float>> sparse_coo;

template<class T>
inline size_t rows_to_samples(const T& longMat) {
    return 0.5*(1 + sqrt(1 + 8*(longMat.rows())));
}

// These are inlined partially to avoid conflicting with the cuda
// versions which have the same prototype
inline long calc_row_idx(const long long k, const long n) {
	return n - 2 - floor(sqrt((double)(-8*k + 4*n*(n-1)-7))/2 - 0.5);
}

inline long calc_col_idx(const long long k, const long i, const long n) {
	return k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
}

inline long long square_to_condensed(long i, long j, long n) {
    assert(j > i);
	return (n*i - ((i*(i+1)) >> 1) + j - 1 - i);
}

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
