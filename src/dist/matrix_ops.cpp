/*
 *
 * matrix_ops.cpp
 * Distance matrix transformations
 *
 */
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <string>
#include <vector>

#include "api.hpp"

// Prototypes for cppying functions run in threads
void square_block(const Eigen::VectorXf &longDists, NumpyMatrix &squareMatrix,
                  const size_t n_samples, const size_t start,
                  const size_t offset, const size_t max_elems);

void rectangle_block(const Eigen::VectorXf &longDists,
                     NumpyMatrix &squareMatrix, const size_t nrrSamples,
                     const size_t nqqSamples, const size_t start,
                     const size_t max_elems);

template <typename T>
std::vector<T> combine_vectors(const std::vector<std::vector<T>> &vec,
                               const size_t len) {
  std::vector<T> all(len);
  auto all_it = all.begin();
  for (size_t i = 0; i < vec.size(); ++i) {
    std::copy(vec[i].cbegin(), vec[i].cend(), all_it);
    all_it += vec[i].size();
  }
  return all;
}

sparse_coo sparsify_dists_by_threshold(const NumpyMatrix &denseDists,
                                       const float distCutoff,
                                       const size_t num_threads) {

  if (distCutoff < 0) {
    throw std::runtime_error("kNN must be > 1 or distCutoff > 0");
  }

  // Parallelisation parameter
  size_t len = 0;

  // ijv vectors
  std::vector<std::vector<float>> dists;
  std::vector<std::vector<long>> i_vec;
  std::vector<std::vector<long>> j_vec;
#pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:len)
  for (long i = 0; i < denseDists.rows(); i++) {
    for (long j = i + 1; j < denseDists.cols(); j++) {
      if (denseDists(i, j) < distCutoff) {
        dists[i].push_back(denseDists(i, j));
        i_vec[i].push_back(i);
        j_vec[i].push_back(j);
      }
    }
    len += i_vec[i].size();
  }
  std::vector<float> dists_all = combine_vectors(dists, len);
  std::vector<long> i_vec_all = combine_vectors(i_vec, len);
  std::vector<long> j_vec_all = combine_vectors(j_vec, len);
  return (std::make_tuple(i_vec_all, j_vec_all, dists_all));
}

NumpyMatrix long_to_square(const Eigen::VectorXf &rrDists,
                           const Eigen::VectorXf &qrDists,
                           const Eigen::VectorXf &qqDists,
                           unsigned int num_threads) {
  // Set up square matrix to move values into
  size_t nrrSamples = rows_to_samples(rrDists);
  size_t nqqSamples = 0;
  if (qrDists.size() > 0) {
    nqqSamples = rows_to_samples(qqDists);
    // If one query, qqDists = [0] or None, so may come back as zero
    if (nqqSamples < 1 && qrDists.size() > 0) {
      nqqSamples = 1;
    }
    if (qrDists.size() != nrrSamples * nqqSamples) {
      throw std::runtime_error(
          "Shape of reference, query and ref vs query matrices mismatches");
    }
  }

  // Initialise matrix and set diagonal to zero
  NumpyMatrix squareDists(nrrSamples + nqqSamples, nrrSamples + nqqSamples);
  for (size_t diag_idx = 0; diag_idx < nrrSamples + nqqSamples; diag_idx++) {
    squareDists(diag_idx, diag_idx) = 0;
  }

// Loop over threads for ref v ref square
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (long distIdx = 0; distIdx < rrDists.rows(); distIdx++) {
    unsigned long i = calc_row_idx(distIdx, nrrSamples);
    unsigned long j = calc_col_idx(distIdx, i, nrrSamples);
    squareDists(i, j) = rrDists[distIdx];
    squareDists(j, i) = rrDists[distIdx];
  }

  if (qqDists.size() > 0) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (long distIdx = 0; distIdx < qqDists.rows(); distIdx++) {
      unsigned long i = calc_row_idx(distIdx, nqqSamples) + nrrSamples;
      unsigned long j =
          calc_col_idx(distIdx, i - nrrSamples, nqqSamples) + nrrSamples;
      squareDists(i, j) = qqDists[distIdx];
      squareDists(j, i) = qqDists[distIdx];
    }
  }

  // Query vs ref rectangles
  if (qrDists.size() > 0) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (long distIdx = 0; distIdx < qrDists.rows(); distIdx++) {
      unsigned long i = distIdx % nrrSamples;
      unsigned long j = distIdx / nrrSamples + nrrSamples;
      squareDists(i, j) = qrDists[distIdx];
      squareDists(j, i) = qrDists[distIdx];
    }
  }

  return squareDists;
}

Eigen::VectorXf square_to_long(const NumpyMatrix &squareDists,
                               const unsigned int num_threads) {
  if (squareDists.rows() != squareDists.cols()) {
    throw std::runtime_error("square_to_long input must be a square matrix");
  }

  long n = squareDists.rows();
  Eigen::VectorXf longDists((n * (n - 1)) >> 1);

// Each inner loop increases in size linearly with outer index
// due to reverse direction
// guided schedules inversely proportional to outer index
#pragma omp parallel for schedule(guided, 1) num_threads(num_threads)
  for (long i = n - 2; i >= 0; i--) {
    for (long j = i + 1; j < n; j++) {
      longDists(square_to_condensed(i, j, n)) = squareDists(i, j);
    }
  }

  return (longDists);
}
