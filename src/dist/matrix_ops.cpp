/*
 *
 * matrix_ops.cpp
 * Distance matrix transformations
 *
 */
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <omp.h>

#include "api.hpp"

const float epsilon = 1E-10;

// Prototypes for cppying functions run in threads
void square_block(const Eigen::VectorXf &longDists,
                  NumpyMatrix &squareMatrix,
                  const size_t n_samples,
                  const size_t start,
                  const size_t offset,
                  const size_t max_elems);

void rectangle_block(const Eigen::VectorXf &longDists,
                     NumpyMatrix &squareMatrix,
                     const size_t nrrSamples,
                     const size_t nqqSamples,
                     const size_t start,
                     const size_t max_elems);

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

sparse_coo sparsify_dists(const NumpyMatrix &denseDists,
                          const float distCutoff,
                          const unsigned long int kNN,
                          const unsigned int num_threads)
{
  if (kNN > 0 && distCutoff > 0)
  {
    throw std::runtime_error("Specify only one of kNN or distCutoff");
  }
  else if (kNN < 1 && distCutoff < 0)
  {
    throw std::runtime_error("kNN must be > 1 or distCutoff > 0");
  }

  // ijv vectors
  std::vector<float> dists;
  std::vector<long> i_vec;
  std::vector<long> j_vec;
  if (distCutoff > 0)
  {
    for (long i = 0; i < denseDists.rows(); i++)
    {
      for (long j = i + 1; j < denseDists.cols(); j++)
      {
        if (denseDists(i, j) < distCutoff)
        {
          dists.push_back(denseDists(i, j));
          i_vec.push_back(i);
          j_vec.push_back(j);
        }
      }
    }
  }
  else if (kNN >= 1)
  {
    // Only add the k nearest (unique) neighbours
    // May be >k if repeats, often zeros
    for (long i = 0; i < denseDists.rows(); i++)
    {
      unsigned long unique_neighbors = 0;
      float prev_value = -1;
      for (auto j : sort_indexes(denseDists.row(i)))
      {
        if (j == i)
        {
          continue; // Ignore diagonal which will always be one of the closest
        }
        bool new_val = abs(denseDists(i, j) - prev_value) < epsilon;
        if (unique_neighbors < kNN || new_val)
        {
          dists.push_back(denseDists(i, j));
          i_vec.push_back(i);
          j_vec.push_back(j);
          if (!new_val)
          {
            unique_neighbors++;
            prev_value = denseDists(i, j);
          }
        }
        else
        {
          break;
        }
      }
    }
  }

  return (std::make_tuple(i_vec, j_vec, dists));
}

// Unnormalised (signed_ distance between a point (x0, y0) and a line defined
// by the two points (xmax, 0) and (0, ymax)
// Divide by 1/sqrt(xmax^2 + ymax^2) to get distance
inline float line_dist(const float x0,
                       const float y0,
                       const float x_max,
                       const float y_max,
                       const int slope)
{
  float boundary_side = 0;
  if (slope == 2)
  {
    boundary_side = y0 * x_max + x0 * y_max - x_max * y_max;
  }
  else if (slope == 0)
  {
    boundary_side = x0 - x_max;
  }
  else if (slope == 1)
  {
    boundary_side = y0 - y_max;
  }

  return boundary_side;
}

Eigen::VectorXf assign_threshold(const NumpyMatrix &distMat,
                                 const int slope,
                                 const float x_max,
                                 const float y_max,
                                 unsigned int num_threads)
{
  Eigen::VectorXf boundary_test(distMat.rows());

#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (long row_idx = 0; row_idx < distMat.rows(); row_idx++)
  {
    float in_tri = line_dist(distMat(row_idx, 0), distMat(row_idx, 1),
                             x_max, y_max, slope);
    float boundary_side;
    if (in_tri == 0)
    {
      boundary_side = 0;
    }
    else if (in_tri > 0)
    {
      boundary_side = 1;
    }
    else
    {
      boundary_side = -1;
    }
    boundary_test[row_idx] = boundary_side;
  }
  return (boundary_test);
}

std::tuple<std::vector<long>, std::vector<long>, std::vector<long>>
threshold_iterate(const NumpyMatrix &distMat,
                  const std::vector<double> &offsets,
                  const int slope,
                  const float x0,
                  const float y0,
                  const float x1,
                  const float y1,
                  const int num_threads)
{
  std::vector<long> i_vec;
  std::vector<long> j_vec;
  std::vector<long> offset_idx;
  const float gradient = (y1 - y0) / (x1 - x0); // == tan(theta)
  const size_t n_samples = rows_to_samples(distMat);

  std::vector<float> boundary_dist(distMat.rows());
  std::vector<long> boundary_order;
  long sorted_idx = 0;
  for (int offset_nr = 0; offset_nr < offsets.size(); ++offset_nr)
  {
    float x_intercept = x0 + offsets[offset_nr] * (1 / std::sqrt(1 + gradient));
    float y_intercept = y0 + offsets[offset_nr] * (gradient / std::sqrt(1 + gradient));
    float x_max, y_max;
    if (slope == 2)
    {
      x_max = x_intercept + y_intercept * gradient;
      y_max = y_intercept + x_intercept / gradient;
    }
    else if (slope == 0)
    {
      x_max = x_intercept;
      y_max = 0;
    }
    else
    {
      x_max = 0;
      y_max = y_intercept;
    }

    // printf("grad:%f xint:%f yint:%f x_max:%f y_max:%f\n",
    //         gradient, x_intercept, y_intercept, x_max, y_max);
    // Calculate the distances and sort them on the first loop entry
    if (offset_nr == 0)
    {
#pragma omp parallel for schedule(static) num_threads(num_threads)
      for (long row_idx = 0; row_idx < distMat.rows(); row_idx++)
      {
        boundary_dist[row_idx] = line_dist(distMat(row_idx, 0), distMat(row_idx, 1), x_max, y_max, slope);
      }
      boundary_order = sort_indexes(boundary_dist);
    }

    long row_idx = boundary_order[sorted_idx];
    while (sorted_idx < boundary_order.size() &&
           line_dist(distMat(row_idx, 0),
                     distMat(row_idx, 1),
                     x_max, y_max, slope) <= 0)
    {
      long i = calc_row_idx(row_idx, n_samples);
      long j = calc_col_idx(row_idx, i, n_samples);
      i_vec.push_back(i);
      j_vec.push_back(j);
      offset_idx.push_back(offset_nr);
      sorted_idx++;
      row_idx = boundary_order[sorted_idx];
    }
  }
  return (std::make_tuple(i_vec, j_vec, offset_idx));
}

NumpyMatrix long_to_square(const Eigen::VectorXf &rrDists,
                           const Eigen::VectorXf &qrDists,
                           const Eigen::VectorXf &qqDists,
                           unsigned int num_threads)
{
  // Set up square matrix to move values into
  size_t nrrSamples = rows_to_samples(rrDists);
  size_t nqqSamples = 0;
  if (qrDists.size() > 0)
  {
    nqqSamples = rows_to_samples(qqDists);
    // If one query, qqDists = [0] or None, so may come back as zero
    if (nqqSamples < 1 && qrDists.size() > 0)
    {
      nqqSamples = 1;
    }
    if (qrDists.size() != nrrSamples * nqqSamples)
    {
      throw std::runtime_error("Shape of reference, query and ref vs query matrices mismatches");
    }
  }

  // Initialise matrix and set diagonal to zero
  NumpyMatrix squareDists(nrrSamples + nqqSamples, nrrSamples + nqqSamples);
  for (size_t diag_idx = 0; diag_idx < nrrSamples + nqqSamples; diag_idx++)
  {
    squareDists(diag_idx, diag_idx) = 0;
  }

// Loop over threads for ref v ref square
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (long distIdx = 0; distIdx < rrDists.rows(); distIdx++)
  {
    unsigned long i = calc_row_idx(distIdx, nrrSamples);
    unsigned long j = calc_col_idx(distIdx, i, nrrSamples);
    squareDists(i, j) = rrDists[distIdx];
    squareDists(j, i) = rrDists[distIdx];
  }

  if (qqDists.size() > 0)
  {
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (long distIdx = 0; distIdx < qqDists.rows(); distIdx++)
    {
      unsigned long i = calc_row_idx(distIdx, nqqSamples) + nrrSamples;
      unsigned long j = calc_col_idx(distIdx, i - nrrSamples, nqqSamples) + nrrSamples;
      squareDists(i, j) = qqDists[distIdx];
      squareDists(j, i) = qqDists[distIdx];
    }
  }

  // Query vs ref rectangles
  if (qrDists.size() > 0)
  {
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (long distIdx = 0; distIdx < qrDists.rows(); distIdx++)
    {
      unsigned long i = static_cast<size_t>(distIdx / (float)nqqSamples + 0.001f);
      unsigned long j = distIdx % nqqSamples + nrrSamples;
      squareDists(i, j) = qrDists[distIdx];
      squareDists(j, i) = qrDists[distIdx];
    }
  }

  return squareDists;
}

Eigen::VectorXf square_to_long(const NumpyMatrix &squareDists,
                               const unsigned int num_threads)
{
  if (squareDists.rows() != squareDists.cols())
  {
    throw std::runtime_error("square_to_long input must be a square matrix");
  }

  long n = squareDists.rows();
  Eigen::VectorXf longDists((n * (n - 1)) >> 1);

// Each inner loop increases in size linearly with outer index
// due to reverse direction
// guided schedules inversely proportional to outer index
#pragma omp parallel for schedule(guided, 1) num_threads(num_threads)
  for (long i = n - 2; i >= 0; i--)
  {
    for (long j = i + 1; j < n; j++)
    {
      longDists(square_to_condensed(i, j, n)) = squareDists(i, j);
    }
  }

  return (longDists);
}
