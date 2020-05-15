/*
 *
 * api.hpp
 * main functions for interacting with sketches
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

#include <Eigen/Dense>
using Eigen::MatrixXf;

#include "reference.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NumpyMatrix;
typedef std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<float>> sparse_coo;

// api.cpp
std::vector<Reference> create_sketches(const std::string& db_name,
                   const std::vector<std::string>& names, 
                   const std::vector<std::vector<std::string>>& files, 
                   const std::vector<size_t>& kmer_lengths,
                   const size_t sketchsize64,
                   const bool use_rc,
                   size_t min_count,
                   const bool exact,
                   const size_t num_threads);

NumpyMatrix query_db(std::vector<Reference>& ref_sketches,
                    std::vector<Reference>& query_sketches,
                    const std::vector<size_t>& kmer_lengths,
                    const bool jaccard,
                    const size_t num_threads);

#ifdef GPU_AVAILABLE
NumpyMatrix query_db_gpu(std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
    const int device_id = 0);
#endif

bool same_db_version(const std::string& db1_name,
                         const std::string& db2_name);

std::vector<Reference> load_sketches(const std::string& db_name,
                                     const std::vector<std::string>& names,
                                     std::vector<size_t> kmer_lengths,
                                     const bool messages = true);

// matrix_ops.cpp
NumpyMatrix long_to_square(const Eigen::VectorXf& rrDists, 
                            const Eigen::VectorXf& qrDists,
                            const Eigen::VectorXf& qqDists,
                            unsigned int num_threads = 1);

sparse_coo sparsify_dists(const NumpyMatrix& denseDists,
                          const float distCutoff,
                          const unsigned long int kNN,
                          const unsigned int num_threads = 1);

Eigen::VectorXf assign_threshold(const NumpyMatrix& distMat,
                                 int slope,
                                 float x_max,
                                 float y_max,
                                 unsigned int num_threads = 1);