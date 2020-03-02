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

typedef Eigen::Matrix<float, Eigen::Dynamic, 2> DistMatrix;

// These are the four functions called by python bindings
std::vector<Reference> create_sketches(const std::string& db_name,
                   const std::vector<std::string>& names, 
                   const std::vector<std::vector<std::string>>& files, 
                   const std::vector<size_t>& kmer_lengths,
                   const size_t sketchsize64,
                   size_t min_count,
                   const size_t num_threads);

DistMatrix query_db(std::vector<Reference>& ref_sketches,
                    std::vector<Reference>& query_sketches,
                    const std::vector<size_t>& kmer_lengths,
                    const size_t num_threads);

// defined in dist.cu
#ifdef GPU_AVAILABLE
DistMatrix query_db_gpu(const std::vector<Reference>& ref_sketches,
	const std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int blockCount,
	const int blockSize,
    const size_t max_device_mem = 0)
#endif

std::vector<Reference> load_sketches(const std::string& db_name,
                                     const std::vector<std::string>& names,
                                     std::vector<size_t> kmer_lengths,
                                     const bool messages = true);
