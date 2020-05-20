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

#include "matrix.hpp"
#include "reference.hpp"

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
    const int device_id = 0,
    const unsigned int num_cpu_threads = 1);
#endif

bool same_db_version(const std::string& db1_name,
                     const std::string& db2_name);

std::vector<Reference> load_sketches(const std::string& db_name,
                                     const std::vector<std::string>& names,
                                     std::vector<size_t> kmer_lengths,
                                     const bool messages = true);

