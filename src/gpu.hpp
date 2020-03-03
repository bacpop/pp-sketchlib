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

#include "reference.hpp"

// defined in dist.cu
#ifdef GPU_AVAILABLE
std::vector<float> query_db_cuda(std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int blockSize,
    const size_t max_device_mem = 0);
#endif

