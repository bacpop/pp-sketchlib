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
DistMatrix query_db_cuda(const std::vector<Reference>& ref_sketches,
	const std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int blockSize,
    const size_t max_device_mem = 0)
#endif

