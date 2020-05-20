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

#ifdef GPU_AVAILABLE
// defined in dist.cu
std::vector<float> query_db_cuda(std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int device_id = 0);
#endif

