/*
 *
 * gpu.hpp
 * functions using CUDA
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

#include "reference.hpp"

typedef std::tuple<RandomStrides, std::vector<float>> FlatRandom;

#ifdef GPU_AVAILABLE
// Structure of flattened vectors
struct SketchStrides {
	size_t bin_stride;
	size_t kmer_stride;
	size_t sample_stride;
	size_t sketchsize64; 
	size_t bbits;
};

struct SketchSlice {
	size_t ref_offset;
	size_t ref_size;
	size_t query_offset;
	size_t query_size;
};

// defined in dist.cu
std::tuple<size_t, size_t> initialise_device(const int device_id);

std::vector<float> dispatchDists(
				   std::vector<Reference>& ref_sketches,
				   std::vector<Reference>& query_sketches,
				   SketchStrides& ref_strides,
				   SketchStrides& query_strides,
				   const FlatRandom& flat_random,
				   const std::vector<uint16_t>& ref_random_idx,
				   const std::vector<uint16_t>& query_random_idx,
				   const SketchSlice& sketch_subsample,
				   const std::vector<size_t>& kmer_lengths,
				   const bool self);
#endif

