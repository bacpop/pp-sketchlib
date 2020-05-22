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

#ifdef GPU_AVAILABLE
// Structure of flattened vectors
struct SketchStrides {
	size_t bin_stride;
	size_t kmer_stride;
	size_t sample_stride;
	size_t sketchsize64; 
	size_t bbits;
};

// defined in dist.cu
std::tuple<size_t, size_t> initialise_device(const int device_id);

std::vector<float> dispatchDists(
				   std::vector<Reference>& ref_sketches,
				   std::vector<Reference>& query_sketches,
				   SketchStrides& ref_strides,
				   SketchStrides& query_strides,
				   const SketchSlice& sketch_subsample,
				   const std::vector<size_t>& kmer_lengths,
				   const bool self);
#endif

