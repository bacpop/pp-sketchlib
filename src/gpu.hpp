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

struct RandomStrides {
	size_t kmer_stride;
	size_t cluster_inner_stride;
	size_t cluster_outer_stride;
};

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

// defined in cuda.cuh
std::tuple<size_t, size_t> initialise_device(const int device_id);

// defined in dist.cu
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

// defined in sketch.cu
class GPUCountMin {
    public:
        GPUCountMin();
        ~GPUCountMin();

#ifdef __NVCC__
		__device__
#endif
        uint8_t add_count_min(uint64_t hash_val, const int k);

        void reset();

    private:
        uint8_t * d_countmin_table;

        // parameters - these are currently hard coded based on a short bacterial genome
        const unsigned int table_width_bits;
        const uint64_t mask;
        const uint32_t table_width;
        const int hash_per_hash = 2;
        const int table_rows = 4;
        const size_t table_cells;
};

class DeviceReads {
    public:
        DeviceReads(const SeqBuf& seq_in, const size_t n_threads);
        ~DeviceReads();

        char * read_ptr() { return d_reads; }
        size_t count() const { return n_reads; }
        size_t length() const { return read_length; }

    private:
        // delete move and copy to avoid accidentally using them
        DeviceReads ( const DeviceReads & ) = delete;
        DeviceReads ( DeviceReads && ) = delete;

        char * d_reads;
        size_t n_reads;
        size_t read_length;
};

std::vector<uint64_t> get_signs(DeviceReads& reads,
                                GPUCountMin& countmin,
                                const int k,
                                const bool use_rc,
                                const uint16_t min_count,
                                const uint64_t binsize,
                                const uint64_t nbins);

#endif

