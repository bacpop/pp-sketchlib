/*
 *
 * gpu.hpp
 * functions using CUDA
 *
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "device_memory.cuh"
#include "gpu_countmin.cuh"
#include "device_reads.cuh"

static const int warp_size = 32;

#ifdef GPU_AVAILABLE
// defined in dist.cu
std::tuple<size_t, size_t, size_t> initialise_device(const int device_id);

std::vector<uint64_t> flatten_by_bins(const std::vector<Reference> &sketches,
                                      const std::vector<size_t> &kmer_lengths,
                                      SketchStrides &strides,
                                      const size_t start_sample_idx,
                                      const size_t end_sample_idx,
                                      const int cpu_threads = 1);

std::vector<uint64_t>
flatten_by_samples(const std::vector<Reference> &sketches,
                   const std::vector<size_t> &kmer_lengths,
                   SketchStrides &strides, const size_t start_sample_idx,
                   const size_t end_sample_idx, const int cpu_threads = 1);

std::vector<float> dispatchDists(std::vector<Reference> &ref_sketches,
                                 std::vector<Reference> &query_sketches,
                                 SketchStrides &ref_strides,
                                 SketchStrides &query_strides,
                                 const FlatRandom &flat_random,
                                 const std::vector<uint16_t> &ref_random_idx,
                                 const std::vector<uint16_t> &query_random_idx,
                                 const SketchSlice &sketch_subsample,
                                 const std::vector<size_t> &kmer_lengths,
                                 const bool self, const int cpu_threads,
                                 const size_t shared_size);

// in sketch.cu
void copyNtHashTablesToDevice();

std::vector<uint64_t> get_signs(DeviceReads &reads, GPUCountMin &countmin,
                                const int k, const bool use_rc,
                                const uint16_t min_count,
                                const uint64_t binsize, const uint64_t nbins,
                                const size_t sample_n,
                                const size_t shared_size_available);

#endif
