#pragma once

#include "reference.hpp"

// Align structs
// https://stackoverflow.com/a/12779757
#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

struct ALIGN(8) RandomStrides {
  size_t kmer_stride;
  size_t cluster_inner_stride;
  size_t cluster_outer_stride;
};

typedef std::tuple<RandomStrides, std::vector<float>> FlatRandom;

#ifdef GPU_AVAILABLE

// Structure of flattened vectors
struct ALIGN(16) SketchStrides {
  size_t bin_stride;
  size_t kmer_stride;
  size_t sample_stride;
  size_t sketchsize64;
  size_t bbits;
};

struct ALIGN(8) SketchSlice {
  size_t ref_offset;
  size_t ref_size;
  size_t query_offset;
  size_t query_size;
};

// Memory on device for each operation
class DeviceMemory {
public:
  DeviceMemory(SketchStrides &ref_strides, SketchStrides &query_strides,
               std::vector<Reference> &ref_sketches,
               std::vector<Reference> &query_sketches,
               const SketchSlice &sample_slice, const FlatRandom &flat_random,
               const std::vector<uint16_t> &ref_random_idx,
               const std::vector<uint16_t> &query_random_idx,
               const std::vector<size_t> &kmer_lengths, long long dist_rows,
               const bool self, const int cpu_threads);

  ~DeviceMemory();

  std::vector<float> read_dists();

  uint64_t *ref_sketches() { return d_ref_sketches; }
  uint64_t *query_sketches() { return d_query_sketches; }
  float *random_table() { return d_random_table; }
  uint16_t *ref_random() { return d_ref_random; }
  uint16_t *query_random() { return d_query_random; }
  int *kmers() { return d_kmers; }
  float *dist_mat() { return d_dist_mat; }

private:
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory(DeviceMemory &&) = delete;

  size_t _n_dists;
  uint64_t *d_ref_sketches;
  uint64_t *d_query_sketches;
  float *d_random_table;
  uint16_t *d_ref_random;
  uint16_t *d_query_random;
  int *d_kmers;
  float *d_dist_mat;
};

#endif