#pragma once

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
