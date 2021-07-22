
#include "device_memory.cuh"
#include "cuda.cuh"
#include "gpu.hpp"

// Sets up data structures and loads them onto the device
DeviceMemory::DeviceMemory(
    SketchStrides &ref_strides, SketchStrides &query_strides,
    std::vector<Reference> &ref_sketches,
    std::vector<Reference> &query_sketches, const SketchSlice &sample_slice,
    const FlatRandom &flat_random, const std::vector<uint16_t> &ref_random_idx,
    const std::vector<uint16_t> &query_random_idx,
    const std::vector<size_t> &kmer_lengths, long long dist_rows,
    const bool self, const int cpu_threads)
    : _n_dists(dist_rows * 2), d_query_sketches(nullptr),
      d_query_random(nullptr) {
  // Set up reference sketches, flatten and copy to device
  std::vector<uint64_t> flat_ref = flatten_by_samples(
      ref_sketches, kmer_lengths, ref_strides, sample_slice.ref_offset,
      sample_slice.ref_offset + sample_slice.ref_size, cpu_threads);
  CUDA_CALL(
      cudaMalloc((void **)&d_ref_sketches, flat_ref.size() * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpy(d_ref_sketches, flat_ref.data(),
                       flat_ref.size() * sizeof(uint64_t), cudaMemcpyDefault));

  // Preload random match chances, which have already been flattened
  CUDA_CALL(cudaMalloc((void **)&d_random_table,
                       std::get<1>(flat_random).size() * sizeof(float)));
  CUDA_CALL(cudaMemcpy(d_random_table, std::get<1>(flat_random).data(),
                       std::get<1>(flat_random).size() * sizeof(float),
                       cudaMemcpyDefault));
  CUDA_CALL(cudaMalloc((void **)&d_ref_random,
                       sample_slice.ref_size * sizeof(uint16_t)));
  CUDA_CALL(
      cudaMemcpy(d_ref_random, ref_random_idx.data() + sample_slice.ref_offset,
                 sample_slice.ref_size * sizeof(uint16_t), cudaMemcpyDefault));

  // If ref v query mode, also flatten query vector and copy to device
  if (!self) {
    std::vector<uint64_t> flat_query = flatten_by_bins(
        query_sketches, kmer_lengths, query_strides, sample_slice.query_offset,
        sample_slice.query_offset + sample_slice.query_size, cpu_threads);
    CUDA_CALL(cudaMalloc((void **)&d_query_sketches,
                         flat_query.size() * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpy(d_query_sketches, flat_query.data(),
                         flat_query.size() * sizeof(uint64_t),
                         cudaMemcpyDefault));

    CUDA_CALL(cudaMalloc((void **)&d_query_random,
                         sample_slice.query_size * sizeof(uint16_t)));
    CUDA_CALL(cudaMemcpy(
        d_query_random, query_random_idx.data() + sample_slice.query_offset,
        sample_slice.query_size * sizeof(uint16_t), cudaMemcpyDefault));
  } else {
    query_strides = ref_strides;
  }

  // Copy or set other arrays needed on device (kmers and distance output)
  std::vector<int> kmer_ints(kmer_lengths.begin(), kmer_lengths.end());
  CUDA_CALL(cudaMalloc((void **)&d_kmers, kmer_ints.size() * sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_kmers, kmer_ints.data(),
                       kmer_ints.size() * sizeof(int), cudaMemcpyDefault));

  CUDA_CALL(cudaMalloc((void **)&d_dist_mat, _n_dists * sizeof(float)));
  CUDA_CALL(cudaMemset(d_dist_mat, 0, _n_dists * sizeof(float)));
}

DeviceMemory::~DeviceMemory() {
  CUDA_CALL(cudaFree(d_ref_sketches));
  CUDA_CALL(cudaFree(d_query_sketches));
  CUDA_CALL(cudaFree(d_random_table));
  CUDA_CALL(cudaFree(d_ref_random));
  CUDA_CALL(cudaFree(d_query_random));
  CUDA_CALL(cudaFree(d_kmers));
  CUDA_CALL(cudaFree(d_dist_mat));
}

std::vector<float> DeviceMemory::read_dists() {
  std::vector<float> dists(_n_dists);
  CUDA_CALL(cudaMemcpy(dists.data(), d_dist_mat, _n_dists * sizeof(float),
                       cudaMemcpyDefault));
  return dists;
}