
#include "gpu_countmin.cuh"
#include "cuda.cuh"

GPUCountMin::GPUCountMin()
  : _table_width_bits(cuda_table_width_bits),
    _table_width(cuda_table_width), _hash_per_hash(cuda_hash_per_hash),
    _table_rows(cuda_table_rows), _table_cells(cuda_table_cells) {
  CUDA_CALL(cudaMalloc((void **)&_d_countmin_table,
                       _table_cells * sizeof(unsigned int)));
  reset();
}

GPUCountMin::~GPUCountMin() { CUDA_CALL(cudaFree(_d_countmin_table)); }


void GPUCountMin::reset() {
  CUDA_CALL(
    cudaMemset(_d_countmin_table, 0, _table_cells * sizeof(unsigned int)));
}

// Create a new hash from an nthash
__device__ inline uint64_t shifthash(const uint64_t hVal, const unsigned k,
  const unsigned i) {
  uint64_t tVal = hVal * (i ^ k * multiSeed);
  tVal ^= tVal >> multiShift;
  return (tVal);
}

// Loop variables are global constants defined in gpu.hpp
__device__ unsigned int add_count_min(unsigned int *d_countmin_table,
                                      uint64_t hash_val, const int k) {
  unsigned int min_count = UINT32_MAX;
  for (int hash_nr = 0; hash_nr < cuda_table_rows; hash_nr += cuda_hash_per_hash) {
    uint64_t current_hash = hash_val;
    for (uint i = 0; i < cuda_hash_per_hash; i++) {
      uint32_t hash_val_masked = current_hash & cuda_table_width;
      unsigned int cell_count =
          atomicInc(d_countmin_table + (hash_nr + i) * cuda_table_width +
                        hash_val_masked,
                    UINT32_MAX) +
          1;
      if (cell_count < min_count) {
        min_count = cell_count;
      }
      current_hash = current_hash >> cuda_table_width_bits;
    }
    hash_val = shifthash(hash_val, k, hash_nr / 2);
  }
  return (min_count);
}