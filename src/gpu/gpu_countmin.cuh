#pragma once

#include "sketch/nthash_tables.hpp"

// parameters - these are currently hard coded based on a 3090 (24Gb RAM)
const unsigned int cuda_table_width_bits =
    30; // 2^30 + 1 = 1073741825 =~ 1 billion k-mers
constexpr uint64_t cuda_table_width{0x3FFFFFFF}; // 30 lowest bits ON
const int cuda_hash_per_hash =
    2; // This should be 2, or the table is likely too narrow
const int cuda_table_rows =
    4; // Number of hashes, should be a multiple of hash_per_hash
constexpr uint64_t cuda_table_cells = cuda_table_rows * cuda_table_width;

class GPUCountMin {
public:
  GPUCountMin();

  ~GPUCountMin();

  unsigned int *get_table() { return _d_countmin_table; }

  void reset();

private:
  // delete move and copy to avoid accidentally using them
  GPUCountMin(const GPUCountMin &) = delete;
  GPUCountMin(GPUCountMin &&) = delete;

  unsigned int *_d_countmin_table;

  const unsigned int _table_width_bits;
  const uint64_t _table_width;
  const int _hash_per_hash;
  const int _table_rows;
  const uint64_t _table_cells;
};

#ifdef __NVCC__
__device__ unsigned int add_count_min(unsigned int *d_countmin_table,
                                      uint64_t hash_val, const int k);
#endif
