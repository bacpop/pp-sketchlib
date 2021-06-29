
// parameters - these are currently hard coded based on a 3090 (24Gb RAM)
const unsigned int table_width_bits = 30; // 2^30 + 1 = 1073741825 =~ 1 billion k-mers
constexpr uint64_t table_width{0x3FFFFFFF};       // 30 lowest bits ON
const int hash_per_hash =
    2; // This should be 2, or the table is likely too narrow
const int table_rows =
    4; // Number of hashes, should be a multiple of hash_per_hash
constexpr uint64_t table_cells = table_rows * table_width;

class GPUCountMin {
public:
  GPUCountMin()
    : _table_width_bits(table_width_bits),
      _table_width(table_width), _hash_per_hash(hash_per_hash),
      _table_rows(table_rows), _table_cells(table_cells) {
    CUDA_CALL(cudaMalloc((void **)&_d_countmin_table,
                        table_cells * sizeof(unsigned int)));
    reset();
  }

  ~GPUCountMin() { CUDA_CALL(cudaFree(_d_countmin_table)); }

  unsigned int *get_table() { return _d_countmin_table; }

  void reset() {
    CUDA_CALL(
      cudaMemset(_d_countmin_table, 0, table_cells * sizeof(unsigned int)));
  }

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

// Loop variables are global constants defined in gpu.hpp
__device__ unsigned int add_count_min(unsigned int *d_countmin_table,
                                      uint64_t hash_val, const int k) {
  unsigned int min_count = UINT32_MAX;
  for (int hash_nr = 0; hash_nr < table_rows; hash_nr += hash_per_hash) {
    uint64_t current_hash = hash_val;
    for (uint i = 0; i < hash_per_hash; i++) {
      uint32_t hash_val_masked = current_hash & table_width;
      unsigned int cell_count =
          atomicInc(d_countmin_table + (hash_nr + i) * table_width +
                        hash_val_masked,
                    UINT32_MAX) +
          1;
      if (cell_count < min_count) {
        min_count = cell_count;
      }
      current_hash = current_hash >> table_width_bits;
    }
    hash_val = shifthash(hash_val, k, hash_nr / 2);
  }
  return (min_count);
}
