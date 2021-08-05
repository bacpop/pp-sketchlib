
/*
 *
 * sketch.cu
 * CUDA version of bindash sketch method
 *
 */

#include <stdint.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

// memcpy_async
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#pragma diag_suppress static_var_with_dynamic_init

#include "cuda.cuh"
#include "gpu.hpp"

// nthash
#include "sketch/nthash_tables.hpp"

// Tables on device
__constant__ uint64_t d_seedTab[256];
__constant__ uint64_t d_A33r[33];
__constant__ uint64_t d_A31l[31];
__constant__ uint64_t d_C33r[33];
__constant__ uint64_t d_C31l[31];
__constant__ uint64_t d_G33r[33];
__constant__ uint64_t d_G31l[31];
__constant__ uint64_t d_T33r[33];
__constant__ uint64_t d_T31l[31];
__constant__ uint64_t d_N33r[33];
__constant__ uint64_t d_N31l[31];
__constant__ uint64_t *d_msTab33r[256];
__constant__ uint64_t *d_msTab31l[256];

// main nthash functions - see nthash.hpp
// All others are built from calling these

__device__ inline uint64_t rol1(const uint64_t v) {
  return (v << 1) | (v >> 63);
}

__device__ inline uint64_t ror1(const uint64_t v) {
  return (v >> 1) | (v << 63);
}

__device__ inline uint64_t rol31(const uint64_t v, unsigned s) {
  s %= 31;
  return ((v << s) | (v >> (31 - s))) & 0x7FFFFFFF;
}

__device__ inline uint64_t rol33(const uint64_t v, unsigned s) {
  s %= 33;
  return ((v << s) | (v >> (33 - s))) & 0x1FFFFFFFF;
}

__device__ inline uint64_t swapbits033(const uint64_t v) {
  uint64_t x = (v ^ (v >> 33)) & 1;
  return v ^ (x | (x << 33));
}

__device__ inline uint64_t swapbits3263(const uint64_t v) {
  uint64_t x = ((v >> 32) ^ (v >> 63)) & 1;
  return v ^ ((x << 32) | (x << 63));
}

// Forward strand hash for first k-mer
__device__ inline void NT64(const char *kmerSeq, const unsigned k,
                            uint64_t &fhVal) {
  fhVal = 0;
  for (int i = k - 1; i >= 0; i--) {
    // Ns are removed, but this is how to check for them
    /*
    if(seedTab[(unsigned char)kmerSeq[i * baseStride]]==seedN) {
        locN=i;
        return false;
    }
    */
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= d_seedTab[(unsigned char)kmerSeq[k - 1 - i]];
  }
  // return true;
}

// Both strand hashes for first k-mer
__device__ inline void NTC64(const char *kmerSeq, const unsigned k,
                             uint64_t &fhVal, uint64_t &rhVal, uint64_t &hVal) {
  hVal = fhVal = rhVal = 0;
  for (int i = (k - 1); i >= 0; i--) {
    // Ns are removed, but this is how to check for them
    /*
    if(seedTab[(unsigned char)kmerSeq[i * baseStride]]==seedN) {
        locN = i;
        return false;
    }
    */
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= d_seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= d_seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }
  hVal = (rhVal < fhVal) ? rhVal : fhVal;
  // return true;
}

// forward-strand ntHash for subsequent sliding k-mers
__device__ inline uint64_t NTF64(const uint64_t fhVal, const unsigned k,
                                 const unsigned char charOut,
                                 const unsigned char charIn) {
  uint64_t hVal = rol1(fhVal);
  hVal = swapbits033(hVal);
  hVal ^= d_seedTab[charIn];
  hVal ^= (d_msTab31l[charOut][k % 31] | d_msTab33r[charOut][k % 33]);
  return hVal;
}

// reverse-complement ntHash for subsequent sliding k-mers
__device__ inline uint64_t NTR64(const uint64_t rhVal, const unsigned k,
                                 const unsigned char charOut,
                                 const unsigned char charIn) {
  uint64_t hVal = rhVal ^ (d_msTab31l[charIn & cpOff][k % 31] |
                           d_msTab33r[charIn & cpOff][k % 33]);
  hVal ^= d_seedTab[charOut & cpOff];
  hVal = ror1(hVal);
  hVal = swapbits3263(hVal);
  return hVal;
}

// bindash functions
const uint64_t SIGN_MOD = (1ULL << 61ULL) - 1ULL;

// countmin and binsign
// using unsigned long long int = uint64_t due to atomicCAS prototype
__device__ void binhash(uint64_t *signs, unsigned int *countmin_table,
                        const uint64_t hash, const uint64_t binsize,
                        const int k, const uint16_t min_count) {
  unsigned long long int sign = hash % SIGN_MOD;
  unsigned long long int binidx = sign / binsize;
  // printf("binidx:%llu sign:%llu\n", binidx, sign);

  // Only consider if the bin is yet to be filled, or is min in bin
  // NB there is a potential race condition here as the bin may be written
  // to by another thread
  unsigned long long int current_bin_val = signs[binidx]; // stall long scoreboard here
  if (current_bin_val == UINT64_MAX || sign < current_bin_val) {
    if (add_count_min(countmin_table, hash, k) >= min_count) {
      unsigned long long int new_bin_val = atomicCAS(
          (unsigned long long int *)signs + binidx, current_bin_val, sign);
      // If the bin val has changed since first reading it in, CAS will not
      // write the new value and will return the new value. In this case, keep
      // trying as long as it's still the bin minimum
      while (new_bin_val != current_bin_val) {
        current_bin_val = new_bin_val;
        if (sign < current_bin_val) {
          new_bin_val = atomicCAS((unsigned long long int *)signs + binidx,
                                  current_bin_val, sign);
        }
      }
    }
  }
  __syncwarp();
}

// hash iterator object
__global__ void process_reads(char *read_seq, const size_t n_reads,
                              const size_t read_length, const int k,
                              uint64_t *signs, const uint64_t binsize,
                              unsigned int *countmin_table, const bool use_rc,
                              const uint16_t min_count, bool use_shared) {
  // Load reads in block into shared memory
  char *read_ptr;
  int read_length_bank_pad = read_length;
  // TODO: another possible optimisation would be to put signs into shared
  // may affect occupancy though
  if (use_shared) {
    const int bank_bytes = 8;
    read_length_bank_pad +=
        read_length % bank_bytes ? bank_bytes - read_length % bank_bytes : 0;
    extern __shared__ char read_shared[];
    auto block = cooperative_groups::this_thread_block();
    size_t n_reads_in_block = blockDim.x;
    if (blockDim.x * (blockIdx.x + 1) > n_reads) {
      n_reads_in_block = n_reads - blockDim.x * blockIdx.x;
    }
    // TODO: better performance if the reads are padded to 4 bytes
    // best performance if aligned to 128
    // NOTE: I think the optimal thing to do here is to align blockSize lots of
    // reads in global memory to 128 when reading in, then read in padded to 4
    // bytes Then can read in all at once with single memcpy_async with size
    // padded to align at 128, and individual reads padded to align at 4
    // NOTE 2: It may just be easiest to pack this into a class/type with
    // 4 chars when reading, or even a DNA alphabet bit vector
    for (int read_idx = 0; read_idx < n_reads_in_block; ++read_idx) {
      // Copies one read into shared
      cooperative_groups::memcpy_async(
          block, read_shared + read_idx * read_length_bank_pad,
          read_seq + read_length * (blockIdx.x * blockDim.x + read_idx),
          sizeof(char) * read_length);
    }
    cooperative_groups::wait(block);
    read_ptr = read_shared;
  } else {
    read_ptr = read_seq + read_length * (blockIdx.x * blockDim.x);
  }

  int read_index = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t fhVal, rhVal, hVal;
  if (read_index < n_reads) {
    // Get first valid k-mer
    if (use_rc) {
      NTC64(read_ptr + threadIdx.x * read_length_bank_pad, k, fhVal, rhVal,
            hVal);
      binhash(signs, countmin_table, hVal, binsize, k, min_count);
    } else {
      NT64(read_ptr + threadIdx.x * read_length_bank_pad, k, fhVal);
      binhash(signs, countmin_table, hVal, binsize, k, min_count);
    }

    // Roll through remaining k-mers in the read
    for (int pos = 0; pos < read_length - k; pos++) {
      fhVal = // stall short scoreboard
          NTF64(fhVal, k, read_ptr[threadIdx.x * read_length_bank_pad + pos],
                read_ptr[threadIdx.x * read_length_bank_pad + pos + k]);
      if (use_rc) {
        rhVal = // stall short scoreboard
            NTR64(rhVal, k, read_ptr[threadIdx.x * read_length_bank_pad + pos],
                  read_ptr[threadIdx.x * read_length_bank_pad + pos + k]);
        hVal = (rhVal < fhVal) ? rhVal : fhVal;
        binhash(signs, countmin_table, hVal, binsize, k, min_count);
      } else {
        binhash(signs, countmin_table, fhVal, binsize, k, min_count);
      }
    }
  }
  __syncwarp();
}

void copyNtHashTablesToDevice() {
  CUDA_CALL(
      cudaMemcpyToSymbolAsync(d_seedTab, seedTab, 256 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_A33r, A33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_A31l, A31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_C33r, C33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_C31l, C31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_G33r, G33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_G31l, G31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_T33r, T33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_T31l, T31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_N33r, N33r, 33 * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_N31l, N31l, 31 * sizeof(uint64_t)));
  CUDA_CALL(cudaDeviceSynchronize());

  uint64_t *A33r_ptr, *A31l_ptr, *C33r_ptr, *C31l_ptr, *G33r_ptr, *G31l_ptr,
      *T33r_ptr, *T31l_ptr, *N33r_ptr, *N31l_ptr;
  CUDA_CALL(cudaGetSymbolAddress((void **)&A33r_ptr, d_A33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&A31l_ptr, d_A31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&C33r_ptr, d_C33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&C31l_ptr, d_C31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&G33r_ptr, d_G33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&G31l_ptr, d_G31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&T33r_ptr, d_T33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&T31l_ptr, d_T31l));
  CUDA_CALL(cudaGetSymbolAddress((void **)&N33r_ptr, d_N33r));
  CUDA_CALL(cudaGetSymbolAddress((void **)&N31l_ptr, d_N31l));

  static const uint64_t *d_addr_msTab33r[256] = {
      N33r_ptr, T33r_ptr, N33r_ptr, G33r_ptr,
      A33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr, // 0..7
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 8..15
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 16..23
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 24..31
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 32..39
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 40..47
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 48..55
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 56..63
      N33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, G33r_ptr, // 64..71
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 72..79
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      T33r_ptr, T33r_ptr, N33r_ptr, N33r_ptr, // 80..87
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 88..95
      N33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, G33r_ptr, // 96..103
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 104..111
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      T33r_ptr, T33r_ptr, N33r_ptr, N33r_ptr, // 112..119
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 120..127
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 128..135
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 136..143
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 144..151
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 152..159
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 160..167
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 168..175
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 176..183
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 184..191
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 192..199
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 200..207
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 208..215
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 216..223
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 224..231
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 232..239
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 240..247
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr,
      N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr // 248..255
  };

  static const uint64_t *d_addr_msTab31l[256] = {
      N31l_ptr, T31l_ptr, N31l_ptr, G31l_ptr,
      A31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr, // 0..7
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 8..15
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 16..23
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 24..31
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 32..39
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 40..47
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 48..55
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 56..63
      N31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, G31l_ptr, // 64..71
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 72..79
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      T31l_ptr, T31l_ptr, N31l_ptr, N31l_ptr, // 80..87
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 88..95
      N31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, G31l_ptr, // 96..103
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 104..111
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      T31l_ptr, T31l_ptr, N31l_ptr, N31l_ptr, // 112..119
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 120..127
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 128..135
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 136..143
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 144..151
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 152..159
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 160..167
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 168..175
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 176..183
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 184..191
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 192..199
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 200..207
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 208..215
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 216..223
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 224..231
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 232..239
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 240..247
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr,
      N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr // 248..255
  };

  CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab31l, d_addr_msTab31l,
                                    256 * sizeof(uint64_t *)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab33r, d_addr_msTab33r,
                                    256 * sizeof(uint64_t *)));
  CUDA_CALL(cudaDeviceSynchronize());
}

// main function called here returns signs vector - rest can be done by
// sketch.cpp
std::vector<uint64_t> get_signs(DeviceReads &reads, GPUCountMin &countmin,
                                const int k, const bool use_rc,
                                const uint16_t min_count,
                                const uint64_t binsize, const uint64_t nbins,
                                const size_t sample_n,
                                const size_t shared_size_available) {
  // Set countmin to zero (already on device)
  countmin.reset();

  // Signs
  std::vector<uint64_t> signs(nbins, UINT64_MAX);
  uint64_t *d_signs;
  CUDA_CALL(cudaMalloc((void **)&d_signs, nbins * sizeof(uint64_t)));
  CUDA_CALL(cudaMemcpy(d_signs, signs.data(), nbins * sizeof(uint64_t),
                       cudaMemcpyDefault));

  // Run process_read kernel, looping over reads loaded into buffer
  //      This runs nthash on read sequence at all k-mer lengths
  //      Check vs signs and countmin on whether to add each
  const size_t blockSize = 32; // best from profiling. Occupancy limited by size of shared mem request
  const int bank_bytes = 8;
  const int read_length_bank_pad =
      reads.length() % bank_bytes ? bank_bytes - reads.length() % bank_bytes
                                  : 0;
  size_t shared_mem_size =
      (read_length_bank_pad + reads.length()) * blockSize * sizeof(char);
  bool use_shared = true;
  if (shared_mem_size > shared_size_available) {
    use_shared = false;
    shared_mem_size = 0;
  }

  reads.reset_buffer();
  while (reads.next_buffer()) {
    size_t blockCount = (reads.buffer_count() + blockSize - 1) / blockSize;
    CUDA_CALL(cudaDeviceSynchronize()); // Make sure copy is finished
    process_reads<<<blockCount, blockSize, shared_mem_size>>>(
        reads.read_ptr(), reads.buffer_count(), reads.length(), k, d_signs,
        binsize, countmin.get_table(), use_rc, min_count, use_shared);

    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error(
          "Error when processing sketches with shared memory");
    }

    // Check for interrupt
    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
  }

  // Copy signs back from device (memcpy syncs)
  CUDA_CALL(cudaMemcpy(signs.data(), d_signs, nbins * sizeof(uint64_t),
                       cudaMemcpyDefault));
  CUDA_CALL(cudaFree(d_signs));

  fprintf(stderr, "%cSample %lu\tk = %d  ", 13, sample_n + 1, k);

  return (signs);
}
