/*
 *
 * dist.cu
 * PopPUNK dists using CUDA
 * nvcc compiled part (try to avoid eigen)
 *
 */

// std
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <pybind11/pybind11.h>
namespace py = pybind11;

// memcpy_async
#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda/barrier>
#include <cooperative_groups.h>
#pragma diag_suppress static_var_with_dynamic_init
#endif

// internal headers
#include "cuda.cuh"
#include "dist/matrix_idx.hpp"
#include "gpu.hpp"
#include "sketch/bitfuncs.hpp"

/******************
 *			          *
 *	Device code   *
 *			          *
 *******************/

// Ternary used in observed_excess
template <class T> __device__ T non_neg_minus(T a, T b) {
  return a > b ? (a - b) : 0;
}

// Calculates excess observations above a random value
template <class T> __device__ T observed_excess(T obs, T exp, T max) {
  T diff = non_neg_minus(obs, exp);
  return (diff * max / (max - exp));
}

// CUDA version of bindash dist function (see dist.cpp)
__device__ float jaccard_dist(const uint64_t *sketch1, const uint64_t *sketch2,
                              const size_t sketchsize64, const size_t bbits,
                              const size_t s1_stride, const size_t s2_stride) {
  size_t samebits = 0;
  for (int i = 0; i < sketchsize64; i++) {
    int bin_index = i * bbits;
    uint64_t bits = ~((uint64_t)0ULL);
    for (int j = 0; j < bbits; j++) {
      // Almost all kernel time is spent on this line
      // (bbits * sketchsize64 * N^2 * 2 8-byte memory loads)
      bits &=
          ~(sketch1[bin_index * s1_stride] ^ sketch2[bin_index * s2_stride]);
      bin_index++;
    }
    samebits += __popcll(bits); // CUDA 64-bit popcnt
  }

  const size_t maxnbits = sketchsize64 * NBITS(uint64_t);
  const size_t expected_samebits = (maxnbits >> bbits);
  size_t intersize = samebits;
  if (!expected_samebits) {
    size_t ret = observed_excess(samebits, expected_samebits, maxnbits);
  }

  size_t unionsize = NBITS(uint64_t) * sketchsize64;
  float jaccard = __fdiv_ru(intersize, unionsize);
  return (jaccard);
}

// Simple linear regression, exact solution
// Avoids use of dynamic memory allocation on device, or
// linear algebra libraries
__device__ void simple_linear_regression(float *core_dist,
                                         float *accessory_dist,
                                         const float xsum, const float ysum,
                                         const float xysum,
                                         const float xsquaresum,
                                         const float ysquaresum, const int n) {
  if (n < 2) {
    *core_dist = 0;
    *accessory_dist = 0;
  } else {
    // CUDA fast-math intrinsics on floats, which give comparable accuracy
    // Speed gain is fairly minimal, as most time spent on Jaccard distance
    // __fmul_ru(x, y) = x * y and rounds up.
    // __fpow(x, a) = x^a give 0 for x<0, so not using here (and it is slow)
    float xbar = xsum / n;
    float ybar = ysum / n;
    float x_diff = xsquaresum - __fmul_ru(xsum, xsum) / n;
    float y_diff = ysquaresum - __fmul_ru(ysum, ysum) / n;
    float xstddev = __fsqrt_ru((xsquaresum - __fmul_ru(xsum, xsum) / n) / n);
    float ystddev = __fsqrt_ru((ysquaresum - __fmul_ru(ysum, ysum) / n) / n);
    float r =
        __fdiv_ru(xysum - __fmul_ru(xsum, ysum) / n, __fsqrt_ru(x_diff * y_diff));
    float beta = __fmul_ru(r, __fdiv_ru(ystddev, xstddev));
    float alpha = __fmaf_ru(-beta, xbar, ybar); // maf: x * y + z

    // Store core/accessory in dists, truncating at zero
    // Memory should be initialised to zero so else block not strictly
    // necessary, but better safe than sorry!
    if (beta < 0) {
      *core_dist = 1 - __expf(beta);
    } else {
      *core_dist = 0;
    }

    if (alpha < 0) {
      *accessory_dist = 1 - __expf(alpha);
    } else {
      *accessory_dist = 0;
    }
  }
}

/******************
 *			      *
 *	Global code   *
 *			      *
 *******************/

// Main kernel functions run on the device,
// but callable from the host

__global__ void calculate_dists(
    const bool self, const uint64_t *ref, const long ref_n,
    const uint64_t *query, const long query_n, const int *kmers,
    const int kmer_n, float *dists, const long long dist_n,
    const float *random_table, const uint16_t *ref_idx_lookup,
    const uint16_t *query_idx_lookup, const SketchStrides ref_strides,
    const SketchStrides query_strides, const RandomStrides random_strides,
    progress_atomics progress, const bool use_shared) {
  // Calculate indices for query, ref and results
  int ref_idx, query_idx, dist_idx;
  if (self) {
    // Blocks have the same i -- calculate blocks needed by each row up
    // to this point (blockIdx.x)
    int blocksDone = 0;
    for (query_idx = 0; query_idx < ref_n; query_idx++) {
      blocksDone += (ref_n + blockDim.x - 2 - query_idx) / blockDim.x;
      if (blocksDone > blockIdx.x) {
        break;
      }
    }
    // j (column) is given by multiplying the blocks needed for this i (row)
    // by the block size, plus offsets of i + 1 and the thread index
    int blocksPerQuery = (ref_n + blockDim.x - 2 - query_idx) / blockDim.x;
    ref_idx = query_idx + 1 + threadIdx.x +
              (blockIdx.x - (blocksDone - blocksPerQuery)) * blockDim.x;

    if (ref_idx < ref_n) {
      // Order of ref/query reversed here to give correct output order
      dist_idx = square_to_condensed(query_idx, ref_idx, ref_n);
    }
  } else {
    int blocksPerQuery = (ref_n + blockDim.x - 1) / blockDim.x;
    query_idx = blockIdx.x / blocksPerQuery;
    ref_idx = (blockIdx.x % blocksPerQuery) * blockDim.x + threadIdx.x;
    dist_idx = query_idx * ref_n + ref_idx;
  }
  __syncwarp();

  const uint64_t *ref_start = ref + ref_idx * ref_strides.sample_stride;
  const uint64_t *query_start = query + query_idx * query_strides.sample_stride;
  const float tolerance =
      __fdividef(5.0f, __int2float_rz(64 * ref_strides.sketchsize64));

  // Calculate Jaccard distances over k-mer lengths
  int kmer_used = 0;
  float xsum = 0;
  float ysum = 0;
  float xysum = 0;
  float xsquaresum = 0;
  float ysquaresum = 0;
  bool stop = false;
  for (int kmer_idx = 0; kmer_idx < kmer_n; kmer_idx++) {
    // Copy query sketch into __shared__ mem
    // Uses all threads *in a single warp* to do the copy
    // NB there is no disadvantage vs using multiple warps, as they would have
    // to wait (see
    // https://stackoverflow.com/questions/15468059/copy-to-the-shared-memory-in-cuda)
    // NB for query these reads will be coalesced, but for ref they won't, as
    // can't coalesce both here (bin inner stride) and in jaccard (sample inner
    // stride)
    const uint64_t *query_ptr;
    extern __shared__ uint64_t query_shared[];
    int query_bin_strides;
#if __CUDACC_VER_MAJOR__ >= 11
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (block.thread_rank() == 0) {
      init(&barrier, block.size()); // Friend function initializes barrier
    }
    block.sync();
#endif
    if (use_shared) {
      size_t sketch_bins = query_strides.bbits * query_strides.sketchsize64;
      size_t sketch_stride = query_strides.bin_stride;
      if (threadIdx.x < warp_size) {
        for (int lidx = threadIdx.x; lidx < sketch_bins; lidx += warp_size) {
#if __CUDACC_VER_MAJOR__ >= 11
          cuda::memcpy_async(query_shared + lidx,
                             query_start + (lidx * sketch_stride),
                             sizeof(uint64_t),
                             barrier);
#else
          query_shared[lidx] = query_start[lidx * sketch_stride];
#endif
        }
      }
      query_ptr = query_shared;
      query_bin_strides = 1;
    } else {
      query_ptr = query_start;
      query_bin_strides = query_strides.bin_stride;
    }
#if __CUDACC_VER_MAJOR__ >= 11
    barrier.arrive_and_wait();
#else
    __syncthreads();
#endif

    // Some threads at the end of the last block will have nothing to do
    // Need to have conditional here to avoid block on __syncthreads() above
    if (ref_idx < ref_n) {
      // Calculate Jaccard distance at current k-mer length
      float jaccard_obs = jaccard_dist(
          ref_start, query_ptr, ref_strides.sketchsize64, ref_strides.bbits,
          ref_strides.bin_stride, query_bin_strides);

      // Adjust for random matches
      float jaccard_expected =
          random_table[kmer_idx * random_strides.kmer_stride +
                       ref_idx_lookup[ref_idx] *
                           random_strides.cluster_inner_stride +
                       query_idx_lookup[query_idx] *
                           random_strides.cluster_outer_stride];
      float jaccard = observed_excess(jaccard_obs, jaccard_expected, 1.0f);
      // Stop regression if distances =~ 0
      if (jaccard < tolerance) {
        // Would normally break here, but gives no advantage on a GPU as causes
        // warp to diverge
        // As the thread blocks are used to load the query in, adding a break
        // would actually cause a stall. So just stop adding
        stop = true;
      } else if (!stop) {
        float y = __logf(jaccard);
        // printf("i:%d j:%d k:%d r:%f jac:%f y:%f\n", ref_idx, query_idx,
        // kmer_idx, jaccard_expected, jaccard_obs, y);

        // Running totals for regression
        kmer_used++;
        int kmer = kmers[kmer_idx];
        xsum += kmer;
        ysum += y;
        xysum += kmer * y;
        xsquaresum += kmer * kmer;
        ysquaresum += y * y;
      }
    }

    // Move to next k-mer length
    ref_start += ref_strides.kmer_stride;
    query_start += query_strides.kmer_stride;
  }

  if (ref_idx < ref_n) {
    // Run the regression, and store results in dists
    simple_linear_regression(dists + dist_idx, dists + dist_n + dist_idx, xsum,
                             ysum, xysum, xsquaresum, ysquaresum, kmer_used);

    update_progress(dist_idx, dist_n, progress);
  }
}

/***************
 *			       *
 *	Host code  *
 *			       *
 ***************/

// Get the blockSize and blockCount for CUDA call
std::tuple<size_t, size_t> getBlockSize(const size_t ref_samples,
                                        const size_t query_samples,
                                        const size_t dist_rows,
                                        const bool self) {
  // Each block processes a single query. As max size is 512 threads
  // per block, may need multiple blocks (non-exact multiples lead
  // to some wasted computation in threads)
  // We take the next multiple of 32 that is larger than the number of
  // reference sketches, up to a maximum of 512
  size_t blockSize =
      std::min(512, 32 * static_cast<int>((ref_samples + 32 - 1) / 32));
  size_t blockCount = 0;
  if (self) {
    for (int i = 0; i < ref_samples; i++) {
      blockCount += (ref_samples + blockSize - 2 - i) / blockSize;
    }
  } else {
    size_t blocksPerQuery = (ref_samples + blockSize - 1) / blockSize;
    blockCount = blocksPerQuery * query_samples;
  }
  return (std::make_tuple(blockSize, blockCount));
}

// Writes a progress meter using the device int which keeps
// track of completed jobs
void reportDistProgress(progress_atomics progress, long long dist_rows) {
  long long progress_blocks = 1 << progressBitshift;
  int now_completed = 0;
  float kern_progress = 0;
  if (dist_rows > progress_blocks) {
    while (now_completed < progress_blocks - 1) {
      if (PyErr_CheckSignals() != 0) {
        *(progress.kill_kernel) = true;
        throw py::error_already_set();
      }
      if (*(progress.blocks_complete) > now_completed) {
        now_completed = *(progress.blocks_complete);
        kern_progress = now_completed / (float)progress_blocks;
        fprintf(stderr, "%cProgress (GPU): %.1lf%%", 13, kern_progress * 100);
      } else {
        usleep(1000);
      }
    }
  }
}

// Initialise device and return info on its memory
std::tuple<size_t, size_t, size_t> initialise_device(const int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));

  size_t mem_free = 0;
  size_t mem_total = 0;
  CUDA_CALL(cudaMemGetInfo(&mem_free, &mem_total));
  int shared_size = 0;
  CUDA_CALL(cudaDeviceGetAttribute(
      &shared_size, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  return (
      std::make_tuple(mem_free, mem_total, static_cast<size_t>(shared_size)));
}

// Main function to run the distance calculations, reading/writing into
// device_arrays Cache preferences: Upper dist memory access is hard to predict,
// so try and cache as much as possible Query uses on-chip cache (__shared__) to
// store query sketch
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
                                 const size_t shared_size) {
  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  // Progress meter
  progress_atomics progress;
  progress.init();

  RandomStrides random_strides = std::get<0>(flat_random);
  long long dist_rows;
  if (self) {
    dist_rows =
        (sketch_subsample.ref_size * (sketch_subsample.ref_size - 1)) >> 1;
  } else {
    dist_rows = sketch_subsample.ref_size * sketch_subsample.query_size;
  }

  // Load memory onto device
  DeviceMemory device_arrays(ref_strides, query_strides, ref_sketches,
                             query_sketches, sketch_subsample, flat_random,
                             ref_random_idx, query_random_idx, kmer_lengths,
                             dist_rows, self, cpu_threads);

  size_t sketch_size_bytes =
      query_strides.sketchsize64 * query_strides.bbits * sizeof(uint64_t);
  bool use_shared = true;
  if (sketch_size_bytes > shared_size) {
    std::cerr << "You are using a large sketch size, which may slow down "
                 "computation on this device"
              << std::endl;
    std::cerr << "Reduce sketch size to "
              << std::floor(64 * shared_size /
                            (query_strides.bbits * sizeof(uint64_t)))
              << " or less for better performance" << std::endl;
    sketch_size_bytes = 0;
    use_shared = false;
  }

  size_t blockSize, blockCount;
  if (self) {
    std::tie(blockSize, blockCount) = getBlockSize(
        sketch_subsample.ref_size, sketch_subsample.ref_size, dist_rows, self);

    // Third argument is the size of __shared__ memory needed by a thread block
    // This is equal to the query sketch size in bytes (at a single k-mer
    // length)
    calculate_dists<<<blockCount, blockSize, sketch_size_bytes>>>(
        self, device_arrays.ref_sketches(), sketch_subsample.ref_size,
        device_arrays.ref_sketches(), sketch_subsample.ref_size,
        device_arrays.kmers(), kmer_lengths.size(), device_arrays.dist_mat(),
        dist_rows, device_arrays.random_table(), device_arrays.ref_random(),
        device_arrays.ref_random(), ref_strides, ref_strides, random_strides,
        progress, use_shared);
  } else {
    std::tie(blockSize, blockCount) =
        getBlockSize(sketch_subsample.ref_size, sketch_subsample.query_size,
                     dist_rows, self);

    // Third argument is the size of __shared__ memory needed by a thread block
    // This is equal to the query sketch size in bytes (at a single k-mer
    // length)
    calculate_dists<<<blockCount, blockSize, sketch_size_bytes>>>(
        self, device_arrays.ref_sketches(), sketch_subsample.ref_size,
        device_arrays.query_sketches(), sketch_subsample.query_size,
        device_arrays.kmers(), kmer_lengths.size(), device_arrays.dist_mat(),
        dist_rows, device_arrays.random_table(), device_arrays.ref_random(),
        device_arrays.query_random(), ref_strides, query_strides,
        random_strides, progress, use_shared);
  }

  // Check for error in kernel launch
  CUDA_CALL(cudaGetLastError());
  reportDistProgress(progress, dist_rows);
  fprintf(stderr, "%cProgress (GPU): 100.0%%\n", 13);
  progress.free();

  // Copy results back to host
  CUDA_CALL(cudaDeviceSynchronize());
  std::vector<float> dist_results = device_arrays.read_dists();

  return (dist_results);
}
