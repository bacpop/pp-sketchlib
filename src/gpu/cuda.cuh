#pragma once

#include <stdexcept>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int progressBitshift = 10; // Update every 2^10 = 1024 dists

static void HandleCUDAError(const char *file, int line,
                            cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
  cudaDeviceSynchronize();
#endif

  if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess) {
    if (status == cudaErrorUnknown) {
      printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
    }
    printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line,
           cudaGetErrorString(status));
    throw std::runtime_error("CUDA error");
  }
}

#define CUDA_CALL(err) (HandleCUDAError(__FILE__, __LINE__, err))

// Use atomic add to update a counter, so progress works regardless of
// dispatch order
__device__ inline void update_progress(long long dist_idx, long long dist_n,
                                       volatile int *blocks_complete) {
  // Progress indicator
  // The >> progressBitshift is a divide by 1024 - update roughly every 0.1%
  if (dist_idx % (dist_n >> progressBitshift) == 0) {
    assert(blocks_complete >= 0);
    atomicAdd((int *)blocks_complete, 1);
    __threadfence_system();
  }
}
