#pragma once

#include <stdexcept>
#include <stdio.h>

#include <type_traits>
static_assert(__CUDACC_VER_MAJOR__ >= 11, "CUDA >=11.0 required");

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int progress_blocks = 1000; // Update 1000

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
#define CUDA_CALL_NOTHROW( err ) (err)

class progress_atomics {
public:
  progress_atomics() {
    CUDA_CALL(cudaMallocManaged(&blocks_complete, sizeof(int)));
    CUDA_CALL(cudaMallocManaged(&kill_kernel, sizeof(bool)));
    *blocks_complete = 0;
    *kill_kernel = false;
  }

  ~progress_atomics() {
    CUDA_CALL(cudaFree((void *)blocks_complete));
    CUDA_CALL(cudaFree(kill_kernel));
  }

  __host__
  void set_kill() {
    *kill_kernel = true;
  }

  __device__
  void check_kill() {
    if (*kill_kernel) {
      __trap();
    }
  }

  __device__
  void tick(const int advance) {
    atomicAdd((int *)blocks_complete, advance);
    __threadfence_system();
  }

  __host__ __device__
  int complete() {
    return *blocks_complete;
  }

private:
  progress_atomics(const progress_atomics &) = delete;
  progress_atomics(progress_atomics &&) = delete;

  volatile int *blocks_complete;
  bool *kill_kernel;
};

// Use atomic add to update a counter, so progress works regardless of
// dispatch order
__device__ inline void update_progress(long long dist_idx, long long dist_n,
                                       progress_atomics *progress) {
  // Progress indicator
  if (dist_idx % (dist_n / progress_blocks) == 0) {
    progress->check_kill();
    progress->tick(1);
  }
}
