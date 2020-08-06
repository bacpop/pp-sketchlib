#pragma once

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cpp11/protect.hpp>

const int progressBitshift = 10; // Update every 2^10 = 1024 dists

static void HandleCUDAError(const char *file, int line,
                            cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
  cudaDeviceSynchronize();
#endif

  if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess) {
    if (status == cudaErrorUnknown) {
      cpp11::stop("%s(%i) An Unknown CUDA Error Occurred :(\n",
                  file, line);
    }
    cpp11::stop("%s(%i) CUDA Error Occurred;\n%s\n",
                file, line, cudaGetErrorString(status));
  }
}

#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))

// Use atomic add to update a counter, so progress works regardless of
// dispatch order
__device__
void update_progress(long long dist_idx,
					 long long dist_n,
					 volatile int * blocks_complete) {
	// Progress indicator
	// The >> progressBitshift is a divide by 1024 - update roughly every 0.1%
	if (dist_idx % (dist_n >> progressBitshift) == 0)
	{
		atomicAdd((int *)blocks_complete, 1);
		__threadfence_system();
	}
}

// Initialise device and return info on its memory
std::tuple<size_t, size_t> initialise_device(const int device_id) {
	cudaSetDevice(device_id);
	cudaDeviceReset();

	size_t mem_free = 0; size_t mem_total = 0;
	cudaMemGetInfo(&mem_free, &mem_total);
	return(std::make_tuple(mem_free, mem_total));
}