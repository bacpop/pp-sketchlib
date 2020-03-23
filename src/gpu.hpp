/*
 *
 * api.hpp
 * main functions for interacting with sketches
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

#ifdef GPU_AVAILABLE
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#endif

#include "reference.hpp"

#ifdef GPU_AVAILABLE
// defined in dist.cu
std::vector<float> query_db_cuda(std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int blockSize,
	const int device_id = 0);

// thrust vector using mallocManaged (supports device page faults)
// from https://tinyurl.com/whykq6o
template<class T>
class managed_allocator : public thrust::device_malloc_allocator<T>
{
  public:
    using value_type = T;

    typedef thrust::device_ptr<T>  pointer;
    inline pointer allocate(size_t n)
    {
      value_type* result = nullptr;
  
      cudaError_t error = cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
      }
  
      return thrust::device_pointer_cast(result);
    }
  
    inline void deallocate(pointer ptr, size_t)
    {
      cudaError_t error = cudaFree(thrust::raw_pointer_cast(ptr));
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
      }
    }
};

#endif

