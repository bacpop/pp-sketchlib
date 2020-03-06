/*
 *
 * dist.cpp
 * PopPUNK dists using CUDA
 *
 */

// std
#include <cstdint>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iomanip>

// cuda
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// internal headers
#include "bitfuncs.hpp"
#include "gpu.hpp"

// Error checking of dynamic memory allocation on device
// https://stackoverflow.com/a/14038590
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

template <class T>
__device__
T non_neg_minus(T a, T b) {
	return a > b ? (a - b) : 0;
}

// CUDA version of bindash dist function
__device__
float jaccard_dist(const uint64_t * sketch1, 
                    const uint64_t * sketch2, 
					const size_t sketchsize64,
                    const size_t bbits) 
{
	size_t samebits = 0;
    for (size_t i = 0; i < sketchsize64; i++) 
    {
		uint64_t bits = ~((uint64_t)0ULL);
		for (size_t j = 0; j < bbits; j++) 
        {
			// iff implementing a bin_stride != 1, change index here
			bits &= ~(sketch1[i * bbits + j] ^ sketch2[i * bbits + j]);
		}

		samebits += __popcll(bits); // CUDA 64-bit popcnt
	}
	const size_t maxnbits = sketchsize64 * NBITS(uint64_t); 
	const size_t expected_samebits = (maxnbits >> bbits);
	size_t intersize = samebits;
	if (!expected_samebits) 
	{
		size_t ret = non_neg_minus(samebits, expected_samebits);
		intersize = ret * maxnbits / (maxnbits - expected_samebits);
	}
	size_t unionsize = NBITS(uint64_t) * sketchsize64;
    float jaccard = intersize/(float)unionsize;
    return(jaccard);
}

// Gets Jaccard distance across k-mer lengths and runs a regression
// to get core and accessory
__device__
void regress_kmers(float *& dists,
				   const long long dist_idx,
				   const uint64_t * ref,
				   const long i, 
				   const uint64_t * query,
				   const long j, 
				   const int * kmers,
				   const int kmer_n,
				   const size_t sketchsize64, 
				   const size_t bbits,
				   const size_t kmer_stride, 
				   const size_t sample_stride)						  
{
	long long ref_offset = i * sample_stride;
	long long query_offset = j * sample_stride;
	float xsum = 0; float ysum = 0; float xysum = 0;
	float xsquaresum = 0; float ysquaresum = 0;
	for (unsigned int kmer_it = 0; kmer_it < kmer_n; ++kmer_it)
    {
		// Get Jaccard distance and move pointers
		float y = logf(jaccard_dist(ref + ref_offset, query + query_offset, sketchsize64, bbits)); 
		ref_offset += kmer_stride;
		query_offset += kmer_stride;
		
		// Running totals
		xsum += kmers[kmer_it]; 
		ysum += y; 
		xysum += kmers[kmer_it] * y;
		xsquaresum += kmers[kmer_it] * kmers[kmer_it];
		ysquaresum += y * y;
    }

	// Simple linear regression
	float xbar = xsum / kmer_n;
	float ybar = ysum / kmer_n;
    float x_diff = xsquaresum - powf(xsum, 2)/kmer_n;
    float y_diff = ysquaresum - powf(ysum, 2)/kmer_n;
	float xstddev = sqrtf((xsquaresum - powf(xsum, 2)/kmer_n)/kmer_n);
	float ystddev = sqrtf((ysquaresum - powf(ysum, 2)/kmer_n)/kmer_n);
	float r = (xysum - (xsum*ysum)/kmer_n) / sqrtf(x_diff*y_diff);
	float beta = r * (ystddev / xstddev);
    float alpha = ybar - beta * xbar;

	// Store core/accessory in dists, truncating at zero
	float core_dist = 0, accessory_dist = 0;
	if (beta < 0)
	{
		core_dist = -expm1f(beta); // 1-exp(beta)
	}
	if (alpha < 0)
	{
		accessory_dist = -expm1f(alpha); // 1-exp(alpha)
	}
	dists[dist_idx*2] = core_dist;
	dists[dist_idx*2 + 1] = accessory_dist;
}

// Functions to convert index position to/from squareform to condensed form
__device__
long calc_row_idx(const long long k, const long n) {
	return n - 2 - floorf(sqrtf(__ll2float_rn(-8*k + 4*n*(n-1)-7))/2 - 0.5);
}

__device__
long calc_col_idx(const long long k, const long i, const long long n) {
	return k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
}

__device__
long long square_to_condensed(long i, long j, long n) {
    assert(i > j);
	return (n*j - ((j*(j+1)) >> 1) + i - 1 - j);
}

// Takes a position in the condensed form distance matrix, converts into an
// i, j for the ref/query vectors. Calls regression with these start points
__global__
void calculate_dists(const uint64_t * ref,
					 const long ref_n,
					 const uint64_t * query,
					 const long query_n,
					 const int * kmers,
					 const int kmer_n,
					 float * dists,
					 const long long dist_n,
					 const size_t sketchsize64, 
					 const size_t bbits,
					 const size_t kmer_stride,
					 const size_t sample_stride)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (long long dist_idx = index; dist_idx < dist_n; dist_idx += stride)
	{
		long i, j;
		if (query == nullptr)
		{
			query = ref;
			i = calc_row_idx(dist_idx, ref_n);
			j = calc_col_idx(dist_idx, i, ref_n);
			if (j <= i)
			{
				continue;
			}
		}
		else if (query != nullptr)
		{
			i = dist_idx % ref_n;
			j = (long)(__fdividef(dist_idx, query_n) + 0.001f);
		}
		regress_kmers(dists, dist_idx,
			ref, i,
			query, j,
			kmers, kmer_n,
			sketchsize64, bbits,
			kmer_stride, sample_stride);
	}
}

// Turn a vector of references into a flattened vector of
// uint64 with strides bins * kmers * samples
thrust::host_vector<uint64_t> flatten_sketches(
	const std::vector<Reference>& sketches,
	const std::vector<size_t>& kmer_lengths,
	const size_t sample_stride)
{
	thrust::host_vector<uint64_t> flat_ref(sample_stride * sketches.size());
	auto flat_ref_it = flat_ref.begin();
	for (auto sample_it = sketches.cbegin(); sample_it != sketches.cend(); sample_it++)
	{
		for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend(); kmer_it++)
		{
			thrust::copy(sample_it->get_sketch(*kmer_it).cbegin(),
						 sample_it->get_sketch(*kmer_it).cend(),
						 flat_ref_it);
            flat_ref_it += sample_it->get_sketch(*kmer_it).size();
		}
	}
	return flat_ref;
}

// Checks bbits, sketchsize and k-mer lengths are identical in
// all sketches
void checkSketchParamsMatch(const std::vector<Reference>& sketches, 
	const std::vector<size_t>& kmer_lengths, 
	const size_t bbits, 
	const size_t sketchsize64)
{
	for (auto sketch_it = sketches.cbegin(); sketch_it != sketches.cend(); sketch_it++)
	{
		if (sketch_it->bbits() != bbits)
		{
			throw std::runtime_error("Mismatching bbits in sketches");
		}
		if (sketch_it->sketchsize64() != sketchsize64)
		{
			throw std::runtime_error("Mismatching sketchsize64 in sketches");
		}
		if (sketch_it->kmer_lengths() != kmer_lengths)
		{
			throw std::runtime_error("Mismatching k-mer lengths in sketches");
		}
	}
}

// Main function callable via API
// Checks inputs
// Copies data to device
// Runs kernel function across distance elements
// Copies and returns results
std::vector<float> query_db_cuda(std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int blockSize,
	const size_t max_device_mem,
    const int device_id)
{
	std::cerr << "Calculating distances on GPU device " << device_id << std::endl;
    
    // Check if ref = query, then run as self mode
	// TODO implement max device mem
	// TODO will involve taking square blocks of distmat
	bool self = false;

	// Check sketches are compatible
	size_t bbits = ref_sketches[0].bbits();
	size_t sketchsize64 = ref_sketches[0].sketchsize64();
	checkSketchParamsMatch(ref_sketches, kmer_lengths, bbits, sketchsize64);
	if (!self)
	{
		checkSketchParamsMatch(query_sketches, kmer_lengths, bbits, sketchsize64);
	}
	
	// long bin_stride = 1; // unused - pass this iff strides of flattened array change
	size_t kmer_stride = sketchsize64 * bbits;
	size_t sample_stride = kmer_stride * kmer_lengths.size();	
	long long dist_rows; long n_samples = 0;
	if (ref_sketches == query_sketches)
    {
		dist_rows = static_cast<long long>(0.5*(ref_sketches.size())*(ref_sketches.size() - 1));
		n_samples = ref_sketches.size(); 
		self = true;
	}
	else
	{
		dist_rows = ref_sketches.size() * query_sketches.size();
		n_samples = ref_sketches.size() + query_sketches.size(); 
	}
	double est_size  = (sample_stride * n_samples * sizeof(uint64_t) + dist_rows * sizeof(float))/(1048576);
	std::cerr << "Estimated device memory required: " << std::fixed << std::setprecision(0) << est_size << "Mb" << std::endl;

	size_t mem_free = 0; size_t mem_total = 0;
	cudaMemGetInfo(&mem_free, &mem_total);
	std::cerr << "Total device memory: " << std::fixed << std::setprecision(0) << mem_total/(1048576) << "Mb" << std::endl;
	std::cerr << "Free device memory: " << std::fixed << std::setprecision(0) << mem_free/(1048576) << "Mb" << std::endl;

	// Initialise device
	cudaSetDevice(device_id);
	cudaDeviceReset();

	// flatten the input sketches and copy ref sketches to device
	thrust::device_vector<int> d_kmers = kmer_lengths;
	int* d_kmers_array = thrust::raw_pointer_cast( &d_kmers[0] );
	thrust::host_vector<uint64_t> flat_ref = flatten_sketches(ref_sketches, kmer_lengths, sample_stride);
	thrust::device_vector<uint64_t> d_ref_sketches = flat_ref;

	// Set up query and distance arrays and copy to device
	uint64_t* d_ref_array = thrust::raw_pointer_cast( &d_ref_sketches[0] );
	uint64_t* d_query_array = nullptr;
	if (!self)
    {
		dist_rows = ref_sketches.size() * query_sketches.size();
		thrust::host_vector<uint64_t> flat_query = flatten_sketches(query_sketches, kmer_lengths, sample_stride);
		thrust::device_vector<uint64_t> d_query_sketches = flat_query;
		d_query_array = thrust::raw_pointer_cast( &d_query_sketches[0] ); 
	}
	thrust::device_vector<float> dist_mat(dist_rows*2, 0);
	float* d_dist_array = thrust::raw_pointer_cast( &dist_mat[0] );

	// Run dists on device
	int blockCount = (dist_rows + blockSize - 1) / blockSize;
	calculate_dists<<<blockCount, blockSize>>>(
		d_ref_array,
		ref_sketches.size(),
		d_query_array,
		query_sketches.size(),
		d_kmers_array,
		kmer_lengths.size(),
		d_dist_array,
		dist_rows,
		sketchsize64,
		bbits,
		kmer_stride,
		sample_stride);
				
	// copy results from device to return
	std::vector<float> dist_results(dist_mat.size());
	try
	{
		thrust::copy(dist_mat.begin(), dist_mat.end(), dist_results.begin());
	}
	catch(thrust::system_error &e)
	{
		// output an error message and exit
		std::cerr << "Error getting result: " << std::endl;
		std::cerr << e.what() << std::endl;
		exit(1);
	}
	return dist_results;
}