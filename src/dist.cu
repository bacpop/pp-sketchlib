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
#include <chrono>
#include <ctime>
#include <ratio>

// cuda
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// internal headers
#include "bitfuncs.hpp"
#include "gpu.hpp"

// mallocManaged for limited device memory
template<class T>
using managed_device_vector = thrust::device_vector<T, managed_allocator<T>>;

// Structure of flattened vectors
struct SketchStrides
{
	size_t bin_stride;
	size_t kmer_stride;
	size_t sample_stride;
	size_t sketchsize64; 
	size_t bbits;
};

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
				   const SketchStrides& s1_strides, 
				   const SketchStrides& s2_strides) 
{
	size_t samebits = 0;
    for (size_t i = 0; i < s1_strides.sketchsize64; i++) 
    {
		uint64_t bits = ~((uint64_t)0ULL);
		for (size_t j = 0; j < s1_strides.bbits; j++) 
        {
			long long bin_index = i * s1_strides.bbits + j;
			bits &= ~(sketch1[bin_index * s1_strides.bin_stride] ^ sketch2[bin_index * s2_strides.bin_stride]);
		}

		samebits += __popcll(bits); // CUDA 64-bit popcnt
	}
	const size_t maxnbits = s1_strides.sketchsize64 * NBITS(uint64_t); 
	const size_t expected_samebits = (maxnbits >> s1_strides.bbits);
	size_t intersize = samebits;
	if (!expected_samebits) 
	{
		size_t ret = non_neg_minus(samebits, expected_samebits);
		intersize = ret * maxnbits / (maxnbits - expected_samebits);
	}
	size_t unionsize = NBITS(uint64_t) * s1_strides.sketchsize64;
    float jaccard = intersize/(float)unionsize;
    return(jaccard);
}

// Gets Jaccard distance across k-mer lengths and runs a regression
// to get core and accessory
__device__
void regress_kmers(float *& dists,
				   const long long dist_idx,
				   const long long dist_n,
				   const uint64_t * ref,
				   const uint64_t * query,
				   const int * kmers,
				   const int kmer_n,
				   const SketchStrides& ref_strides,						  
				   const SketchStrides& query_strides)						  
{

	float xsum = 0; float ysum = 0; float xysum = 0;
	float xsquaresum = 0; float ysquaresum = 0;
	for (unsigned int kmer_it = 0; kmer_it < kmer_n; ++kmer_it)
    {
		// Get Jaccard distance and move pointers
		float y = __logf(jaccard_dist(ref, query, ref_strides, query_strides)); 
		ref += ref_strides.kmer_stride;
		query += query_strides.kmer_stride;
		
		// Running totals
		xsum += kmers[kmer_it]; 
		ysum += y; 
		xysum += kmers[kmer_it] * y;
		xsquaresum += kmers[kmer_it] * kmers[kmer_it];
		ysquaresum += y * y;
    }

	// Simple linear regression
	// Here I use CUDA fast-math intrinsics on floats, which give comparable accuracy
	// --use-fast-math compile option also possible, but gives less control
	// __fmul_ru(x, y) = x * y and rounds up. 
	// __fpow(x, a) = x^a give 0 for x<0, so not using here (and it is slow)
	// could also replace add / subtract, but becomes less readable
	float xbar = xsum / kmer_n;
	float ybar = ysum / kmer_n;
    float x_diff = xsquaresum - __fmul_ru(xsum, xsum)/kmer_n;
    float y_diff = ysquaresum - __fmul_ru(ysum, ysum)/kmer_n;
	float xstddev = __fsqrt_ru((xsquaresum - __fmul_ru(xsum, xsum)/kmer_n)/kmer_n);
	float ystddev = __fsqrt_ru((ysquaresum - __fmul_ru(ysum, ysum)/kmer_n)/kmer_n);
	float r = __fdiv_ru(xysum - __fmul_ru(xsum, ysum)/kmer_n,  __fsqrt_ru(x_diff*y_diff));
	float beta = __fmul_ru(r, __fdiv_ru(ystddev, xstddev));
    float alpha = ybar - __fmul_ru(beta, xbar);

	// Store core/accessory in dists, truncating at zero
	float core_dist = 0, accessory_dist = 0;
	if (beta < 0)
	{
		core_dist = 1 - __expf(beta);
	}
	if (alpha < 0)
	{
		accessory_dist = 1 - __expf(alpha);
	}
	dists[dist_idx] = core_dist;
	dists[dist_n + dist_idx] = accessory_dist;
}

// Functions to convert index position to/from squareform to condensed form
__device__
long calc_row_idx(const long long k, const long n) {
	// __ll2float_rn() casts long long to float, rounding to nearest
	return n - 2 - floor(__dsqrt_rn(__ll2double_rz(-8*k + 4*n*(n-1)-7))/2 - 0.5);
}

__device__
long calc_col_idx(const long long k, const long i, const long n) {
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
					 const SketchStrides ref_strides,
					 const SketchStrides query_strides)
{
	// TODO implement different iteration for query vs ref
	// TODO allocate __shared here for ref, constant for a block of threads
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
			j = __float2ll_rz(__fdividef(dist_idx, ref_n) + 0.001f);
		}
		regress_kmers(dists, dist_idx, dist_n,
			ref + i * ref_strides.sample_stride,
			query + j * query_strides.sample_stride,
			kmers, kmer_n,
			ref_strides, query_strides);

		// Progress
		if (dist_idx % (dist_n/1000) == 0)
		{
			printf("%cProgress (GPU): %.1lf%%", 13, (float)dist_idx/dist_n * 100);
		}
	}
}

// Turn a vector of references into a flattened vector of
// uint64 with strides bins * kmers * samples
thrust::host_vector<uint64_t> flatten_by_bins(
	const std::vector<Reference>& sketches,
	const std::vector<size_t>& kmer_lengths,
	SketchStrides& strides)
{
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch[kmer_lengths[0]].size());
	strides.bin_stride = 1;
	strides.kmer_stride = strides.bin_stride * num_bins;
	strides.sample_stride = strides.kmer_stride * kmer_lengths.size();
	
	thrust::host_vector<uint64_t> flat_ref(strides.sample_stride * sketches.size());
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

// Turn a vector of queries into a flattened vector of
// uint64 with strides samples * bins * kmers
thrust::host_vector<uint64_t> flatten_by_samples(
	const std::vector<Reference>& sketches,
	const std::vector<size_t>& kmer_lengths,
	SketchStrides& strides)
{
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch[kmer_lengths[0]].size());
	strides.sample_stride = 1;
	strides.bin_stride = sketches.size();
	strides.kmer_stride = strides.bin_stride * num_bins;

	// Stride by bins then restride by samples
	// This is 4x faster than striding by samples in the first place, presumably
	// because many fewer dereferences are being used
	SketchStrides old_strides = strides;
	thrust::host_vector<uint64_t> flat_bins = flatten_by_bins(sketches, kmer_lengths, old_strides);
	thrust::host_vector<uint64_t> flat_ref(strides.kmer_stride * kmer_lengths.size());
	auto flat_ref_it = flat_ref.begin();
	for (size_t kmer_idx = 0; kmer_idx < kmer_lengths.size(); kmer_idx++)
	{
		for (size_t bin_idx = 0; bin_idx < num_bins; bin_idx++)
		{
			for (size_t sample_idx = 0; sample_idx < sketches.size(); sample_idx++)
			{
				*flat_ref_it = flat_bins[sample_idx * old_strides.sample_stride + \
										 bin_idx * old_strides.bin_stride + \
										 kmer_idx * old_strides.kmer_stride];
				flat_ref_it++; 
			}
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
    const int device_id)
{
	std::cerr << "Calculating distances on GPU device " << device_id << std::endl;
	// Initialise device
	cudaSetDevice(device_id);
	cudaDeviceReset();

	// Check sketches are compatible
	bool self = false;
	size_t bbits = ref_sketches[0].bbits();
	size_t sketchsize64 = ref_sketches[0].sketchsize64();
	checkSketchParamsMatch(ref_sketches, kmer_lengths, bbits, sketchsize64);
	
	// Set up memory on device
	SketchStrides ref_strides;
	ref_strides.bbits = bbits;
	ref_strides.sketchsize64 = sketchsize64;
	long long dist_rows; long n_samples = 0;
	if (ref_sketches == query_sketches)
    {
		self = true;
		dist_rows = static_cast<long long>(0.5*(ref_sketches.size())*(ref_sketches.size() - 1));
		n_samples = ref_sketches.size(); 
	}
	else
	{
		checkSketchParamsMatch(query_sketches, kmer_lengths, bbits, sketchsize64);
		dist_rows = ref_sketches.size() * query_sketches.size();
		n_samples = ref_sketches.size() + query_sketches.size(); 
	}
	double est_size  = (bbits * sketchsize64 * kmer_lengths.size() * n_samples * sizeof(uint64_t) + \
						dist_rows * sizeof(float));
	std::cerr << "Estimated device memory required: " << std::fixed << std::setprecision(0) << est_size/(1048576) << "Mb" << std::endl;

	size_t mem_free = 0; size_t mem_total = 0;
	cudaMemGetInfo(&mem_free, &mem_total);
	std::cerr << "Total device memory: " << std::fixed << std::setprecision(0) << mem_total/(1048576) << "Mb" << std::endl;
	std::cerr << "Free device memory: " << std::fixed << std::setprecision(0) << mem_free/(1048576) << "Mb" << std::endl;

	// Data structures for host and device
	std::chrono::steady_clock::time_point a = std::chrono::steady_clock::now();
	SketchStrides query_strides = ref_strides;
	uint64_t *d_ref_array = nullptr, *d_query_array = nullptr;
	managed_device_vector<uint64_t> d_managed_ref_sketches;
	thrust::device_vector<uint64_t> d_ref_sketches, d_query_sketches;

	// Set up reference sketches, flatten and copy to device
	thrust::host_vector<uint64_t> flat_ref = flatten_by_samples(ref_sketches, kmer_lengths, ref_strides);
	if (est_size > mem_free * 0.9)
	{
		// Try managedMalloc is device memory likely to be exceeded
		if (self)
		{
			d_managed_ref_sketches = flat_ref;	
			d_ref_array = thrust::raw_pointer_cast( &d_managed_ref_sketches[0] );
		}
		else
		{
			throw std::runtime_error("Using greater than device memory is unsupport for query mode. "
				 					 "Split your input into smaller chunks");	
		}
	}
	else
	{
		d_ref_sketches = flat_ref;
		d_ref_array = thrust::raw_pointer_cast( &d_ref_sketches[0] );	
	}

	// If needed, flatten query vector and copy to device
	if (!self)
	{
		thrust::host_vector<uint64_t> flat_query = flatten_by_samples(query_sketches, kmer_lengths, query_strides);
		d_query_sketches = flat_query;
		d_query_array = thrust::raw_pointer_cast( &d_query_sketches[0] ); 
	}
	else
	{
		query_strides = ref_strides;
	}

	// Copy other arrays needed on device (kmers and distance output)
	thrust::device_vector<int> d_kmers = kmer_lengths;
	int* d_kmers_array = thrust::raw_pointer_cast( &d_kmers[0] );
	thrust::device_vector<float> dist_mat(dist_rows*2, 0);
	float* d_dist_array = thrust::raw_pointer_cast( &dist_mat[0] );

	// Cache preferences:
	// Upper dist memory access is hard to predict, so try and cache as much
	// as possible
	// Query uses cache to store sketch
	if (self)
	{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	}
	else
	{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	}

	// Run dists on device
	cudaDeviceSynchronize();
	std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
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
		ref_strides,
	    query_strides);
				
	// copy results from device to return
	cudaDeviceSynchronize();
	std::chrono::steady_clock::time_point c = std::chrono::steady_clock::now();
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
	std::chrono::steady_clock::time_point d = std::chrono::steady_clock::now();
	printf("%cProgress (GPU): 100.0%%", 13);
	std::cout << std::endl << "" << std::endl;

	// Report timings of each step
	std::chrono::duration<double> load_time = std::chrono::duration_cast<std::chrono::duration<double> >(b-a);
	std::chrono::duration<double> calc_time = std::chrono::duration_cast<std::chrono::duration<double> >(c-b);
	std::chrono::duration<double> save_time = std::chrono::duration_cast<std::chrono::duration<double> >(d-c);

	std::cout << "Loading: " << load_time.count()<< "s" << std::endl;
	std::cout << "Distances: " << calc_time.count()<< "s" << std::endl;
	std::cout << "Saving: " << save_time.count()<< "s" << std::endl;

	return dist_results;
}