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
#include <tuple>
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

const int WARP_SIZE = 32;
const int selfBlockSize = 32;
const float mem_epsilon = 0.05;

struct DeviceMemory {
	thrust::device_vector<uint64_t> ref_sketches;	
	thrust::device_vector<uint64_t> query_sketches;	
	thrust::device_vector<float> ref_random;	
	thrust::device_vector<float> query_random;	
	thrust::device_vector<int> kmers;	
	thrust::device_vector<float> dist_mat;	
};

// Structure of flattened vectors
struct SketchStrides {
	size_t bin_stride;
	size_t kmer_stride;
	size_t sample_stride;
	size_t sketchsize64; 
	size_t bbits;
};

/******************
*			      *
*	Device code   *
*			      *	
*******************/

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

// Ternary used in observed_excess
template <class T>
__device__
T non_neg_minus(T a, T b) {
	return a > b ? (a - b) : 0;
}

// Calculates excess observations above a random value
template <class T>
__device__
T observed_excess(T obs, T exp, T max) {
	T diff = non_neg_minus(obs, exp);
	return(diff * max / (max - exp));
}

// CUDA version of bindash dist function (see dist.cpp)
__device__
float jaccard_dist(const uint64_t * sketch1, 
                   const uint64_t * sketch2, 
				   const SketchStrides& s1_strides, 
				   const SketchStrides& s2_strides) 
{
	size_t samebits = 0;
    for (int i = 0; i < s1_strides.sketchsize64; i++) 
    {
		uint64_t bits = ~((uint64_t)0ULL);
		for (int j = 0; j < s1_strides.bbits; j++) 
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
		size_t ret = observed_excess(samebits, expected_samebits, maxnbits);
	}
	size_t unionsize = NBITS(uint64_t) * s1_strides.sketchsize64;
    float jaccard = __fdiv_ru(intersize, unionsize);
    return(jaccard);
}

// Simple linear regression, exact solution
// Avoids use of dynamic memory allocation on device, or
// linear algebra libraries
__device__
void simple_linear_regression(float * const &core_dist,
				              float * const &accessory_dist,
							  const float xsum,
							  const float ysum,
							  const float xysum,
							  const float xsquaresum,
							  const float ysquaresum,
							  const int n)
{
	// Here I use CUDA fast-math intrinsics on floats, which give comparable accuracy
	// --use-fast-math compile option also possible, but gives less control
	// __fmul_ru(x, y) = x * y and rounds up. 
	// __fpow(x, a) = x^a give 0 for x<0, so not using here (and it is slow)
	// could also replace add / subtract, but becomes less readable
	float xbar = xsum / n;
	float ybar = ysum / n;
    float x_diff = xsquaresum - __fmul_ru(xsum, xsum)/n;
    float y_diff = ysquaresum - __fmul_ru(ysum, ysum)/n;
	float xstddev = __fsqrt_ru((xsquaresum - __fmul_ru(xsum, xsum)/n)/n);
	float ystddev = __fsqrt_ru((ysquaresum - __fmul_ru(ysum, ysum)/n)/n);
	float r = __fdiv_ru(xysum - __fmul_ru(xsum, ysum)/n,  __fsqrt_ru(x_diff*y_diff));
	float beta = __fmul_ru(r, __fdiv_ru(ystddev, xstddev));
    float alpha = __fmaf_ru(-beta, xbar, ybar); // maf: x * y + z

	// Store core/accessory in dists, truncating at zero
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

/******************
*			      *
*	Global code   *
*			      *	
*******************/

// Main kernel functions run on the device, 
// but callable from the host

// To calculate distance of query sketches from a panel
// of references
__global__
void calculate_query_dists(const uint64_t * ref,
					 const long ref_n,
					 const uint64_t * query,
					 const long query_n,
					 const int * kmers,
					 const int kmer_n,
					 float * dists,
					 const long long dist_n,
					 const float * random_match_ref,
					 const float * random_match_query,
					 const SketchStrides ref_strides,
					 const SketchStrides query_strides) {
	// Calculate indices for query, ref and results
	long blocksPerQuery = (ref_n + blockDim.x - 1) / blockDim.x;
	long query_idx = __float2int_rz(__fdividef(blockIdx.x, blocksPerQuery) + 0.001f);
	long ref_idx = (blockIdx.x % blocksPerQuery) * blockDim.x + threadIdx.x;
	long dist_idx = query_idx * ref_n + ref_idx;
	const uint64_t* ref_start = ref + ref_idx * ref_strides.sample_stride;
	const uint64_t* query_start = query + query_idx * query_strides.sample_stride;
	
	// Calculate Jaccard distances over k-mer lengths
	float xsum = 0; float ysum = 0; float xysum = 0;
	float xsquaresum = 0; float ysquaresum = 0;
	for (int kmer_idx = 0; kmer_idx < kmer_n; kmer_idx++)
	{
		// Copy query sketch into __shared__ mem (on chip) for faster access within block
		// Hopefully this doesn't suffer from bank conflicts as the sketch2 access in
		// jaccard_distance() should result in a broadcast
		// Uses all threads *in a single warp* to do the copy
		// NB there is no disadvantage vs using multiple warps, as they would have to wait
		// (see https://stackoverflow.com/questions/15468059/copy-to-the-shared-memory-in-cuda)
		extern __shared__ uint64_t query_shared[];
		if (threadIdx.x < WARP_SIZE) {
			for (long lidx = threadIdx.x; lidx < query_strides.bbits * query_strides.sketchsize64; lidx += WARP_SIZE) {
				query_shared[lidx] = query_start[lidx * query_strides.bin_stride];
			}
		}
		__syncthreads();
	
		// Some threads at the end of the last block will have nothing to do
		// Need to have conditional here to avoid block on __syncthreads() above
		if (ref_idx < ref_n)
		{
			// Calculate Jaccard distance at current k-mer length
			float jaccard_obs = jaccard_dist(ref_start, query_start, ref_strides, query_strides);

			// Adjust for random matches
			float r1 = random_match_ref[kmer_idx * ref_n + ref_idx];
			float r2 = random_match_query[kmer_idx * query_n + query_idx];
			float jaccard_expected = (r1 * r2) / (r1 + r2 - r1 * r2);
			float y = __logf(observed_excess(jaccard_obs, jaccard_expected, 1.0f));

			// Running totals for regression
			xsum += kmers[kmer_idx]; 
			ysum += y; 
			xysum += kmers[kmer_idx] * y;
			xsquaresum += kmers[kmer_idx] * kmers[kmer_idx];
			ysquaresum += y * y;
		}

		// Move to next k-mer length
		ref_start += ref_strides.kmer_stride;
		query_start += query_strides.kmer_stride;
	}

	if (ref_idx < ref_n)
	{
		// Run the regression, and store results in dists
		simple_linear_regression(dists + dist_idx,
								 dists + dist_n + dist_idx,
								 xsum,
								 ysum,
								 xysum,
								 xsquaresum,
								 ysquaresum,
								 kmer_n);

		// Progress indicator
		// The >> 10 is a divide by 1024 - update roughly every 0.1%
		if (dist_idx % (dist_n >> 10) == 0) 
		{
			printf("%cProgress (GPU): %.1lf%%", 13, (float)dist_idx/dist_n * 100);
		}
	}

}

// Takes a position in the condensed form distance matrix, converts into an
// i, j for the ref/query vectors. Calls regression with these start points
__global__
void calculate_self_dists(const uint64_t * ref,
					      const long ref_n,
					      const int * kmers,
					      const int kmer_n,
					      float * dists,
						  const long long dist_n,
						  const float * random_match,
					      const SketchStrides ref_strides)
{
	// Grid-stride loop
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (long long dist_idx = index; dist_idx < dist_n; dist_idx += stride)
	{
		long i, j;
		i = calc_row_idx(dist_idx, ref_n);
		j = calc_col_idx(dist_idx, i, ref_n);
		if (j <= i)
		{
			continue;
		}
		
		// Set pointers to start of sketch i, j
		const uint64_t* ref_start = ref + i * ref_strides.sample_stride;
		const uint64_t* query_start = ref + j * ref_strides.sample_stride;

		float xsum = 0; float ysum = 0; float xysum = 0;
		float xsquaresum = 0; float ysquaresum = 0;
		for (int kmer_idx = 0; kmer_idx < kmer_n; ++kmer_idx)
		{
			// Get Jaccard distance and move pointers to next k-mer
			float jaccard_obs = jaccard_dist(ref_start, query_start, ref_strides, ref_strides); 
			ref_start += ref_strides.kmer_stride;
			query_start += ref_strides.kmer_stride;

			// Adjust for random matches
			float r1 = random_match[kmer_idx * ref_n + i];
			float r2 = random_match[kmer_idx * ref_n + j];
			float jaccard_expected = (r1 * r2) / (r1 + r2 - r1 * r2);
			float y = __logf(observed_excess(jaccard_obs, jaccard_expected, 1.0f));
			
			// Running totals for regression
			xsum += kmers[kmer_idx]; 
			ysum += y; 
			xysum += kmers[kmer_idx] * y;
			xsquaresum += kmers[kmer_idx] * kmers[kmer_idx];
			ysquaresum += y * y;
		}
		
		// Run the regression, and store results in dists
		simple_linear_regression(dists + dist_idx,
								 dists + dist_n + dist_idx,
								 xsum,
								 ysum,
								 xysum,
								 xsquaresum,
								 ysquaresum,
								 kmer_n);

		// Progress indicator
		// The >> 10 is a divide by 1024 - update roughly every 0.1%
		if (dist_idx % (dist_n >> 10) == 0) 
		{
			printf("%cProgress (GPU): %.1lf%%", 13, (float)dist_idx/dist_n * 100);
		}
	}
}

/***************
*			   *
*	Host code  *
*			   *	
***************/

// Turn a vector of references into a flattened vector of
// uint64 with strides bins * kmers * samples
thrust::host_vector<uint64_t> flatten_by_bins(
	const std::vector<Reference>& sketches,
	const std::vector<size_t>& kmer_lengths,
	SketchStrides& strides)
{
	// Set strides structure
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch[kmer_lengths[0]].size());
	strides.bin_stride = 1;
	strides.kmer_stride = strides.bin_stride * num_bins;
	strides.sample_stride = strides.kmer_stride * kmer_lengths.size();
	
	// Iterate over each dimension to flatten
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
	// Set strides
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch[kmer_lengths[0]].size());
	strides.sample_stride = 1;
	strides.bin_stride = sketches.size();
	strides.kmer_stride = strides.bin_stride * num_bins;

	// Stride by bins then restride by samples
	// This is 4x faster than striding by samples by looping over References vector, 
	// presumably because many fewer dereferences are being used
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

// Calculates the random match probability for all sketches at all k-mer lengths
thrust::host_vector<float> preloadRandom(std::vector<Reference>& sketches, 
								 		 const std::vector<size_t>& kmer_lengths) {
	thrust::host_vector<float> random_sample_strided(sketches.size() * kmer_lengths.size());
	for (unsigned int sketch_idx = 0; sketch_idx < sketches.size(); sketch_idx++) {
		for (unsigned int kmer_idx = 0; kmer_idx < kmer_lengths.size(); kmer_idx++) {
			random_sample_strided[kmer_idx * sketches.size() + sketch_idx] = 
				(float)sketches[sketch_idx].random_match(kmer_lengths[kmer_idx]);
		}
	}
	return random_sample_strided;
}

DeviceMemory loadDeviceMemory(SketchStrides& ref_strides,
					  SketchStrides& query_strides,
					  std::vector<Reference>& ref_sketches,
					  std::vector<Reference>& query_sketches,
					  const SketchSlice& sample_slice,
					  const std::vector<size_t>& kmer_lengths,
					  long long dist_rows,
					  const bool self) {
	DeviceMemory loaded;

	// Need to (or easiest to) make temporary copies until we get
	// std::span in C++20

	// I think this use of pointers is not leaking memory - but 
	// should check whether new and unique_ptr is better
	std::unique_ptr<std::vector<Reference>> ref_subsample;
	if (sample_slice.ref_size < ref_sketches.size()) {
		ref_subsample.reset(new \
			std::vector<Reference>(ref_sketches.begin() + sample_slice.ref_offset,
								   ref_sketches.begin() + sample_slice.ref_offset + sample_slice.ref_size));
	} else {
		ref_subsample.reset(&ref_sketches);
	}

	std::unique_ptr<std::vector<Reference>> query_subsample;
	if (!self && sample_slice.query_size < query_sketches.size()) {
			query_subsample.reset(new \
				std::vector<Reference>(query_sketches.begin() + sample_slice.query_offset,
									   query_sketches.begin() + sample_slice.query_offset + sample_slice.query_size));
	} else {
		query_subsample.reset(&query_sketches);
	}

	// Set up reference sketches, flatten and copy to device
	thrust::host_vector<uint64_t> flat_ref = flatten_by_samples(*ref_subsample, kmer_lengths, ref_strides);
	loaded.ref_sketches = flat_ref;

	// If ref v query mode, also flatten query vector and copy to device
	if (!self)
	{
		thrust::host_vector<uint64_t> flat_query = flatten_by_bins(*query_subsample, kmer_lengths, query_strides);
		loaded.query_sketches = flat_query;
	}

	// Preload random match chances
	loaded.ref_random = preloadRandom(*ref_subsample, kmer_lengths);
	if (!self) {
		thrust::host_vector<float> query_random = preloadRandom(*query_subsample, kmer_lengths);
		loaded.query_random = query_random;
	}

	// Copy other arrays needed on device (kmers and distance output)
	loaded.kmers = kmer_lengths;
	loaded.dist_mat.resize(dist_rows*2, 0);

	return(loaded);
}

// Checks bbits, sketchsize and k-mer lengths are identical in
// all sketches
// throws runtime_error if mismatches (should be ensured in passing
// code)
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

// Get the blockSize and blockCount for CUDA call
std::tuple<size_t, size_t> getBlockSize(const size_t ref_samples,
										const size_t query_samples,
									    const size_t dist_rows) {
	size_t blockSize, blockCount;
	if (query_samples > 0) {
		// Each block processes a single query. As max size is 512 threads
		// per block, may need multiple blocks (non-exact multiples lead
		// to some wasted computation in threads)
		// We take the next multiple of 32 that is larger than the number of
		// reference sketches, up to a maximum of 512
		blockSize = std::min(512, (int)(32 * (ref_samples + 32 - 1) / 32));
		size_t blocksPerQuery = (ref_samples + blockSize - 1) / blockSize;
		blockCount = blocksPerQuery * query_samples;
	} else {
		// Empirically a blockSize (selfBlockSize global const) of 32 or 256 seemed best
		blockSize = selfBlockSize;
		blockCount = (dist_rows + blockSize - 1) / blockSize;
	}
	return(std::make_tuple(blockSize, blockCount));
} 

// Run the distance calculations, reading/writing into device_arrays
// Cache preferences:
// Upper dist memory access is hard to predict, so try and cache as much
// as possible
// Query uses on-chip cache (__shared__) to store query sketch
// std::chrono::steady_clock::time_point b;
void dispatchDists(DeviceMemory& device_arrays,
				   std::vector<Reference>& ref_sketches,
				   std::vector<Reference>& query_sketches,
				   SketchStrides& ref_strides,
				   SketchStrides& query_strides,
				   const SketchSlice& sketch_subsample,
				   const std::vector<size_t>& kmer_lengths,
				   const bool self) {
	if (self) {
		// square 'self' block
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		
		long long chunk_dist_rows = static_cast<long long>(
										0.5*(sketch_subsample.ref_size)*(sketch_subsample.ref_size - 1));
		device_arrays = loadDeviceMemory(
			ref_strides,
			query_strides,
			ref_sketches,
			query_sketches,
			sketch_subsample,
			kmer_lengths,
			chunk_dist_rows,
			true);

		// cudaDeviceSynchronize();	
		// b = std::chrono::steady_clock::now()

		size_t blockSize, blockCount;
		std::tie(blockSize, blockCount) = getBlockSize(sketch_subsample.ref_size, 
													   sketch_subsample.ref_size,
													   chunk_dist_rows);
		calculate_self_dists<<<blockCount, selfBlockSize>>>
			(
				thrust::raw_pointer_cast(&device_arrays.ref_sketches[0]),
				sketch_subsample.ref_size,
				thrust::raw_pointer_cast(&device_arrays.kmers[0]),
				kmer_lengths.size(),
				thrust::raw_pointer_cast(&device_arrays.dist_mat[0]),
				chunk_dist_rows,
				thrust::raw_pointer_cast(&device_arrays.ref_random[0]),
				ref_strides
			);
	} else {
		// 'query' block
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); 
		cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

		long long chunk_dist_rows = sketch_subsample.ref_size * sketch_subsample.query_size;
		device_arrays = loadDeviceMemory(
			ref_strides,
			query_strides,
			ref_sketches,
			query_sketches,
			sketch_subsample,
			kmer_lengths,
			chunk_dist_rows,
			false);
			
		size_t blockSize, blockCount;
		std::tie(blockSize, blockCount) = getBlockSize(sketch_subsample.ref_size, 
													   sketch_subsample.query_size,
													   chunk_dist_rows);
		
		// cudaDeviceSynchronize();	
		// b = std::chrono::steady_clock::now()

		// Third argument is the size of __shared__ memory needed by a thread block
		// This is equal to the query sketch size in bytes (at a single k-mer length)
		calculate_query_dists<<<blockCount, blockSize, 
								query_strides.sketchsize64*query_strides.bbits*sizeof(uint64_t)>>>
		(
			thrust::raw_pointer_cast(&device_arrays.ref_sketches[0]),
			ref_sketches.size(),
			thrust::raw_pointer_cast(&device_arrays.query_sketches[0]),
			query_sketches.size(),
			thrust::raw_pointer_cast(&device_arrays.kmers[0]),
			kmer_lengths.size(),
			thrust::raw_pointer_cast(&device_arrays.dist_mat[0]),
			chunk_dist_rows,
			thrust::raw_pointer_cast(&device_arrays.ref_random[0]),
			thrust::raw_pointer_cast(&device_arrays.query_random[0]),
			ref_strides,
			query_strides
		);
	}
	printf("%cProgress (GPU): 100.0%%", 13);
	std::cout << std::endl << "" << std::endl;
}

// Main function callable via API
// Checks inputs
// Flattens sketches
// Copies flattened sketches to device
// Runs kernel function across distance elements
// Copies and returns results
NumpyMatrix query_db_cuda(std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int device_id,
	const unsigned int num_cpu_threads)
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
	
	// Set up sketch information and sizes
	SketchStrides ref_strides;
	ref_strides.bbits = bbits;
	ref_strides.sketchsize64 = sketchsize64;
	SketchStrides query_strides = ref_strides;

	long long dist_rows; long n_samples = 0;
	if (ref_sketches == query_sketches)
    {
		self = true;
		dist_rows = static_cast<long long>(0.5*(ref_sketches.size())*(ref_sketches.size() - 1));
		n_samples = ref_sketches.size(); 
	}
	else
	{
		// Also check query sketches are compatible
		checkSketchParamsMatch(query_sketches, kmer_lengths, bbits, sketchsize64);
		dist_rows = ref_sketches.size() * query_sketches.size();
		n_samples = ref_sketches.size() + query_sketches.size(); 
	}
	double est_size  = (bbits * sketchsize64 * kmer_lengths.size() * n_samples * sizeof(uint64_t) + \ // Size of sketches
						kmer_lengths.size() * n_samples * sizeof(float) + \                           // Size of random matches
						dist_rows * 2 * sizeof(float));												  // Size of distance matrix
	std::cerr << "Estimated device memory required: " << std::fixed << std::setprecision(0) << est_size/(1048576) << "Mb" << std::endl;

	size_t mem_free = 0; size_t mem_total = 0;
	cudaMemGetInfo(&mem_free, &mem_total);
	std::cerr << "Total device memory: " << std::fixed << std::setprecision(0) << mem_total/(1048576) << "Mb" << std::endl;
	std::cerr << "Free device memory: " << std::fixed << std::setprecision(0) << mem_free/(1048576) << "Mb" << std::endl;

	if (est_size > mem_free * (1 - mem_epsilon) && !self) {
		throw std::runtime_error("Using greater than device memory is unsupported for query mode. "
							     "Split your input into smaller chunks");	
	}

	// Ready to run dists on device
	DeviceMemory device_arrays;
	SketchSlice sketch_subsample;
	unsigned int chunks = 1;
	std::vector<float> dist_results(dist_rows * 2);
	NumpyMatrix coreSquare, accessorySquare; 
	if (self)
	{
		// To prevent memory being exceeded, total distance matrix is split up into
		// chunks which do fit in memory. These are iterated over in the same order
		// as a square distance matrix. The i = j chunks are 'self', i < j can be skipped
		// as they contain only lower triangle values, i > j work as query vs ref
		chunks = floor(est_size / (mem_free * (1 - mem_epsilon))) + 1;
		size_t calc_per_chunk = n_samples / chunks;
		unsigned int num_big_chunks = n_samples % chunks;

		// Only allocate these square matrices if they are needed
		if (chunks > 1) {
			coreSquare.resize(n_samples, n_samples);
			accessorySquare.resize(n_samples, n_samples);
		}
		unsigned int total_chunks = (chunks * (chunks + 1)) >> 1;
		unsigned int chunk_count = 0;

		sketch_subsample.ref_offset = 0; 
		for (unsigned int chunk_i = 0; chunk_i < chunks; chunk_i++) {
			sketch_subsample.ref_size = calc_per_chunk;
			if (chunk_i < num_big_chunks) {
				sketch_subsample.ref_size++;
			}
			
			sketch_subsample.query_offset = sketch_subsample.ref_size; 
			for (unsigned int chunk_j = chunk_i; chunk_j < chunks; chunk_j++) {
				printf("Running chunk %ud of %ud\n", ++chunk_count, total_chunks);
				sketch_subsample.query_size = calc_per_chunk;
				if (chunk_j < num_big_chunks) {
					sketch_subsample.query_size++;
				}
				
				if (chunk_i == chunk_j) {
					// 'self' blocks
					dispatchDists(device_arrays,
						ref_sketches,
						ref_sketches,
						ref_strides,
						query_strides,
						sketch_subsample,
						kmer_lengths,
						true);
				} else {
					// 'query' block
					dispatchDists(device_arrays,
						ref_sketches,
						query_sketches,
						ref_strides,
						query_strides,
						sketch_subsample,
						kmer_lengths,
						false);
				}
				sketch_subsample.query_offset += sketch_subsample.query_size; 

				// Read intermediate dists out
				if (chunks > 1) {
					try {
						// Copy results from device into Nx2 matrix
						std::vector<float> block_results;
						thrust::copy(device_arrays.dist_mat.begin(), device_arrays.dist_mat.end(), block_results.begin());
						NumpyMatrix blockMat = \
							Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::RowMajor> >(block_results.data(),block_results.size()/2,2);
						
						// Convert each long form column of Nx2 matrix into square distance matrix
						// Add this square matrix into the correct submatrix (block) of the final square matrix
						longToSquareBlock(coreSquare,
										  accessorySquare,
										  sketch_subsample,
										  block_results,
										  num_cpu_threads);

					} catch (thrust::system_error &e) {
						std::cerr << "Error getting result: " << std::endl;
						std::cerr << e.what() << std::endl;
						exit(1);
					}
					
				}

			}
			sketch_subsample.ref_offset += sketch_subsample.ref_size; 
		}

	}
	else
	{
		sketch_subsample.ref_size = ref_sketches.size();
		sketch_subsample.query_size = query_sketches.size();
		dispatchDists(device_arrays,
			ref_sketches,
			query_sketches,
			ref_strides,
			query_strides,
			sketch_subsample,
			kmer_lengths,
			false);	
	}
	// cudaDeviceSynchronize();
	// std::chrono::steady_clock::time_point c = std::chrono::steady_clock::now();
	
	// copy results from device back to host
	// try and keep Eigen code in .cpp files (http://eigen.tuxfamily.org/dox-devel///TopicCUDA.html)
	NumpyMatrix dists_ret_matrix;
	if (self && chunks > 1) {
		// Chunked computation yields square matrix, which needs to be converted back to long
		// form
		dists_ret_matrix = twoColumnSquareToLong(coreSquare,
												 accessorySquare,
												 num_cpu_threads);
	} else {
		try {
			// Single chunks just need to be moved from the device into the return vector
			// CUDA code now returns column major data (i.e. all core dists, then all accessory dists)
			// to try and coalesce writes.
			// NB: almost all other code is row major (i.e. sample core then accessory, then next sample)
			thrust::copy(device_arrays.dist_mat.begin(), device_arrays.dist_mat.end(), dist_results.begin());
			dists_ret_matrix = \
				Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::RowMajor> >(dist_results.data(),dist_results.size()/2,2);
		} catch (thrust::system_error &e) {
			// output a non-threatening but likely inaccurate error message and exit
			// e.g. 'trivial_device_copy D->H failed: unspecified launch failure'
			// error will have occurred elsewhere as launch is async, but better to catch 
			// and deal with it here
			std::cerr << "Error getting result: " << std::endl;
			std::cerr << e.what() << std::endl;
			exit(1);
		}
	}

	/* Code used to time in development:
	// Report timings of each step
	std::chrono::steady_clock::time_point d = std::chrono::steady_clock::now();
	std::chrono::duration<double> load_time = std::chrono::duration_cast<std::chrono::duration<double> >(b-a);
	std::chrono::duration<double> calc_time = std::chrono::duration_cast<std::chrono::duration<double> >(c-b);
	std::chrono::duration<double> save_time = std::chrono::duration_cast<std::chrono::duration<double> >(d-c);

	std::cout << "Loading: " << load_time.count()<< "s" << std::endl;
	std::cout << "Distances: " << calc_time.count()<< "s" << std::endl;
	std::cout << "Saving: " << save_time.count()<< "s" << std::endl;
	*/

	return dists_ret_matrix;
}