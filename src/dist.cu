/*
 *
 * dist.cu
 * PopPUNK dists using CUDA
 * nvcc compiled part (try to avoid eigen)
 *
 */

// std
#include <cstdint>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iomanip>

// cuda
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// internal headers
#include "bitfuncs.hpp"
#include "gpu.hpp"

const int selfBlockSize = 32;
const int progressBitshift = 10; // Update every 2^10 = 1024 dists

// Memory on device for each operation
struct DeviceMemory {
	thrust::device_vector<uint64_t> ref_sketches;
	thrust::device_vector<uint64_t> query_sketches;
	thrust::device_vector<float> random_table;
	thrust::device_vector<uint16_t> ref_random;
	thrust::device_vector<uint16_t> query_random;
	thrust::device_vector<int> kmers;
	thrust::device_vector<float> dist_mat;
};

/******************
*			      *
*	Device code   *
*			      *
*******************/

// Error checking of dynamic memory allocation on device
// https://stackoverflow.com/a/14038590
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__host__ __device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
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
    assert(j > i);
	return (n*i - ((i*(i+1)) >> 1) + j - 1 - i);
}

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
					 const float * random_table,
					 const uint16_t * ref_idx_lookup,
					 const uint16_t * query_idx_lookup,
					 const SketchStrides ref_strides,
					 const SketchStrides query_strides,
					 const RandomStrides random_strides,
					 volatile int * blocks_complete) {
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
		if (threadIdx.x < warpSize) {
			for (long lidx = threadIdx.x; lidx < query_strides.bbits * query_strides.sketchsize64; lidx += warpSize) {
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
			float jaccard_expected = random_table[kmer_idx * random_strides.kmer_stride +
												  ref_idx_lookup[ref_idx] * random_strides.cluster_inner_stride +
												  query_idx_lookup[query_idx] * random_strides.cluster_outer_stride];
			float y = __logf(observed_excess(jaccard_obs, jaccard_expected, 1.0f));
			//printf("i:%ld j:%ld k:%d r1:%f r2:%f jac:%f y:%f\n", ref_idx, query_idx, kmer_idx, r1, r2, jaccard_obs, y);

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

		update_progress(dist_idx, dist_n, blocks_complete);
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
						  const float * random_table,
						  const uint16_t * ref_idx_lookup,
						  const SketchStrides ref_strides,
						  const RandomStrides random_strides,
						  volatile int * blocks_complete)
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
			float jaccard_expected = random_table[kmer_idx * random_strides.kmer_stride +
												  ref_idx_lookup[i] * random_strides.cluster_inner_stride +
												  ref_idx_lookup[j] * random_strides.cluster_outer_stride];
			float y = __logf(observed_excess(jaccard_obs, jaccard_expected, 1.0f));
			//printf("i:%ld j:%ld k:%d r_idx:%ld r:%f jac_obs:%f jac_adj:%f y:%f\n",
			//  i, j, kmer_idx,
			//	kmer_idx * random_strides.kmer_stride +
			//	ref_idx_lookup[i] * random_strides.cluster_inner_stride +
			//	ref_idx_lookup[j] * random_strides.cluster_outer_stride,
			//  jaccard_expected, jaccard_obs, jaccard_expected, y);

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

		update_progress(dist_idx, dist_n, blocks_complete);
	}
}

/***************
*			   *
*	Host code  *
*			   *
***************/

// Initialise device and return info on its memory
std::tuple<size_t, size_t> initialise_device(const int device_id) {
	cudaSetDevice(device_id);
	cudaDeviceReset();

	size_t mem_free = 0; size_t mem_total = 0;
	cudaMemGetInfo(&mem_free, &mem_total);
	return(std::make_tuple(mem_free, mem_total));
}

// Turn a vector of references into a flattened vector of
// uint64 with strides bins * kmers * samples
thrust::host_vector<uint64_t> flatten_by_bins(
	const std::vector<Reference>& sketches,
	const std::vector<size_t>& kmer_lengths,
	SketchStrides& strides,
	const size_t start_sample_idx,
	const size_t end_sample_idx)
{
	// Set strides structure
	size_t num_sketches = end_sample_idx - start_sample_idx;
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch(kmer_lengths[0]).size());
	assert(end_sample_idx > start_sample_idx);
	strides.bin_stride = 1;
	strides.kmer_stride = strides.bin_stride * num_bins;
	strides.sample_stride = strides.kmer_stride * kmer_lengths.size();

	// Iterate over each dimension to flatten
	thrust::host_vector<uint64_t> flat_ref(strides.sample_stride * num_sketches);
	auto flat_ref_it = flat_ref.begin();
	for (auto sample_it = sketches.cbegin() + start_sample_idx;
		 sample_it != sketches.cend() - (sketches.size() - end_sample_idx);
		 sample_it++)
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
	SketchStrides& strides,
	const size_t start_sample_idx,
	const size_t end_sample_idx)
{
	// Set strides
	size_t num_sketches = end_sample_idx - start_sample_idx;
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch(kmer_lengths[0]).size());
	strides.sample_stride = 1;
	strides.bin_stride = num_sketches;
	strides.kmer_stride = strides.bin_stride * num_bins;

	// Stride by bins then restride by samples
	// This is 4x faster than striding by samples by looping over References vector,
	// presumably because many fewer dereferences are being used
	SketchStrides old_strides = strides;
	thrust::host_vector<uint64_t> flat_bins = flatten_by_bins(sketches,
															  kmer_lengths,
															  old_strides,
															  start_sample_idx,
															  end_sample_idx);
	thrust::host_vector<uint64_t> flat_ref(strides.kmer_stride * kmer_lengths.size());
	auto flat_ref_it = flat_ref.begin();
	for (size_t kmer_idx = 0; kmer_idx < kmer_lengths.size(); kmer_idx++) {
		for (size_t bin_idx = 0; bin_idx < num_bins; bin_idx++) {
			for (size_t sample_idx = 0; sample_idx < num_sketches; sample_idx++) {
				*flat_ref_it = flat_bins[sample_idx * old_strides.sample_stride + \
										 bin_idx * old_strides.bin_stride + \
										 kmer_idx * old_strides.kmer_stride];
				flat_ref_it++;
			}
		}
	}

	return flat_ref;
}

// Sets up data structures and loads them onto the device
DeviceMemory loadDeviceMemory(SketchStrides& ref_strides,
					  SketchStrides& query_strides,
					  std::vector<Reference>& ref_sketches,
					  std::vector<Reference>& query_sketches,
					  const SketchSlice& sample_slice,
					  const FlatRandom& flat_random,
					  const std::vector<uint16_t>& ref_random_idx,
					  const std::vector<uint16_t>& query_random_idx,
					  const std::vector<size_t>& kmer_lengths,
					  long long dist_rows,
					  const bool self) {
	DeviceMemory loaded;
	try {
		// Set up reference sketches, flatten and copy to device
		thrust::host_vector<uint64_t> flat_ref =
			flatten_by_samples(ref_sketches,
								kmer_lengths,
								ref_strides,
								sample_slice.ref_offset,
								sample_slice.ref_offset + sample_slice.ref_size);
		loaded.ref_sketches = flat_ref; // copies to device

		// Preload random match chances, which have already been flattened
		loaded.random_table = std::get<1>(flat_random);
		loaded.ref_random.resize(sample_slice.ref_size);
		thrust::copy(ref_random_idx.begin() + sample_slice.ref_offset,
					ref_random_idx.begin() + sample_slice.ref_offset + sample_slice.ref_size,
					loaded.ref_random.begin());

		// If ref v query mode, also flatten query vector and copy to device
		if (!self) {
			thrust::host_vector<uint64_t> flat_query =
				flatten_by_bins(query_sketches,
								kmer_lengths,
								query_strides,
								sample_slice.query_offset,
								sample_slice.query_offset + sample_slice.query_size);
			loaded.query_sketches = flat_query;

			loaded.query_random.resize(sample_slice.query_size);
			thrust::copy(query_random_idx.begin() + sample_slice.query_offset,
					     query_random_idx.begin() + sample_slice.query_offset + sample_slice.query_size,
						 loaded.query_random.begin());
		}

		// Copy other arrays needed on device (kmers and distance output)
		loaded.kmers = kmer_lengths;
		loaded.dist_mat.resize(dist_rows*2, 0);
	} catch (thrust::system_error &e) {
		std::cerr << "Error loading sketches onto GPU: " << std::endl;
		std::cerr << e.what() << std::endl;
		exit(1);
	}

	return(loaded);
}

// Get the blockSize and blockCount for CUDA call
std::tuple<size_t, size_t> getBlockSize(const size_t ref_samples,
										const size_t query_samples,
										const size_t dist_rows,
									    const bool self) {
	size_t blockSize, blockCount;
	if (self) {
		// Empirically a blockSize (selfBlockSize global const) of 32 or 256 seemed best
		blockSize = selfBlockSize;
		blockCount = (dist_rows + blockSize - 1) / blockSize;
	} else {
		// Each block processes a single query. As max size is 512 threads
		// per block, may need multiple blocks (non-exact multiples lead
		// to some wasted computation in threads)
		// We take the next multiple of 32 that is larger than the number of
		// reference sketches, up to a maximum of 512
		blockSize = std::min(512, (int)(32 * (ref_samples + 32 - 1) / 32));
		size_t blocksPerQuery = (ref_samples + blockSize - 1) / blockSize;
		blockCount = blocksPerQuery * query_samples;
	}
	return(std::make_tuple(blockSize, blockCount));
}

// Writes a progress meter using the device int which keeps
// track of completed jobs
void reportProgress(volatile int * blocks_complete,
					long long dist_rows) {
	long long progress_blocks = 1 << progressBitshift;
	int now_completed = 0; float kern_progress = 0;
	if (dist_rows > progress_blocks) {
		while (now_completed < progress_blocks - 1) {
			if (*blocks_complete > now_completed) {
				now_completed = *blocks_complete;
				kern_progress = now_completed / (float)progress_blocks;
				fprintf(stderr, "%cProgress (GPU): %.1lf%%", 13, kern_progress * 100);
			} else {
				usleep(1000);
			}
		}
	}
}

// Main function to run the distance calculations, reading/writing into device_arrays
// Cache preferences:
// Upper dist memory access is hard to predict, so try and cache as much
// as possible
// Query uses on-chip cache (__shared__) to store query sketch
std::vector<float> dispatchDists(
	std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	SketchStrides& ref_strides,
	SketchStrides& query_strides,
	const FlatRandom& flat_random,
	const std::vector<uint16_t>& ref_random_idx,
	const std::vector<uint16_t>& query_random_idx,
	const SketchSlice& sketch_subsample,
	const std::vector<size_t>& kmer_lengths,
	const bool self) {

	// Progress meter
	volatile int *blocks_complete;
	cdpErrchk( cudaMallocManaged(&blocks_complete, sizeof(int)) );
	*blocks_complete = 0;

	RandomStrides random_strides = std::get<0>(flat_random);
	DeviceMemory device_arrays;
	long long dist_rows;
	if (self) {
		// square 'self' block
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		dist_rows = static_cast<long long>(
						0.5*(sketch_subsample.ref_size)*(sketch_subsample.ref_size - 1));
		device_arrays = loadDeviceMemory(
			ref_strides,
			query_strides,
			ref_sketches,
			query_sketches,
			sketch_subsample,
			flat_random,
			ref_random_idx,
			query_random_idx,
			kmer_lengths,
			dist_rows,
			self);

		size_t blockSize, blockCount;
		std::tie(blockSize, blockCount) = getBlockSize(sketch_subsample.ref_size,
													   sketch_subsample.ref_size,
													   dist_rows,
													   self);
		calculate_self_dists<<<blockCount, blockSize>>>
			(
				thrust::raw_pointer_cast(&device_arrays.ref_sketches[0]),
				sketch_subsample.ref_size,
				thrust::raw_pointer_cast(&device_arrays.kmers[0]),
				kmer_lengths.size(),
				thrust::raw_pointer_cast(&device_arrays.dist_mat[0]),
				dist_rows,
				thrust::raw_pointer_cast(&device_arrays.random_table[0]),
				thrust::raw_pointer_cast(&device_arrays.ref_random[0]),
				ref_strides,
				random_strides,
				blocks_complete
			);
		reportProgress(blocks_complete, dist_rows);
	} else {
		// 'query' block
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

		dist_rows = sketch_subsample.ref_size * sketch_subsample.query_size;
		device_arrays = loadDeviceMemory(
			ref_strides,
			query_strides,
			ref_sketches,
			query_sketches,
			sketch_subsample,
			flat_random,
			ref_random_idx,
			query_random_idx,
			kmer_lengths,
			dist_rows,
			self);

		size_t blockSize, blockCount;
		std::tie(blockSize, blockCount) = getBlockSize(sketch_subsample.ref_size,
													   sketch_subsample.query_size,
													   dist_rows,
													   self);

		// Third argument is the size of __shared__ memory needed by a thread block
		// This is equal to the query sketch size in bytes (at a single k-mer length)
		calculate_query_dists<<<blockCount, blockSize,
								query_strides.sketchsize64*query_strides.bbits*sizeof(uint64_t)>>>
		(
			thrust::raw_pointer_cast(&device_arrays.ref_sketches[0]),
			sketch_subsample.ref_size,
			thrust::raw_pointer_cast(&device_arrays.query_sketches[0]),
			sketch_subsample.query_size,
			thrust::raw_pointer_cast(&device_arrays.kmers[0]),
			kmer_lengths.size(),
			thrust::raw_pointer_cast(&device_arrays.dist_mat[0]),
			dist_rows,
			thrust::raw_pointer_cast(&device_arrays.random_table[0]),
			thrust::raw_pointer_cast(&device_arrays.ref_random[0]),
			thrust::raw_pointer_cast(&device_arrays.query_random[0]),
			ref_strides,
			query_strides,
			random_strides,
			blocks_complete
		);
		reportProgress(blocks_complete, dist_rows);
	}

	// Copy results back to host
	std::vector<float> dist_results(dist_rows * 2);
	try {
		thrust::copy(device_arrays.dist_mat.begin(), device_arrays.dist_mat.end(), dist_results.begin());
	} catch (thrust::system_error &e) {
		// output a non-threatening but likely inaccurate error message and exit
		// e.g. 'trivial_device_copy D->H failed: unspecified launch failure'
		// error will have occurred elsewhere as launch is async, but better to catch
		// and deal with it here
		std::cerr << "Error getting result: " << std::endl;
		std::cerr << e.what() << std::endl;
		// cdpErrchk( cudaFree(blocks_complete) ); // Not needed, not an array
		exit(1);
	}

	cudaDeviceSynchronize();
	// cdpErrchk( cudaFree(blocks_complete) ); // Not needed, not an array
	fprintf(stderr, "%cProgress (GPU): 100.0%%\n", 13);

	return(dist_results);
}


