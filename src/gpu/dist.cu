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

// internal headers
#include "sketch/bitfuncs.hpp"
#include "gpu.hpp"
#include "cuda.cuh"

// Memory on device for each operation
class DeviceMemory {
	public:
		// Defined below
		DeviceMemory(SketchStrides& ref_strides,
			SketchStrides& query_strides,
			std::vector<Reference>& ref_sketches,
			std::vector<Reference>& query_sketches,
			const SketchSlice& sample_slice,
			const FlatRandom& flat_random,
			const std::vector<uint16_t>& ref_random_idx,
			const std::vector<uint16_t>& query_random_idx,
			const std::vector<size_t>& kmer_lengths,
			long long dist_rows,
			const bool self);

		~DeviceMemory() {
			CUDA_CALL(cudaFree(d_ref_sketches));
			CUDA_CALL(cudaFree(d_query_sketches));
			CUDA_CALL(cudaFree(d_random_table));
			CUDA_CALL(cudaFree(d_ref_random));
			CUDA_CALL(cudaFree(d_query_random));
			CUDA_CALL(cudaFree(d_kmers));
			CUDA_CALL(cudaFree(d_dist_mat));
		}

		std::vector<float> read_dists() {
			cudaDeviceSynchronize();
			std::vector<float> dists(_n_dists);
			CUDA_CALL(cudaMemcpy(dists.data(),
								 d_dist_mat,
								 _n_dists * sizeof(float),
								 cudaMemcpyDefault));
            return dists;
		}

		uint64_t * ref_sketches() { return d_ref_sketches; }
		uint64_t * query_sketches() { return d_query_sketches; }
		float * random_table() { return d_random_table; }
		uint16_t * ref_random() { return d_ref_random; }
		uint16_t * query_random() { return d_query_random; }
		int * kmers() { return d_kmers; }
		float * dist_mat() { return d_dist_mat; }

	private:
		DeviceMemory ( const DeviceMemory & ) = delete;
        DeviceMemory ( DeviceMemory && ) = delete;

		size_t _n_dists;
		uint64_t * d_ref_sketches;
		uint64_t * d_query_sketches;
		float * d_random_table;
		uint16_t * d_ref_random;
		uint16_t * d_query_random;
		int * d_kmers;
		float * d_dist_mat;
};

/******************
*			      *
*	Device code   *
*			      *
*******************/

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
				   const size_t sketchsize64,
				   const size_t bbits,
				   const size_t s1_stride,
				   const size_t s2_stride)
{
	size_t samebits = 0;
    for (int i = 0; i < sketchsize64; i++)
    {
		int bin_index = i * bbits;
		uint64_t bits = ~((uint64_t)0ULL);
		for (int j = 0; j < bbits; j++) {
			// Almost all kernel time is spent on this line
			// (bbits * sketchsize64 * N^2 * 2 8-byte memory loads)
			bits &= ~(sketch1[bin_index * s1_stride] ^ sketch2[bin_index * s2_stride]);
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
							  const int n) {
	// CUDA fast-math intrinsics on floats, which give comparable accuracy
	// Speed gain is fairly minimal, as most time spent on Jaccard distance
	// __fmul_ru(x, y) = x * y and rounds up.
	// __fpow(x, a) = x^a give 0 for x<0, so not using here (and it is slow)
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

/******************
*			      *
*	Global code   *
*			      *
*******************/

// Main kernel functions run on the device,
// but callable from the host

__global__
void calculate_dists(const bool self,
					 const uint64_t * ref,
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
		query_idx = __float2int_rz(__fdividef(blockIdx.x, blocksPerQuery) + 0.001f);
		ref_idx = (blockIdx.x % blocksPerQuery) * blockDim.x + threadIdx.x;
		dist_idx = query_idx * ref_n + ref_idx;
	}
	__syncwarp();

	const uint64_t* ref_start = ref + ref_idx * ref_strides.sample_stride;
	const uint64_t* query_start = query + query_idx * query_strides.sample_stride;

	// Calculate Jaccard distances over k-mer lengths
	float xsum = 0; float ysum = 0; float xysum = 0;
	float xsquaresum = 0; float ysquaresum = 0;
	for (int kmer_idx = 0; kmer_idx < kmer_n; kmer_idx++) {
		// Copy query sketch into __shared__ mem
		// Uses all threads *in a single warp* to do the copy
		// NB there is no disadvantage vs using multiple warps, as they would have to wait
		// (see https://stackoverflow.com/questions/15468059/copy-to-the-shared-memory-in-cuda)
		// NB for query these reads will be coalesced, but for ref they won't, as can't
		// coalesce both here (bin inner stride) and in jaccard (sample inner stride)
		extern __shared__ uint64_t query_shared[];
		size_t sketch_bins = query_strides.bbits * query_strides.sketchsize64;
		size_t sketch_stride = query_strides.bin_stride;
		if (threadIdx.x < warp_size) {
			for (int lidx = threadIdx.x; lidx < sketch_bins; lidx += warp_size) {
				query_shared[lidx] = query_start[lidx * sketch_stride];
			}
		}
		__syncthreads();

		// Some threads at the end of the last block will have nothing to do
		// Need to have conditional here to avoid block on __syncthreads() above
		if (ref_idx < ref_n) {
			// Calculate Jaccard distance at current k-mer length
			float jaccard_obs = jaccard_dist(ref_start, query_shared,
											 ref_strides.sketchsize64,
											 ref_strides.bbits,
											 ref_strides.bin_stride,
											 1);

			// Adjust for random matches
			float jaccard_expected = random_table[kmer_idx * random_strides.kmer_stride +
												  ref_idx_lookup[ref_idx] * random_strides.cluster_inner_stride +
												  query_idx_lookup[query_idx] * random_strides.cluster_outer_stride];
			float y = __logf(observed_excess(jaccard_obs, jaccard_expected, 1.0f));
			//printf("i:%d j:%d k:%d r:%f jac:%f y:%f\n", ref_idx, query_idx, kmer_idx, jaccard_expected, jaccard_obs, y);

			// Running totals for regression
			int kmer = kmers[kmer_idx];
			xsum += kmer;
			ysum += y;
			xysum += kmer * y;
			xsquaresum += kmer * kmer;
			ysquaresum += y * y;
		}

		// Move to next k-mer length
		ref_start += ref_strides.kmer_stride;
		query_start += query_strides.kmer_stride;
	}

	if (ref_idx < ref_n) {
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
	__syncwarp();
}

/***************
*			   *
*	Host code  *
*			   *
***************/

// Gives strides aligned to the warp size (32)
inline size_t warpPad(const size_t stride) {
	return(stride + (stride % warp_size ? warp_size - stride % warp_size : 0));
}

// Turn a vector of references into a flattened vector of
// uint64 with strides bins * kmers * samples
std::vector<uint64_t> flatten_by_bins(
	const std::vector<Reference>& sketches,
	const std::vector<size_t>& kmer_lengths,
	SketchStrides& strides,
	const size_t start_sample_idx,
	const size_t end_sample_idx) {
	// Input checks
	size_t num_sketches = end_sample_idx - start_sample_idx;
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch(kmer_lengths[0]).size());
	assert(end_sample_idx > start_sample_idx);

	// Set strides structure
	strides.bin_stride = 1;
	strides.kmer_stride = warpPad(strides.bin_stride * num_bins);
	// warpPad not needed here, as k-mer stride already a multiple of warp size
	strides.sample_stride = strides.kmer_stride * kmer_lengths.size();

	// Iterate over each dimension to flatten
	std::vector<uint64_t> flat_ref(strides.sample_stride * num_sketches);
	auto flat_ref_it = flat_ref.begin();
	for (auto sample_it = sketches.cbegin() + start_sample_idx;
		 sample_it != sketches.cend() - (sketches.size() - end_sample_idx);
		 sample_it++) {
		for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend(); kmer_it++) {
			std::copy(sample_it->get_sketch(*kmer_it).cbegin(),
						 sample_it->get_sketch(*kmer_it).cend(),
						 flat_ref_it);
            flat_ref_it += strides.kmer_stride;
		}
	}
	return flat_ref;
}

// Turn a vector of queries into a flattened vector of
// uint64 with strides samples * bins * kmers
std::vector<uint64_t> flatten_by_samples(
	const std::vector<Reference>& sketches,
	const std::vector<size_t>& kmer_lengths,
	SketchStrides& strides,
	const size_t start_sample_idx,
	const size_t end_sample_idx) {
	// Set strides
	size_t num_sketches = end_sample_idx - start_sample_idx;
	const size_t num_bins = strides.sketchsize64 * strides.bbits;
	assert(num_bins == sketches[0].get_sketch(kmer_lengths[0]).size());
	strides.sample_stride = 1;
	strides.bin_stride = warpPad(num_sketches);
	strides.kmer_stride = strides.bin_stride * num_bins;

	// Stride by bins then restride by samples
	// This is 4x faster than striding by samples by looping over References vector,
	// presumably because many fewer dereferences are being used
	SketchStrides old_strides = strides;
	std::vector<uint64_t> flat_bins = flatten_by_bins(sketches,
															  kmer_lengths,
															  old_strides,
															  start_sample_idx,
															  end_sample_idx);
	std::vector<uint64_t> flat_ref(strides.kmer_stride * kmer_lengths.size());
	auto flat_ref_it = flat_ref.begin();
	for (size_t kmer_idx = 0; kmer_idx < kmer_lengths.size(); kmer_idx++) {
		for (size_t bin_idx = 0; bin_idx < num_bins; bin_idx++) {
			for (size_t sample_idx = 0; sample_idx < num_sketches; sample_idx++) {
				*flat_ref_it = flat_bins[sample_idx * old_strides.sample_stride + \
										 bin_idx * old_strides.bin_stride + \
										 kmer_idx * old_strides.kmer_stride];
				flat_ref_it++;
			}
			flat_ref_it += strides.bin_stride - num_sketches;
		}
	}

	return flat_ref;
}

// Sets up data structures and loads them onto the device
DeviceMemory::DeviceMemory(SketchStrides& ref_strides,
					  SketchStrides& query_strides,
					  std::vector<Reference>& ref_sketches,
					  std::vector<Reference>& query_sketches,
					  const SketchSlice& sample_slice,
					  const FlatRandom& flat_random,
					  const std::vector<uint16_t>& ref_random_idx,
					  const std::vector<uint16_t>& query_random_idx,
					  const std::vector<size_t>& kmer_lengths,
					  long long dist_rows,
					  const bool self)
	: _n_dists(dist_rows * 2),
	  d_query_sketches(nullptr),
	  d_query_random(nullptr) {
	// Set up reference sketches, flatten and copy to device
	std::vector<uint64_t> flat_ref = flatten_by_samples
		(
			ref_sketches,
			kmer_lengths,
			ref_strides,
			sample_slice.ref_offset,
			sample_slice.ref_offset + sample_slice.ref_size
		);
	CUDA_CALL(cudaMalloc((void**)&d_ref_sketches,
						 flat_ref.size() * sizeof(uint64_t)));
	CUDA_CALL(cudaMemcpy(d_ref_sketches, flat_ref.data(),
						 flat_ref.size() * sizeof(uint64_t),
						 cudaMemcpyDefault));

	// Preload random match chances, which have already been flattened
	CUDA_CALL(cudaMalloc((void**)&d_random_table,
						  std::get<1>(flat_random).size() * sizeof(float)));
	CUDA_CALL(cudaMemcpy(d_random_table, std::get<1>(flat_random).data(),
						 std::get<1>(flat_random).size() * sizeof(float),
						 cudaMemcpyDefault));
	CUDA_CALL(cudaMalloc((void**)&d_ref_random,
						  sample_slice.ref_size * sizeof(uint16_t)));
	CUDA_CALL(cudaMemcpy(d_ref_random,
						 ref_random_idx.data() + sample_slice.ref_offset,
						 sample_slice.ref_size * sizeof(uint16_t),
						 cudaMemcpyDefault));

	// If ref v query mode, also flatten query vector and copy to device
	if (!self) {
		std::vector<uint64_t> flat_query = flatten_by_bins
		(
			query_sketches,
			kmer_lengths,
			query_strides,
			sample_slice.query_offset,
			sample_slice.query_offset + sample_slice.query_size
		);
		CUDA_CALL(cudaMalloc((void**)&d_query_sketches,
							 flat_query.size() * sizeof(uint64_t)));
		CUDA_CALL(cudaMemcpy(d_query_sketches, flat_query.data(),
							 flat_query.size() * sizeof(uint64_t),
						     cudaMemcpyDefault));

		CUDA_CALL(cudaMalloc((void**)&d_query_random,
							 sample_slice.query_size * sizeof(uint16_t)));
		CUDA_CALL(cudaMemcpy(d_query_random,
						     query_random_idx.data() + sample_slice.query_offset,
							 sample_slice.query_size * sizeof(uint16_t),
							 cudaMemcpyDefault));
	} else {
		query_strides = ref_strides;
	}

	// Copy or set other arrays needed on device (kmers and distance output)
	CUDA_CALL(cudaMalloc((void**)&d_kmers,
						 kmer_lengths.size() * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_kmers, kmer_lengths.data(),
						 kmer_lengths.size() * sizeof(int),
						 cudaMemcpyDefault));

	CUDA_CALL(cudaMalloc((void**)&d_dist_mat,
						 _n_dists * sizeof(float)));
	CUDA_CALL(cudaMemset(d_dist_mat, 0, _n_dists * sizeof(float)));
}

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
	size_t blockSize = std::min(512, 32 * static_cast<int>((ref_samples + 32 - 1) / 32));
	size_t blockCount = 0;
	if (self) {
		for (int i = 0; i < ref_samples; i++) {
			blockCount += (ref_samples + blockSize - 2 - i) / blockSize;
		}
	} else {
		size_t blocksPerQuery = (ref_samples + blockSize - 1) / blockSize;
		blockCount = blocksPerQuery * query_samples;
	}
	return(std::make_tuple(blockSize, blockCount));
}

// Writes a progress meter using the device int which keeps
// track of completed jobs
void reportDistProgress(volatile int * blocks_complete,
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

// Initialise device and return info on its memory
std::tuple<size_t, size_t> initialise_device(const int device_id) {
	cudaSetDevice(device_id);
	cudaDeviceReset();

	size_t mem_free = 0; size_t mem_total = 0;
	cudaMemGetInfo(&mem_free, &mem_total);
	return(std::make_tuple(mem_free, mem_total));
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
	// Note this is a preference, which will be overridden if more __shared__
	// space is needed
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// Progress meter
	volatile int *blocks_complete;
	CUDA_CALL( cudaMallocManaged(&blocks_complete, sizeof(int)) );
	*blocks_complete = 0;

	RandomStrides random_strides = std::get<0>(flat_random);
	long long dist_rows;
	if (self) {
		dist_rows = (sketch_subsample.ref_size * (sketch_subsample.ref_size - 1)) >> 1;
	} else {
		dist_rows = sketch_subsample.ref_size * sketch_subsample.query_size;
	}

	// Load memory onto device
	DeviceMemory device_arrays
	(
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
		self
	);

	size_t blockSize, blockCount;
	if (self) {
		std::tie(blockSize, blockCount) =
			getBlockSize(sketch_subsample.ref_size,
						 sketch_subsample.ref_size,
						 dist_rows,
						 self);

		// Third argument is the size of __shared__ memory needed by a thread block
		// This is equal to the query sketch size in bytes (at a single k-mer length)
		calculate_dists<<<blockCount, blockSize,
						query_strides.sketchsize64*query_strides.bbits*sizeof(uint64_t)>>>
			(
				self,
				device_arrays.ref_sketches(),
				sketch_subsample.ref_size,
				device_arrays.ref_sketches(),
				sketch_subsample.ref_size,
				device_arrays.kmers(),
				kmer_lengths.size(),
				device_arrays.dist_mat(),
				dist_rows,
				device_arrays.random_table(),
				device_arrays.ref_random(),
				device_arrays.ref_random(),
				ref_strides,
				ref_strides,
				random_strides,
				blocks_complete
			);
	} else {
		std::tie(blockSize, blockCount) =
			getBlockSize(sketch_subsample.ref_size,
						 sketch_subsample.query_size,
						 dist_rows,
						 self);

		// Third argument is the size of __shared__ memory needed by a thread block
		// This is equal to the query sketch size in bytes (at a single k-mer length)
		calculate_dists<<<blockCount, blockSize,
						query_strides.sketchsize64*query_strides.bbits*sizeof(uint64_t)>>>
			(
				self,
				device_arrays.ref_sketches(),
				sketch_subsample.ref_size,
				device_arrays.query_sketches(),
				sketch_subsample.query_size,
				device_arrays.kmers(),
				kmer_lengths.size(),
				device_arrays.dist_mat(),
				dist_rows,
				device_arrays.random_table(),
				device_arrays.ref_random(),
				device_arrays.query_random(),
				ref_strides,
				query_strides,
				random_strides,
				blocks_complete
			);
	}

	reportDistProgress(blocks_complete, dist_rows);

	// Copy results back to host
	std::vector<float> dist_results = device_arrays.read_dists();

	fprintf(stderr, "%cProgress (GPU): 100.0%%\n", 13);

	return(dist_results);
}
