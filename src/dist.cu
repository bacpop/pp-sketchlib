/*
 *
 * dist.cpp
 * PopPUNK dists using CUDA
 *
 */

// std
#include <cstdint>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

// cuda
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// internal headers
#include "bitfuncs.hpp"

static void
CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
	const char *statement, cudaError_t err) 
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

__device__
template <class T>
T non_neg_minus(T a, T b) {
	return a > b ? (a - b) : 0;
}

// CUDA version of bindash dist function
__device__
size_t jaccard_dist(const uint64_t * sketch1, 
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
		size_t intersize = ret * maxnbits / (maxnbits - expected_samebits);
	}
	size_t unionsize = NBITS(uint64_t) * sketchsize64;
    double jaccard = intersize/(double)unionsize;
    return(jaccard)
}

// Gets Jaccard distance across k-mer lengths and runs a regression
// to get core and accessory
__device__
void regress_kmers(float *& dists,
				   const long long dist_idx
				   const uint64_t * ref,
				   const long i, 
				   const uint64_t * query,
				   const long j, 
				   const int * kmers,
				   const int kmer_n,
				   const size_t sketchsize64, 
				   const size_t bbits,
				   const long kmer_stride, 
				   const long sample_stride)						  
{
    // Vector for Jaccard dists 
	float * y;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&y, sizeof(float) * kmer_n));
	
	long long ref_offset = i * sample_stride;
	long long query_offset = j * sample_stride
	for (unsigned int kmer_it = 0; kmer_it < kmer_n; ++kmer_it)
    {
		y[i] = log(jaccard_dist(ref + ref_offset, query + query_offset, sketchsize64, bbits));
		ref_offset += kmer_stride;
		query_offset += kmer_stride;
    }

	// Simple linear regression
	// Maybe BLAS routines would be more efficient
	float xsum = 0; float ysum = 0; float xysum = 0;
	float xsquaresum = 0; float ysquaresum = 0;
	for (unsigned int i = 0; i < N; ++i)
	{
		xsum += kmers[i]; 
		ysum += y[i]; 
		xysum += kmers[i] * y[i];
		xsquaresum = kmers[i] * kmers[i];
		ysquaresum = y[i] * y[i];
	}
	CUDA_CHECK_RETURN(cudaFree(y));

	float xbar = xsum / N;
	float ybar = xyum / N;
    float xy = xysum - xbar*ybar;
    float x_diff = (xsquaresum / N) - pow(xbar, 2);
    float y_diff = (ysquaresum / N) - pow(ybar, 2);
	float xstddev = sqrt(x_diff);
	float ystddev = sqrt(y_diff);
	double beta = xy * (1/sqrt(x_diff*y_diff)) * (ystddev / xstddev);
    double alpha = ybar - beta * xbar;

	// Store core/accessory in dists, truncating at zero
	float core_dist = 0, accessory_dist = 0;
	if (beta < 0)
	{
		dists[dist_idx*2] = 1 - exp(beta);
	}
	if (alpha < 0)
	{
		dists[dist_idx*2 + 1] = 1 - exp(alpha);
	}
}

// Functions to convert index position to/from squareform to condensed form
__device__
long long calc_row_idx(const long long k, const long long n) {
	return static_cast<long long>(ceil((0.5) * (- sqrt(-8*k + 4 *pow(n,2) -4*n - 7) + 2*n -1) - 1));
}

__device__
long elem_in_i_rows(const long i, const long long n) {
	return (i * (n - 1 - i) + ((i*(i + 1)) >> 1));
}

__device__
long calc_col_idx(const long long k, const long i, const long long n) {
	return (n - elem_in_i_rows(i + 1, n) + k);
}

__device__
long long square_to_condensed(long long i, long long j, long long n) {
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
					 float *&dists,
					 const long long dist_n,
					 const size_t sketchsize64, 
					 const size_t bbits,
					 const long kmer_stride,
					 const long sample_stride)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (long long dist_idx = index; dist_idx < dist_n; i += stride)
	{
		if (query == nullptr)
		{
			long i = calc_row_idx(dist_idx, dist_n)
			long j = calc_col_idx(dist_idx, i, dist_n)
			if (j <= i)
			{
				continue;
			}
		}
		else (query != nullptr)
		{
			long i = dist_idx % ref_n;
			long j = floor(dist_idx / query_n);
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
	const std::vector<size_t>& kmer_lengths)
{
	thrust::host_vector<uint64_t> flat_ref(sample_stride * ref_sketches.size());
	auto flat_ref_it = flat_ref.begin();
	for (auto sample_it = sketches.cbegin(); sample_it != sketches.cend(); sample_it++)
	{
		for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend(); kmer_it++)
		{
			thrust::copy(sample_it->get_sketch(*kmer_it).cbegin(),
						 sample_it->get_sketch(*kmer_it).cend(),
						 flat_ref_it)
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
DistMatrix query_db_gpu(const std::vector<Reference>& ref_sketches,
	const std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
	const int blockSize,
    const size_t max_device_mem)
{
	std::cerr << "Calculating distances on GPU device 0" << std::endl;
    
    // Check if ref = query, then run as self mode
	// TODO implement max device mem
	// TODO will involve taking square blocks of distmat
	bool self = False;
	std::sort(ref_sketches.begin(), ref_sketches.end());
	std::sort(query_sketches.begin(), query_sketches.end())
	
	long long dist_rows;
	if (ref_sketches == query_sketches)
    {
		dist_rows = static_cast<long long>(0.5*(ref_sketches.size())*(ref_sketches.size() - 1));
		self = True;
	}
	else
	{
		dist_rows = ref_sketches.size() * query_sketches.size();
	}
	double est_size  = (sample_stride * n_samples * sizeof(uint64_t) + dist_rows * sizeof(float))/(1048576);
	std::cerr << "Estimated device memory: " << std::setprecision(1) << est_size << "Mb" << std::endl;
	
	// Check sketches are compatible
	size_t bbits = ref_sketches[0].bbits();
	size_t sketchsize64 = ref_sketches[0].sketchsize64();
	checkSketchParamsMatch(ref_sketches, kmer_lengths, bbits, sketchsize64);
	if (!self)
	{
		checkSketchParamsMatch(query_sketches, kmer_lengths, bbits, sketchsize64);
	}

	// Initialise device
	cudaSetDevice(0);
	cudaDeviceReset();

	// flatten the input sketches and copy ref sketches to device
	// long bin_stride = 1; // unused - pass this iff strides of flattened array change
	long kmer_stride = sketchsize64 * bbits;
	long sample_stride = kmer_stride * kmer_lengths.size();
	thrust::device_vector<int> d_kmers = kmer_lengths;
	int* d_kmers_array = thrust::raw_pointer_cast( &d_kmers[0] );
	thrust::host_vector<uint64_t> flat_ref = flatten_sketches(ref_sketches, kmer_lengths);
	thrust::device_vector<uint64_t> d_ref_sketches = flat_ref;

	// Set up query and distance arrays and copy to device
	uint64_t* d_ref_array = thrust::raw_pointer_cast( &d_ref_sketches[0] );
	uint64_t* d_query_array = nullptr;
	size_t dist_rows;
	if (!self)
    {
		dist_rows = ref_sketches.size() * query_sketches.size();
		thrust::host_vector<uint64_t> flat_query = flatten_sketches(query_sketches, kmer_lengths);
		d_query_array = thrust::raw_pointer_cast( &flat_query[0] ); 
	}
	thrust::device_vector<float> dist_mat(dist_rows*2, 0);
	float* d_dist_array = thrust::raw_pointer_cast( &dist_mat[0] );

	// Run dists on device
	int blockCount = (dist_rows + blockSize - 1) / dist_rows;
	calculate_dists<<<blockCount, blockSize>>>(
		d_ref_array,
		ref_sketches.size(),
		d_query_array,
		query_sketches.size(),
		d_kmers_array,
		kmers.size(),
		dist_mat,
		dist_rows,
		kmer_stride,
		sample_stride)
				
	// copy results from device, and convert to return type
	std::vector<float> dist_results = dist_mat; 
	DistMatrix dists_ret = 
		Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::RowMajor> >(dist_results.data(),dist_rows,2);

    return dists_ret;
}