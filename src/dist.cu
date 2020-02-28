/*
 *
 * dist.cpp
 * PopPUNK dists using CUDA
 *
 */

#include <cstdint>
#include <stdlib.h>
#include <iostream>

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

__device__
size_t jaccard_dist(const uint64_t * sketch1, 
                    const uint64_t * sketch2, 
					const size_t sketchsize64,
					const size_t start_offset,
                    const size_t bbits) 
{
	size_t samebits = 0;

    for (size_t i = 0; i < sketchsize64; i++) 
    {
		uint64_t bits = ~((uint64_t)0ULL);
		for (size_t j = 0; j < bbits; j++) 
        {
			size_t index = (i + (start_offset * sketchsize64)) * bbits + j;
			bits &= ~(sketch1[index] ^ sketch2[index]);
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

__device__
void regress_kmers(float * dists,
				   const uint64_t * r1, 
                   const uint64_t * r2, 
				   const int * kmers,
				   const size_t sketchsize64, 
				   const size_t bbits,
				   const int N)
{
    // Vector of points 
	float * y;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&y, sizeof(float) * N));
    for (unsigned int i = 0; i < N; ++i)
    {
        y[i] = log(jaccard_dist(r1, r2, i, sketchsize64, nbits));
    }

	// Simple linear regression
	// Maybe BLAS routines would be more efficient
	float xsum = 0; float ysum = 0; float xysum = 0;
	float xsquaresum = 0; float ysquaresum = 0;
	for (unsigned int i = 0; i < N; ++i)
	{
		xsum += x[i]; 
		ysum += y[i]; 
		xysum += x[i] * y[i];
		xsquaresum = x[i] * x[i];
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
	double beta = xy * (1/pow(x_diff*y_diff, 0.5)) * (ystddev / xstddev);
    double alpha = ybar - beta * xbar;

	// Store core/accessory in dists, truncating at zero
	float core_dist = 0, accessory_dist = 0;
	if (beta < 0)
	{
		dists[i*2] = 1 - exp(beta);
	}
	if (alpha < 0)
	{
		dists[i*2 + 1] = 1 - exp(alpha);
	}
}

// TODO fix this (or combine into above)
// needs to skip where appropriate, and give correct start index and size in flattened vector
__global__
void calculate_dists(int n, 
					 uint64_t *ref, 
					 uint64_t *query,
					 float *dists)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		// may be useful to relate i to i,j in upper triangle/rectangle
		regress_kmers(dists, ref, i)
	}
}

// TODO
// function which loads block of sketches and resukts into device memory
//		flatten usigs for each k-mer into single vector, check indexing is ok
//		further flatten into samples (separately for ref and query; nullptr if ref == query)
// main function which
//		takes Reference objects, loads blocks into memory, and runs parallel calculate_dists
//		reads results back into host memory (eigen matrix for answer)
//		reads in the next block
//		frees memory
//		remember to add in deviceInit and deviceSync lines