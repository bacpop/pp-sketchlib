/*
 *
 * gpu_api.cpp
 * PopPUNK dists using CUDA
 * gcc compiled part (uses Eigen)
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

// internal headers
#include "bitfuncs.hpp"
#include "gpu.hpp"
#include "matrix.hpp"

const float mem_epsilon = 0.05;

template<class T>
inline T samples_to_rows(const T samples) {
    return ((samples * (samples - 1)) >> 1);
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

void printSlice(const SketchSlice& s) {
	std::cerr << s.ref_offset << std::endl;
	std::cerr << s.ref_size << std::endl;
	std::cerr << s.query_offset << std::endl;
	std::cerr << s.query_size << std::endl;
}

// Functions to join matrices when doing computation in chunks
void longToSquareBlock(NumpyMatrix& coreSquare,
                       NumpyMatrix& accessorySquare,
                       const SketchSlice& sketch_subsample,
                       NumpyMatrix& blockMat,
                       const bool self,
                       const unsigned int num_threads) {
    // Convert each long form column of Nx2 matrix into square distance matrix
    // Add this square matrix into the correct submatrix (block) of the final square matrix
    Eigen::VectorXf dummy_query_ref;
    Eigen::VectorXf dummy_query_query;
    
    NumpyMatrix square_form;
    if (self) {
        square_form = long_to_square(blockMat.col(0), 
                                    dummy_query_ref, 
                                    dummy_query_query,
                                    num_threads);
    } else {
        square_form = Eigen::Map<NumpyMatrix, 0, Eigen::InnerStride<2>>(blockMat.data(), 
                                    sketch_subsample.ref_size, 
                                    sketch_subsample.query_size);
    }
    coreSquare.block(sketch_subsample.ref_offset, 
                        sketch_subsample.query_offset,
                        sketch_subsample.ref_size, 
                        sketch_subsample.query_size) = square_form; 

    if (self) {
        square_form = long_to_square(blockMat.col(1), 
                                    dummy_query_ref, 
                                    dummy_query_query,
                                    num_threads);
    } else {
        square_form = Eigen::Map<NumpyMatrix, 0, Eigen::InnerStride<2>>(blockMat.data()+1, 
                                            sketch_subsample.ref_size, 
                                            sketch_subsample.query_size);

    }
    accessorySquare.block(sketch_subsample.ref_offset, 
                        sketch_subsample.query_offset,
                        sketch_subsample.ref_size, 
                        sketch_subsample.query_size) = square_form; 
}

NumpyMatrix twoColumnSquareToLong(const NumpyMatrix& coreSquare,
                       const NumpyMatrix& accessorySquare,
                       const unsigned int num_threads) {
    Eigen::VectorXf core_dists = square_to_long(coreSquare, num_threads);
    Eigen::VectorXf accessory_dists = square_to_long(accessorySquare, num_threads);
    
    NumpyMatrix dists_ret_matrix(samples_to_rows(coreSquare.rows()), 2);
    dists_ret_matrix << core_dists, accessory_dists; // Join columns 
    return(dists_ret_matrix);
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
	size_t mem_free, mem_total;
	std::tie(mem_free, mem_total) = initialise_device(device_id);
    std::cerr << "Calculating distances on GPU device " << device_id << std::endl;

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
	double est_size  = (bbits * sketchsize64 * kmer_lengths.size() * n_samples * sizeof(uint64_t) + // Size of sketches
						kmer_lengths.size() * n_samples * sizeof(float) +                           // Size of random matches
						dist_rows * 2 * sizeof(float));							    				// Size of distance matrix
	std::cerr << "Estimated device memory required: " << std::fixed << std::setprecision(0) << est_size/(1048576) << "Mb" << std::endl;
	std::cerr << "Total device memory: " << std::fixed << std::setprecision(0) << mem_total/(1048576) << "Mb" << std::endl;
	std::cerr << "Free device memory: " << std::fixed << std::setprecision(0) << mem_free/(1048576) << "Mb" << std::endl;

	if (est_size > mem_free * (1 - mem_epsilon) && !self) {
		throw std::runtime_error("Using greater than device memory is unsupported for query mode. "
							     "Split your input into smaller chunks");	
	}

	// Ready to run dists on device
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
		chunks = 2;
        size_t calc_per_chunk = n_samples / chunks;
		unsigned int num_big_chunks = n_samples % chunks;

		// Only allocate these square matrices if they are needed
		if (chunks > 1) {
			coreSquare.resize(n_samples, n_samples);
			accessorySquare.resize(n_samples, n_samples);
		}
		unsigned int total_chunks = (chunks * (chunks + 1)) >> 1;
		unsigned int chunk_count = 0;

		sketch_subsample.query_offset = 0;
		for (unsigned int chunk_i = 0; chunk_i < chunks; chunk_i++) {
			sketch_subsample.query_size = calc_per_chunk;
			if (chunk_i < num_big_chunks) {
				sketch_subsample.query_size++;
			}
			
			sketch_subsample.ref_offset = sketch_subsample.query_offset; 
			for (unsigned int chunk_j = chunk_i; chunk_j < chunks; chunk_j++) {
				std::cerr << "Running chunk " << ++chunk_count << " of " << total_chunks << std::endl;
                sketch_subsample.ref_size = calc_per_chunk;
				if (chunk_j < num_big_chunks) {
					sketch_subsample.ref_size++;
				}
				printSlice(sketch_subsample);
				
                if (chunk_i == chunk_j) {
					// 'self' blocks
					dist_results = dispatchDists(
						ref_sketches,
						ref_sketches,
						ref_strides,
						query_strides,
						sketch_subsample,
						kmer_lengths,
						true);
				} else {
					// 'query' block
					dist_results = dispatchDists(
						ref_sketches,
						query_sketches,
						ref_strides,
						query_strides,
						sketch_subsample,
						kmer_lengths,
						false);
				}
				sketch_subsample.ref_offset += sketch_subsample.ref_size; 
                printf("Returned\n");

				// Read intermediate dists out
				if (chunks > 1) {
                    // Copy results from device into Nx2 matrix
                    NumpyMatrix blockMat = \
                        Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::ColMajor> > \
                            (dist_results.data(),dist_results.size()/2,2);
                    printf("Mapped\n");
                    
                    // Convert each long form column of Nx2 matrix into square distance matrix
                    // Add this square matrix into the correct submatrix (block) of the final square matrix
                    if (self) {

                    }
                    longToSquareBlock(coreSquare,
                                        accessorySquare,
                                        sketch_subsample,
                                        blockMat,
                                        chunk_i == chunk_j,
                                        num_cpu_threads);
                    printf("To square\n");
				} 
			}
			sketch_subsample.query_offset += sketch_subsample.query_size; 
		}

	}
	else
	{
		sketch_subsample.ref_size = ref_sketches.size();
		sketch_subsample.query_size = query_sketches.size();
        sketch_subsample.ref_offset = 0;
        sketch_subsample.query_offset = 0;

		dist_results = dispatchDists(ref_sketches,
                                    query_sketches,
                                    ref_strides,
                                    query_strides,
                                    sketch_subsample,
                                    kmer_lengths,
                                    false);	
	}
	
	NumpyMatrix dists_ret_matrix;
    if (self && chunks > 1) {
		// Chunked computation yields square matrix, which needs to be converted back to long
		// form
		dists_ret_matrix = twoColumnSquareToLong(coreSquare,
												 accessorySquare,
												 num_cpu_threads);
	} else {
        dists_ret_matrix = \
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::ColMajor> >(dist_results.data(),dist_results.size()/2,2);
    } 

	return dists_ret_matrix;
}