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
#include "random_match.hpp"

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
		square_form = Eigen::Map<NumpyMatrix, 0, Eigen::InnerStride<2>>(
            blockMat.data(),
            sketch_subsample.ref_size,
            sketch_subsample.query_size);
    }
 	// Only update the upper triangle
	coreSquare.block(sketch_subsample.query_offset,
                        sketch_subsample.ref_offset,
                        sketch_subsample.query_size,
                        sketch_subsample.ref_size) = square_form;

    if (self) {
        square_form = long_to_square(blockMat.col(1),
                                    dummy_query_ref,
                                    dummy_query_query,
                                    num_threads);
    } else {
        square_form = Eigen::Map<NumpyMatrix, 0, Eigen::InnerStride<2>>(
                blockMat.data()+1,
                sketch_subsample.ref_size,
                sketch_subsample.query_size);

    }
	accessorySquare.block(sketch_subsample.query_offset,
                        sketch_subsample.ref_offset,
                        sketch_subsample.query_size,
                        sketch_subsample.ref_size) = square_form;
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
	RandomMC& random_match,
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
	double est_size  = (bbits * sketchsize64 * kmer_lengths.size() * n_samples * sizeof(uint64_t) +    // Size of sketches
						kmer_lengths.size() * std::pow(random_match.n_clusters(), 2) * sizeof(float) + // Size of random matches
						n_samples * sizeof(uint16_t) +
						dist_rows * 2 * sizeof(float));							    				   // Size of distance matrix
	std::cerr << "Estimated device memory required: " << std::fixed << std::setprecision(0) << est_size/(1048576) << "Mb" << std::endl;
	std::cerr << "Total device memory: " << std::fixed << std::setprecision(0) << mem_total/(1048576) << "Mb" << std::endl;
	std::cerr << "Free device memory: " << std::fixed << std::setprecision(0) << mem_free/(1048576) << "Mb" << std::endl;

	if (est_size > mem_free * (1 - mem_epsilon) && !self) {
		throw std::runtime_error("Using greater than device memory is unsupported for query mode. "
							     "Split your input into smaller chunks");
	}

	// Turn the random matches into an array (same for any ref, query or subsample thereof)
	FlatRandom flat_random = random_match.flattened_random(kmer_lengths, ref_sketches[0].seq_length());
	std::vector<uint16_t> ref_random_idx = random_match.lookup_array(ref_sketches);

	// Ready to run dists on device
	SketchSlice sketch_subsample;
	unsigned int chunks = 1;
	std::vector<float> dist_results(dist_rows * 2);
	NumpyMatrix coreSquare, accessorySquare;
	if (self)
	{
		// To prevent memory being exceeded, total distance matrix is split up into
		// chunks which do fit in memory. The most is needed in the 'corners' where
		// two separate lots of sketches are loaded, hence the factor of two below

		// These are iterated over in the same order as a square distance matrix.
		// The i = j chunks are 'self', i < j can be skipped
		// as they contain only lower triangle values, i > j work as query vs ref
		chunks = floor((est_size * 2) / (mem_free * (1 - mem_epsilon))) + 1;
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
				if (total_chunks > 1) {
					std::cerr << "Running chunk " << ++chunk_count << " of " << total_chunks << std::endl;
				}

	            sketch_subsample.ref_size = calc_per_chunk;
				if (chunk_j < num_big_chunks) {
					sketch_subsample.ref_size++;
				}

				dist_results = dispatchDists(
						ref_sketches,
						ref_sketches,
						ref_strides,
						query_strides,
						flat_random,
						ref_random_idx,
						ref_random_idx,
						sketch_subsample,
						kmer_lengths,
						chunk_i == chunk_j);

				// Read intermediate dists out
				if (chunks > 1) {
					NumpyMatrix blockMat = \
                        Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::ColMajor> > \
                            (dist_results.data(),dist_results.size()/2,2);

                    // Convert each long form column of Nx2 matrix into square distance matrix
                    // Add this square matrix into the correct submatrix (block) of the final square matrix
                    longToSquareBlock(coreSquare,
                                        accessorySquare,
                                        sketch_subsample,
                                        blockMat,
                                        chunk_i == chunk_j,
                                        num_cpu_threads);
				}
				sketch_subsample.ref_offset += sketch_subsample.ref_size;
			}
			sketch_subsample.query_offset += sketch_subsample.query_size;
		}

	}
	else
	{
		std::vector<uint16_t> query_random_idx =
			random_match.lookup_array(query_sketches);

		sketch_subsample.ref_size = ref_sketches.size();
		sketch_subsample.query_size = query_sketches.size();
        sketch_subsample.ref_offset = 0;
        sketch_subsample.query_offset = 0;

		dist_results = dispatchDists(ref_sketches,
                                    query_sketches,
                                    ref_strides,
                                    query_strides,
									flat_random,
									ref_random_idx,
									query_random_idx,
                                    sketch_subsample,
                                    kmer_lengths,
                                    false);
	}

	NumpyMatrix dists_ret_matrix;
    if (self && chunks > 1) {
		// Chunked computation yields square matrix, which needs to be converted back to long
		// form
		Eigen::VectorXf core_dists = square_to_long(coreSquare, num_cpu_threads);
		Eigen::VectorXf accessory_dists = square_to_long(accessorySquare, num_cpu_threads);

		dists_ret_matrix.resize(samples_to_rows(coreSquare.rows()), 2);
		dists_ret_matrix << core_dists, accessory_dists; // Join columns
	} else {
        dists_ret_matrix = \
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::ColMajor> >(dist_results.data(),dist_results.size()/2,2);
    }

	return dists_ret_matrix;
}