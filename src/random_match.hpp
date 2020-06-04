/*
 *
 * random_match.hpp
 * Header file for random_match.cpp
 *
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <tuple>

#include "robin_hood.h"
#include "matrix.hpp"

class Reference;

class RandomMC {
	public:
		RandomMC(); // no adjustment
        RandomMC(const bool use_rc); // no MC - use simple Bernoulli prob
		RandomMC(const std::vector<Reference>& sketches, 
				   const std::vector<size_t>& kmer_lengths,
				   unsigned int n_clusters,
				   const unsigned int n_MC,
				   const bool use_rc,
				   const int num_threads);

		double random_match(const Reference& r1, const Reference& r2, const size_t kmer_len) const;
        size_t closest_cluster(const Reference& ref) const;
		void add_query(const Reference& query);
		// TODO add flatten functions here too
		// will need a lookup table (array) from sample_idx -> random_match_idx

	private:
		unsigned int _n_clusters;
		bool _no_adjustment;
        bool _no_MC;
		bool _use_rc;
		
		// name index -> cluster ID
		robin_hood::unordered_node_map<std::string, uint16_t> _cluster_table;
		std::vector<std::string> _representatives;
		// k-mer idx -> square matrix of matches, idx = cluster
		robin_hood::unordered_node_map<size_t, NumpyMatrix> _matches; 

        NumpyMatrix _cluster_centroids;
};