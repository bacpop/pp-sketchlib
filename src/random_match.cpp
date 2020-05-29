/*
 *
 * random_match.cpp
 * Monte Carlo simulation of random match probabilites
 *
 */

#include <cstdint>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <random>
#include <memory>

#include "asa058.hpp"
#include "reference.hpp"

// A, C, G, T
// Just in case we ever add N (or U, heaven forbid)
#define N_BASES = 4
const char BASEMAP[N_BASES] = {'A', 'C', 'G', 'T'};

const int max_iter = 300;
const int max_tries = 5;

class RandomMC {
	public:
		RandomMC(const std::vector<Reference>& sketches, const unsigned int n_clusters);

		double random_match(const std::string& name1, const std::string& name2) const;
		// TODO add flatten functions here too
		// will need a lookup table from sample_idx -> random_match_idx

	private:
		unsigned int _n_clusters;	
		
		// TODO may be easier to get rid of hashes and use sketch index instead? (considering GPU)
		// name index -> cluster ID
		robin_hood::unordered_node_map<std::string, uint16_t> _cluster_table;
		std::vector<std::string> _representatives;
		// k-mer idx -> cluster ID (vector position) -> square matrix of matches, idx = cluster
		robin_hood::unordered_node_map<size_t, std::vector<Eigen::EigenXd matches>> _matches; 
};

RandomMC::RandomMC(const std::vector<Reference>& sketches, 
				   const unsigned int n_clusters,
				   const unsigned int n_MC) : _n_clusters(n_clusters) {
    _cluster_table = cluster_frequencies(sketches, _n_clusters);
	
	// Pick representative sequences, generate random sequence from them
	unsigned int found = 0;
	_representatives.reserve(n_clusters);
	std::fill(_representatives.begin(), _representatives.end(), "")
	for (auto sketch_it = sketches.begin(); sketch_it) {
		if (_representatives[_cluster_table[sketch_it->name()]].empty()) {
			_representatives[_cluster_table[sketch_it->name()]] = sketch_it->name(); 
			SeqBuf random_seq(generate_random_sequence(*sketch_it)); // TODO need to make a SeqBuf constructor from string
			if (++found >= n_clusters) {							// TODO ?? generate how many random seqs? ?? //
				break;
			}
		}
	}

	// sketch random sequences at each k-mer length (use api.hpp)

	// calculate jaccard distances at each k-mer length (use api.hpp)

	// store these dists in an eigen matrix
}

RandomMC::random_match(const std::string& name1, const std::string& name2) const {

}

std::vector<size_t> random_ints(const size_t n_samples) {
	std::vector<size_t> random_idx(sketches.size());
	std::iota(random_idx.begin(), random_idx.end(), 0);
    std::shuffle(random_idx.begin(), random_idx.end(), std::mt19937{std::random_device{}()});
	std::sort(random_idx.begin(), random_idx.begin() + n_samples);
	random_idx.erase(random_idx.begin() + n_clusters + 1, random_idx.end());
	return(random_idx);
}

// k-means
robin_hood::unordered_node_map<std::string, uint16_t> cluster_frequencies(
	const std::vector<Reference>& sketches,
	const unsigned int n_clusters) {
	
	// Build the input matrix in the right form
	auto data = std::make_unique<double[]>(sketches.size() * N_BASES); // column major
	for (size_t idx = 0; idx < sketches.size(); idx++) {
		std::vector<double> base_ref = sketches[idx].base_composition();
		// Cannot distinguish A/T G/C if strand unknown
		if (ref.rc()) {
			base_ref[0] = 0.5*(base_ref[0] + base_ref[3]);
			base_ref[1] = 0.5*(base_ref[1] + base_ref[2]);
			base_ref[2] = base_ref[1];
			base_ref[3] = base_ref[0];
		}
		
		for (size_t base = 0; base < base_ref.size(); base++) {
			data[idx + base * sketches.size()] = base_ref[base];
		}
	}

	// Output arrays
	auto assignment = std::make_unique<int[]>(sketches.size());
	auto cluster_counts = std::make_unique<int[]>(n_clusters);
	auto wss = std::make_unique<double[]>(n_clusters);

	auto centroids = std::make_unique<double[]>(n_clusters * N_BASES);
	int* success; *success = -1;
	int tries = 0;
	while(*success && tries < max_tries) {
		tries++;
		
		// Pick random centroids for each attempt
		std::vector<size_t> centroid_samples = random_ints(n_clusters);
		for (size_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
			for (size_t base = 0; base < base_ref.size(); base++) {
				centroids[centroid_idx + base * n_clusters] = data[centroid_samples[centroid_idx] + base * n_clusters];
			}	
		}
		
		// Run k-means
		kmns(data, sketches.size(), N_BASES, centroids, n_clusters, 
		     assignment, cluster_counts, max_iter, wss, success);
		if (*success == 3) {
			throw std::runtime_error("Error with k-means input");
		}
	}
	if (tries == max_tries) {
		throw std::runtime_error("Could not cluster base frequencies");
	}

	robin_hood::unordered_node_map<std::string, uint16_t> cluster_map;
	for (size_t sketch_idx = 0; sketch_idx < sketches.size(); sketch_idx++) {
		cluster_map[sketches[sketch_idx].name()] = assignment[sketch_idx]; 
	}

	return cluster_map;	
}

std::string generate_random_sequence(const Reference& ref_seq) {
	std::vector<double> base_f = ref_seq.base_composition()
	std::default_random_engine generator;
	std::discrete_distribution<int> base_dist {base_f[0], base_f[1], base_f[2], base_f[3]};   
	
	std::ostringstream random_seq_buffer;
	for (size_t i = 0; size_t < ref_seq.seq_length(); size_t++) {
		random_seq_buffer << BASEMAP[base_dist(generator)];
	}

	return(std::string(random_seq_buffer.str()));
}