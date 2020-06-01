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
#include "api.hpp"

// A, C, G, T
// Just in case we ever add N (or U, heaven forbid)
#define N_BASES = 4
const char BASEMAP[N_BASES] = {'A', 'C', 'G', 'T'};
const char RCMAP[N_BASES] = {'T', 'G', 'C', 'A'};

const int max_iter = 300;
const int max_tries = 5;

// Functions used in construction
std::vector<size_t> random_ints(const size_t k_draws, const size_t n_samples);
std::tuple<robin_hood::unordered_node_map<std::string, uint16_t>,
			NumpyMatrix> cluster_frequencies cluster_frequencies(
	const std::vector<Reference>& sketches,
	const unsigned int n_clusters);
std::string generate_random_sequence(const Reference& ref_seq, const bool use_rc);

RandomMC::RandomMC() : _n_clusters(0), _no_adjustment(true), _use_rc(false), _no_MC(false);

RandomMC::RandomMC(const bool use_rc) : _n_clusters(0), _no_adjustment(false), _use_rc(use_rc), _no_MC(true);

// Constructor - generates random match chances by Monte Carlo, sampling probabilities
// from the input sketches
RandomMC::RandomMC(const std::vector<Reference>& sketches, 
				   const std::vector<size_t>& kmer_lengths,
				   const unsigned int n_clusters,
				   const unsigned int n_MC,
				   const int num_threads) : _n_clusters(n_clusters), _no_adjustment(false), _use_rc(use_rc), _no_MC(false) {
	size_t sketchsize64 = sketches[0].sketchsize64();
	_use_rc = sketches[0].rc();
 
    // Run k-means on the base frequencies, save the results in a hash table   
	std::tie(_cluster_table, _cluster_centroids) = cluster_frequencies(sketches, _n_clusters);
	
	// Pick representative sequences, generate random sequence from them
	unsigned int found = 0; 
	_representatives.reserve(n_clusters);
	std::fill(_representatives.begin(), _representatives.end(), "")
	const std::vector<std::vector<Reference>::iterator representatives_it;
	for (auto sketch_it = sketches.begin(); sketch_it) {
		if (sketch_it->sketchsize64 != sketchsize64 || sketch_it->rc != _use_rc) {
			throw std::runtime_error("Sketches have incompatible sizes or strand settings");
		}
		if (_representatives[_cluster_table[sketch_it->name()]].empty()) {
			_representatives[_cluster_table[sketch_it->name()]] = sketch_it->name(); 
			representatives_it.push_back(sketch_it);	
			if (++found >= n_clusters) {
				break;
			}
		}
	}

	// Generate random sequences and sketch them (in parallel)
	// TODO need to ensure these matches are not adjusted!
	std::vector<Reference> random_seqs(representatives.size() * n_MC);
	omp_set_num_threads(num_threads);
	#pragma omp parallel for simd collapse(2) schedule(static)
	for (unsigned int r_idx = 0; r_idx < _representatives.size(); r_idx++) {
		for (unsigned int copies = 0; copies < n_MC; copies++) {
				SeqBuf random_seq(generate_random_sequence(*(representatives_it + r_idx), use_rc), 
						*(representatives_it + r_idx)->base_composition(),
						0, kmer_lengths.back());
				random_seqs[r_idx * n_MC + copies] = \
					Reference("random" + std::to_string(r_idx * n_MC + copies), 
							  random_seq, kmer_lengths, sketchsize64, use_rc, 0, false);
		}
	}

	// calculate jaccard distances at each k-mer length (use api.hpp)
	NumpyMatrix random = query_db(random_seqs, random_seqs, kmer_lengths, true, num_threads);

	// store these dists in an eigen matrix
	Eigen::VectorXf dummy_query_ref;
    Eigen::VectorXf dummy_query_query;
	for (unsigned int kmer_idx = 0; kmer_idx < kmer_lengths.size()l kmer_idx++) {
		NumpyMatrix random_full = long_to_square(random.col(kmer_idx), dummy_query_ref, dummy_query_query, num_threads);
		NumpyMatrix random_mean;
		random_mean.resize(n_clusters, n_clusters);
		unsigned int blockSize = n_MC * (n_MC - 1); // upper and lower triangle
		#pragma omp parallel for simd collapse(2) schedule(static)
		for (unsigned int i = 0; i < n_MC; i++) {
			for (unsigned int j = 0; j < n_MC; j++) {
				random_mean(i, j) = random_full.block(i*n_MC, j*n_MC, n_MC, n_MC).sum()/blockSize;
			}
		}
		_matches[kmer_lengths[kmer_idx]] = random_mean; 
	}
}

// Get the random match chance between two samples at a given k-mer length
// Will use Bernoulli estimate if MC was not run
float RandomMC::random_match(const Reference& r1, const Reference& r2, const size_t kmer_len) const {
	float random_chance = 0;
	if (_no_MC) {
		int rc_factor = _use_rc ? 2 : 1; // If using the rc, may randomly match on the other strand
		float avg_length = (r1.seq_length() + r2.seq_length()) / 2;
		random_chance = avg_length / (avg_length + rc_factor * std::pow(0.25, -kmer_len));
	} else if (!_no_adjustment) {
		uint16_t cluster1 = _cluster_table[r1.name()]; 
		uint16_t cluster2 = _cluster_table[r2.name()];
		random_chance = _matches[kmer_len](cluster1, cluster2) 
	}
	return(random_chance);
}

// find nearest neighbour
size_t RandomMC::closest_cluster(const Reference& ref) const {
	MatrixXf::Index index;
	Eigen::VectorXf v = ref.base_composition;
  	(_cluster_centroids.rowwise() - v).colwise().squaredNorm().minCoeff(&index);
	return(index);
}

/*
*
* Internal functions
*
*/

// Draw k samples from [0, n]
std::vector<size_t> random_ints(const size_t k_draws, const size_t n_samples) {
	std::vector<size_t> random_idx(sketches.size());
	std::iota(random_idx.begin(), random_idx.end(), 0);
    std::shuffle(random_idx.begin(), random_idx.end(), std::mt19937{std::random_device{}()});
	std::sort(random_idx.begin(), random_idx.begin() + n_samples);
	random_idx.erase(random_idx.begin() + k_draws + 1, random_idx.end());
	return(random_idx);
}

// k-means
std::tuple<robin_hood::unordered_node_map<std::string, uint16_t>,
			NumpyMatrix> cluster_frequencies(
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

	Eigen::Map<NumpyMatrix> centroids_matrix(centroids, n_clusters, N_BASES);

	return std::make_tuple(cluster_map, centroids_matrix);	
}

// Bernoulli random draws - each base independent
std::string generate_random_sequence(const Reference& ref_seq, const bool use_rc) {
	std::vector<double> base_f = ref_seq.base_composition()
	std::default_random_engine generator;
	std::discrete_distribution<int> base_dist {base_f[0], base_f[1], base_f[2], base_f[3]};   
	
	std::ostringstream random_seq_buffer;
	for (size_t i = 0; size_t < ref_seq.seq_length(); size_t++) {
		if (use_rc && i > (ref_seq.seq_length()/2)) {
			random_seq_buffer << RCMAP[base_dist(generator)];
		} else {
			random_seq_buffer << BASEMAP[base_dist(generator)];
		}
	}

	return(std::string(random_seq_buffer.str()));
}
