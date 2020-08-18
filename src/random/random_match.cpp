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
#include <algorithm>

#include "random_match.hpp"
#include "rng.hpp"
#include "api.hpp"

// A, C, G, T
// Just in case we ever add N (or U, heaven forbid)
#define N_BASES 4
const char BASEMAP[N_BASES] = {'A', 'C', 'G', 'T'};
const char RCMAP[N_BASES] = {'T', 'G', 'C', 'A'};

// k-mer length parameters
const size_t min_kmer = 6;
const size_t max_kmer = 31;

// k-means parameters
const int max_iter = 300;
const int max_tries = 5;

// Functions used in construction
double csrs(const size_t k, const bool use_rc, const size_t l1, const size_t l2);
std::tuple<robin_hood::unordered_node_map<std::string, uint16_t>,
			NumpyMatrix> cluster_frequencies(
	const std::vector<Reference>& sketches,
	const unsigned int n_clusters,
	const unsigned int num_threads);
std::vector<double> apply_rc(const Reference& ref);
uint16_t nearest_neighbour(const Reference& ref, const NumpyMatrix& cluster_centroids);
std::vector<std::string> generate_random_sequence(const Eigen::VectorXf& base_f,
												  const size_t seq_length,
												  const bool use_rc,
												  Xoshiro& generator);

RandomMC::RandomMC() : _n_clusters(0), _no_adjustment(true), _no_MC(false),
					   _use_rc(false), _min_k(0), _max_k(default_max_k) {}

RandomMC::RandomMC(const bool use_rc) : _n_clusters(0), _no_adjustment(false),
				   _no_MC(true), _use_rc(use_rc), _min_k(0),
				   _max_k(default_max_k) {}

// Constructor - generates random match chances by Monte Carlo, sampling probabilities
// from the input sketches
RandomMC::RandomMC(const std::vector<Reference>& sketches,
				   unsigned int n_clusters,
				   const unsigned int n_MC,
				   const bool use_rc,
				   const int num_threads)
	: _no_adjustment(false), _no_MC(false), _use_rc(use_rc) {
	std::cerr << "Calculating random match chances using Monte Carlo" << std::endl;

	if (n_clusters >= sketches.size()) {
		std::cerr << "Cannot make more base frequency clusters than sketches" << std::endl;
		n_clusters = sketches.size() - 1;
	}
	if (n_clusters < 2) {
		throw std::runtime_error("Cannot make this few base frequency clusters");
	}
	_n_clusters = n_clusters;

	size_t sketchsize64 = sketches[0].sketchsize64();
	_use_rc = sketches[0].rc();

    // Run k-means on the base frequencies, save the results in a hash table
	std::tie(_cluster_table, _cluster_centroids) = cluster_frequencies(sketches, _n_clusters, num_threads);

	// Decide which k-mer lengths to use assuming equal base frequencies
	RandomMC default_adjustment(use_rc);
	std::vector<size_t> kmer_lengths = {min_kmer};
	size_t kmer_size = min_kmer + 1;
	const double min_random = static_cast<double>(1)/(sketchsize64 * 64);
	while (kmer_size <= max_kmer) {
		double match_chance = default_adjustment.random_match(sketches[0],
															  0,
															  sketches[0].seq_length(),
															  kmer_size);
		if (match_chance > min_random) {
			kmer_lengths.push_back(kmer_size);
		} else {
			break;
		}
		kmer_size++;
	}
	_min_k = kmer_lengths.front();
	_max_k = kmer_lengths.back();

	// Generate random sequences and sketch them (in parallel)
	std::vector<std::vector<Reference>> random_seqs(n_clusters);
	Xoshiro generator(1);
	#pragma omp parallel for collapse(2) firstprivate(generator) schedule(static) num_threads(num_threads)
	for (unsigned int r_idx = 0; r_idx < n_clusters; r_idx++) {
		for (unsigned int copies = 0; copies < n_MC; copies++) {
				// Ensure generated sequence uses uncorrelated streams of RNG
				for (int rng_jump = 0; rng_jump < omp_get_thread_num(); rng_jump++) {
					generator.jump();
				}
				// Make the sequence
				SeqBuf random_seq(
					generate_random_sequence(_cluster_centroids.row(r_idx),
											 sketches[0].seq_length(),
											 use_rc, generator));
				// Sketch it
				random_seqs[r_idx].push_back(
					Reference("random" + std::to_string(r_idx * n_MC + copies),
							  random_seq, kmer_lengths, sketchsize64, use_rc, 0, false));
		}
	}

	RandomMC no_adjust;
	// printf("kmer\tMC\tformula\n");
	for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++) {
		NumpyMatrix matches = NumpyMatrix::Zero(n_clusters, n_clusters);
		for (unsigned int i = 0; i < n_clusters; i++) {
			for (unsigned int j = i; j < n_clusters; j++) {
				double dist_sum = 0;
				#pragma omp parallel for simd collapse(2) schedule(static) reduction(+:dist_sum)
				for (unsigned int copy_i = 0; copy_i < n_MC; copy_i++) {
					for (unsigned int copy_j = 0; copy_j < n_MC; copy_j++) {
						dist_sum += random_seqs[i][copy_i].jaccard_dist(random_seqs[j][copy_j], *kmer_it, no_adjust);
					}
				}
				int dist_count = n_MC * n_MC;
				// Diagonal contains n_MC self comparisons (== 1) which need to be removed
				if (i == j) {
					dist_sum -= n_MC;
					dist_count -= n_MC;
				}
				matches(i, j) = dist_sum/dist_count;
				matches(j, i) = matches(i, j);
				//printf("%lu\t%f\t%f\n", *kmer_it, matches(i, j), default_adjustment.random_match(random_seqs[i][0],
				//												random_seqs[j][0],
				//												*kmer_it));
			}
		}
		_matches[*kmer_it] = matches;
	}

}

size_t RandomMC::min_supported_k(const size_t seq_length) const {
	size_t min_k_signal = _min_k;
	if (!_no_MC) {
		for (size_t k = min_k_signal; k < _max_k; k++) {
			double random = 0;
			if (_no_adjustment) {
				random = csrs(k, _use_rc, seq_length, seq_length);
			} else {
				random = _matches.at(k).mean();
			}
			if (random < 1) {
				min_k_signal = k;
				break;
			}
		}
	}
	return min_k_signal;
}

// This is used for query v ref, so query lookup is not repeated
double RandomMC::random_match(const Reference& r1, const uint16_t q_cluster_id,
						      const size_t q_length, const size_t kmer_len) const {
	double random_chance = 0;
	if (_no_MC) {
		random_chance = csrs(kmer_len, _use_rc, r1.seq_length(), q_length);
	} else if (!_no_adjustment && kmer_len < _max_k) {
		// Longer k-mer lengths here are set to zero
		const uint16_t r_cluster_id = _cluster_table.at(r1.name());
		random_chance = _matches.at(kmer_len)(r_cluster_id, q_cluster_id);
	}
	return(random_chance);
}

double RandomMC::random_match(const Reference& r1, const Reference& r2,
						      const size_t kmer_len) const {
	uint16_t q_cluster_id = 0;
	if (!_no_MC && !_no_adjustment) {
		q_cluster_id = _cluster_table.at(r2.name());
	}
	return random_match(r1, q_cluster_id, r2.seq_length(), kmer_len);
}

std::vector<double> RandomMC::random_matches(const Reference& r1,
	const uint16_t q_cluster_id, const size_t q_length,
	const std::vector<size_t>& kmer_lengths) const {

	std::vector<double> random;
	for (auto kmer_len = kmer_lengths.cbegin(); kmer_len != kmer_lengths.cend(); ++kmer_len) {
		random.push_back(random_match(r1, q_cluster_id, q_length, *kmer_len));
	}
	return random;
}

// Returns zero if there are no clusters defined
uint16_t RandomMC::closest_cluster(const Reference& ref) const {
	uint16_t closest_cluster_id = 0;
	if (!_no_MC && !_no_adjustment) {
		closest_cluster_id = nearest_neighbour(ref, _cluster_centroids);
	}
	return(closest_cluster_id);
}

/*
size_t RandomMC::closest_cluster(const arma::vec& bases) const {
	double* ptr = bases.memptr();
	return(closest_cluster(ptr, _cluster_centroids));
}
*/

void RandomMC::add_query(const Reference& query) {
	_cluster_table[query.name()] = closest_cluster(query);
}

// Helper functions for loading onto GPU
// Get an array index in order of sample name
std::vector<uint16_t> RandomMC::lookup_array(const std::vector<Reference>& sketches) const {
	std::vector<uint16_t> lookup;
	lookup.reserve(sketches.size());
	for (auto &sketch : sketches) {
		if (_no_adjustment || _no_MC) {
			lookup.push_back(0); // If no MC, lookup tables have only one index (0)
		} else {
			auto in_table = _cluster_table.find(sketch.name());
			if (in_table == _cluster_table.end()) {
				// Queries need to be added
				lookup.push_back(closest_cluster(sketch));
			} else {
				// References are already in the table
				lookup.push_back(in_table->second);
			}
		}
	}
	return lookup;
}

// Flatten chosen random structure
// access: kmer_idx * kmer_stride + ref_idx * inner_stride + query_idx * outer_stride
FlatRandom RandomMC::flattened_random(
	const std::vector<size_t>& kmer_lengths,
	const size_t default_length) const {
	size_t matrix_size = _n_clusters * _n_clusters;
	RandomStrides strides;

	std::vector<float> flat;
	// No adjustment -> return 0
	if (_no_adjustment) {
		strides = {0, 0, 0};
		flat.push_back(0);
	} else {
		for (auto &k : kmer_lengths) {
			if (_no_MC) {
				// Keep one entry based on equal base frequencies
				strides = {1, 0, 0};
				flat.push_back(csrs(k, _use_rc, default_length, default_length));
			} else {
				// Full structure, indexed with lookup_array
				strides = {matrix_size, _n_clusters, 1};
				if (k < _min_k) {
					throw std::runtime_error("Trying to choose a k-mer length below the minimum allowed\n");
				} else if (k <= _max_k) {
					std::copy(_matches.at(k).data(),
							_matches.at(k).data() + matrix_size,
							std::back_inserter(flat));
				} else {
					std::fill_n(std::back_inserter(flat), matrix_size, 0);
				}
			}
		}
	}
	return(std::make_tuple(strides, flat));
}

/*
*
* Internal functions
*
*/

double csrs(const size_t k, const bool use_rc, const size_t l1, const size_t l2) {
	// This is what we're doing, written more clearly
	// int rc_factor = _use_rc ? 2 : 1; // If using the rc, may randomly match on the other strand
	// size_t match_chance = rc_factor * std::pow(N_BASES, (double)-kmer_len)
	//float j1 = r1.seq_length() / (r1.seq_length() + rc_factor * std::pow(N_BASES, (double)-kmer_len));
	//float j2 = r2.seq_length() / (r2.seq_length() + rc_factor * std::pow(N_BASES, (double)-kmer_len));
	// use bitshift to calculate 4^k
	double random_chance = 0;
	size_t match_chance = (size_t)1 << ((k - 1) * 2 + (use_rc ? 1 : 0));
	double j1 = 1 - std::pow(1 - (double)1/match_chance, (double)l1);
	double j2 = 1 - std::pow(1 - (double)1/match_chance, (double)l2);
	if (j1 > 0 && j2 > 0) {
		random_chance = (j1 * j2) / (j1 + j2 - j1 * j2);
	}
	return(random_chance);
}

std::vector<double> apply_rc(const Reference& ref) {
	std::vector<double> base_ref = ref.base_composition();
	// Cannot distinguish A/T G/C if strand unknown
	if (ref.rc()) {
			base_ref[0] = 0.5*(base_ref[0] + base_ref[3]);
			base_ref[1] = 0.5*(base_ref[1] + base_ref[2]);
			base_ref[2] = base_ref[1];
			base_ref[3] = base_ref[0];
	}
	return base_ref;
}

// find nearest neighbour centroid id
uint16_t nearest_neighbour(const Reference& ref, const NumpyMatrix& cluster_centroids) {
	std::vector<double> bases = apply_rc(ref);
	Eigen::MatrixXf::Index index;
	Eigen::Map<Eigen::VectorXd> vd(bases.data(), bases.size());
	Eigen::RowVectorXf vf = vd.cast<float>();
	(cluster_centroids.rowwise() - vf).rowwise().squaredNorm().minCoeff(&index);
	return((uint16_t)index);
}

arma::uvec random_ints(const size_t k_draws, const size_t max_n) {
	std::vector<arma::uword> random_idx(max_n);
	std::iota(random_idx.begin(), random_idx.end(), 0);
    // std::shuffle(random_idx.begin(), random_idx.end(), std::mt19937{std::random_device{}()}); // non-deterministic
    std::shuffle(random_idx.begin(), random_idx.end(), std::mt19937{1}); // deterministic
	std::sort(random_idx.begin(), random_idx.begin() + k_draws);
	random_idx.erase(random_idx.begin() + k_draws, random_idx.end());
	return(arma::uvec(random_idx));
}

// k-means
std::tuple<robin_hood::unordered_node_map<std::string, uint16_t>,
			NumpyMatrix> cluster_frequencies(
	const std::vector<Reference>& sketches,
	const unsigned int n_clusters,
	const unsigned int num_threads) {

	// Build the input matrix in the right form
	arma::mat data(N_BASES, sketches.size(), arma::fill::zeros);
	for (size_t idx = 0; idx < sketches.size(); idx++) {
		std::vector<double> base_ref = apply_rc(sketches[idx]);
		for (size_t base = 0; base < base_ref.size(); base++) {
			data(base, idx) = base_ref[base];
		}
	}

	arma::mat means;
	bool success = false;
	int tries = 0;
	omp_set_num_threads(num_threads);
	while(!success && tries < max_tries) {
		tries++;
		// Also uses openmp
		success = arma::kmeans(means, data, n_clusters, arma::random_subset, max_iter, false);
	}
	if (!success) {
		std::cerr << "Could not cluster base frequencies; using randomly chosen samples" << std::endl;
		means = data.cols(random_ints(n_clusters, sketches.size()));
	}

	// Build the return types
	// doubles returned need to be cast to float in numpy matrix
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		 centroids_matrix_d(means.memptr(), n_clusters, N_BASES);
	NumpyMatrix centroids_matrix = centroids_matrix_d.cast<float>();

	robin_hood::unordered_node_map<std::string, uint16_t> cluster_map;
	// #pragma omp parallel for schedule(static)
	for (size_t sketch_idx = 0; sketch_idx < sketches.size(); sketch_idx++) {
		cluster_map[sketches[sketch_idx].name()] = \
			nearest_neighbour(sketches[sketch_idx], centroids_matrix);
	}

	return std::make_tuple(cluster_map, centroids_matrix);
}

// Bernoulli random draws - each base independent
std::vector<std::string> generate_random_sequence(const Eigen::VectorXf& base_f,
												  const size_t seq_length,
												  const bool use_rc,
												  Xoshiro& generator) {
	std::discrete_distribution<int> base_dist {base_f[0], base_f[1], base_f[2], base_f[3]};

	// Better simulation:
	// draw number of contigs N ~ Pois(mean(nr_contigs))
	// draw lengths, probably from an empirical distribution
	// (exp doesn't fit well, there's usually one long contig)
	// draw direction from std::bernoulli_distribution rc_dist(0.5);

	std::ostringstream random_seq_buffer;
	for (size_t i = 0; i < seq_length; i++) {
		if (use_rc && i > seq_length / 2) {
			random_seq_buffer << RCMAP[base_dist(generator)];
		} else {
			random_seq_buffer << BASEMAP[base_dist(generator)];
		}
	}

	std::vector<std::string> random = {std::string(random_seq_buffer.str())};
	return(random);
}
