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

#include "asa136.hpp"
#include "api.hpp"

// A, C, G, T
// Just in case we ever add N (or U, heaven forbid)
#define N_BASES 4
const char BASEMAP[N_BASES] = {'A', 'C', 'G', 'T'};
const char RCMAP[N_BASES] = {'T', 'G', 'C', 'A'};

const int max_iter = 300;
const int max_tries = 5;

// Functions used in construction
std::vector<size_t> random_ints(const size_t k_draws, const size_t n_samples);
std::tuple<robin_hood::unordered_node_map<std::string, uint16_t>,
			NumpyMatrix> cluster_frequencies(
	const std::vector<Reference>& sketches,
	const unsigned int n_clusters);
std::vector<std::string> generate_random_sequence(const Reference& ref_seq, const bool use_rc);

RandomMC::RandomMC() : _n_clusters(0), _no_adjustment(true), _no_MC(false), _use_rc(false) {}

RandomMC::RandomMC(const bool use_rc) : _n_clusters(0), _no_adjustment(false), _no_MC(true), _use_rc(use_rc) {}

// Constructor - generates random match chances by Monte Carlo, sampling probabilities
// from the input sketches
RandomMC::RandomMC(const std::vector<Reference>& sketches, 
				   const std::vector<size_t>& kmer_lengths,
				   const unsigned int n_clusters,
				   const unsigned int n_MC,
				   const bool use_rc,
				   const int num_threads) : _n_clusters(n_clusters), _no_adjustment(false), _no_MC(false), _use_rc(use_rc) {
	size_t sketchsize64 = sketches[0].sketchsize64();
	_use_rc = sketches[0].rc();
 
    // Run k-means on the base frequencies, save the results in a hash table   
	std::tie(_cluster_table, _cluster_centroids) = cluster_frequencies(sketches, _n_clusters);
	
	// Pick representative sequences, generate random sequence from them
	unsigned int found = 0; 
	_representatives.reserve(n_clusters);
	std::fill(_representatives.begin(), _representatives.end(), "");
	std::vector<size_t> representatives_idx;
	for (auto sketch_it = sketches.begin(); sketch_it != sketches.end(); sketch_it++) {
		if (sketch_it->sketchsize64() != sketchsize64 || sketch_it->rc() != _use_rc) {
			throw std::runtime_error("Sketches have incompatible sizes or strand settings");
		}
		if (_representatives[_cluster_table[sketch_it->name()]].empty()) {
			_representatives[_cluster_table[sketch_it->name()]] = sketch_it->name(); 
			representatives_idx.push_back(sketch_it - sketches.begin());	
			if (++found >= n_clusters) {
				break;
			}
		}
	}

	// Generate random sequences and sketch them (in parallel)
	// TODO need to ensure these matches are not adjusted!
	omp_set_num_threads(num_threads);
	std::vector<std::vector<Reference>> random_seqs(n_clusters);
	#pragma omp parallel for simd collapse(2) schedule(static)
	for (unsigned int r_idx = 0; r_idx < n_clusters; r_idx++) {
		for (unsigned int copies = 0; copies < n_MC; copies++) {
				SeqBuf random_seq(generate_random_sequence(sketches[representatives_idx[r_idx]], use_rc), 
						sketches[representatives_idx[r_idx]].base_composition(),
						0, kmer_lengths.back());
				random_seqs[r_idx].push_back(
					Reference("random" + std::to_string(r_idx * n_MC + copies), 
							  random_seq, kmer_lengths, sketchsize64, use_rc, 0, false));
		}
	}

	RandomMC no_adjust;
	for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++) {
		NumpyMatrix matches = NumpyMatrix::Zero(n_clusters, n_clusters);
		for (unsigned int i = 0; i < n_clusters; i++) {
			for (unsigned int j = i; j < n_clusters; j++) {
				double dist_sum = 0;
				#pragma omp parallel for simd collapse(2) schedule(static) reduction(+:dist_sum)
				for (unsigned int copy_i = 0; copy_i < n_MC; copy_i++) {
					for (unsigned int copy_j = 0; copy_j < n_MC; copy_j++)
						dist_sum += random_seqs[i][copy_i].jaccard_dist(random_seqs[j][copy_j], *kmer_it, no_adjust);
				}
				int dist_count = n_MC * n_MC;
				if (i == j) {
					dist_count -= n_MC; // diagonal is zero
				}
				matches(i, j) = dist_sum/dist_count;
				matches(j, i) = matches(i, j);
			}
		}
		_matches[*kmer_it] = matches;	
	}

}

// Get the random match chance between two samples at a given k-mer length
// Will use Bernoulli estimate if MC was not run
float RandomMC::random_match(const Reference& r1, const Reference& r2, const size_t kmer_len) const {
	float random_chance = 0;
	if (_no_MC) {
		int rc_factor = _use_rc ? 2 : 1; // If using the rc, may randomly match on the other strand
		float j1 = r1.seq_length() / (r1.seq_length() + rc_factor * std::pow(0.25, -kmer_len));
		float j2 = r2.seq_length() / (r2.seq_length() + rc_factor * std::pow(0.25, -kmer_len));
		random_chance = (j1 * j2) / (j1 + j2 - j1 * j2);
	} else if (!_no_adjustment) {
		const uint16_t cluster1 = _cluster_table.at(r1.name()); 
		const uint16_t cluster2 = _cluster_table.at(r2.name());
		random_chance = _matches.at(kmer_len)(cluster1, cluster2);
	}
	return(random_chance);
}

// find nearest neighbour
size_t RandomMC::closest_cluster(const Reference& ref) const {
	Eigen::MatrixXf::Index index;
	double* ptr = &ref.base_composition()[0];
	Eigen::Map<Eigen::VectorXd> vd(ptr, ref.base_composition().size());
	Eigen::RowVectorXf vf = vd.cast<float>();
  	(_cluster_centroids.rowwise() - vf).colwise().squaredNorm().minCoeff(&index);
	return((size_t)index);
}

void RandomMC::add_query(const Reference& query) {
	_cluster_table[query.name()] = closest_cluster(query);
}

/*
*
* Internal functions
*
*/

// Draw k samples from [0, n]
std::vector<size_t> random_ints(const size_t k_draws, const size_t n_samples) {
	std::vector<size_t> random_idx(n_samples);
	std::iota(random_idx.begin(), random_idx.end(), 0);
    std::shuffle(random_idx.begin(), random_idx.end(), std::mt19937{std::random_device{}()});
	std::sort(random_idx.begin(), random_idx.begin() + k_draws);
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
		if (sketches[idx].rc()) {
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
		std::vector<size_t> centroid_samples = random_ints(n_clusters, sketches.size());
		for (size_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++) {
			for (size_t base = 0; base < N_BASES; base++) {
				centroids[centroid_idx + base * n_clusters] = data[centroid_samples[centroid_idx] + base * n_clusters];
			}	
		}
		
		// Run k-means
		kmns(data.get(), sketches.size(), N_BASES, centroids.get(), n_clusters, 
		     assignment.get(), cluster_counts.get(), max_iter, wss.get(), success);
		if (*success == 3) {
			throw std::runtime_error("Error with k-means input");
		}
	}
	if (tries == max_tries) {
		throw std::runtime_error("Could not cluster base frequencies");
	}

	// Build the return types
	robin_hood::unordered_node_map<std::string, uint16_t> cluster_map;
	for (size_t sketch_idx = 0; sketch_idx < sketches.size(); sketch_idx++) {
		cluster_map[sketches[sketch_idx].name()] = assignment[sketch_idx]; 
	}

	// doubles returned need to be cast to float in numpy matrix
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		 centroids_matrix_d(centroids.get(), n_clusters, N_BASES);
	NumpyMatrix centroids_matrix = centroids_matrix_d.cast<float>();
	/*{
		double* centroids_ptr = centroids.get();
		float* centroids_f = static_cast<float*>(centroids_ptr);
		new (&centroids_matrix) Eigen::Map<NumpyMatrix>(centroids_f);
	} */
	
	return std::make_tuple(cluster_map, centroids_matrix);	
}

// Bernoulli random draws - each base independent
std::vector<std::string> generate_random_sequence(const Reference& ref_seq, const bool use_rc) {
	std::vector<double> base_f = ref_seq.base_composition();
	std::default_random_engine generator;
	std::discrete_distribution<int> base_dist {base_f[0], base_f[1], base_f[2], base_f[3]};   
	
	std::ostringstream random_seq_buffer;
	for (size_t i = 0; i < ref_seq.seq_length(); i++) {
		if (use_rc && i > (ref_seq.seq_length()/2)) {
			random_seq_buffer << RCMAP[base_dist(generator)];
		} else {
			random_seq_buffer << BASEMAP[base_dist(generator)];
		}
	}

	std::vector<std::string> random = {std::string(random_seq_buffer.str())};
	return(random);
}
