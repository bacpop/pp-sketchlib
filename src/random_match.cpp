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

#include "asa058.hpp"
#include "reference.hpp"

const char BASEMAP[4] = {'A', 'C', 'G', 'T'};

class RandomMC {
	public:
		RandomMC(const std::vector<Reference>& sketches);

		double random_match(const std::string& name1, const std::string& name2) const;

	private:
		// name index -> cluster ID
		robin_hood::unordered_node_map<std::string, uint16_t> cluster_table;
		// k-mer idx -> cluster ID (vector position) -> square matrix of matches, idx = cluster
		robin_hood::unordered_node_map<size_t, std::vector<Eigen::EigenXd matches>> matches; 
};

// k-means
void cluster_frequencies(const std::vector<Reference>& ) {

}

std::string generate_random_sequence(const BaseComp<double>& base_f,
							  		 const size_t length) {
	std::default_random_engine generator;
	std::discrete_distribution<int> base_dist {base_f.a, base_f.c, base_f.g, base_f.t};   
	
	std::ostringstream random_seq_buffer;
	for (size_t i = 0; size_t < length; size_t++) {
		random_seq_buffer << BASEMAP[base_dist(generator)];
	}

	return(std::string(random_seq_buffer.str()));
}