/*
 *
 * random_match.hpp
 * Header file for random_match.cpp
 *
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "dist/matrix.hpp"
#include "gpu/gpu.hpp"
#include "robin_hood.h"

const unsigned int default_max_k = 101;
const unsigned int default_n_clusters = 3;
const unsigned int default_n_MC = 5;

class Reference;

enum class RandomType { NoAdjust, EqualBase, MonteCarlo };

class RandomMC {
public:
  // no adjustment
  RandomMC();
  // no MC - use simple Bernoulli prob
  RandomMC(const bool use_rc);
  // set up MC from sketches
  RandomMC(const std::vector<Reference> &sketches, unsigned int n_clusters,
           const unsigned int n_MC, const bool codon_phased, const bool use_rc,
           const int num_threads);
  // load MC from database (see database.cpp)
  RandomMC(const bool use_rc, const unsigned int min_k,
           const unsigned int max_k,
           const robin_hood::unordered_node_map<std::string, uint16_t>
               &cluster_table,
           const robin_hood::unordered_node_map<size_t, NumpyMatrix> &matches,
           const NumpyMatrix &cluster_centroids)
      : _n_clusters(cluster_centroids.rows()), _no_adjustment(false),
        _no_MC(false), _use_rc(use_rc), _min_k(min_k), _max_k(max_k),
        _cluster_table(cluster_table), _matches(matches),
        _cluster_centroids(cluster_centroids) {}

  RandomType mode() const {
    if (_no_adjustment) {
      return RandomType::NoAdjust;
    } else if (_no_MC) {
      return RandomType::EqualBase;
    } else {
      return RandomType::MonteCarlo;
    }
  };

  // Use in ref v query mode
  size_t min_supported_k(const size_t seq_length) const;
  double random_match(const Reference &r1, const uint16_t q_cluster_id,
                      const size_t q_length, const size_t kmer_len) const;
  std::vector<double>
  random_matches(const Reference &r1, const uint16_t q_cluster_id,
                 const size_t q_length,
                 const std::vector<size_t> &kmer_lengths) const;
  // Use in ref v ref mode
  double random_match(const Reference &r1, const Reference &r2,
                      const size_t kmer_len) const;

  // Other functions for adding new data in
  uint16_t closest_cluster(const Reference &ref) const;
  void add_query(const Reference &query);
  bool check_present(const std::vector<Reference> &sketches, bool update);

  // GPU helper functions to flatten
  std::vector<uint16_t>
  lookup_array(const std::vector<Reference> &sketches) const;
  FlatRandom flattened_random(const std::vector<size_t> &kmer_lengths,
                              const size_t default_length) const;

  // functions for saving
  robin_hood::unordered_node_map<std::string, uint16_t> cluster_table() const {
    return _cluster_table;
  }
  robin_hood::unordered_node_map<size_t, NumpyMatrix> matches() const {
    return _matches;
  }
  NumpyMatrix cluster_centroids() const { return _cluster_centroids; }
  std::tuple<unsigned int, unsigned int> k_range() const {
    return std::make_tuple(_min_k, _max_k);
  }
  bool use_rc() const { return _use_rc; }
  unsigned int n_clusters() const { return _n_clusters; }

  // Overloads
  bool operator==(const RandomMC &rhs) const {
    return _cluster_table == rhs.cluster_table() && _matches == rhs.matches() &&
           _cluster_centroids == rhs.cluster_centroids();
  }
  bool operator!=(const RandomMC &rhs) const { return !(*this == rhs); }

private:
  unsigned int _n_clusters;
  bool _no_adjustment;
  bool _no_MC;
  bool _use_rc;

  unsigned int _min_k;
  unsigned int _max_k;

  // name index -> cluster ID
  robin_hood::unordered_node_map<std::string, uint16_t> _cluster_table;
  // k-mer idx -> square matrix of matches, idx = cluster
  robin_hood::unordered_node_map<size_t, NumpyMatrix> _matches;

  NumpyMatrix _cluster_centroids;
};