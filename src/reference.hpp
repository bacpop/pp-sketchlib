/*
 *
 * reference.hpp
 * Header file for reference.cpp
 *
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <tuple>
#include "robin_hood.h"

#define ARMA_ALLOW_FAKE_GCC
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

const size_t def_bbits = 14; // = log2(sketch size) where sketch size = 64 * sketchsize64
const size_t def_sketchsize64 = 156;

#include "sketch/sketch.hpp"

class RandomMC;

class Reference
{
public:
  Reference();
  // Read and run sketch
  Reference(const std::string &name,
            SeqBuf &sequence,
            const KmerSeeds &kmers,
            const size_t sketchsize64,
            const bool codon_phased,
            const bool use_rc,
            const uint8_t min_count,
            const bool exact);

  // For loading from DB
  Reference(const std::string &name,
            const size_t bbits,
            const size_t sketchsize64,
            const size_t seq_size,
            const std::vector<double> bases,
            const unsigned long int missing_bases);

  // Initialise from GPU sketch
  Reference(const std::string &name,
            robin_hood::unordered_map<int, std::vector<uint64_t>> &sketch,
            const size_t bbits,
            const size_t sketchsize64,
            const size_t seq_size,
            const BaseComp<double> &bases,
            const unsigned long int missing_bases,
            const bool use_rc,
            const bool densified);

  const std::vector<uint64_t> &get_sketch(const int kmer_len) const;
  void add_kmer_sketch(const std::vector<uint64_t> &sketch, const int kmer_len);
  void remove_kmer_sketch(const size_t kmer_len);
  double jaccard_dist(Reference &query, const int kmer_len, const double random_jaccard);
  double jaccard_dist(Reference &query, const int kmer_len, const RandomMC &random);
  template <typename T>
  std::tuple<float, float> core_acc_dist(Reference &query, const T &random);
  template <typename T>
  std::tuple<float, float> core_acc_dist(Reference &query, const arma::mat &kmers, const T &random);
  std::vector<size_t> kmer_lengths() const;

  std::string name() const { return _name; }
  size_t bbits() const { return _bbits; }
  size_t sketchsize64() const { return _sketchsize64; }
  size_t seq_length() const { return _seq_size; }
  bool densified() const { return _densified; }
  bool rc() const { return _use_rc; }
  std::vector<double> base_composition() const { return {_bases.a, _bases.c, _bases.g, _bases.t}; }
  unsigned long int missing_bases() const { return _missing_bases; }

  // For sorting, order by name
  friend bool operator<(Reference const &a, Reference const &b)
  {
    return a._name < b._name;
  }
  friend bool operator==(Reference const &a, Reference const &b)
  {
    return a._name == b._name;
  }

private:
  // Info
  std::string _name;
  size_t _bbits;
  size_t _sketchsize64;
  bool _use_rc;

  // Sequence statistics
  size_t _seq_size;
  unsigned long int _missing_bases;
  bool _densified;

  // Proportion of each base
  BaseComp<double> _bases;

  // sketch - map keys are k-mer length
  robin_hood::unordered_map<int, std::vector<uint64_t>> usigs;
};

template <class T>
arma::mat kmer2mat(const T &kmers);

// Defined in linear_regression.cpp
std::tuple<float, float> regress_kmers(Reference *r1,
                                       Reference *r2,
                                       const arma::mat &kmers,
                                       const std::vector<double> &random);
std::tuple<float, float> regress_kmers(Reference *r1,
                                       Reference *r2,
                                       const arma::mat &kmers,
                                       const RandomMC &random);
