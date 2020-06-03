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
#include <unordered_map>
#include <string>
#include <tuple>

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>


#include "seqio.hpp"

class RandomMC; 

class Reference
{
    public:
        Reference();
        Reference(const std::string& name, 
                  const SeqBuf& sequence, 
                  const std::vector<size_t>& kmer_lengths,
                  const size_t sketchsize64, 
                  const bool use_rc,
                  const uint8_t min_count,
                  const bool exact); // read and run sketch

        Reference(const std::string& name,
                  const size_t bbits,
                  const size_t sketchsize64,
                  const size_t seq_size,
                  const std::vector<double> bases,
                  const unsigned long int missing_bases); // For loading from DB
        
        const std::vector<uint64_t> & get_sketch(const int kmer_len) const;
        void add_kmer_sketch(const std::vector<uint64_t>& sketch, const int kmer_len);
        void remove_kmer_sketch(const size_t kmer_len);
        double jaccard_dist(Reference &query, const int kmer_len, const RandomMC& random);
        std::tuple<float, float> core_acc_dist(Reference &query, const RandomMC& random);
        std::tuple<float, float> core_acc_dist(Reference &query, const arma::mat& kmers, const RandomMC& random);
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
        friend bool operator < (Reference const & a, Reference const & b)
        { return a._name < b._name; }
        friend bool operator == (Reference const & a, Reference const & b)
        { return a._name == b._name; }

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
        std::unordered_map<int, std::vector<uint64_t>> usigs;
};

template <class T>
arma::mat kmer2mat(const T& kmers);

// Defined in linear_regression.cpp
std::tuple<float, float> regress_kmers(Reference * r1, 
                                       Reference * r2, 
                                       const arma::mat& kmers,
                                       const RandomMC& random);
