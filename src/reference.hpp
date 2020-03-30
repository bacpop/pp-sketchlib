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

class Reference
{
    public:
        Reference();
        Reference(const std::string& name, 
                  const std::vector<std::string>& filenames, 
                  const std::vector<size_t>& kmer_lengths,
                  const size_t sketchsize64, 
                  const bool use_rc,
                  const uint8_t min_count,
                  const bool exact); // read and run sketch

        Reference(const std::string& name,
                  const size_t bbits,
                  const size_t sketchsize64,
                  const size_t seq_size); // For loading from DB
        
        const std::vector<uint64_t> & get_sketch(const int kmer_len) const;
        void add_kmer_sketch(const std::vector<uint64_t>& sketch, const int kmer_len);
        void remove_kmer_sketch(const size_t kmer_len);
        double jaccard_dist(const Reference &query, const int kmer_len) const;
        std::tuple<float, float> core_acc_dist(const Reference &query) const;
        std::tuple<float, float> core_acc_dist(const Reference &query, const arma::mat& kmers) const;
        std::vector<size_t> kmer_lengths() const;

        std::string name() const { return _name; }
        size_t bbits() const { return _bbits; }
        size_t sketchsize64() const { return _sketchsize64; }
        size_t seq_length() const { return _seq_size; }

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
        size_t _seq_size;

        // sketch - map keys are k-mer length
        std::unordered_map<int, std::vector<uint64_t>> usigs;
};

template <class T>
arma::mat kmer2mat(const T& kmers);

// Defined in linear_regression.cpp
std::tuple<float, float> regress_kmers(const Reference * r1, 
                                       const Reference * r2, 
                                       const arma::mat& kmers);
