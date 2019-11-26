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

class Reference
{
    public:
        Reference(const std::string& name, 
                  const std::string& filename, 
                  const std::vector<size_t>& kmer_lengths); // read and run sketch
        Reference(const std::string& name,
                  const size_t bbits,
                  const size_t sketchsize64,
                  const int seed); // For loading from DB
        
        const std::vector<uint64_t> & get_sketch(const int kmer_len) const;
        void add_kmer_sketch(const std::vector<uint64_t>& sketch, const int kmer_len);
        double jaccard_dist(const Reference &query, const int kmer_len);
        std::tuple<float, float> Reference::core_acc_dist(const Reference &query);
        std::vector<int> kmer_lengths() const;

        std::string name() const { return _name; }
        size_t bbits() const { return _bbits; }
        size_t sketchsize64() const { return _sketchsize64; }
        int seed() const { return _seed; }

    private:
        // Info
        std::string _name;
        size_t _bbits;
        size_t _sketchsize64;
        bool _isstrandpreserved;
        int _seed;

        // sketch - map keys are k-mer length
        std::unordered_map<int, std::vector<uint64_t>> usigs;
};
