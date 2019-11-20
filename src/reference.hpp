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
#include <memory>
#include <unordered_map>
#include <string>

class Reference
{
    public:
        Reference(const std::string& _name, 
                  const std::string& filename, 
                  const std::vector<size_t>& kmer_lengths); // read and run sketch
        
        const std::vector<uint64_t> & get_sketch(const int kmer_len) const;
        double dist(const Reference &query, const int kmer_len);
        void save();
        void load();

    private:
        // Info
        std::string name;
        size_t bbits;
        size_t sketchsize64;
        bool isstrandpreserved;

        // sketch - map keys are k-mer length
        std::unordered_map<int, std::vector<uint64_t>> usigs;
};
