/*
 *
 * countmin.hpp
 * Class implementing countmin sketch for reads
 *
 */
#pragma once

// C/C++/C++11/C++17 headers
#include <stdint.h>
#include <cstddef>
#include <unordered_map>

class CountMin 
{
    public:
        CountMin(const uint8_t min_count);

        uint8_t add_count(uint64_t doublehash);
        uint8_t probe(uint64_t doublehash);
        bool above_min(const uint64_t doublehash);

    private:
        std::unordered_map<uint64_t, uint8_t> hash_table;
        uint8_t _min_count;
};
