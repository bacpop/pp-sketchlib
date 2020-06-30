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
#include <cmath>
#include <array>

#include "stHashIterator.hpp"
#include "robin_hood.h"

// Countmin parameters
const unsigned int table_width_bits = 24;
constexpr uint64_t mask{ 0xFFFFFF }; // 24 lowest bits ON (to set 25 0x1FFFFFF; 26 0x3FFFFFF; 27 0x7FFFFFF)
const long table_width = (long)mask; // 2^24 + 1 = 16777216
const unsigned int hash_per_hash = (int)floor(64/table_width_bits); // This should be 2, or the table is likely too narrow
const size_t table_rows = 4; // Number of hashes, should be a multiple of hash_per_hash

class KmerCounter
{
    public:
        KmerCounter(const uint8_t min_count, const size_t num_hashes);
        virtual ~KmerCounter() = 0;

        uint8_t min_count() const { return _min_count; }
        size_t num_hashes() const { return _num_hashes_needed; }

        bool above_min(stHashIterator& hash);
        virtual uint8_t add_count(stHashIterator& hash) = 0;

    protected:
        uint8_t _min_count;
        size_t _num_hashes_needed;

};

class CountMin : public KmerCounter
{
    public:
        CountMin(const uint8_t min_count);

        uint8_t add_count(stHashIterator& hash) override;

    private:
        std::array<std::array<uint8_t, table_width>, table_rows> hash_table;
};

class HashCounter : public KmerCounter
{
    public:
        HashCounter(const uint8_t min_count);

        uint8_t add_count(stHashIterator& hash) override;
        uint8_t probe(stHashIterator& hash);

    private:
        robin_hood::unordered_flat_map<uint64_t, uint8_t> hash_table;
};
