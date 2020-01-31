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
#include <array>

#include "ntHashIterator.hpp"
#include "robin_hood.h"

const long table_width = 16777216; // 2^24
const size_t table_rows = 6;

class KmerCounter 
{
    public:
        KmerCounter(const uint8_t min_count, const size_t num_hashes);
        virtual ~KmerCounter() = 0;

        uint8_t min_count() const { return _min_count; }
        size_t num_hashes() const { return _num_hashes_needed; }

        bool above_min(ntHashIterator& hash);
        virtual uint8_t add_count(ntHashIterator& hash) = 0;

    protected:
        uint8_t _min_count;
        size_t _num_hashes_needed;

};

class CountMin : public KmerCounter 
{
    public:
        CountMin(const uint8_t min_count);

        uint8_t add_count(ntHashIterator& hash) override;
    
    private:
        std::array<std::array<uint8_t, table_rows>, table_width> hash_table;
};

class HashCounter : public KmerCounter 
{
    public:
        HashCounter(const uint8_t min_count);
        
        uint8_t add_count(ntHashIterator& hash) override;
        uint8_t probe(ntHashIterator& hash);

    private:
        robin_hood::unordered_flat_map<uint64_t, uint8_t> hash_table;
};
