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
#include <array>

const long table_width = 2097152; // 2^21
const int table_rows = 3;

class KmerCounter 
{
    public:
        KmerCounter(const uint8_t min_count);
        virtual ~KmerCounter() = 0;

        uint8_t min_count() const { return _min_count; }

        bool above_min(const uint64_t doublehash);
        virtual uint8_t add_count(uint64_t doublehash) = 0;

    protected:
        uint8_t _min_count;
};

class CountMin : public KmerCounter 
{
    public:
        CountMin(const uint8_t min_count);

        uint8_t add_count(uint64_t doublehash) override;
    
    private:
        std::array<std::array<uint8_t, table_rows>, table_width> hash_table;
};

class HashCounter : public KmerCounter 
{
    public:
        HashCounter(const uint8_t min_count);
        
        uint8_t add_count(uint64_t doublehash) override;
        uint8_t probe(uint64_t doublehash);

    private:
        std::unordered_map<uint64_t, uint8_t> hash_table;
};
