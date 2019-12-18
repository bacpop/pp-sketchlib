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

const long table_width = 2097152; // 2^21
const int table_rows = 3;

class CountMin 
{
    public:
        CountMin(const uint8_t min_count);

        uint8_t add_count(uint64_t doublehash);
        bool above_min(const uint64_t doublehash);

    private:
        std::array<std::array<uint8_t, table_rows>, table_width> hash_table;
        uint8_t _min_count;
};
