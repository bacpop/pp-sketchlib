/*
*
* countmin.cpp
* Class implementing countmin sketch for reads
*
*/

#include "countmin.hpp"

// C/C++/C++11/C++17 headers
#include <iterator>
#include <limits>

#include "countmin.hpp"

constexpr uint64_t mask{ 0x1FFFFF }; // 21 lowest bits ON

// Constructors

KmerCounter::KmerCounter(const uint8_t min_count)
:_min_count(min_count)
{
}

CountMin::CountMin(const uint8_t min_count)
:KmerCounter(min_count)
{
    for (auto row_it = hash_table.begin(); row_it != hash_table.end(); row_it++)
    {
        row_it->fill(0);
    }
}

HashCounter::HashCounter(const uint8_t min_count)
:KmerCounter(min_count)
{
}

uint8_t CountMin::add_count(uint64_t doublehash)
{
    uint8_t min_count = 0;
    for (unsigned int hash_nr = 0; hash_nr < table_rows; hash_nr++)
    {
        long hash = doublehash & mask;
        doublehash = doublehash >> 21;
        if (hash_table[hash_nr][hash] < std::numeric_limits<uint8_t>::max())
        {
            if (++hash_table[hash_nr][hash] > min_count)
            {
                min_count = hash_table[hash_nr][hash];
            }
        }
        else
        {
            min_count = std::numeric_limits<uint8_t>::max();
        }
        
    }
    return(min_count);
}

bool KmerCounter::above_min(const uint64_t doublehash)
{
    return (add_count(doublehash) > _min_count);
}

uint8_t HashCounter::add_count(uint64_t doublehash)
{
    auto table_val = hash_table.find(doublehash);
    if (table_val == hash_table.end())
    {
        hash_table[doublehash] = 1;
    }
    else if (hash_table[doublehash] < std::numeric_limits<uint8_t>::max())
    {
        table_val->second++;
    }
    return(table_val->second);
}

uint8_t HashCounter::probe(uint64_t doublehash)
{
    return(hash_table[doublehash]);
}
