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

CountMin::CountMin(const uint8_t min_count)
:_min_count(min_count)
{
}

uint8_t CountMin::add_count(uint64_t doublehash)
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


// Duplicates code, just used in testing, remove later
uint8_t CountMin::probe(uint64_t doublehash)
{
    return(hash_table[doublehash]);
}

bool CountMin::above_min(const uint64_t doublehash)
{
    return (add_count(doublehash) >= _min_count);
}