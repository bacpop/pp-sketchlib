
/*
 *
 * sketch.cpp
 * bindash sketch method
 *
 */

#include <tuple>
#include <random>
#include <vector>
#include <exception>
#include <memory>
#include <unordered_map>
#include <iostream>

#include "ntHashIterator.hpp"

#include "sketch.hpp"

#include "bitfuncs.hpp"
#include "countmin.hpp"

const uint64_t SIGN_MOD = (1ULL << 61ULL) - 1ULL; 

inline uint64_t doublehash(uint64_t hash1, uint64_t hash2) { return (hash1 + hash2) % SIGN_MOD; }

// Universal hashing function for densifybin
uint64_t univhash(uint64_t s, uint64_t t) 
{
	uint64_t x = (1009) * s + (1000*1000+3) * t;
	return (48271 * x + 11) % ((1ULL << 31) - 1);
}

void binsign(std::vector<uint64_t> &signs, 
             const uint64_t sign, 
             const uint64_t binsize) 
{
	uint64_t binidx = sign / binsize;
	signs[binidx] = MIN(signs[binidx], sign);
}

void fillusigs(std::vector<uint64_t>& usigs, const std::vector<uint64_t> &signs, size_t bbits) 
{
	for (size_t signidx = 0; signidx < signs.size(); signidx++) 
    {
		uint64_t sign = signs[signidx];
		int leftshift = (signidx % NBITS(uint64_t));
		for (size_t i = 0; i < bbits; i++) 
        {
			uint64_t orval = (BITATPOS(sign, i) << leftshift);
			usigs[signidx/NBITS(uint64_t) * bbits + i] |= orval;
		}
	}
}

int densifybin(std::vector<uint64_t> &signs) 
{
	uint64_t minval = UINT64_MAX;
	uint64_t maxval = 0;
	for (auto sign : signs) { 
		minval = MIN(minval, sign);
		maxval = MAX(maxval, sign);
	}
	if (UINT64_MAX != maxval) { return 0; }
	if (UINT64_MAX == minval) { return -1; }
	for (uint64_t i = 0; i < signs.size(); i++) 
    {
		uint64_t j = i;
		uint64_t nattempts = 0;
		while (UINT64_MAX == signs[j]) 
        {
			j = univhash(i, nattempts) % signs.size();
			nattempts++;
		}
		signs[i] = signs[j];
	}
	return 1;
}

std::vector<uint64_t> sketch(const std::string & name,
                             SeqBuf &seq,
                             const uint64_t sketchsize, 
                             const size_t kmer_len, 
                             const size_t bbits,
                             const uint8_t min_count)
{
    const uint64_t nbins = sketchsize * NBITS(uint64_t);
    const uint64_t binsize = (SIGN_MOD + nbins - 1ULL) / nbins;
    std::vector<uint64_t> usigs(sketchsize * bbits, 0);
    std::vector<uint64_t> signs(sketchsize * NBITS(uint64_t), UINT64_MAX); // carry over

    // This is needed as we don't get optional until C++17
    HashCounter * read_counter = nullptr;
    CountMin * test_counter = nullptr;
	unsigned h = 1;
    if (seq.is_reads() && min_count > 0)
    {
        read_counter = new HashCounter(min_count);
        test_counter = new CountMin(min_count);
        if (read_counter->num_hashes() > 0)
        {
            h = read_counter->num_hashes(); 
        }
        h = test_counter->num_hashes();
    }

    // Rolling hash through string
    long long added = 0, correct = 0;
    robin_hood::unordered_flat_map<uint64_t, bool> added_table;
    while (!seq.eof()) 
    {
        ntHashIterator hashIt(*(seq.getseq()), h, kmer_len);
        while (hashIt != hashIt.end())
        {
            auto hash = (*hashIt)[0] % SIGN_MOD;
            uint8_t rc = read_counter->add_count(hashIt);
            uint8_t tc = test_counter->add_count(hashIt);
            /* if (tc != rc)
            {
                std::cerr << hash << "\t" << (int)rc << "\t" << (int)tc << std::endl;
            } */
            if (test_counter == nullptr || tc == test_counter->min_count())
            {
                if (added_table.find(hash) == added_table.end())
                {
                    added++;
                    added_table[hash] = true;
                    binsign(signs, hash, binsize);
                }
            }
            if (read_counter == nullptr || rc == read_counter->min_count())
            {
                correct++;
            }
            ++hashIt;
        }
        seq.move_next_seq();
    }

    // Free memory from read_counter
    delete read_counter;
    delete test_counter;
    std::cerr << correct << "\t" << added << std::endl;

    // Apply densifying function
    int res = densifybin(signs);
    if (res != 0) 
    {
        std::cerr << "Warning: the genome " << name << " is densified with flag " << res << std::endl;
    }
    fillusigs(usigs, signs, bbits);
    
    seq.reset();

    return(usigs);
}


