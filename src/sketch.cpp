
/*
 *
 * sketch.cpp
 * bindash sketch method
 *
 */

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

size_t estimatesize(std::vector<uint64_t> &signs)
{
    uint64_t minhash = signs[0];
    return(static_cast<size_t>(SIGN_MOD / (double)minhash));
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

std::tuple<std::vector<uint64_t>, size_t, bool> sketch(SeqBuf &seq,
                                                        const uint64_t sketchsize, 
                                                        const size_t kmer_len, 
                                                        const size_t bbits,
                                                        const bool use_canonical,
                                                        const uint8_t min_count,
                                                        const bool exact)
{
    const uint64_t nbins = sketchsize * NBITS(uint64_t);
    const uint64_t binsize = (SIGN_MOD + nbins - 1ULL) / nbins;
    std::vector<uint64_t> usigs(sketchsize * bbits, 0);
    std::vector<uint64_t> signs(sketchsize * NBITS(uint64_t), UINT64_MAX); // carry over

    // nullptr is used as we don't get optional until C++17
    KmerCounter * read_counter = nullptr;
    unsigned h = 1;
    if (seq.is_reads() && min_count > 0)
    {
        if (exact)
        {
            read_counter = new HashCounter(min_count);
        }
        else
        {
            read_counter = new CountMin(min_count);
            h = read_counter->num_hashes();
        }
    }

    // Rolling hash through string
    while (!seq.eof()) 
    {
        ntHashIterator hashIt(*(seq.getseq()), h, kmer_len, use_canonical);
        while (hashIt != hashIt.end())
        {
            auto hash = (*hashIt)[0] % SIGN_MOD;
            if (read_counter == nullptr || read_counter->add_count(hashIt) >= read_counter->min_count())
            {
                binsign(signs, hash, binsize);
            }
            ++hashIt;
        }
        seq.move_next_seq();
    }

    // Free memory from read_counter
    delete read_counter;

    // Apply densifying function
    int densified = densifybin(signs);
    fillusigs(usigs, signs, bbits);
    
    size_t seq_size = estimatesize(signs);
    seq.reset();

    return(std::make_tuple(usigs, seq_size, densified != 0));
}


