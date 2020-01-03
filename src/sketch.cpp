
/*
 *
 * sketch.cpp
 * bindash sketch method
 *
 */

#include <tuple>
#include <vector>
#include <exception>
#include <memory>
#include <iostream>

#include <xxhash.h>

#include "sketch.hpp"

#include "bitfuncs.hpp"
#include "countmin.hpp"

const uint64_t SIGN_MOD = (1ULL << 61ULL) - 1ULL; 

typedef std::tuple<uint64_t, uint64_t> hash_vals;

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

void binupdate(std::vector<uint64_t> &signs,
               HashCounter * read_counter,
		       hash_vals& hf, 
		       bool isstrandpreserved, 
               uint64_t binsize)
{
    // std::cout << std::get<0>(hf).hashvalue % SIGN_MOD << "\t" << std::get<1>(hf).hashvalue % SIGN_MOD << std::endl;
    auto signval = std::get<0>(hf) % SIGN_MOD;
    
    // Take canonical k-mer
    if (!isstrandpreserved) 
    {
        auto signval2 = std::get<1>(hf) % SIGN_MOD;
        signval = doublehash(signval, signval2);
    }

    if (read_counter == nullptr || read_counter->add_count(signval) == read_counter->min_count())
    {
        binsign(signs, signval, binsize);
    }
}

std::vector<uint64_t> sketch(const std::string & name,
                             SeqBuf &seq,
                             const uint64_t sketchsize, 
                             const size_t kmer_len, 
                             const size_t bbits,
                             const bool isstrandpreserved,
                             const int hashseed,
                             const uint8_t min_count)
{
    const uint64_t nbins = sketchsize * NBITS(uint64_t);
    const uint64_t binsize = (SIGN_MOD + nbins - 1ULL) / nbins;
    std::vector<uint64_t> usigs(sketchsize * bbits, 0);
    std::vector<uint64_t> signs(sketchsize * NBITS(uint64_t), UINT64_MAX); // carry over

    // This is needed as we don't get optional until C++17
    HashCounter * read_counter = nullptr;
    if (seq.is_reads())
    {
        read_counter = new HashCounter(min_count);
    }

    // Rolling hash through string
    while (!seq.eof()) 
    {
        XXH64_hash_t fwd_hash = XXH64(seq.get_fwd(kmer_len).data(), kmer_len * sizeof(char), hashseed);
        XXH64_hash_t rev_hash = XXH64(seq.get_rev(kmer_len).data(), kmer_len * sizeof(char), hashseed);
        hash_vals hf = std::make_tuple(fwd_hash, rev_hash);
        binupdate(signs, read_counter, hf, isstrandpreserved, binsize);
        seq.move_next(kmer_len);
    }

    // Free memory from read_counter
    delete read_counter;

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

