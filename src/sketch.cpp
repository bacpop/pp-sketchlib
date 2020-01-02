
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

#include <cyclichash.h>

#include "sketch.hpp"

#include "bitfuncs.hpp"
#include "countmin.hpp"

const uint64_t SIGN_MOD = (1ULL << 61ULL) - 1ULL; 

typedef std::tuple<CyclicHash<uint64_t>, CyclicHash<uint64_t>> rollinghash;

inline uint64_t doublehash(uint64_t hash1, uint64_t hash2) { return (hash1 + hash2) % SIGN_MOD; }

const unsigned char RCMAP[256] = {
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
'T', 66,'G', 68, 69, 70,'C',
 72, 73, 74, 75, 76, 77, 78,
 79, 80, 81,     82, 83,'A',
 85, 86, 87,     88, 89, 90,
 91, 92, 93,     94, 95, 96,
't', 98,'g',100,101,102,'c',
104,105,106,107,108,109,110,
111,112,113,    114,115,'a',
117,118,119,    120,121,122,
123,124,125,126,127,
128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,
144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,
176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,
208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,
240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255};

// Universal hashing function for densifybin
uint64_t univhash(uint64_t s, uint64_t t) 
{
	uint64_t x = (1009) * s + (1000*1000+3) * t;
	return (48271 * x + 11) % ((1ULL << 31) - 1);
}

rollinghash init_hashes(const size_t kmer_len, const int hashseed)
{
    std::minstd_rand0 g1(hashseed);
	auto s1 = g1();
	auto s2 = g1();

    // Initialise hash function
    CyclicHash<uint64_t> hf(kmer_len, s1, s2, 64);
    CyclicHash<uint64_t> hfrc(kmer_len, s1, s2, 64);

    return(std::make_tuple(hf, hfrc));
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
		       rollinghash& hf, 
		       bool isstrandpreserved, 
               uint64_t binsize)
{
    // std::cout << std::get<0>(hf).hashvalue % SIGN_MOD << "\t" << std::get<1>(hf).hashvalue % SIGN_MOD << std::endl;
    auto signval = std::get<0>(hf).hashvalue % SIGN_MOD;
    
    // Take canonical k-mer
    if (!isstrandpreserved) 
    {
        auto signval2 = std::get<1>(hf).hashvalue % SIGN_MOD;
        signval = doublehash(signval, signval2);
    }
    
    if (read_counter == nullptr || read_counter->above_min(signval))
    {
        binsign(signs, signval, binsize);
    }
}

void hashupdate(SeqBuf & seq, 
               std::vector<uint64_t> &signs,
               HashCounter * read_counter, 
		       rollinghash& hf, 
		       bool isstrandpreserved, 
               uint64_t binsize) 
{
    std::get<0>(hf).update(seq.getout(), seq.getnext());
	std::get<1>(hf).reverse_update(RCMAP[(int)seq.getnext()], RCMAP[(int)seq.getout()]);
	// std::cout << seq.getout() << "\t" << seq.getnext() << std::endl;
	// std::cout << RCMAP[(int)seq.getout()] << "\t" << RCMAP[(int)seq.getnext()] << std::endl;

    binupdate(signs, read_counter, hf, isstrandpreserved, binsize);
}

void hashinit(SeqBuf & seq, 
              std::vector<uint64_t> &signs, 
              HashCounter * read_counter,
		      rollinghash& hf, 
		      bool isstrandpreserved, 
              uint64_t binsize,
              size_t kmer_len) 
{
	std::get<0>(hf).reset();
    std::get<1>(hf).reset();
    std::get<0>(hf).hashvalue = 0;
	std::get<1>(hf).hashvalue = 0;
	
    std::string rev_comp;
    for (size_t k = 0; k < kmer_len; ++k) 
    {
        std::get<0>(hf).eat(seq.getnext());
        // std::cout << seq.getnext(); 
        rev_comp += (RCMAP[(int)seq.getnext()]);
        
        // At end of loop this move_next() leaves next unhashed - so call hashupdate below
        bool looped = seq.move_next(kmer_len); 
        if (looped)
        {
            throw std::runtime_error("Hashing sequence shorter than k-mer length");
        }
	}
    // std::cout << std::endl;
    
    for (auto k_rev = rev_comp.crbegin(); k_rev != rev_comp.crend(); k_rev++)
    {
        // std::cout << *k_rev; 
        std::get<1>(hf).eat(*k_rev); 
    }
    // std::cout << std::endl;
    binupdate(signs, read_counter, hf, isstrandpreserved, binsize);
    hashupdate(seq, signs, read_counter, hf, isstrandpreserved, binsize);
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

    rollinghash hf = init_hashes(kmer_len, hashseed); 
    hashinit(seq, signs, read_counter, hf, isstrandpreserved, binsize, kmer_len);
    
    // Rolling hash through string
    while (!seq.eof()) 
    {
        bool looped = seq.move_next(kmer_len);
        if (looped && ! seq.eof()) 
        { 
            hashinit(seq, signs, read_counter, hf, isstrandpreserved, binsize, kmer_len);
        }
        else
        {
            hashupdate(seq, signs, read_counter, hf, isstrandpreserved, binsize);
        }
        
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









