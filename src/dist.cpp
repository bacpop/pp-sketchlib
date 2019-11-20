/*
 *
 * dist.cpp
 * bindash dist method
 *
 */

#include <cstdint>
#include <stdlib.h>

#include "dist.hpp"

#define BITATPOS(x, pos) ((x & (1ULL << pos)) >> pos)
#define NBITS(x) (8*sizeof(x))
#define ROUNDDIV(a, b) (((a) + (b)/2) / (b))

// Start of macros and method copied from https://github.com/kimwalisch/libpopcnt

#ifdef __GNUC__
	#define GNUC_PREREQ(x, y) \
			(__GNUC__ > x || (__GNUC__ == x && __GNUC_MINOR__ >= y))
#else
	#define GNUC_PREREQ(x, y) 0
#endif

#ifndef __has_builtin
	#define __has_builtin(x) 0
#endif

/*
 * This uses fewer arithmetic operations than any other known
 * implementation on machines with fast multiplication.
 * It uses 12 arithmetic operations, one of which is a multiply.
 * http://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
 */
static inline uint64_t popcount64(uint64_t x)
{
	uint64_t m1 = 0x5555555555555555ll;
	uint64_t m2 = 0x3333333333333333ll;
	uint64_t m4 = 0x0F0F0F0F0F0F0F0Fll;
	uint64_t h01 = 0x0101010101010101ll;

	x -= (x >> 1) & m1;
	x = (x & m2) + ((x >> 2) & m2);
	x = (x + (x >> 4)) & m4;

	return (x * h01) >> 56;
}

// End of macros and method copied from https://github.com/kimwalisch/libpopcnt

template <class T>
T non_neg_minus(T a, T b) {
	return a > b ? (a - b) : 0;
}

size_t calc_intersize(const Reference &r1, 
                      const Reference &r2, 
                      const size_t kmer_len, 
                      const size_t sketchsize64, 
                      const size_t bbits) 
{
	// assert (e1.usigs.size() == e2.usigs.size());	
	// assert (e1.usigs.size() == sketchsize64 * bbits);
	size_t samebits = 0;
	const std::vector<uint64_t>& sketch1 = r1.get_sketch(kmer_len);
	const std::vector<uint64_t>& sketch2 = r2.get_sketch(kmer_len);
    
    for (size_t i = 0; i < sketchsize64; i++) 
    {
		uint64_t bits = ~((uint64_t)0ULL);
		// std::cout << "bits = " << std::hex << bits << std::endl;
		for (size_t j = 0; j < bbits; j++) 
        {
			// assert(e1.usigs.size() > i * bbits + j || !fprintf(stderr, "i=%lu j=%lu bbits=%lu vsize=%lu\n", i, j, bbits, e1.usigs.size()));
			bits &= ~(sketch1[i * bbits + j] ^ sketch2[i * bbits + j]);
			// std::cout << " bits = " << std::hex << bits << std::endl;
		}

#if GNUC_PREREQ(4, 2) || __has_builtin(__builtin_popcountll) 
		samebits += __builtin_popcountll(bits);
#else
		samebits += popcount64(bits)
#endif
	}
	// std::cout << " samebits = " << std::hex << samebits << std::endl;
	const size_t maxnbits = sketchsize64 * NBITS(uint64_t); 
	const size_t expected_samebits = (maxnbits >> bbits);
	if (expected_samebits) {
		return samebits;
	}
	size_t ret = non_neg_minus(samebits, expected_samebits);
	return ret * maxnbits / (maxnbits - expected_samebits);
}
