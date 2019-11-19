
#include <chrono>
#include <random>

#include "rollinghashcpp/cyclichash.h"

typedef std::tuple<CyclicHash<uint64_t>, CyclicHash<uint64_t>> rollinghash;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define NBITS(x) (8*sizeof(x))
#define BITATPOS(x, pos) ((x & (1ULL << pos)) >> pos)
const uint64_t SIGN_MOD = (1ULL << 61ULL) - 1ULL; 

const int hashseed = 86;

inline uint64_t doublehash(uint64_t hash1, uint64_t hash2) { return (hash1 + hash2) % SIGN_MOD; }

std::vector<uint64_t> sketch(const std::string & name, 
                             const uint64_t sketchsize, 
                             const size_t kmer_len, 
                             const size_t bbits,
                             const bool isstrandpreserved)
{

    const uint64_t nbins = sketchsize * NBITS(uint64_t);
    const uint64_t binsize = (SIGN_MOD + nbins - 1ULL) / nbins;
    std::vector<uint64_t> usigs(sketchsize * bbits, 0);
    std::vector<uint64_t> signs(sketchsize * NBITS(uint64_t), UINT64_MAX); // carry over
    
    rollinghash hf = init_hashes(kmer_len); 
    hashinit(cbuf, hf, kmer_len);
    
    // Rolling hash through string
    while (cbuf.nseqs() < nseqs) 
    {
        cbuf.eatnext();
        if (cbuf.ceof()) 
        { 
            break; 
        }
        hashupdate(cbuf, signs, hf, isstrandpreserved, binsize);
    }
    
    // Apply densifying function
    int res = densifybin(signs);
    if (res != 0) 
    {
        std::cerr << "Warning: the genome " << name << " is densified with flag " << res <<  std::endl;
    }
    fillusigs(usigs, signs, bbits);
    
    return(usigs)
}

void hashinit(CBuf & cbuf, 
              rollinghash& hf
              size_t kmer_len) 
{
	std::get<0>(hf).reset();
    std::get<1>(hf).reset();
    std::get<0>(hf).hashvalue = 0;
	std::get<1>(hf).hashvalue = 0;
	
    for (size_t k = 0; k < kmerlen; ++k) 
    {
		std::get<0>(hf).eat(cbuf.getith(k));
		std::get<1>(hf).eat(RCMAP[(int)cbuf.getith(kmerlen - k - 1)]);
	}
}

// TODO: Remove hashinit function from main loop, instead call here when buffer rolls
// over to next sequence. Need suitable returns from seqio function calls to find when
// this is needed

void hashupdate(CBuf & cbuf, 
                std::vector<uint64_t> &signs, 
		        rollinghash& hf, 
		        bool isstranspreserved, 
                uint64_t binsize) 
{
	std::get<0>(hf).update(cbuf.getout(), cbuf.getnewest());
	std::get<1>(hf).reverse_update(RCMAP[(int)cbuf.getnewest()], RCMAP[(int)cbuf.getout()]);
	
    if (cbuf.slen >= cbuf.size) {
		auto signval = std::get<0>(hf).hashvalue % SIGN_MOD;
		
        // Take canonical k-mer
        if (!isstranspreserved) 
        {
			auto signval2 = std::get<1>(hf).hashvalue % SIGN_MOD;
			signval = doublehash(signval, signval2);
		}
		binsign(signs, signval, binsize);
	}
}

rollinghash init_hashes(const size_t kmer_len)
{
    std::minstd_rand0 g1(hashseed);
	auto s1 = g1();
	auto s2 = g1();

    // Initialise hash function
    CyclicHash<uint64_t> hf(kmer_len, s1, s2, 64);
    CyclicHash<uint64_t> hfrc(kmer_len, s1, s2, 64);

    return(std::make_tuple(hf, hfrc));
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
#ifdef CANONICAL_DENSIFICATION
	while (sign < signs[binidx]) 
    {
		signs[binidx] = sign;
		if (0 == binidx) { break; }
		binidx--;
	}
#else
	signs[binidx] = MIN(signs[binidx], sign);
#endif
}