
/*
 *
 * sketch.cu
 * CUDA version of bindash sketch method
 *
 */

#include <thrust/device_vector.h>
#include <thrust/copy.h>

// nthash
#include "nthash_tables.hpp"

// main nthash functions - see nthash.hpp
// All others are built from calling these

__device__
inline uint64_t rol1(const uint64_t v) {
    return (v << 1) | (v >> 63);
}

__device__
inline uint64_t ror1(const uint64_t v) {
    return (v >> 1) | (v << 63);
}

__device__
inline uint64_t rol31(const uint64_t v, unsigned s) {
    s%=31;
    return ((v << s) | (v >> (31 - s))) & 0x7FFFFFFF;
}

__device__
inline uint64_t rol33(const uint64_t v, unsigned s) {
    s%=33;
    return ((v << s) | (v >> (33 - s))) & 0x1FFFFFFFF;
}

__device__
inline uint64_t swapbits033(const uint64_t v) {
    uint64_t x = (v ^ (v >> 33)) & 1;
    return v ^ (x | (x << 33));
}

__device__
inline uint64_t swapbits3263(const uint64_t v) {
    uint64_t x = ((v >> 32) ^ (v >> 63)) & 1;
    return v ^ ((x << 32) | (x << 63));
}

// Forward strand hash for first k-mer
__device__
inline void NT64(const char *kmerSeq, const unsigned k,
                  uint64_t& fhVal,
                  const size_t baseStride) {
    fhVal=0;
    for(int i = k - 1; i >= 0; i--) {
        // Ns are removed, but this is how to check for them
        /*
        if(seedTab[(unsigned char)kmerSeq[i * baseStride]]==seedN) {
            locN=i;
            return false;
        }
        */
        fhVal = rol1(fhVal);
        fhVal = swapbits033(fhVal);
        fhVal ^= seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];
    }
    //return true;
}

// Both strand hashes for first k-mer
__device__
inline void NTC64(const char *kmerSeq, const unsigned k,
                  uint64_t& fhVal, uint64_t& rhVal,
                  uint64_t& hVal,
                  const size_t baseStride) {
    hVal=fhVal=rhVal=0;
    for(int i = (k - 1); i >= 0; i--) {
        // Ns are removed, but this is how to check for them
        /*
        if(seedTab[(unsigned char)kmerSeq[i * baseStride]]==seedN) {
            locN = i;
            return false;
        }
        */
        fhVal = rol1(fhVal);
        fhVal = swapbits033(fhVal);
        fhVal ^= seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];

        rhVal = rol1(rhVal);
        rhVal = swapbits033(rhVal);
        rhVal ^= seedTab[(unsigned char)kmerSeq[i * baseStride]&cpOff];
    }
    hVal = (rhVal<fhVal) ? rhVal : fhVal;
    //return true;
}

// forward-strand ntHash for subsequent sliding k-mers
__device__
inline uint64_t NTF64(const uint64_t fhVal, const unsigned k,
                      const unsigned char charOut, const unsigned char charIn) {
    uint64_t hVal = rol1(fhVal);
    hVal = swapbits033(hVal);
    hVal ^= seedTab[charIn];
    hVal ^= (msTab31l[charOut][k%31] | msTab33r[charOut][k%33]);
    return hVal;
}

// reverse-complement ntHash for subsequent sliding k-mers
__device__
inline uint64_t NTR64(const uint64_t rhVal, const unsigned k,
                      const unsigned char charOut, const unsigned char charIn) {
    uint64_t hVal = rhVal ^ (msTab31l[charIn&cpOff][k%31] | msTab33r[charIn&cpOff][k%33]);
    hVal ^= seedTab[charOut&cpOff];
    hVal = ror1(hVal);
    hVal = swapbits3263(hVal);
    return hVal;
}

// Create a new hash from an nthash
__device__
inline uint64_t shifthash(const uint64_t hVal, const unsigned k) {
    uint64_t tVal = hVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    return(tVal);
}

// Countmin
// parameters - these are currently hard coded based on a short bacterial genome
const unsigned int table_width_bits = 27;
constexpr uint64_t mask{ 0x3FFFFFF }; // 27 lowest bits ON
const uint32_t table_width = (uint32_t)mask; // 2^27 + 1 = 134217729 =~ 134M
const int hash_per_hash = 2; // This should be 2, or the table is likely too narrow
const int table_rows = 4; // Number of hashes, should be a multiple of hash_per_hash

uint16_t add_count_min(uint64_t hash_val, uint16_t * hash_table, const int k) {
    uint16_t min_count = std::numeric_limits<uint16_t>::max();
    for (int hash_nr = 0; hash_nr < table_rows; hash_nr += hash_per_hash) {
        uint64_t current_hash = hash_val;
        for (uint i = 0; i < hash_per_hash; i++)
        {
            uint32_t hash_val_masked = current_hash & mask;
            uint16_t cell_count = atomicInc(hash_table + (hash_nr + i) * table_width + hash_val_masked,
                                            std::numeric_limits<uint16_t>::max());
            if (cell_count < min_count) {
                min_count = cell_count;
            }
            current_hash = current_hash >> table_width_bits;
        }
        hash_val = shifthash(hash_val, k);
    }
    return(min_count);
}

// bindash functions
const uint64_t SIGN_MOD = (1ULL << 61ULL) - 1ULL;

// countmin and binsign
__device__
void binhash(uint64_t *& signs,
             uint16_t *& countmin_table,
             uint64_t sign,
             const uint64_t binsize,
             const int k) {
    sign = sign % SIGN_MOD;
    uint64_t binidx = sign / binsize;
    if (signs[binidx] == UINT64_MAX || sign < signs[binidx]) {
        if (add_count_min(sign, countmin_table, k) >= min_count) {
            signs[binidx] = sign;
        }
    }
}

// hash iterator object
__global__
void process_reads(const unsigned char * read_seq,
                   const size_t n_reads,
                   const size_t read_length,
                   const int k,
                   const uint64_t *& signs,
                   const uint64_t binsize,
                   const bool use_rc,
                   volatile int * blocks_complete) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (long long read_idx = index; dist_idx < n_reads; dist_idx += stride)
	{
        unsigned pos = 0;
        uint64_t fhVal, rhVal, hVal;

        // Set pointers to start of read
        const unsigned char* read_start =
            read_seq + read_idx * read_strides.sample_stride;

        // Get first valid k-mer
        if (use_rc) {
            NTC64(read_start + pos * read_strides.base_stride,
                  k, fhVal, rhVal, hVal, read_strides.base_stride);
            binhash(signs, hVal, binsize);
        } else {
            NT64(read_start + pos * read_strides.base_stride,
                k, fhVal, read_strides.base_stride);
            binhash(signs, hVal, binsize);
        }

        while (pos < read_length - k + 1) {
            fhVal = NTF64(fhVal, k,
                read_start + (pos - 1) * read_strides.base_stride
                read_start + (pos - 1 + k) * read_strides.base_stride);
            if (use_rc) {
                rhVal = NTR64(rhVal, k,
                    read_start + (pos - 1) * read_strides.base_stride
                    read_start + (pos - 1 + k) * read_strides.base_stride);
                hVal = (rhVal<fhVal) ? rhVal : fhVal;
                binhash(signs, hVal, binsize);
            } else {
                binhash(signs, fhVal, binsize);
            }
        }

        // update progress meter
        update_progress(read_idx, n_reads, blocks_complete);
    }
}

// bindash/sketchlib

// main function called here returns signs vector - rest can be done by sketch.cpp
std::vector<uint64_t> get_signs() {
    // Progress meter
	volatile int *blocks_complete;
	cdpErrchk( cudaMallocManaged(&blocks_complete, sizeof(int)) );
    *blocks_complete = 0;

    // TODO
    // Transpose sequence vectors to flattened array
    //      Remove any reads with Ns
    // Create a flattened countmin filter
    // Copy these onto device
    const uint64_t nbins = sketchsize * NBITS(uint64_t);
    const uint64_t binsize = (SIGN_MOD + nbins - 1ULL) / nbins;
    std::vector<uint64_t> signs(sketchsize * NBITS(uint64_t), UINT64_MAX);
    thrust::device_vector<uint64_t> d_signs = signs;

    // Run process_read kernel
    //      This runs nthash on read sequence at all k-mer lengths
    //      Check vs signs and countmin on whether to add each
    //      (get this working for a single k-mer length first)
    const size_t blockSize = selfBlockSize;
    const size_t blockCount = (n_reads + blockSize - 1) / blockSize;
    process_reads<<<blockCount, blockSize>>>
			(
				thrust::raw_pointer_cast(&device_arrays.ref_sketches[0]),
				sketch_subsample.ref_size,
				thrust::raw_pointer_cast(&device_arrays.kmers[0]),
				kmer_lengths.size(),
				thrust::raw_pointer_cast(&device_arrays.dist_mat[0]),
				dist_rows,
				thrust::raw_pointer_cast(&device_arrays.random_table[0]),
				thrust::raw_pointer_cast(&device_arrays.ref_random[0]),
				ref_strides,
				random_strides,
				blocks_complete
            );

    // TODO - add these to separate .cu file and header
    reportProgress(blocks_complete, dist_rows);

    // Copy signs back from device
    try {
        signs = d_signs;
    } catch (thrust::system_error &e) {
		std::cerr << "Error getting result: " << std::endl;
		std::cerr << e.what() << std::endl;
		exit(1);
	}

    fprintf(stderr, "%cProgress (GPU): 100.0%%\n", 13);

    return(signs);
}

