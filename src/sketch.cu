
/*
 *
 * sketch.cu
 * CUDA version of bindash sketch method
 *
 */

#include <stdint.h>

#include "gpu.hpp"
#include "cuda.cuh"

// nthash
#include "nthash_tables.hpp"

// Tables on device
__constant__ uint64_t d_seedTab[256];
__constant__ uint64_t d_A33r[33];
__constant__ uint64_t d_A31l[31];
__constant__ uint64_t d_C33r[33];
__constant__ uint64_t d_C31l[31];
__constant__ uint64_t d_G33r[33];
__constant__ uint64_t d_G31l[31];
__constant__ uint64_t d_T33r[33];
__constant__ uint64_t d_T31l[31];
__constant__ uint64_t d_N33r[33];
__constant__ uint64_t d_N31l[31];
__constant__ uint64_t * d_msTab33r[256];
__constant__ uint64_t * d_msTab31l[256];

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
        fhVal ^= d_seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];
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
        fhVal ^= d_seedTab[(unsigned char)kmerSeq[(k - 1 - i) * baseStride]];

        rhVal = rol1(rhVal);
        rhVal = swapbits033(rhVal);
        rhVal ^= d_seedTab[(unsigned char)kmerSeq[i * baseStride]&cpOff];
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
    hVal ^= d_seedTab[charIn];
    hVal ^= (d_msTab31l[charOut][k%31] | d_msTab33r[charOut][k%33]);
    return hVal;
}

// reverse-complement ntHash for subsequent sliding k-mers
__device__
inline uint64_t NTR64(const uint64_t rhVal, const unsigned k,
                      const unsigned char charOut, const unsigned char charIn) {
    uint64_t hVal = rhVal ^ (d_msTab31l[charIn&cpOff][k%31] | d_msTab33r[charIn&cpOff][k%33]);
    hVal ^= d_seedTab[charOut&cpOff];
    hVal = ror1(hVal);
    hVal = swapbits3263(hVal);
    return hVal;
}

// Create a new hash from an nthash
__device__
inline uint64_t shifthash(const uint64_t hVal, const unsigned k,
                          const unsigned i) {
    uint64_t tVal = hVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    return(tVal);
}

// parameters - these are currently hard coded based on a short bacterial genome
const unsigned int table_width_bits = 27; // 2^27 + 1 = 134217729 =~ 134M
constexpr uint64_t mask{ 0x7FFFFFF }; // 27 lowest bits ON
const uint32_t table_width = static_cast<uint32_t>(mask);
const int hash_per_hash = 2; // This should be 2, or the table is likely too narrow
const int table_rows = 4; // Number of hashes, should be a multiple of hash_per_hash
constexpr size_t table_cells = table_rows * table_width;

// Countmin
// See countmin.cpp
GPUCountMin::GPUCountMin() :
         _table_width_bits(table_width_bits),
         _mask(mask),
         _table_width(table_width),
         _hash_per_hash(hash_per_hash),
         _table_rows(table_rows),
         _table_cells(table_cells) {
    CUDA_CALL(cudaMalloc((void**)&_d_countmin_table, table_cells * sizeof(unsigned int)));
    reset();
}

GPUCountMin::~GPUCountMin() {
    CUDA_CALL( cudaFree(_d_countmin_table));
}

// Loop variables are global constants defined in gpu.hpp
__device__
unsigned int add_count_min(unsigned int * d_countmin_table,
                           uint64_t hash_val, const int k) {
    unsigned int min_count = UINT32_MAX;
    for (int hash_nr = 0; hash_nr < table_rows; hash_nr += hash_per_hash) {
        uint64_t current_hash = hash_val;
        for (uint i = 0; i < hash_per_hash; i++)
        {
            uint32_t hash_val_masked = current_hash & mask;
            unsigned int cell_count =
                atomicInc(d_countmin_table + (hash_nr + i) * table_width +
                          hash_val_masked, UINT32_MAX) + 1;
            if (cell_count < min_count) {
                min_count = cell_count;
            }
            __syncwarp();
            current_hash = current_hash >> 27;
        }
        hash_val = shifthash(hash_val, k, hash_nr / 2);
    }
    return(min_count);
}

void GPUCountMin::reset() {
    CUDA_CALL(cudaMemset(_d_countmin_table, 0, table_cells * sizeof(unsigned int)));
}

// bindash functions
const uint64_t SIGN_MOD = (1ULL << 61ULL) - 1ULL;

// countmin and binsign
__device__
void binhash(uint64_t * signs,
             unsigned int * countmin_table,
             const uint64_t hash,
             const uint64_t binsize,
             const int k,
             const uint16_t min_count) {
    uint64_t sign = hash % SIGN_MOD;
    uint64_t binidx = sign / binsize;
    //printf("binidx:%llu sign:%llu\n", binidx, sign);

    // Only consider if the bin is yet to be filled, or is min in bin
    if (signs[binidx] == UINT64_MAX || sign < signs[binidx]) {
        if (add_count_min(countmin_table, hash,  k) >= min_count) {
            signs[binidx] = sign;
        }
    }
    __syncwarp();
}

// hash iterator object
__global__
void process_reads(char * read_seq,
                   const size_t n_reads,
                   const size_t read_length,
                   const int k,
                   uint64_t * signs,
                   const uint64_t binsize,
                   unsigned int * countmin_table,
                   const bool use_rc,
                   const uint16_t min_count,
                   volatile int * blocks_complete) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (long long read_idx = index; read_idx < n_reads; read_idx += stride)
	{
        uint64_t fhVal, rhVal, hVal;

        // Set pointers to start of read
        //printf("k:%d read_idx: %lld\n", (int)k, read_idx);
        char* read_start = read_seq + read_idx;

        // Get first valid k-mer
        if (use_rc) {
            NTC64(read_start,
                  k, fhVal, rhVal, hVal, n_reads);
            binhash(signs, countmin_table, hVal, binsize, k, min_count);
        } else {
            NT64(read_start,
                 k, fhVal, n_reads);
            binhash(signs, countmin_table, hVal, binsize, k, min_count);
        }

        // Roll through remaining k-mers in the read
        for (int pos = 0; pos < read_length - k; pos++) {
            fhVal = NTF64(fhVal, k,
                read_start[pos * n_reads],
                read_start[(pos + k) * n_reads]);
            if (use_rc) {
                rhVal = NTR64(rhVal, k,
                    read_start[pos * n_reads],
                    read_start[(pos + k) * n_reads]);
                hVal = (rhVal<fhVal) ? rhVal : fhVal;
                binhash(signs, countmin_table, hVal, binsize, k, min_count);
            } else {
                binhash(signs, countmin_table, fhVal, binsize, k, min_count);
            }
        }

        // update progress meter
        update_progress(read_idx, n_reads, blocks_complete);
        __syncwarp();
    }
}

// Writes a progress meter using the device int which keeps
// track of completed jobs
void reportSketchProgress(volatile int * blocks_complete,
                          int k,
                          long long n_reads) {
	long long progress_blocks = 1 << progressBitshift;
	int now_completed = 0; float kern_progress = 0;
	if (n_reads > progress_blocks) {
		while (now_completed < progress_blocks - 1) {
			if (*blocks_complete > now_completed) {
				now_completed = *blocks_complete;
				kern_progress = now_completed / (float)progress_blocks;
                fprintf(stderr, "%ck = %d (%.1lf%%)", 13, k, kern_progress * 100);
			} else {
				usleep(1000);
			}
		}
	}
}


DeviceReads::DeviceReads(const SeqBuf& seq_in,
                         const size_t n_threads): d_reads(nullptr) {
    CUDA_CALL(cudaFree(d_reads)); // Initialises device if needed

    std::vector<char> flattened_reads = seq_in.as_square_array(n_threads);
    n_reads = seq_in.n_full_seqs();
    read_length = seq_in.max_length();
    CUDA_CALL( cudaMalloc((void**)&d_reads, flattened_reads.size() * sizeof(char)));
    CUDA_CALL( cudaMemcpy(d_reads, flattened_reads.data(), flattened_reads.size() * sizeof(char),
                            cudaMemcpyDefault));
}

DeviceReads::~DeviceReads() {
    CUDA_CALL(cudaFree(d_reads));
}

void copyNtHashTablesToDevice() {
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_seedTab, seedTab, 256 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_A33r, A33r, 33 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_A31l, A31l, 31 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_C33r, C33r, 33 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_C31l, C31l, 31 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_G33r, G33r, 33 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_G31l, G31l, 31 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_T33r, T33r, 33 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_T31l, T31l, 31 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_N33r, N33r, 33 * sizeof(uint64_t)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_N31l, N31l, 31 * sizeof(uint64_t)));
    CUDA_CALL(cudaDeviceSynchronize());

    uint64_t *A33r_ptr, *A31l_ptr,
             *C33r_ptr, *C31l_ptr,
             *G33r_ptr, *G31l_ptr,
             *T33r_ptr, *T31l_ptr,
             *N33r_ptr, *N31l_ptr;
    CUDA_CALL(cudaGetSymbolAddress((void**)&A33r_ptr, d_A33r));
    CUDA_CALL(cudaGetSymbolAddress((void**)&A31l_ptr, d_A31l));
    CUDA_CALL(cudaGetSymbolAddress((void**)&C33r_ptr, d_C33r));
    CUDA_CALL(cudaGetSymbolAddress((void**)&C31l_ptr, d_C31l));
    CUDA_CALL(cudaGetSymbolAddress((void**)&G33r_ptr, d_G33r));
    CUDA_CALL(cudaGetSymbolAddress((void**)&G31l_ptr, d_G31l));
    CUDA_CALL(cudaGetSymbolAddress((void**)&T33r_ptr, d_T33r));
    CUDA_CALL(cudaGetSymbolAddress((void**)&T31l_ptr, d_T31l));
    CUDA_CALL(cudaGetSymbolAddress((void**)&N33r_ptr, d_N33r));
    CUDA_CALL(cudaGetSymbolAddress((void**)&N31l_ptr, d_N31l));

    static const uint64_t *d_addr_msTab33r[256] = {
        N33r_ptr, T33r_ptr, N33r_ptr, G33r_ptr, A33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr, // 0..7
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 8..15
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 16..23
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 24..31
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 32..39
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 40..47
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 48..55
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 56..63
        N33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, G33r_ptr, // 64..71
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 72..79
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, T33r_ptr, T33r_ptr, N33r_ptr, N33r_ptr, // 80..87
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 88..95
        N33r_ptr, A33r_ptr, N33r_ptr, C33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, G33r_ptr, // 96..103
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 104..111
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, T33r_ptr, T33r_ptr, N33r_ptr, N33r_ptr, // 112..119
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 120..127
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 128..135
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 136..143
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 144..151
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 152..159
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 160..167
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 168..175
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 176..183
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 184..191
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 192..199
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 200..207
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 208..215
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 216..223
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 224..231
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 232..239
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, // 240..247
        N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r_ptr, N33r  // 248..255
    };

    static const uint64_t *d_addr_msTab31l[256] = {
        N31l_ptr, T31l_ptr, N31l_ptr, G31l_ptr, A31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr, // 0..7
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 8..15
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 16..23
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 24..31
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 32..39
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 40..47
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 48..55
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 56..63
        N31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, G31l_ptr, // 64..71
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 72..79
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, T31l_ptr, T31l_ptr, N31l_ptr, N31l_ptr, // 80..87
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 88..95
        N31l_ptr, A31l_ptr, N31l_ptr, C31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, G31l_ptr, // 96..103
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 104..111
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, T31l_ptr, T31l_ptr, N31l_ptr, N31l_ptr, // 112..119
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 120..127
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 128..135
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 136..143
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 144..151
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 152..159
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 160..167
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 168..175
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 176..183
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 184..191
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 192..199
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 200..207
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 208..215
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 216..223
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 224..231
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 232..239
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, // 240..247
        N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l_ptr, N31l  // 248..255
    };

    CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab31l, d_addr_msTab31l, 256 * sizeof(uint64_t*)));
    CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab33r, d_addr_msTab33r, 256 * sizeof(uint64_t*)));
    CUDA_CALL(cudaDeviceSynchronize());
}

// main function called here returns signs vector - rest can be done by sketch.cpp
std::vector<uint64_t> get_signs(DeviceReads& reads, // use seqbuf.as_square_array() to get this
                                GPUCountMin& countmin,
                                const int k,
                                const bool use_rc,
                                const uint16_t min_count,
                                const uint64_t binsize,
                                const uint64_t nbins) {
    // Progress meter
	volatile int *blocks_complete;
	CUDA_CALL( cudaMallocManaged(&blocks_complete, sizeof(int)) );
    *blocks_complete = 0;

    // Set countmin to zero (already on device)
    countmin.reset();

    // Signs
    std::vector<uint64_t> signs(nbins, UINT64_MAX);
    uint64_t * d_signs;
    CUDA_CALL( cudaMalloc((void**)&d_signs, nbins * sizeof(uint64_t)));
    CUDA_CALL( cudaMemcpy(d_signs, signs.data(), nbins * sizeof(uint64_t),
                          cudaMemcpyDefault));

    // Run process_read kernel
    //      This runs nthash on read sequence at all k-mer lengths
    //      Check vs signs and countmin on whether to add each
    //      (get this working for a single k-mer length first)
    const size_t blockSize = 32;
    const size_t blockCount = (reads.count() + blockSize - 1) / blockSize;
    process_reads<<<blockCount, blockSize>>> (
        reads.read_ptr(),
        reads.count(),
        reads.length(),
        k,
        d_signs,
        binsize,
        countmin.get_table(),
        use_rc,
        min_count,
        blocks_complete
    );

    reportSketchProgress(blocks_complete, k, reads.count());

    // Copy signs back from device
    cudaDeviceSynchronize();
    CUDA_CALL( cudaMemcpy(signs.data(), d_signs, nbins * sizeof(uint64_t),
                          cudaMemcpyDefault));
    CUDA_CALL( cudaFree(d_signs));

    fprintf(stderr, "%ck = %d (100%%)", 13, k);

    return(signs);
}

