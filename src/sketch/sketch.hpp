/*
 *
 * sketch.hpp
 * bindash sketch method
 *
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

#include "robin_hood.h"

#include "seqio.hpp"

typedef robin_hood::unordered_flat_map<size_t, std::vector<unsigned>> KmerSeeds;

KmerSeeds generate_seeds(std::vector<size_t> kmer_lengths,
                         const bool codon_phased);

std::tuple<std::vector<uint64_t>, double, bool>
sketch(SeqBuf &seq, const uint64_t sketchsize,
       const std::vector<unsigned> &kmer_seed, const size_t bbits,
       const bool codon_phased = false, const bool use_canonical = true,
       const uint8_t min_count = 0, const bool exact = false);

#ifdef GPU_AVAILABLE
class GPUCountMin;

std::tuple<robin_hood::unordered_map<int, std::vector<uint64_t>>, size_t, bool>
sketch_gpu(const std::shared_ptr<SeqBuf> &seq, GPUCountMin &countmin,
           const uint64_t sketchsize, const std::vector<size_t> &kmer_lengths,
           const size_t bbits, const bool use_canonical,
           const uint8_t min_count, const size_t sample_n,
           const size_t cpu_threads, const size_t shared_size);
#endif