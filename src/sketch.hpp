/*
 *
 * sketch.hpp
 * bindash sketch method
 *
 */
#pragma once

#include <cstdint>
#include <cstddef>

#include "seqio.hpp"

std::vector<uint64_t> sketch(const std::string & name,
                             SeqBuf &seq,
                             size_t &seq_size,
                             const uint64_t sketchsize, 
                             const size_t kmer_len, 
                             const size_t bbits,
                             const bool use_canonical = true,
                             const uint8_t min_count = 0,
                             const bool exact = false);