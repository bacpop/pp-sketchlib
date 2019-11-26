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
                             const uint64_t sketchsize, 
                             const size_t kmer_len, 
                             const size_t bbits,
                             const bool isstrandpreserved,
                             const int hashseed);