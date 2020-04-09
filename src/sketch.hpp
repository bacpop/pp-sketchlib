/*
 *
 * sketch.hpp
 * bindash sketch method
 *
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <tuple>

#include "seqio.hpp"

std::tuple<std::vector<uint64_t>, size_t, bool> sketch(SeqBuf &seq,
                                                        const uint64_t sketchsize, 
                                                        const size_t kmer_len, 
                                                        const size_t bbits,
                                                        const bool use_canonical = true,
                                                        const uint8_t min_count = 0,
                                                        const bool exact = false);