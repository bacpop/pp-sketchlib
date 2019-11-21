/*
 *
 * dist.hpp
 * bindash dist method
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

size_t calc_intersize(const std::vector<uint64_t> * sketch1, 
                      const std::vector<uint64_t> * sketch2, 
                      const size_t kmer_len, 
                      const size_t sketchsize64, 
                      const size_t bbits);