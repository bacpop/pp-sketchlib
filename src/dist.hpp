/*
 *
 * dist.hpp
 * bindash dist method
 *
 */

#include <cstddef>

#include "reference.hpp"

size_t calc_intersize(const Reference &r1, 
                      const Reference &r2, 
                      const size_t kmer_len, 
                      const size_t sketchsize64, 
                      const size_t bbits);