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

template <class T>
inline T non_neg_minus(T a, T b)
{
  return a > b ? (a - b) : 0;
}

template <class T, class U>
inline T observed_excess(T obs, T exp, U max)
{
  T diff = non_neg_minus(obs, exp);
  return (diff * max / (max - exp));
}

size_t calc_intersize(const std::vector<uint64_t> *sketch1,
                      const std::vector<uint64_t> *sketch2,
                      const size_t sketchsize64,
                      const size_t bbits);