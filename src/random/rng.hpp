// Thread-safe RNG from http://prng.di.unimi.it/ using:
// xoshiro256starstar.c
// splitmix64.c

#pragma once

#include <vector>
#include <climits>
#include <limits>
#include <cstddef>
#include <cstdint>

#define XOSHIRO_WIDTH 4

class Xoshiro
{
public:
  // Definitions to be used as URNG in C++11
  typedef size_t result_type;
  static constexpr size_t min() { return std::numeric_limits<uint64_t>::min(); }
  static constexpr size_t max() { return std::numeric_limits<uint64_t>::max(); }
  uint64_t operator()(); // generate random number U(min, max)

  // Constructor
  Xoshiro(uint64_t seed);

  // Change internal state
  void set_seed(uint64_t seed);
  void jump();
  void long_jump();

private:
  uint64_t _state[XOSHIRO_WIDTH];
};
