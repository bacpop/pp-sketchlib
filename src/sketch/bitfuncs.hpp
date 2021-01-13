/*
 *
 * bitfuncs.hpp
 * inline functions for bit manipulation
 *
 */
#pragma once

#define BITATPOS(x, pos) ((x & (1ULL << pos)) >> pos)
#define NBITS(x) (8 * sizeof(x))
#define ROUNDDIV(a, b) (((a) + (b) / 2) / (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))