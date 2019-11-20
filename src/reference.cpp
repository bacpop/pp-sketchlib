/*
 * File: reference.cpp
 *
 * Sketch sequences, save and load
 *
 */

#include "reference.hpp"

#include "seqio.hpp"
#include "sketch.hpp"

// bit function defs
#define BITATPOS(x, pos) ((x & (1ULL << pos)) >> pos)
#define NBITS(x) (8*sizeof(x))
#define ROUNDDIV(a, b) (((a) + (b)/2) / (b))

const size_t def_bbits = 14;
const size_t def_sketchsize64 = 32;
const bool def_isstrandpreserved = false;

// Initialisation
Reference::Reference(const std::string& _name, 
                     const std::string& filename, 
                     const std::vector<size_t>& kmer_lengths)
   :name(_name), bbits(def_bbits), sketchsize64(def_sketchsize64), isstrandpreserved(def_isstrandpreserved)
{
    // Read in sequence
    SeqBuf sequence(filename);

    for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++)
    {
        usigs[*kmer_it] = sketch(_name, sequence, sketchsize64, *kmer_it, bbits, isstrandpreserved);
    }
}
    

