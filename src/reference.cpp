/*
 * File: reference.cpp
 *
 * Sketch sequences, save and load
 *
 */

#include <iostream>

#include "reference.hpp"

#include "sketch.hpp"
#include "dist.hpp"

const size_t def_bbits = 14;
const size_t def_sketchsize64 = 32;
const bool def_isstrandpreserved = false;

#include "bitfuncs.hpp"

// Initialisation
Reference::Reference(const std::string& _name, 
                     const std::string& filename, 
                     const std::vector<size_t>& kmer_lengths)
   :name(_name), bbits(def_bbits), sketchsize64(def_sketchsize64), isstrandpreserved(def_isstrandpreserved)
{
    // Read in sequence
    SeqBuf sequence(filename, kmer_lengths.back());
    if (sequence.nseqs() == 0)
    {
        throw std::runtime_error(filename + " contains no sequence");
    }

    for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++)
    {
        usigs[*kmer_it] = sketch(_name, sequence, sketchsize64, *kmer_it, bbits, isstrandpreserved);
    }
    // SeqBuf containing sequences will get deleted here
    // usigs (the sketch) will be retained
}

double Reference::dist(const Reference &query, const int kmer_len)
{
    size_t intersize = calc_intersize(&this->get_sketch(kmer_len), 
                                      &query.get_sketch(kmer_len), 
                                      sketchsize64, 
                                      bbits);
	size_t unionsize = NBITS(uint64_t) * sketchsize64;
    double jaccard = intersize/(double)unionsize;
    return(jaccard);
}

const std::vector<uint64_t> & Reference::get_sketch(const int kmer_len) const
{
    try
    {
        return usigs.at(kmer_len);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Kmer length " + std::to_string(kmer_len) + " not found in sketch");
    }
}
