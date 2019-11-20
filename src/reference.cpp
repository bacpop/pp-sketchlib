/*
 * File: reference.cpp
 *
 * Sketch sequences, save and load
 *
 */

#include "reference.hpp"

#include "seqio.hpp"
#include "sketch.hpp"
#include "dist.hpp"

const size_t def_bbits = 14;
const size_t def_sketchsize64 = 32;
const bool def_isstrandpreserved = false;

#define NBITS(x) (8*sizeof(x))

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

double Reference::dist(const Reference &query, const int kmer_len)
{
    size_t intersize = calc_intersize(*this, query, sketchsize64, kmer_len, bbits);
	size_t unionsize = NBITS(uint64_t) * sketchsize64;
    return(intersize/(double)unionsize);
}

const std::vector<uint64_t> & Reference::get_sketch(const int kmer_len) const
{
    try
    {
        return usigs.at(kmer_len);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Kmer length " << kmer_len << " not found in sketch" << std::endl;
    }
}
