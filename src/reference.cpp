/*
 * File: seqio.cpp
 *
 * Uses seqan3 to read in sequence files
 *
 */

#include "reference.hpp"

// Initialisation
Reference::Reference(const std::string& _name, 
                     const std::string& filename, 
                     const std::vector<size_t>& kmer_len)
   :name(_name)
{
    // Read in sequence
    SeqBuf sequence(filename);

    const size_t bbits = 14;
    const size_t sketchsize64 = 32;
    const bool isstranspreserved = false;
    
    for (auto kmer_it = kmer_len.first(); kmer_it != kmer_len.last(); kmer_it++)
    {
        usigs[*kmer_it] = sketch(_name, sketchsize, *kmer_it, bbits)
    }
}
    

