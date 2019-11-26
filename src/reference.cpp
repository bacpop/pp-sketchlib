/*
 * File: reference.cpp
 *
 * Sketch sequences, save and load
 *
 */

#include <iostream>
#include <algorithm>

#include "reference.hpp"

#include "sketch.hpp"
#include "dist.hpp"

const size_t def_bbits = 14;
const size_t def_sketchsize64 = 32;
const bool def_isstrandpreserved = false;
const int def_hashseed = 86;

#include "bitfuncs.hpp"

auto key_selector = [](auto pair){return pair.first;};
auto value_selector = [](auto pair){return pair.second;};

// Initialisation
Reference::Reference(const std::string& name, 
                     const std::string& filename, 
                     const std::vector<size_t>& kmer_lengths)
   :_name(name), _bbits(def_bbits), _sketchsize64(def_sketchsize64), 
    _isstrandpreserved(def_isstrandpreserved), _seed(def_hashseed)
{
    // Read in sequence
    SeqBuf sequence(filename, kmer_lengths.back());
    if (sequence.nseqs() == 0)
    {
        throw std::runtime_error(filename + " contains no sequence");
    }

    for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++)
    {
        usigs[*kmer_it] = sketch(_name, sequence, _sketchsize64, *kmer_it, _bbits, _isstrandpreserved, _seed);
    }
    // SeqBuf containing sequences will get deleted here
    // usigs (the sketch) will be retained
}

double Reference::dist(const Reference &query, const int kmer_len)
{
    size_t intersize = calc_intersize(&this->get_sketch(kmer_len), 
                                      &query.get_sketch(kmer_len), 
                                      _sketchsize64, 
                                      _bbits);
	size_t unionsize = NBITS(uint64_t) * _sketchsize64;
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

const std::vector<int> const Reference::kmer_lengths()
{
    std::vector<int> keys(usigs.size());
    std::transform(usigs.begin(), usigs.end(), keys.begin(), key_selector);
    return keys;
}