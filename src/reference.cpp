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
const size_t def_sketchsize64 = 156;

#include "bitfuncs.hpp"

auto key_selector = [](auto pair){return pair.first;};
// auto value_selector = [](auto pair){return pair.second;};

// Initialisation
Reference::Reference()
   :_bbits(def_bbits),  
    _sketchsize64(def_sketchsize64)
{
}

Reference::Reference(const std::string& name, 
                     const std::vector<std::string>& filenames, 
                     const std::vector<size_t>& kmer_lengths,
                     const size_t sketchsize64,
                     const uint8_t min_count)
   :_name(name), 
    _bbits(def_bbits),  
    _sketchsize64(sketchsize64)
{
    // Read in sequence
    SeqBuf sequence(filenames, kmer_lengths.back());
    if (sequence.nseqs() == 0)
    {
        throw std::runtime_error(filenames.at(0) + " contains no sequence");
    }

    for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++)
    {
        usigs[*kmer_it] = sketch(_name, sequence, sketchsize64, *kmer_it, _bbits, min_count);
    }
    // SeqBuf containing sequences will get deleted here
    // usigs (the sketch) will be retained
}

Reference::Reference(const std::string& name,
                     const size_t bbits,
                     const size_t sketchsize64)
   :_name(name), _bbits(bbits), _sketchsize64(sketchsize64)
{
}

double Reference::jaccard_dist(const Reference &query, const int kmer_len) const
{
    size_t intersize = calc_intersize(&this->get_sketch(kmer_len), 
                                      &query.get_sketch(kmer_len), 
                                      _sketchsize64, 
                                      _bbits);
	size_t unionsize = NBITS(uint64_t) * _sketchsize64;
    double jaccard = intersize/(double)unionsize;
    return(jaccard);
}

std::tuple<float, float> Reference::core_acc_dist(const Reference &query) const
{
    std::vector<size_t> kmers = this->kmer_lengths();
    if (kmers != query.kmer_lengths())
    {
        throw std::runtime_error("Incompatible k-mer lengths");
    }

    std::tuple<float, float> core_acc = regress_kmers(this, 
                                                      &query, 
                                                      kmer2mat(kmers)); 
    return(core_acc);
}

// Without k-mer sizes check
std::tuple<float, float> Reference::core_acc_dist(const Reference &query, 
                                                  const arma::mat &kmers) const
{
    std::tuple<float, float> core_acc = regress_kmers(this, 
                                                      &query, 
                                                      kmers); 
    return(core_acc);
}

const std::vector<uint64_t> & Reference::get_sketch(const int kmer_len) const
{
    try
    {
        return usigs.at(kmer_len);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Kmer length " + std::to_string(kmer_len) + " not found in sketch " + _name);
    }
}

void Reference::add_kmer_sketch(const std::vector<uint64_t>& sketch, const int kmer_len)
{
    usigs[kmer_len] = sketch;
}

void Reference::remove_kmer_sketch(const size_t kmer_len)
{
    usigs.erase(kmer_len);
}

std::vector<size_t> Reference::kmer_lengths() const 
{
    std::vector<size_t> keys(usigs.size());
    std::transform(usigs.begin(), usigs.end(), keys.begin(), key_selector);
    std::sort(keys.begin(), keys.end());
    return keys;
}

template <class T>
arma::mat kmer2mat(const T& kmers)
{
    arma::mat X(kmers.size(), 2, arma::fill::ones);
    for (size_t i = 0; i < kmers.size(); i++)
    {
        X(i, 1) = kmers[i];
    }
    return X;
}