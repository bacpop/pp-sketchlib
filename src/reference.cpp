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

const size_t def_bbits = 14; // = log2(sketch size) where sketch size = 64 * sketchsize64
const size_t def_sketchsize64 = 156;

#include "bitfuncs.hpp"

auto key_selector = [](auto pair){return pair.first;};
// auto value_selector = [](auto pair){return pair.second;};

// Initialisation
Reference::Reference()
   :_bbits(def_bbits),  
    _sketchsize64(def_sketchsize64),
    _use_rc(true),
    _seq_size(0),
    _densified(false),
    _match_probs(0)
{
}

Reference::Reference(const std::string& name, 
                     const std::vector<std::string>& filenames, 
                     const std::vector<size_t>& kmer_lengths,
                     const size_t sketchsize64,
                     const bool use_rc,
                     const uint8_t min_count,
                     const bool exact)
   :_name(name), 
    _bbits(def_bbits),  
    _sketchsize64(sketchsize64),
    _use_rc(use_rc),
    _seq_size(0),
    _densified(false),
    _match_probs(0)
{
    // Read in sequence
    SeqBuf sequence(filenames, kmer_lengths.back());
    if (sequence.nseqs() == 0)
    {
        throw std::runtime_error(filenames.at(0) + " contains no sequence");
    }
    _bases = sequence.get_composition();
    _missing_bases = sequence.missing_bases();

    double minhash_sum = 0;
    for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++)
    {
        double minhash = 0; bool densified;
        std::tie(usigs[*kmer_it], minhash, densified) = 
            sketch(sequence, sketchsize64, *kmer_it, _bbits, _use_rc, min_count, exact);
        
        minhash_sum += minhash;
        _densified |= densified; // Densified at any k-mer length
    }
    
    // 1/N =~ 1/E(Y) where Yi = minhash in [0,1] for k-mer i
    // See https://www.cs.princeton.edu/courses/archive/fall13/cos521/lecnotes/lec4final.pdf
    if (sequence.is_reads()) {
        _seq_size = static_cast<size_t>((double)kmer_lengths.size() / minhash_sum);
    } else {
        _seq_size = _bases.total;
    }

    // SeqBuf containing sequences will get deleted here
    // usigs (the sketch) will be retained
}

Reference::Reference(const std::string& name,
                     const size_t bbits,
                     const size_t sketchsize64,
                     const size_t seq_size,
                     const std::vector<double> bases
                     const unsigned long int missing_bases)
   :_name(name), _bbits(bbits), _sketchsize64(sketchsize64), _use_rc(true), 
   _seq_size(seq_size), _missing_bases(missing_bases), _densified(false), 
   _match_probs(0)
{
    _bases.a = bases[0];
    _bases.c = bases[1];
    _bases.g = bases[2];
    _bases.t = bases[3];
}

double Reference::random_match(const int kmer_len)
{
    if (_match_probs == 0)
    {
        _match_probs = std::pow(_bases.a, 2) +
                        std::pow(_bases.c, 2) + 
                        std::pow(_bases.g, 2) + 
                        std::pow(_bases.t, 2); 
    }
    double r1 = _seq_size / (_seq_size + std::pow(_match_probs, -kmer_len));
    return r1;
}

double Reference::jaccard_dist(Reference &query, const int kmer_len)
{
    size_t intersize = calc_intersize(&this->get_sketch(kmer_len), 
                                      &query.get_sketch(kmer_len), 
                                      _sketchsize64, 
                                      _bbits);
	size_t unionsize = NBITS(uint64_t) * _sketchsize64;
    double jaccard_obs = intersize/(double)unionsize;
    
    double r1 = this->random_match(kmer_len);
    double r2 = query.random_match(kmer_len);
    double jaccard_expected = (r1 * r2) / (r1 + r2 - r1 * r2);
    
    double jaccard = observed_excess(jaccard_obs, jaccard_expected, 1);
    return(jaccard);
}

std::tuple<float, float> Reference::core_acc_dist(Reference &query)
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
std::tuple<float, float> Reference::core_acc_dist(Reference &query, 
                                                  const arma::mat &kmers)
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