/*
 * File: reference.cpp
 *
 * Sketch sequences, save and load
 *
 */

#include <iostream>
#include <algorithm>

#include "reference.hpp"

#include "dist/dist.hpp"
#include "random/random_match.hpp"

#include "sketch/bitfuncs.hpp"

auto key_selector = [](auto pair) { return pair.first; };
// auto value_selector = [](auto pair){return pair.second;};

// Initialisation
Reference::Reference()
    : _bbits(def_bbits),
      _sketchsize64(def_sketchsize64),
      _use_rc(true),
      _seq_size(0),
      _densified(false) {}

Reference::Reference(const std::string &name,
                     SeqBuf &sequence,
                     const KmerSeeds &kmers,
                     const size_t sketchsize64,
                     const bool codon_phased,
                     const bool use_rc,
                     const uint8_t min_count,
                     const bool exact)
    : _name(name),
      _bbits(def_bbits),
      _sketchsize64(sketchsize64),
      _use_rc(use_rc),
      _seq_size(0),
      _densified(false)
{
  if (sequence.nseqs() == 0)
  {
    throw std::runtime_error(name + " contains no sequence");
  }
  _bases = sequence.get_composition();
  _missing_bases = sequence.missing_bases();

  double minhash_sum = 0.0;
  for (auto kmer_it = kmers.cbegin(); kmer_it != kmers.cend(); ++kmer_it)
  {
    double minhash = 0;
    bool densified;
    std::tie(usigs[kmer_it->first], minhash, densified) =
        sketch(sequence, sketchsize64, kmer_it->second, _bbits, codon_phased,
               _use_rc, min_count, exact);

    minhash_sum += minhash;
    _densified |= densified; // Densified at any k-mer length
  }

  // 1/N =~ 1/E(Y) where Yi = minhash in [0,1] for k-mer i
  // See https://www.cs.princeton.edu/courses/archive/fall13/cos521/lecnotes/lec4final.pdf
  if (sequence.is_reads())
  {
    _seq_size = static_cast<size_t>((double)usigs.size() / minhash_sum);
  }
  else
  {
    _seq_size = _bases.total;
  }
}

Reference::Reference(const std::string &name,
                     const size_t bbits,
                     const size_t sketchsize64,
                     const size_t seq_size,
                     const std::vector<double> bases,
                     const unsigned long int missing_bases)
    : _name(name), _bbits(bbits), _sketchsize64(sketchsize64), _use_rc(true),
      _seq_size(seq_size), _missing_bases(missing_bases), _densified(false)
{
  _bases.a = bases[0];
  _bases.c = bases[1];
  _bases.g = bases[2];
  _bases.t = bases[3];
}

// Initialise from GPU sketch
Reference::Reference(const std::string &name,
                     robin_hood::unordered_map<int, std::vector<uint64_t>> &sketch,
                     const size_t bbits,
                     const size_t sketchsize64,
                     const size_t seq_size,
                     const BaseComp<double> &bases,
                     const unsigned long int missing_bases,
                     const bool use_rc,
                     const bool densified)
    : _name(name), _bbits(bbits), _sketchsize64(sketchsize64), _use_rc(use_rc),
      _seq_size(seq_size), _missing_bases(missing_bases), _densified(densified),
      _bases(bases), usigs(sketch) {}

double Reference::jaccard_dist(Reference &query, const int kmer_len, const double random_jaccard)
{
  size_t intersize = calc_intersize(&this->get_sketch(kmer_len),
                                    &query.get_sketch(kmer_len),
                                    _sketchsize64,
                                    _bbits);
  size_t unionsize = NBITS(uint64_t) * _sketchsize64;

  double jaccard_obs = intersize / (double)unionsize;
  double jaccard = observed_excess(jaccard_obs, random_jaccard, 1);
  return (jaccard);
}

double Reference::jaccard_dist(Reference &query, const int kmer_len,
                               const RandomMC &random)
{
  return (jaccard_dist(query, kmer_len,
                       random.random_match(*this, query, kmer_len)));
}

template <typename T>
std::tuple<float, float>
Reference::core_acc_dist(Reference &query, const T &random)
{
  std::vector<size_t> kmers = this->kmer_lengths();
  if (kmers != query.kmer_lengths())
  {
    throw std::runtime_error("Incompatible k-mer lengths");
  }

  std::tuple<float, float> core_acc = regress_kmers(this,
                                                    &query,
                                                    kmer2mat(kmers),
                                                    random);
  return (core_acc);
}

// Instantiate templates here so they can be used in other files
template std::tuple<float, float> Reference::core_acc_dist<RandomMC>(Reference &query,
                                                                     const RandomMC &random);
template std::tuple<float, float> Reference::core_acc_dist<std::vector<double>>(Reference &query,
                                                                                const std::vector<double> &random);

// Without k-mer sizes check
template <typename T>
std::tuple<float, float> Reference::core_acc_dist(Reference &query,
                                                  const arma::mat &kmers,
                                                  const T &random)
{
  std::tuple<float, float> core_acc = regress_kmers(this,
                                                    &query,
                                                    kmers,
                                                    random);
  return (core_acc);
}

// Instantiate templates here so they can be used in other files
template std::tuple<float, float> Reference::core_acc_dist<RandomMC>(Reference &query,
                                                                     const arma::mat &kmers,
                                                                     const RandomMC &random);
template std::tuple<float, float> Reference::core_acc_dist<std::vector<double>>(Reference &query,
                                                                                const arma::mat &kmers,
                                                                                const std::vector<double> &random);

const std::vector<uint64_t> &Reference::get_sketch(const int kmer_len) const
{
  try
  {
    return usigs.at(kmer_len);
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error(
        "Kmer length " +
        std::to_string(kmer_len) +
        " not found in sketch " +
        _name);
  }
}

void Reference::add_kmer_sketch(const std::vector<uint64_t> &sketch,
                                const int kmer_len)
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
arma::mat kmer2mat(const T &kmers)
{
  arma::mat X(kmers.size(), 2, arma::fill::ones);
  for (size_t i = 0; i < kmers.size(); i++)
  {
    X(i, 1) = kmers[i];
  }
  return X;
}

template arma::mat
kmer2mat<std::vector<size_t>>(const std::vector<size_t> &kmers);