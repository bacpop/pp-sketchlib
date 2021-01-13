
#include <fstream>
#include <iostream>
#include <algorithm> // sort
#include <string>

#include <emscripten/bind.h>
using namespace emscripten;

// This is using version in repo, as conda version has bugged version of
// throws
// Change back to <nlohmann/json.hpp> when conda version is updated
#include <json.hpp>
using json = nlohmann::json;

#include "version.h"
#include "sketch/seqio.hpp"
#include "sketch/sketch.hpp"

std::string json_sketch(const std::string file,
                        const size_t kmer_min,
                        const size_t kmer_max,
                        const size_t kmer_step,
                        const int bbits,
                        const int sketchsize64,
                        const bool codon_phased,
                        const bool use_rc)
{
  std::vector<size_t> kmer_lengths;
  for (size_t k = kmer_min; k <= kmer_max; k += kmer_step)
  {
    kmer_lengths.push_back(k);
  }
  KmerSeeds kmer_seeds = generate_seeds(kmer_lengths, codon_phased);

  printf("Reading\n");
  printf("%s\n", file.c_str());
  SeqBuf sequence({file}, kmer_lengths.back());
#ifndef NOEXCEPT
  if (sequence.nseqs() == 0)
  {
    throw std::runtime_error(file + " contains no sequence");
  }
#else
  if (sequence.nseqs() == 0)
  {
    abort();
  }
#endif

  printf("Sketching\n");
  double minhash_sum = 0;
  bool densified = false;
  json sketch_json;
  for (auto kmer_it = kmer_seeds.cbegin(); kmer_it != kmer_seeds.cend(); ++kmer_it)
  {
    double minhash = 0;
    bool k_densified = false;
    std::vector<uint64_t> kmer_sketch;
    std::tie(kmer_sketch, minhash, densified) =
        sketch(sequence, sketchsize64, kmer_it->second, bbits,
               codon_phased, use_rc, 0, false);
    sketch_json[std::to_string(kmer_it->first)] = kmer_sketch;

    minhash_sum += minhash;
    densified |= k_densified; // Densified at any k-mer length
  }
  sketch_json["codon_phased"] = codon_phased;
  sketch_json["densified"] = densified;
  sketch_json["bbits"] = bbits;
  sketch_json["sketchsize64"] = sketchsize64;
  BaseComp<double> composition = sequence.get_composition();
  sketch_json["length"] = composition.total;
  sketch_json["bases"] = {composition.a, composition.c, composition.g, composition.t};
  sketch_json["missing_bases"] = sequence.missing_bases();
  sketch_json["version"] = SKETCH_VERSION;

  std::string s = sketch_json.dump();
  return s;
}

EMSCRIPTEN_BINDINGS(sketchlib)
{
  function("sketch", &json_sketch);
}