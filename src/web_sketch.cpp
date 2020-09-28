
#include <fstream>
#include <iostream>
#include <algorithm> // sort
#include <string>

#include "json.hpp"
using json = nlohmann::json;

#include "version.h"
#include "seqio.hpp"
#include "sketch.hpp"

// Defaults to change when I figure out how to do this
const int sketchsize64 = 156;
const std::vector<size_t> kmer_lengths {15, 18, 21, 24, 27, 31};
const std::vector<std::string> filenames = {"12673_8_24.fa"};
const std::string name = "test";

const int bbits = 14;
const uint8_t min_count = 0;
const bool exact = false;
const bool codon_phased = false;
const bool use_rc = true;

int main (int argc, char* argv[]) {
    printf("Reading\n");
    SeqBuf sequence(filenames, kmer_lengths.back());
    if (sequence.nseqs() == 0) {
        throw std::runtime_error(filenames.at(0) + " contains no sequence");
    }

    KmerSeeds kmer_seeds = generate_seeds(kmer_lengths, false);
    double minhash_sum = 0;
    bool densified = false;
    json sketch_json;

    printf("Sketching\n");
    for (auto kmer_it = kmer_seeds.cbegin(); kmer_it != kmer_seeds.cend(); ++kmer_it) {
        double minhash = 0; bool k_densified;
        std::vector<uint64_t> kmer_sketch;
        std::tie(kmer_sketch, minhash, densified) =
            sketch(sequence, sketchsize64, kmer_it->second, bbits,
                   codon_phased, use_rc, min_count, exact);
        sketch_json[std::to_string(kmer_it->first)] = kmer_sketch;

        minhash_sum += minhash;
        densified |= k_densified; // Densified at any k-mer length
    }
    sketch_json["bbits"] = bbits;
    sketch_json["sketchsize64"] = sketchsize64;
    BaseComp<double> composition = sequence.get_composition();
    sketch_json["length"] = composition.total;
    sketch_json["bases"] = {composition.a, composition.c, composition.g, composition.t};
    sketch_json["missing_bases"] = sequence.missing_bases();
    sketch_json["version"] = SKETCH_VERSION;

    printf("Sketch json\n");
    std::string s = sketch_json.dump();
    std::cout << s << std::endl;

    return 0;
}

