
#include <fstream>
#include <iostream>
#include <algorithm> // sort

#include "seqio.hpp"
#include "sketch.hpp"

// Defaults to change when I figure out how to do this
const int sketchsize64 = 156;
const std::vector<size_t> kmer_lengths {15, 18, 21, 24, 27, 31};
const std::vector<std::string> filenames = {"12673_8_24.fa"};
const std::string name = "test";

const int bbits = 156;
const int min_count = 0;
const bool exact = false;

int main (int argc, char* argv[])
{
    printf("Reading\n");
    SeqBuf sequence(filenames, kmer_lengths.back());
    if (sequence.nseqs() == 0)
    {
        throw std::runtime_error(filenames.at(0) + " contains no sequence");
    }

    printf("Sketching\n");
    const size_t sketch_size = sketchsize64 * bbits;
    std::vector<uint64_t> sketches(sketch_size * kmer_lengths.size());
    auto sketch_it = sketches.begin();
    for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++)
    {
        std::vector<uint64_t> kmer_sketch = sketch(name, sequence, sketchsize64, *kmer_it, bbits, min_count, exact);
        std::copy(kmer_sketch.begin(), kmer_sketch.end(), sketch_it);
        sketch_it += sketch_size;
    }

    printf("First bin\n");
    std::cout << sketches[0] << std::endl;

    return 0;
}

