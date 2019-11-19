/*
 *
 * seqio.hpp
 * Header file for seqio
 *
 */

// C/C++/C++11/C++17 headers
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <iterator>
#include <vector>
#include <experimental/filesystem>

// seqan3 headers
#include <seqan3/io/sequence_file/input.hpp>

using namespace seqan3;

// bit function defs
#define BITATPOS(x, pos) ((x & (1ULL << pos)) >> pos)
#define NBITS(x) (8*sizeof(x))
#define ROUNDDIV(a, b) (((a) + (b)/2) / (b))

class Reference
{
    public:
        Reference(std::filesystem::path const & reference_path); // read and run sketch
        
        void save();
        void load();

    private:
        // Info
        std::string name;

        // sketch - map keys are k-mer length
        std::map<int, std::vector<uint64_t>> usigs;
};

// functions
