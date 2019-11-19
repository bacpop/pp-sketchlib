
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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

class SeqBuf 
{
    public:
        SeqBuf(std::string& filename);
        
	    unsigned char getout() const { return out; }
	    size_it nseqs() const { return sequence.size(); }
     
        uint64_t eatnext();
	    unsigned char getnewest();
	    unsigned char getith(size_t i);


    private:
    	std::vector<dna5_vector> sequence;
        std::vector<dna5_vector> rev_comp_sequence;

        std::unique_ptr<dna5_vector> current_seq;
        std::unique_ptr<dna5_vector> current_revseq;
        std::unique_ptr<dna5> current_base;
        std::unique_ptr<dna5> current_revbase;
        
        char out;
};

void read_fasta(std::filesystem::path const & reference_path,
                    assembly_storage_t & storage);