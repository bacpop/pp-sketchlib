/*
 *
 * seqio.hpp
 * Sequence reader and buffer class
 *
 */
#pragma once

// C/C++/C++11/C++17 headers
#include <cstddef>
#include <string>
#include <vector>
#include <iterator>

#include <zlib.h>
#include <stdio.h>
#include <string.h>
#include "kseq.h"

class SeqBuf 
{
    public:
        SeqBuf(const std::vector<std::string>& filenames, const size_t kmer_len);

	    unsigned char getnext() const { return *next_base; }
	    unsigned char getout() const { return *out_base; }
	    size_t nseqs() const { return sequence.size(); }
        bool eof() const { return end; }

        bool move_next(size_t word_length);
        void reset();
     

    private:
        std::vector<std::string> sequence;

        std::vector<std::string>::iterator current_seq;
        std::string::iterator next_base;
        std::string::iterator out_base;

        bool end;
};
