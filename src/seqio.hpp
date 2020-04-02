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

template<class T>
struct BaseComp {
    T a;
    T c;
    T g;
    T t;
    BaseComp():a(0), c(0), g(0), t(0) { }
}

class SeqBuf 
{
    public:
        SeqBuf(const std::vector<std::string>& filenames, const size_t kmer_len);

	    unsigned char getnext() const { return *next_base; }
	    unsigned char getout() const { return *out_base; }
        std::vector<std::string>::iterator getseq() const { return current_seq; }
	    size_t nseqs() const { return sequence.size(); }
        bool eof() const { return end; }
        bool is_reads() const { return _reads; }
        BaseComp get_composition() const { return _bases; }

        bool move_next(size_t word_length);
        void move_next_seq() { ++current_seq; end = current_seq == sequence.end() ? true : false; };
        void reset();
     

    private:
        std::vector<std::string> sequence;

        std::vector<std::string>::iterator current_seq;
        std::string::iterator next_base;
        std::string::iterator out_base;

        BaseComp<double> _bases;

        bool end;
        bool _reads;
};
