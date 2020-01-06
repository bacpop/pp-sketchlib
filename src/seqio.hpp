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
#include <memory>

#include <zlib.h>
#include <stdio.h>
#include <string.h>
#include "kseq.h"

class SeqBuf 
{
    public:
        SeqBuf(const std::vector<std::string>& filenames, const size_t kmer_len);

	    char const * get_fwd() const { return sequence[current_seq].c_str() + current_base; }
	    char const * get_rev(const size_t kmer_len) const { return rc_sequence[current_seq].c_str() - kmer_len - current_base + rc_sequence[current_seq].size() + 1; }
	    size_t nseqs() const { return sequence.size(); }
        bool eof() const { return _end; }
        bool is_reads() const { return _reads; }

        void move_next(const size_t kmer_len);
        void reset();

    private:
        std::vector<std::string> sequence;
        std::vector<std::string> rc_sequence;

        size_t current_seq;
        size_t current_base;

        bool _end;
        bool _reads;
};
