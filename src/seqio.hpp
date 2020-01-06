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
#include <nonstd/string_view.hpp> // O(1) substr operations

#include <zlib.h>
#include <stdio.h>
#include <string.h>
#include "kseq.h"

class SeqBuf 
{
    public:
        SeqBuf(const std::vector<std::string>& filenames, const size_t kmer_len);

	    nonstd::string_view get_fwd(const size_t kmer_len) const { return seq_view.substr(current_base, kmer_len); }
	    nonstd::string_view get_rev(const size_t kmer_len) const { return rcseq_view.substr(seq_view.size() - kmer_len - current_base, kmer_len); }
	    size_t nseqs() const { return sequence.size(); }
        bool eof() const { return _end; }
        bool is_reads() const { return _reads; }

        void move_next(const size_t kmer_len);
        void reset();

    private:
        std::vector<std::string> sequence;
        std::vector<std::string> rc_sequence;

        nonstd::string_view seq_view;
        nonstd::string_view rcseq_view;
        size_t current_seq;
        size_t current_base;

        bool _end;
        bool _reads;
};
