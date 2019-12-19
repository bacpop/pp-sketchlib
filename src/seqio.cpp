/*
 *
 * seqio.cpp
 * Sequence reader and buffer class
 *
 */

#include "seqio.hpp"
#include <stdint.h>
KSEQ_INIT(gzFile, gzread)

// C/C++/C++11/C++17 headers
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iterator>
#include <utility>

// code from https://stackoverflow.com/questions/735204/convert-a-string-in-c-to-upper-case
char ascii_toupper_char(char c) {
    return ('a' <= c && c <= 'z') ? c^0x20 : c;    // ^ autovectorizes to PXOR: runs on more ports than paddb
}

size_t strtoupper_autovec(char *dst, const char *src) {
    size_t len = strlen(src);
    for (size_t i=0 ; i<len ; ++i) {
        dst[i] = ascii_toupper_char(src[i]);  // gcc does the vector range check with psubusb / pcmpeqb instead of pcmpgtb
    }
    return len;
}

SeqBuf::SeqBuf(const std::vector<std::string>& filenames, const size_t kmer_len)
{
    /* 
    *   Reads entire sequence to memory
    *   May be faster as hashing at multiple k-mers?
    *   May be better to treat a C strings
    */
    _reads = false;
    for (auto name_it = filenames.begin(); name_it != filenames.end(); name_it++)
    {
        // from kseq.h
        gzFile fp = gzopen(name_it->c_str(), "r");
        kseq_t *seq = kseq_init(fp);
        int l;
        while ((l = kseq_read(seq)) >= 0) 
        {
            if (strlen(seq->seq.s) >= kmer_len)
            {
                // Need to allocate memory for long C string array
                char * upper_seq = new char[strlen(seq->seq.s)]; 
                strtoupper_autovec(upper_seq, seq->seq.s);
                sequence.push_back(upper_seq);
                delete[] upper_seq;
            }
            
            // Presence of any quality scores - assume reads as input
            if (seq->qual.l)
            {
                _reads = true;
            }
        }
        
        // If put back into object, move this to destructor below
        kseq_destroy(seq);
        gzclose(fp);
    }
    this->reset();
}

void SeqBuf::reset()
{
    /* 
    *   Returns to start of sequences
    */
    if (sequence.size() > 0)
    {
        current_seq = sequence.begin();
        next_base = current_seq->begin();
        out_base = current_seq->end();
    }
    end = false;
}

bool SeqBuf::move_next(size_t word_length)
{
    /* 
    *   Moves along to next character in sequence and reverse complement
    *   Loops around to next sequence if end reached
    *   Keeps track of base before k-mer length 
    */
    bool next_seq = false;
    if (!end)
    {
        next_base++;
        
        if (next_base == current_seq->end())
        {
            current_seq++;
            next_seq = true;
            if (current_seq == sequence.end())
            {
                end = true;
            }
            else
            {
                next_base = current_seq->begin();
                out_base = current_seq->end();
            }
        }
        else
        {
            if (out_base != current_seq->end())
            {
                out_base++;
            }
            else if ((next_base - word_length) >= current_seq->begin())
            {
                out_base = current_seq->begin();
            }
        }
    }
    return next_seq;
}


