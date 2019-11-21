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

SeqBuf::SeqBuf(const std::string& filename)
{
    /* 
    *   Reads entire sequence to memory
    *   May be faster as hashing at multiple k-mers?
    *   May be better to treat a C strings
    */
    
    // from kseq.h
    fp = gzopen(filename.c_str(), "r");
    kseq_t *seq = kseq_init(fp);
    int l;
    while ((l = kseq_read(seq)) >= 0) 
    {
        sequence.push_back(seq->seq.s);
    }
    
    // If put back into object, move this to destructor below
    kseq_destroy(seq);
}

SeqBuf::~SeqBuf()
{
    gzclose(fp);
}

void SeqBuf::reset()
{
    /* 
    *   Returns to start of sequences
    */
    current_seq = sequence.begin();
    next_base = current_seq->begin();
    out_base = current_seq->end();
    end = false;
}

bool SeqBuf::eat(size_t word_length)
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
            next_seq = true;
            if (current_seq == sequence.end())
            {
                end = true;
            }
            else
            {
                next_base = ++current_seq->begin();
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


