/*
 *
 * seqio.cpp
 * Sequence reader and buffer class
 *
 */

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
#include <utility>
#include <experimental/filesystem>

// seqan3 headers
#include "seqan3/range/view/all.hpp"
#include "seqan3/std/ranges"
#include "seqan3/io/sequence_file/all.hpp"

#include "seqio.hpp"

using namespace seqan3;

SeqBuf::SeqBuf(const std::string& filename)
{
    /* 
    *   Reads in sequence to memory
    */
    sequence_file_input reference_in{filename};
    for (auto & [seq, id, qual] : reference_in)
    {
        // ids.push_back(std::move(id));
        sequence.push_back(std::move(seq));
        rev_comp_sequence.push_back(sequence.back() | std::view::reverse | seqan3::view::complement);
    }

    this->reset();
}

void SeqBuf::reset()
{
    /* 
    *   Returns to start of sequences
    */
    current_seq = sequence.begin();
    current_base = current_seq->begin();
    current_revseq = rev_comp_sequence.begin();
    current_revbase = current_revseq->begin();
    base_out = nullptr;
    revbase_out = nullptr;
    end = false;
}

bool SeqBuf::eat(size_t word_length)
{
    /* 
    *   Moves along to next character in sequence and reverse complement
    *   Loops around to next sequence if end reached
    *   Keeps track of base before k-mer length 
    */
    current_base++;
    current_revbase++;
    
    bool next_seq = false;
    if (current_base == current_seq->end() || current_revbase == current_revseq->end())
    {
        next_seq = true;
        if (current_seq == sequence.end() || current_revseq == rev_comp_sequence.end())
        {
            end = true;
            current_base = nullptr;
            current_revbase = nullptr;
        }
        else
        {
            current_base = ++current_seq->begin();
            current_revbase = ++current_revseq->begin();
        }
        
        base_out = nullptr;
        revbase_out = nullptr;
    }
    else
    {
        if (base_out != nullptr)
        {
            base_out++;
        }
        else if ((current_base - word_length) >= current_seq->begin())
        {
            base_out = current_seq->begin();
        }

        if (revbase_out != nullptr)
        {
            revbase_out++;
        }
        else if ((current_revbase - word_length) >= current_revseq->begin())
        {
            revbase_out = current_revseq->begin();
        }
    } 

    return next_seq;
}


