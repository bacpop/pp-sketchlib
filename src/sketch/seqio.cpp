/*
 *
 * seqio.cpp
 * Sequence reader and buffer class
 *
 */

#include "seqio.hpp"
#include <stdint.h>
KSEQ_INIT(gzFile, gzread)

// C++ headers
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <utility>

#include "bitfuncs.hpp"

// Stop counting after 1M bases. Useful for reads
const size_t max_composition_sample = 1000000;

// code from
// https://stackoverflow.com/questions/735204/convert-a-string-in-c-to-upper-case
char ascii_toupper_char(char c) {
  return ('a' <= c && c <= 'z')
             ? c ^ 0x20
             : c; // ^ autovectorizes to PXOR: runs on more ports than paddb
}

void track_composition(const char c, BaseComp<size_t> &bases) {
  switch (c) {
  case 'A':
    bases.a++;
    break;
  case 'C':
    bases.c++;
    break;
  case 'G':
    bases.g++;
    break;
  case 'T':
  case 'U':
    bases.t++;
    break;
  }
}

SeqBuf::SeqBuf() : _N_count(0), _max_length(0), _reads(false) {}

SeqBuf::SeqBuf(const std::vector<std::string> &filenames, const size_t kmer_len)
    : _N_count(0), _max_length(0), _reads(false) {
  /*
   *   Reads entire sequence to memory
   */
  BaseComp<size_t> base_counts = BaseComp<size_t>();

  size_t seq_idx = 0;
  for (auto name_it = filenames.begin(); name_it != filenames.end();
       name_it++) {
    // from kseq.h
    gzFile fp = gzopen(name_it->c_str(), "r");
    kseq_t *seq = kseq_init(fp);
    int l;
    while ((l = kseq_read(seq)) >= 0) {
      size_t seq_len = strlen(seq->seq.s);
      if (seq_len > _max_length) {
        _max_length = seq_len;
      }
      if (seq_len >= kmer_len) {
        sequence.push_back(seq->seq.s);
        bool has_N = false;
        for (char &c : sequence.back()) {
          // Convert all to uppercase. ntHash is ok with either
          // but this saves two checks on base compositions
          c = ascii_toupper_char(c);
          base_counts.total++;
          if (base_counts.total < max_composition_sample) {
            track_composition(c, base_counts);
          }
          if (c == 'N') {
            _N_count++;
            has_N = true;
          }
        }
        if (!has_N) {
          _full_index.push_back(seq_idx);
        }
        seq_idx++;
      }

      // Presence of any quality scores - assume reads as input
      if (!_reads && seq->qual.l) {
        _reads = true;
      }
    }

    // If put back into object, move this to destructor below
    kseq_destroy(seq);
    gzclose(fp);
  }
  double total = (double)(MIN(base_counts.total, max_composition_sample));

  _bases.a = base_counts.a / (double)total;
  _bases.c = base_counts.c / (double)total;
  _bases.g = base_counts.g / (double)total;
  _bases.t = base_counts.t / (double)total;
  _bases.total = base_counts.total;

  this->reset();
}

SeqBuf::SeqBuf(const std::vector<std::string> &sequence_in)
    : sequence(sequence_in), _reads(false) {
  this->reset();
}

void SeqBuf::reset() {
  /*
   *   Returns to start of sequences
   */
  if (sequence.size() > 0) {
    current_seq = sequence.begin();
    next_base = current_seq->begin();
    out_base = current_seq->end();
  }
  end = false;
}

void SeqBuf::load_seqs(std::vector<char> &buffer, const size_t start,
                       const size_t end) const {
  size_t buf_idx = 0;
  for (size_t read_idx = start; read_idx < n_full_seqs() && read_idx < end;
       read_idx++) {
    std::string seq = sequence[_full_index[read_idx]];
    std::memcpy(buffer.data() + buf_idx, seq.data(), seq.size());
    buf_idx += seq.size();
  }
}

bool SeqBuf::move_next(size_t word_length) {
  /*
   *   Moves along to next character in sequence and reverse complement
   *   Loops around to next sequence if end reached
   *   Keeps track of base before k-mer length
   */
  bool next_seq = false;
  if (!end) {
    next_base++;

    if (next_base == current_seq->end()) {
      current_seq++;
      next_seq = true;
      if (current_seq == sequence.end()) {
        end = true;
      } else {
        next_base = current_seq->begin();
        out_base = current_seq->end();
      }
    } else {
      if (out_base != current_seq->end()) {
        out_base++;
      } else if ((next_base - word_length) >= current_seq->begin()) {
        out_base = current_seq->begin();
      }
    }
  }
  return next_seq;
}
