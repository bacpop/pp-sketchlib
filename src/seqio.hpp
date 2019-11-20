/*
 *
 * seqio.hpp
 * Sequence reader and buffer class
 *
 */

// C/C++/C++11/C++17 headers
#define __STDC_FORMAT_MACROS
#include <string>
#include <vector>

// seqan3 headers
#include <seqan3/alphabet/all.hpp>

class SeqBuf 
{
    public:
        SeqBuf(const std::string& filename);
        
	    unsigned char getout() const { return base_out->to_char(); }
	    unsigned char getrevout() const { return revbase_out->to_char(); }
	    unsigned char getnext() const { return current_base->to_char(); }
	    unsigned char getrevnext() const { return current_revbase->to_char(); }
	    size_t nseqs() const { return sequence.size(); }
        bool eof() const { return end; }

        bool eat(size_t word_length);
        void reset();
     

    private:
    	std::vector<seqan3::alphabet::nucelotide::dna5::dna5_vector> sequence;
        std::vector<seqan3::alphabet::nucelotide::dna5::dna5_vector> rev_comp_sequence;

        std::unique_ptr<seqan3::alphabet::nucelotide::dna5::dna5_vector> current_seq;
        std::unique_ptr<seqan3::alphabet::nucelotide::dna5::dna5_vector> current_revseq;
        std::unique_ptr<seqan3::alphabet::nucelotide::dna5> current_base;
        std::unique_ptr<seqan3::alphabet::nucelotide::dna5> current_revbase;
        std::unique_ptr<seqan3::alphabet::nucelotide::dna5> base_out;
        std::unique_ptr<seqan3::alphabet::nucelotide::dna5> revbase_out;

        bool end;
};
