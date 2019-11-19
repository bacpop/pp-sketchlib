



#include "seqio.hpp"


SeqBuf::SeqBuf(std::string& filename)
    :buf_idx(0), out('\0')
{
    // TODO make sure the sequence read is not repeated for every k-mer length
    sequence_file_input reference_in{filename};
    for (auto & [seq, id, qual] : reference_in)
    {
        // ids.push_back(std::move(id));
        sequence.push_back(std::move(seq));
        rev_comp_sequence.push_back(sequence.back() | std::view::reverse | seqan3::view::complement);
    }

    reset(self);
}

void SeqBuf::reset()
{
    out = '\0';
    current_seq = sequence.begin();
    current_base = current_seq->begin();
    current_revseq = rev_comp_sequence.begin();
    current_revbase = current_revseq->begin(); 
}



bool CBuf::ceof() {
	return bufidx + 1 >= readsize && readsize < BUFSIZE;
}


uint64_t CBuf::eatnext() {
	unsigned char ch2 = fgetc_visible();
	if ('>' == ch2 || '@' == ch2) {
		while (!ceof() && (ch2 = fgetc_buf()) != '\n'); // { printf("%c,", ch2); }
		slen = 0;
		_nseqs++;
	} else if ('+' == ch2) {
		for (unsigned int i = 0; i < 2; i++) {
			while (!ceof() && (ch2 = fgetc_buf()) != '\n'); // { printf("(%u,%c), ", i, ch2); }
		}
		slen = 0;
	} else {
		slen++;
		chfreqs[(unsigned int)ch2]++;
	}
	return slen;
};


unsigned char SeqBuf::getith(size_t i) {
	return (current_base + i)->to_char();
    buf[(idx + size + i) % size];
};

unsigned char CBuf::getnewest() {
	return buf[(idx + size - 1) % size];
};


