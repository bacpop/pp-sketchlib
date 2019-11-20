
#include <iostream>

#include "reference.hpp"

int main (int argc, char* argv[])
{
    std::vector<size_t> kmer_lengths {13, 17, 21, 25};
    Reference ref("12754_5#72", "~/Documents/listeria/12754_5#72.contigs_velvet.fa", kmer_lengths);
    Reference query("12754_5#71", "~/Documents/listeria/12754_5#71.contigs_velvet.fa", kmer_lengths);

    std::cout << ref.dist(ref, 13) << std::endl;
    std::cout << ref.dist(query, 13) << std::endl;
    std::cout << ref.dist(query, 17) << std::endl;
    std::cout << query.dist(ref, 17) << std::endl;

    return 0;
}