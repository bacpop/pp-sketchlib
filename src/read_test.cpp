
#include <fstream>
#include <iostream>

#include "reference.hpp"
#include "database.hpp"
#include "api.hpp"
#include "countmin.hpp"
#include "seqio.hpp"

int main (int argc, char* argv[])
{
    // Runs a test of functionality
    
    std::vector<size_t> kmer_lengths {15, 17, 19, 21, 23, 25, 27, 29};
    Reference ref(argv[1], {argv[2], argv[3]}, kmer_lengths, 156, 20);
    Reference query(argv[3], {argv[4], argv[5]}, kmer_lengths, 156, 20);

    std::cout << ref.jaccard_dist(query, 15) << std::endl;
    std::cout << ref.jaccard_dist(query, 29) << std::endl;

    auto core_acc = ref.core_acc_dist(query); 
    std::cout << std::get<0>(core_acc) << "\t" << std::get<1>(core_acc) << std::endl;

    // Need to test count-min elsewhere

    return 0;
}

