
#include <fstream>
#include <iostream>

#include "reference.hpp"
#include "database.hpp"
#include "random_match.hpp"
#include "api.hpp"
#include "countmin.hpp"
#include "seqio.hpp"

int main (int argc, char* argv[])
{
    // Runs a test of functionality
    
    std::vector<size_t> kmer_lengths {15, 29};
    Reference ref(argv[1], {argv[2]}, kmer_lengths, 156, 20);
    return 0;
    Reference query(argv[3], {argv[4], argv[5]}, kmer_lengths, 156, 20);

    RandomMC random(true);
    std::cout << ref.jaccard_dist(query, 15, random) << std::endl;
    std::cout << ref.jaccard_dist(query, 29, random) << std::endl;

    auto core_acc = ref.core_acc_dist<RandomMC>(query, random); 
    std::cout << std::get<0>(core_acc) << "\t" << std::get<1>(core_acc) << std::endl;

    return 0;
}

