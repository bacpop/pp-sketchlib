
#include <iostream>

#include "reference.hpp"
#include "database.hpp"

int main (int argc, char* argv[])
{
    std::vector<size_t> kmer_lengths {13, 17};
    Reference ref(argv[1], argv[2], kmer_lengths);
    // Reference ref_copy(argv[1], argv[2], kmer_lengths);
    Reference query(argv[3], argv[4], kmer_lengths);

    std::cout << ref.dist(ref, 13) << std::endl;      // Should be 1
    // std::cout << ref.dist(ref_copy, 13) << std::endl; // Should be 1 (test of consistent randomness in sketch)
    std::cout << ref.dist(query, 13) << std::endl;
    std::cout << ref.dist(query, 17) << std::endl;
    std::cout << query.dist(ref, 17) << std::endl;

    Database sketch_db("sketch.h5");
    sketch_db.add_sketch(ref);
    sketch_db.add_sketch(query);

    return 0;
}

void create_db() 
{
    // Sketches a set of genomes
}

void query_db() 
{
    // Calculates distances against another database
}