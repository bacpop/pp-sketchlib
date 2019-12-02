
#include <iostream>

#include "reference.hpp"
#include "database.hpp"
#include "api.hpp"

int main (int argc, char* argv[])
{
    // Runs a test of functionality
    
    std::vector<size_t> kmer_lengths {13, 17};
    Reference ref(argv[1], argv[2], kmer_lengths, 32);
    // Reference ref_copy(argv[1], argv[2], kmer_lengths);
    Reference query(argv[3], argv[4], kmer_lengths, 32);

    std::cout << ref.jaccard_dist(ref, 13) << std::endl;      // Should be 1
    // std::cout << ref.dist(ref_copy, 13) << std::endl; // Should be 1 (test of consistent randomness in sketch)
    std::cout << ref.jaccard_dist(query, 13) << std::endl;
    std::cout << ref.jaccard_dist(query, 17) << std::endl;
    std::cout << query.jaccard_dist(ref, 17) << std::endl;

    Database sketch_db("sketch.h5");
    sketch_db.add_sketch(ref);
    sketch_db.add_sketch(query);

    Reference ref_read = sketch_db.load_sketch(argv[1]);
    Reference query_read = sketch_db.load_sketch(argv[3]);
    std::cout << ref_read.jaccard_dist(query_read, 13) << std::endl;

    MatrixXd dists = create_db("full.h5",
                               {argv[1], argv[3]}, 
                               {argv[2], argv[4]}, 
                               kmer_lengths,
                               32,
                               2);
    std::cout << dists << std::endl;

    return 0;
}

