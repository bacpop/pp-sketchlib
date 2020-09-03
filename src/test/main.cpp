
#include <fstream>
#include <iostream>

#include "reference.hpp"
#include "database/database.hpp"
#include "api.hpp"

int main (int argc, char* argv[])
{
    // Runs a test of functionality

    std::vector<size_t> kmer_lengths {15, 17, 19, 21, 23, 25, 27, 29};
    KmerSeeds kmer_seeds = generate_seeds(kmer_lengths, false);
    SeqBuf ref_seq({argv[2]}, kmer_lengths.back());
    Reference ref(argv[1], ref_seq, kmer_seeds, 156, false, true, 0, false);
    // Reference ref_copy(argv[1], argv[2], kmer_lengths);
    SeqBuf query_seq({argv[4]}, kmer_lengths.back());
    Reference query(argv[3], query_seq, kmer_seeds, 156, false, true, 0, false);

    RandomMC random_match(true);

    std::cout << ref.jaccard_dist(ref, 15, random_match) << std::endl;      // Should be 1
    std::cout << ref.jaccard_dist(query, 15, random_match) << std::endl;
    std::cout << ref.jaccard_dist(query, 29, random_match) << std::endl;
    std::cout << query.jaccard_dist(ref, 29, random_match) << std::endl;

    auto core_acc = ref.core_acc_dist<RandomMC>(query, random_match);
    std::cout << std::get<0>(core_acc) << "\t" << std::get<1>(core_acc) << std::endl;

    Database sketch_db("sketch.h5");
    sketch_db.add_sketch(ref);
    sketch_db.add_sketch(query);

    Reference ref_read = sketch_db.load_sketch(argv[1]);
    Reference query_read = sketch_db.load_sketch(argv[3]);
    std::cout << ref_read.jaccard_dist(query_read, 15, random_match) << std::endl;

    std::vector<Reference> ref_sketches = create_sketches("full",
                               {argv[1], argv[3]},
                               {{argv[2]}, {argv[4]}},
                               kmer_lengths,
                               156,
                               false,
                               true,
                               0,
                               false,
                               2);
    RandomMC basic_adjust(true);
    NumpyMatrix dists = query_db(ref_sketches,
                              ref_sketches,
                              kmer_lengths,
                              basic_adjust,
                              false,
                              2);

    std::cout << dists << std::endl;

    std::ifstream rfile(argv[5]);
    std::string name, file;
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> files;
    while (rfile >> name >> file)
    {
        names.push_back(name);
        std::vector<std::string> file_list = {file};
        files.push_back(file_list);
    }

    create_sketches("listeria",
                    names,
                    files,
                    kmer_lengths,
                    156,
                    false,
                    true,
                    0,
                    false,
                    4);

    HighFive::File h5_db = open_h5("listeria.h5");
    Database listeria_db(h5_db);
    std::vector<Reference> listeria_sketches;
    for (auto name_it = names.cbegin(); name_it != names.cend(); name_it++)
    {
        listeria_sketches.push_back(listeria_db.load_sketch(*name_it));
    }

    // Save random matches to db
    RandomMC random(listeria_sketches, 3, 5, false, true, 4);
    try {
        listeria_db.save_random(random);
    } catch (const std::exception& e) {
        std::cerr << "Not writing random matches" << std::endl;
        std::cout << e.what() << std::endl;
    }
    RandomMC random_retrived = listeria_db.load_random(true);
    if (random_retrived != random) {
        throw std::runtime_error("Saving and loading random from DB failed!");
    }

    NumpyMatrix listeria_dists = query_db(listeria_sketches,
                            listeria_sketches,
                            kmer_lengths,
                            random,
                            false,
                            4);
    std::cout << listeria_dists << std::endl;

    // Check ref v query works
    std::vector<Reference> query_sketches(2);
    std::copy(listeria_sketches.begin(), listeria_sketches.begin() + 2,
              query_sketches.begin());
    NumpyMatrix query_dists = query_db(listeria_sketches,
                            query_sketches,
                            kmer_lengths,
                            random,
                            false,
                            4);
    std::cout << query_dists << std::endl;

    return 0;
}

