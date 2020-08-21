
#include <fstream>
#include <iostream>

#include "reference.hpp"
#include "database/database.hpp"
#include "random/random_match.hpp"
#include "api.hpp"
#include "sketch/countmin.hpp"
#include "sketch/seqio.hpp"

int main(int argc, char* argv[])
{
    // Runs a test of functionality
    std::ifstream rfile(argv[1]);
    std::string name, file;
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> files;
    while (rfile >> name >> file)
    {
        names.push_back(name);
        std::vector<std::string> file_list = {file};
        files.push_back(file_list);
    }

    HighFive::File h5_db = open_h5("listeria.h5");
    Database listeria_db(h5_db);
    std::vector<Reference> listeria_sketches;
    for (auto name_it = names.cbegin(); name_it != names.cend(); name_it++)
    {
        listeria_sketches.push_back(listeria_db.load_sketch(*name_it));
    }
    RandomMC random_retrived = listeria_db.load_random(true);

    std::vector<size_t> kmer_lengths {15, 17, 19, 21, 23, 25, 27, 29};

#ifdef GPU_AVAILABLE
    NumpyMatrix listeria_dists =
      query_db_cuda(
        listeria_sketches,
	    listeria_sketches,
	    kmer_lengths,
	    random_retrived,
	    0,
	    4);
    std::cout << listeria_dists << std::endl;
#endif
    return 0;
}

