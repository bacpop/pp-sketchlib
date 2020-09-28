
#include <fstream>
#include <iostream>

#include "api.hpp"

int main (int argc, char* argv[])
{
    // Runs a test of functionality
    NumpyMatrix square(6,6);
    square << 0, 1, 2, 3, 10, 20,
              1, 0, 4, 5, 10, 20,
              2, 4, 0, 6, 10, 20,
              3, 5, 6, 0, 10, 20,
              10, 10, 10, 10, 0, 8,
              20, 20, 20, 20, 8, 0;
    std::vector<long> i_vec, j_vec;
    std::vector<float> dists;
    sparse_coo sparse_dists_cutoff = sparsify_dists(square, 7, 0, 1);
    std::tie(i_vec, j_vec, dists) = sparse_dists_cutoff;
    std::cout << "Dists < 7" << std::endl;
    for (unsigned int it = 0; it < i_vec.size(); it++) {
        std::cout << i_vec[it] << "\t" << j_vec[it] << "\t" << dists[it] << std::endl;
    }

    sparse_coo sparse_dists_3nn = sparsify_dists(square, 0, 3, 1);
    std::tie(i_vec, j_vec, dists) = sparse_dists_3nn;
    std::cout << "3-NN" << std::endl;
    for (unsigned int it = 0; it < i_vec.size(); it++) {
        std::cout << i_vec[it] << "\t" << j_vec[it] << "\t" << dists[it] << std::endl;
    }




    return 0;
}

