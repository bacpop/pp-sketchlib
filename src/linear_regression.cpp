/*
 * linear_regression.cpp
 * Regresses k-mer lengths and matches
 *
 */

#include <limits>
#include <math.h>

#include "reference.hpp"
#include <dlib/optimization.h>
typedef dlib::matrix<double,0,1> column_vector;

const double convergence_limit = 1e-7;
const dlib::matrix<double,2,1> x_lower = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
const dlib::matrix<double,2,1> x_upper = {0, 0};

std::tuple<float, float> regress_kmers(const Reference * r1, 
                                       const Reference * r2, 
                                       const dlib::matrix<double,0,2>& kmers)
{
    column_vector y_vec(kmers.nr());
    for (unsigned int i = 0; i < y_vec.nr(); ++i)
    {
        y_vec(i) = log(r1->jaccard_dist(*r2, kmers(i, 1))); 
    }
    LinearLink linear_fit(kmers, y_vec);

    dlib::matrix<double,2,1> starting_point = {-0.01, -0.01};

    try
    {
        dlib::find_min_box_constrained(
            dlib::bfgs_search_strategy(),
            dlib::objective_delta_stop_strategy(convergence_limit),
            [&linear_fit](const column_vector& a) {
                return linear_fit.likelihood(a);
            },
            [&linear_fit](const column_vector& b) {
                return linear_fit.gradient(b);
            },
            starting_point,
            x_lower,
            x_upper);

        // Store core/accessory in dists
        return(std::make_tuple(1 - exp(starting_point(1)), 1 - exp(starting_point(0))));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << "Fitting k-mer gradient failed, matches:" << std::endl;
        for (unsigned int i = 0; i < y_vec.nr(); ++i)
        {
            std::cerr << "\t" << y_vec(i);
        }
        std::cerr << std::endl << "Check for low quality genomes" << std::endl;
        exit(1);
    }
}

// Makes a design matrix
dlib::matrix<double,0,2> add_intercept(const column_vector& kmer_vec)
{
    dlib::matrix<double> design(kmer_vec.nr(), 2);
    dlib::set_colm(design, 0) = dlib::ones_matrix<double>(kmer_vec.size(), 1);
    dlib::set_colm(design, 1) = kmer_vec;
    return(design);
}