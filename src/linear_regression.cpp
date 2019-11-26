/*
 * linear_regression.cpp
 * Regresses k-mer lengths and matches
 *
 */

#include "linear_regression.hpp"
#include <dlib/optimization.h>

#include <math.h>

const double convergence_limit = 1e-7;

const column_vector x_lower(2, -std::numeric_limits<double>::infinity());
const column_vector x_upper(2, 0);

std::tuple<float, float> core_acc = regress_kmers(const Reference * r1, 
                                                  const Reference * r2, 
                                                  const column_vector& kmers) // TODO pass this once to avoid a loop
{
    column_vector y_vec(kmers.nr());
    for (unsigned int i = 0; i < y_vec.nr(); ++i)
    {
        y_vec(i) = log(r1.jaccard_dist(r2, kmers[i])); 
    }
    LinearLink linear_fit(klist, y_vec);

    column_vector starting_point(2);
    starting_point(0) = -0.01;
    starting_point(1) = 0;

    try
    {
        dlib::find_max_box_constrained(
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
        return(std::make_tuple(1 - exp(starting_point(1)), exp(starting_point(0))));
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