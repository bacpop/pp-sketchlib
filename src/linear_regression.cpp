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

const double convergence_limit = 1e-10;
const dlib::matrix<double,2,1> x_lower = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
const dlib::matrix<double,2,1> x_upper = {0, 0};

double residual(const std::pair<dlib::matrix<double,2,1>, double>& data,
                const dlib::matrix<double,2,1>& parameters)
{
    return (data.first(0) * parameters(0) + data.first(1) * parameters(1)) - data.second;
}

dlib::matrix<double,2,1> gradient(const std::pair<dlib::matrix<double,2,1>, double>& data,
                                  const dlib::matrix<double,2,1>& parameters)
{
    return(data.first);
}

std::tuple<float, float> regress_kmers(const Reference * r1, 
                                       const Reference * r2, 
                                       const std::vector<size_t>& kmers)
{
    std::vector<std::pair<dlib::matrix<double,2,1>, double> > data_samples;    
    for (unsigned int i = 0; i < kmers.size(); ++i)
    {
        data_samples.push_back(std::make_pair(dlib::matrix<double,2,1>{1, (double)kmers[i]},
                                              log(r1->jaccard_dist(*r2, kmers[i])))); 
    }
    //LinearLink linear_fit(kmers, y_vec);

    dlib::matrix<double,2,1> starting_point = {0, 0};

    try
    {
/*         dlib::find_min_box_constrained(
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
            x_upper); */
        dlib::solve_least_squares_lm(
            dlib::objective_delta_stop_strategy(convergence_limit),
            residual,
            gradient,
            data_samples,
            starting_point);

        // Store core/accessory in dists
        return(std::make_tuple(1 - exp(starting_point(1)), 1 - exp(starting_point(0))));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << "Fitting k-mer gradient failed, matches:" << std::endl;
        for (unsigned int i = 0; i < data_samples.size(); ++i)
        {
            std::cerr << "\t" << data_samples[i].second;
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