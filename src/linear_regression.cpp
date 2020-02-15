/*
 * linear_regression.cpp
 * Regresses k-mer lengths and matches
 *
 */
#include <math.h>
#include <algorithm>
#include <limits>

#include "reference.hpp"
#include <dlib/matrix.h>
#include <dlib/optimization.h>

typedef dlib::matrix<double,2,1> two_vec;

const double convergence_limit = 1e-10;
const two_vec x_lower = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
const two_vec x_upper = {0, 0};

// Linear model y = ax
double residual(const std::pair<two_vec, double>& data,
                const two_vec& parameters)
{
    return (dlib::trans(data.first) * parameters) - data.second;
}

// Jacobian is constant dy/da = x
two_vec gradient(const std::pair<two_vec, double>& data,
                 const two_vec& parameters)
{
    return(data.first);
}

std::tuple<float, float> regress_kmers(const Reference * r1, 
                                       const Reference * r2, 
                                       const std::vector<size_t>& kmers)
{
    // Vector of points 
    // Each point is an input/output tuple (k-mer length)/(jaccard dist)
    std::vector<std::pair<two_vec, double> > data_samples;    
    for (unsigned int i = 0; i < kmers.size(); ++i)
    {
        data_samples.push_back(std::make_pair(two_vec{1, (double)kmers[i]},
                                              log(r1->jaccard_dist(*r2, kmers[i])))); 
    }

    two_vec starting_point = {0, 0};
    try
    {
        dlib::solve_least_squares_lm(
            dlib::objective_delta_stop_strategy(convergence_limit),
            residual,
            gradient,
            data_samples,
            starting_point);

        // Store core/accessory in dists, truncating at zero
        float core_dist = 0, accessory_dist = 0;
        if (starting_point(1) < x_upper(1))
        {
            core_dist = 1 - exp(starting_point(1));
        }
        if (starting_point(0) < x_upper(0))
        {
            accessory_dist = 1 - exp(starting_point(0));
        }
        return(std::make_tuple(core_dist, accessory_dist));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << "Fitting k-mer gradient failed, for:" << r1->name() << "vs." << r2->name() << std::endl;
        for (unsigned int i = 0; i < data_samples.size(); ++i)
        {
            std::cerr << data_samples[i].first(1) << "\t" << data_samples[i].second << std::endl;
        }
        std::cerr << std::endl << "Check for low quality genomes" << std::endl;
        exit(1);
    }
}
