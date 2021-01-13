/*
 * linear_regression.cpp
 * Regresses k-mer lengths and matches
 *
 */
#include <math.h>
#include <algorithm>
#include <limits>

#include "reference.hpp"

const float core_upper = 0;
const float accessory_upper = 0;

std::tuple<float, float> fit_slope(const arma::mat &kmers,
                                   const arma::vec &dists,
                                   Reference *r1,
                                   Reference *r2)
{

  // Store core/accessory in dists, truncating at zero
  float core_dist = 0, accessory_dist = 0;
  static const double tolerance = (5.0 / (r1->sketchsize64() * 64));
  try
  {
    arma::uvec truncation = arma::find(dists < tolerance, 1, "first");
    arma::colvec slopes;
    if (truncation.n_elem > 0)
    {
      slopes = arma::solve(kmers.head_rows(truncation[0]),
                           log(dists.head(truncation[0])));
    }
    else
    {
      slopes = arma::solve(kmers, log(dists));
    }

    if (slopes(1) < core_upper)
    {
      core_dist = 1 - exp(slopes(1));
    }
    if (slopes(0) < accessory_upper)
    {
      accessory_dist = 1 - exp(slopes(0));
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    std::cerr << "Fitting k-mer gradient failed, for:" << r1->name() << "vs." << r2->name() << std::endl;
    std::cerr << dists << std::endl;
    std::cerr << std::endl
              << "Check for low quality genomes" << std::endl;
    exit(1);
  }
  return (std::make_tuple(core_dist, accessory_dist));
}

std::tuple<float, float> regress_kmers(Reference *r1,
                                       Reference *r2,
                                       const arma::mat &kmers,
                                       const RandomMC &random)
{
  // Vector of points
  arma::vec dists(kmers.n_rows);
  for (unsigned int i = 0; i < dists.n_elem; ++i)
  {
    dists[i] = r1->jaccard_dist(*r2, (int)kmers.at(i, 1), random);
  }
  return (fit_slope(kmers, dists, r1, r2));
}

// Using random vector
std::tuple<float, float> regress_kmers(Reference *r1,
                                       Reference *r2,
                                       const arma::mat &kmers,
                                       const std::vector<double> &random)
{
  // Vector of points
  arma::vec dists(kmers.n_rows);
  for (unsigned int i = 0; i < dists.n_elem; ++i)
  {
    dists[i] = r1->jaccard_dist(*r2, (int)kmers.at(i, 1), random[i]);
  }
  return (fit_slope(kmers, dists, r1, r2));
}
