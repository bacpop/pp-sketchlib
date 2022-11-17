/*
 * linear_regression.cpp
 * Regresses k-mer lengths and matches
 *
 */
#include <algorithm>
#include <limits>
#include <math.h>
#include <iostream>

#include "reference.hpp"

const float core_upper = 0;
const float accessory_upper = 0;

std::tuple<float, float> fit_slope(const Eigen::MatrixXf &kmers,
                                   const Eigen::VectorXf &dists,
                                   Reference *r1, Reference *r2) {

  // Store core/accessory in dists, truncating at zero
  float core_dist = 0, accessory_dist = 0;
  static const double tolerance = (5.0 / (r1->sketchsize64() * 64));
  try {
    Eigen::VectorXf slopes;
    for (int i = 0; i < dists.size(); ++i) {
      if (dists(i) < tolerance) {
        if (i == 0) {
          throw std::runtime_error("No non-zero Jaccard distances");
        }
        Eigen::VectorXf dists_truncation = dists(Eigen::seqN(0, i));
        Eigen::VectorXf kmer_truncation = kmers(Eigen::seqN(0, i), Eigen::all);
        slopes = (kmer_truncation.transpose() * kmer_truncation).ldlt().solve(kmer_truncation.transpose() * (dists_truncation.log()));
        break;
      } else if (i == dists.size() - 1) {
        // See https://eigen.tuxfamily.org/dox/group__LeastSquares.html
        slopes = (kmers.transpose() * kmers).ldlt().solve(kmers.transpose() * (dists.log()));
      }
    }

    if (slopes(1) < core_upper) {
      core_dist = 1 - exp(slopes(1));
    }
    if (slopes(0) < accessory_upper) {
      accessory_dist = 1 - exp(slopes(0));
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "Fitting k-mer gradient failed, for:" << r1->name() << "vs."
              << r2->name() << std::endl;
    std::cerr << dists << std::endl;
    std::cerr << std::endl << "Check for low quality genomes" << std::endl;
    exit(1);
  }
  return (std::make_tuple(core_dist, accessory_dist));
}

std::tuple<float, float> regress_kmers(Reference *r1, Reference *r2,
                                       const Eigen::MatrixXf &kmers,
                                       const RandomMC &random) {
  // Vector of points
  Eigen::VectorXf dists(kmers.size());
  for (unsigned int i = 0; i < kmers.size(); ++i) {
    dists(i) = r1->jaccard_dist(*r2, kmers[i], random);
  }
  return (fit_slope(kmers, dists, r1, r2));
}

// Using random vector
std::tuple<float, float> regress_kmers(Reference *r1, Reference *r2,
                                       const Eigen::MatrixXf &kmers,
                                       const std::vector<double> &random) {
  // Vector of points
  Eigen::VectorXf dists(kmers.size());
  for (unsigned int i = 0; i < dists.size(); ++i) {
    dists[i] = r1->jaccard_dist(*r2, kmers[i], random[i]);
  }
  return (fit_slope(kmers, dists, r1, r2));
}
