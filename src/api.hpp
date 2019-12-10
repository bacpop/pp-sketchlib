/*
 *
 * api.hpp
 * main functions for interacting with sketches
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

#include <Eigen/Dense>
using Eigen::MatrixXf;

#include "reference.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, 2> DistMatrix;

// These are the three functions called by python bindings
std::vector<Reference> create_sketches(const std::string& db_name,
                   const std::vector<std::string>& names, 
                   const std::vector<std::string>& files, 
                   const std::vector<size_t>& kmer_lengths,
                   const size_t sketchsize64,
                   const size_t num_threads);

DistMatrix query_db(std::vector<Reference>& ref_sketches,
                    std::vector<Reference>& query_sketches,
                    const std::vector<size_t>& kmer_lengths,
                    const size_t num_threads);

std::vector<Reference> load_sketches(const std::string& db_name,
                                     const std::vector<std::string>& names,
                                     std::vector<size_t> kmer_lengths,
                                     const bool messages = true);

// Simple class to iterate over upper triangle of distance matrix
class upperTriIterator
{
    public:
        upperTriIterator(const std::vector<Reference>& sketches);
        upperTriIterator(const std::vector<Reference>& sketches, 
                         const std::vector<Reference>::const_iterator& ref_start,
                         const std::vector<Reference>::const_iterator& query_start,
                         const bool query_forwards);

        void advance();

        std::vector<Reference>::const_iterator getRefIt() const
            { return _ref_it; };
        std::vector<Reference>::const_iterator getQueryIt() const
            { return _query_it; };
        bool forwards() const { return _query_forwards; };

    private:
        bool _query_forwards;
        std::vector<Reference>::const_iterator _end_it;
        std::vector<Reference>::const_iterator _ref_it;
        std::vector<Reference>::const_iterator _query_it;
};