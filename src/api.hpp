/*
 *
 * api.hpp
 * main functions for interacting with sketches
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <string>

#include <Eigen/Dense>
using Eigen::MatrixXd;

MatrixXd create_db(std::string& db_name,
               std::vector<std::string>& names, 
               std::vector<std::string>& files, 
               std::vector<size_t>& kmer_lengths,
               const size_t sketchsize64,
               const size_t num_threads);