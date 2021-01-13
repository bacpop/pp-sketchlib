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
#include <sys/stat.h>

#include "dist/matrix.hpp"
#include "random/random_match.hpp"
#include "reference.hpp"

inline bool file_exists(const std::string &name)
{
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

std::vector<Reference> create_sketches(const std::string &db_name,
                                       const std::vector<std::string> &names,
                                       const std::vector<std::vector<std::string>> &files,
                                       const std::vector<size_t> &kmer_lengths,
                                       const size_t sketchsize64,
                                       const bool codon_phased,
                                       const bool use_rc,
                                       size_t min_count,
                                       const bool exact,
                                       const size_t num_threads);

NumpyMatrix query_db(std::vector<Reference> &ref_sketches,
                     std::vector<Reference> &query_sketches,
                     const std::vector<size_t> &kmer_lengths,
                     RandomMC &random_chance,
                     const bool jaccard,
                     const size_t num_threads);

#ifdef GPU_AVAILABLE
// defined in gpu_api.cpp
std::vector<Reference> create_sketches_cuda(const std::string &db_name,
                                            const std::vector<std::string> &names,
                                            const std::vector<std::vector<std::string>> &files,
                                            const std::vector<size_t> &kmer_lengths,
                                            const size_t sketchsize64,
                                            const bool use_rc,
                                            size_t min_count,
                                            const size_t cpu_threads = 1,
                                            const int device_id = 0);

NumpyMatrix query_db_cuda(std::vector<Reference> &ref_sketches,
                          std::vector<Reference> &query_sketches,
                          const std::vector<size_t> &kmer_lengths,
                          RandomMC &random_chance,
                          const int device_id = 0,
                          const unsigned int num_cpu_threads = 1);
#endif

bool same_db_version(const std::string &db1_name,
                     const std::string &db2_name);
std::tuple<std::string, bool> get_db_attr(const std::string &db1_name);

std::vector<Reference> load_sketches(const std::string &db_name,
                                     const std::vector<std::string> &names,
                                     std::vector<size_t> kmer_lengths,
                                     const bool messages = true);

RandomMC calculate_random(const std::vector<Reference> &sketches,
                          const std::string &db_name,
                          const unsigned int n_clusters,
                          const unsigned int n_MC,
                          const bool codon_phased,
                          const bool use_rc,
                          const int num_threads);

RandomMC get_random(const std::string &db_name,
                    const bool use_rc_default);
