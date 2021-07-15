/*
 * api.cpp
 * Main functions for running sketches
 *
 */

#include <algorithm>
#include <limits>

#include <H5Cpp.h>
#include <omp.h>
#include <pybind11/pybind11.h>

#include "api.hpp"
#include "database/database.hpp"
#include "gpu/gpu.hpp"
#include "reference.hpp"
#include "sketch/progress.hpp"

using namespace Eigen;
namespace py = pybind11;

const int progressBitshift = 10;

bool same_db_version(const std::string &db1_name, const std::string &db2_name) {
  // Open databases
  Database db1(db1_name + ".h5");
  Database db2(db2_name + ".h5");

  return (db1.check_version(db2));
}

std::tuple<std::string, bool> get_db_attr(const std::string &db1_name) {
  Database db(db1_name + ".h5");
  return (std::make_tuple(db.version(), db.codon_phased()));
}

// Create sketches, save to file
std::vector<Reference> create_sketches(
    const std::string &db_name, const std::vector<std::string> &names,
    const std::vector<std::vector<std::string>> &files,
    const std::vector<size_t> &kmer_lengths, const size_t sketchsize64,
    const bool codon_phased, const bool use_rc, size_t min_count,
    const bool exact, const size_t num_threads) {
  // Store sketches in vector
  std::vector<Reference> sketches;

  // Try loading sketches from file
  bool resketch = true;
  if (file_exists(db_name + ".h5")) {
    sketches = load_sketches(db_name, names, kmer_lengths);
    if (sketches.size() == names.size()) {
      resketch = false;
    }
  }

  // If not found or not matching, sketch from scratch
  if (resketch) {
    sketches.resize(names.size());

    // Truncate min_count if above 8 bit range
    if (min_count > std::numeric_limits<uint8_t>::max()) {
      min_count = std::numeric_limits<uint8_t>::max();
    }

    size_t num_sketch_threads = num_threads;
    if (sketches.size() < num_threads) {
      num_sketch_threads = sketches.size();
    }

    std::cerr << "Sketching " << names.size() << " genomes using "
              << num_sketch_threads << " thread(s)" << std::endl;

    if (codon_phased) {
      std::cerr << "NB: codon phased seeds are ON" << std::endl;
    }
    KmerSeeds kmer_seeds = generate_seeds(kmer_lengths, codon_phased);

    ProgressMeter sketch_progress(names.size());
    bool interrupt = false;
    std::vector<std::runtime_error> errors;
#pragma omp parallel for schedule(dynamic, 5) num_threads(num_threads)
    for (unsigned int i = 0; i < names.size(); i++) {
      if (interrupt || PyErr_CheckSignals() != 0) {
        interrupt = true;
      } else {
        try {
          SeqBuf seq_in(files[i], kmer_lengths.back());
          sketches[i] = Reference(names[i], seq_in, kmer_seeds, sketchsize64,
                                  codon_phased, use_rc, min_count, exact);
        } catch (const std::runtime_error &e) {
#pragma omp critical
          {
            errors.push_back(e);
            interrupt = true;
          }
        }
      }

      if (omp_get_thread_num() == 0) {
        sketch_progress.tick(num_threads);
      }
    }
    sketch_progress.finalise();

    // Handle errors including Ctrl-C from python
    if (interrupt) {
      for (auto i = errors.cbegin(); i != errors.cend(); ++i) {
        std::cout << i->what() << std::endl;
      }
      if (errors.size()) {
        throw std::runtime_error("Errors during sketching");
      } else {
        throw py::error_already_set();
      }
    }

    // Save sketches and check for densified sketches
    std::cerr << "Writing sketches to file" << std::endl;
    Database sketch_db = new_db(db_name + ".h5", use_rc, codon_phased);
    for (auto sketch_it = sketches.begin(); sketch_it != sketches.end();
         sketch_it++) {
      sketch_db.add_sketch(*sketch_it);
      if (sketch_it->densified()) {
        std::cerr << "NOTE: " << sketch_it->name() << " required densification"
                  << std::endl;
      }
    }
  }

  return sketches;
}

// Calculates distances against another database
// Input is vectors of sketches
NumpyMatrix query_db(std::vector<Reference> &ref_sketches,
                     std::vector<Reference> &query_sketches,
                     const std::vector<size_t> &kmer_lengths,
                     RandomMC &random_chance, const bool jaccard,
                     const size_t num_threads) {
  if (ref_sketches.size() < 1 or query_sketches.size() < 1) {
    throw std::runtime_error("Query with empty ref or query list!");
  }
  if (kmer_lengths[0] <
      random_chance.min_supported_k(ref_sketches[0].seq_length())) {
    throw std::runtime_error("Smallest k-mer has no signal above random "
                             "chance; increase minimum k-mer length");
  }

  // Check all references are in the random object, add if not
  bool missing = random_chance.check_present(ref_sketches, true);
  if (missing) {
    std::cerr
        << "Some members of the reference database were not found "
           "in its random match chances. Consider refreshing with addRandom"
        << std::endl;
  }

  std::cerr << "Calculating distances using " << num_threads << " thread(s)"
            << std::endl;

  NumpyMatrix distMat;
  size_t dist_cols;
  if (jaccard) {
    dist_cols = kmer_lengths.size();
  } else {
    dist_cols = 2;
  }

  // These could be the same but out of order, which could be dealt with
  // using a sort, except the return order of the distances wouldn't be as
  // expected. self iff ref_names == query_names as input
  bool interrupt = false;
  if (ref_sketches == query_sketches) {
    // calculate dists
    size_t dist_rows = static_cast<size_t>(0.5 * (ref_sketches.size()) *
                                           (ref_sketches.size() - 1));
    distMat.resize(dist_rows, dist_cols);

    arma::mat kmer_mat = kmer2mat<std::vector<size_t>>(kmer_lengths);

    // Set up progress meter
    size_t progress_blocks = 1 << progressBitshift;
    size_t update_every = dist_rows >> progressBitshift;
    if (progress_blocks > dist_rows || update_every < 1) {
      progress_blocks = dist_rows;
      update_every = 1;
    }
    ProgressMeter dist_progress(progress_blocks, true);
    int progress = 0;

    // Iterate upper triangle
#pragma omp parallel for schedule(dynamic, 5) num_threads(num_threads) shared(progress)
    for (size_t i = 0; i < ref_sketches.size(); i++) {
      if (interrupt || PyErr_CheckSignals() != 0) {
        interrupt = true;
      } else {
        for (size_t j = i + 1; j < ref_sketches.size(); j++) {
          size_t pos = square_to_condensed(i, j, ref_sketches.size());
          if (jaccard) {
            for (unsigned int kmer_idx = 0; kmer_idx < kmer_lengths.size();
                 kmer_idx++) {
              distMat(pos, kmer_idx) = ref_sketches[i].jaccard_dist(
                  ref_sketches[j], kmer_lengths[kmer_idx], random_chance);
            }
          } else {
            std::tie(distMat(pos, 0), distMat(pos, 1)) =
                ref_sketches[i].core_acc_dist<RandomMC>(
                    ref_sketches[j], kmer_mat, random_chance);
          }
          if (pos % update_every == 0) {
#pragma omp atomic
              progress++;
              dist_progress.tick(1);
          }
        }
      }
    }
    dist_progress.finalise();
  } else {
    // If ref != query, make a thread queue, with each element one ref
    // calculate dists
    size_t dist_rows = ref_sketches.size() * query_sketches.size();
    distMat.resize(dist_rows, dist_cols);

    // Prepare objects used in distance calculations
    arma::mat kmer_mat = kmer2mat<std::vector<size_t>>(kmer_lengths);
    std::vector<size_t> query_lengths(query_sketches.size());
    std::vector<uint16_t> query_random_idxs(query_sketches.size());

#pragma omp parallel for simd schedule(static) num_threads(num_threads)
    for (unsigned int q_idx = 0; q_idx < query_sketches.size(); q_idx++) {
      query_lengths[q_idx] = query_sketches[q_idx].seq_length();
      query_random_idxs[q_idx] =
          random_chance.closest_cluster(query_sketches[q_idx]);
    }

    ProgressMeter dist_progress(dist_rows, true);
#pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
    for (unsigned int q_idx = 0; q_idx < query_sketches.size(); q_idx++) {
      for (unsigned int r_idx = 0; r_idx < ref_sketches.size(); r_idx++) {
        if (interrupt || PyErr_CheckSignals() != 0) {
          interrupt = true;
        } else {
          const long dist_row = q_idx * ref_sketches.size() + r_idx;
          if (jaccard) {
            for (unsigned int kmer_idx = 0; kmer_idx < kmer_lengths.size();
                 kmer_idx++) {
              double jaccard_random = random_chance.random_match(
                  ref_sketches[r_idx], query_random_idxs[q_idx],
                  query_lengths[q_idx], kmer_lengths[kmer_idx]);
              distMat(dist_row, kmer_idx) = query_sketches[q_idx].jaccard_dist(
                  ref_sketches[r_idx], kmer_lengths[kmer_idx], jaccard_random);
            }
          } else {
            std::vector<double> jaccard_random = random_chance.random_matches(
                ref_sketches[r_idx], query_random_idxs[q_idx],
                query_lengths[q_idx], kmer_lengths);
            std::tie(distMat(dist_row, 0), distMat(dist_row, 1)) =
                query_sketches[q_idx].core_acc_dist<std::vector<double>>(
                    ref_sketches[r_idx], kmer_mat, jaccard_random);
          }
          if (omp_get_thread_num() == 0) {
            dist_progress.tick(num_threads);
          }
        }
      }
    }
    dist_progress.finalise();
  }

  // Handle Ctrl-C from python
  if (interrupt) {
    throw py::error_already_set();
  }

  return (distMat);
}

// Load sketches from a HDF5 file
// Returns empty vector on failure
std::vector<Reference> load_sketches(const std::string &db_name,
                                     const std::vector<std::string> &names,
                                     std::vector<size_t> kmer_lengths,
                                     const bool messages) {
  // Vector of set size to store results
  std::vector<Reference> sketches(names.size());
  std::sort(kmer_lengths.begin(), kmer_lengths.end());

  /* Turn off HDF5 error messages */
  /* getAutoPrint throws an unknown exception when
       called from python, but is ok from C++ */
#ifndef PYTHON_EXT
  H5E_auto2_t errorPrinter;
  void **clientData = nullptr;
  H5::Exception::getAutoPrint(errorPrinter, clientData);
#endif
  H5::Exception::dontPrint();

  try {
    // Open as read only
    Database prev_db(db_name + ".h5");

    if (messages) {
      std::cerr << "Looking for existing sketches in " + db_name + ".h5"
                << std::endl;
    }
    size_t i = 0;
    for (auto name_it = names.cbegin(); name_it != names.end(); name_it++) {
      sketches[i] = prev_db.load_sketch(*name_it);

      // Remove unwanted k-mer lengths from sketch dict
      auto loaded_sizes = sketches[i].kmer_lengths();
      std::sort(loaded_sizes.begin(), loaded_sizes.end());
      auto kmer_it = kmer_lengths.begin();
      auto loaded_it = loaded_sizes.begin();
      while (kmer_it != kmer_lengths.end() && loaded_it != loaded_sizes.end()) {
        if (*kmer_it == *loaded_it) {
          kmer_it++;
        } else {
          sketches[i].remove_kmer_sketch(*loaded_it);
        }
        loaded_it++;
      }
      // throw if any of the requested k-mer lengths were not found
      if (kmer_it != kmer_lengths.end()) {
        std::stringstream old_kmers, new_kmers;
        std::copy(loaded_sizes.begin(), loaded_sizes.end(),
                  std::ostream_iterator<size_t>(old_kmers, ","));
        std::copy(kmer_lengths.begin(), kmer_lengths.end(),
                  std::ostream_iterator<size_t>(new_kmers, ","));

        std::string err_message = "k-mer lengths in old database (";
        err_message += old_kmers.str() + ") do not match those requested (" +
                       new_kmers.str() + ")";
        throw std::runtime_error(err_message);
      }

      i++;
    }
  } catch (const HighFive::Exception &e) {
    // Triggered if sketch not found
    std::cerr << "Missing sketch: " << e.what() << std::endl;
    sketches.clear();
  } catch (const std::exception &e) {
    // Triggered if k-mer lengths mismatch
    std::cerr << "Mismatched data: " << e.what() << std::endl;
    sketches.clear();
  }
  // Other errors (likely not safe to continue)
  catch (...) {
    std::cerr << "Error in reading previous database" << std::endl;
    sketches.clear();
    throw std::runtime_error("Database read error");
  }

#ifndef PYTHON_EXT
  /* Restore previous error handler */
  H5::Exception::setAutoPrint(errorPrinter, clientData);
#endif
  return (sketches);
}

RandomMC calculate_random(const std::vector<Reference> &sketches,
                          const std::string &db_name,
                          const unsigned int n_clusters,
                          const unsigned int n_MC, const bool codon_phased,
                          const bool use_rc, const int num_threads) {
  RandomMC random(sketches, n_clusters, n_MC, codon_phased, use_rc,
                  num_threads);

  // Save to the database provided
  Database db(db_name + ".h5", true);
  db.save_random(random);

  return (random);
}

RandomMC get_random(const std::string &db_name, const bool use_rc_default) {
  Database db(db_name + ".h5");
  RandomMC random = db.load_random(use_rc_default);
  return (random);
}
