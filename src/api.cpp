/*
 * api.cpp
 * Main functions for running sketches
 *
 */

#include <algorithm>
#include <limits>
#include <queue>

#include <H5Cpp.h>
#include <omp.h>
#include <pybind11/pybind11.h>

#include "api.hpp"
#include "database/database.hpp"
#include "gpu/gpu.hpp"
#include "reference.hpp"
#include "sketch/progress.hpp"
#include "sketch/bitfuncs.hpp"

using namespace Eigen;
namespace py = pybind11;

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
    size_t done_count = 0;
    bool interrupt = false;
    std::vector<std::runtime_error> errors;
#pragma omp parallel for schedule(dynamic, 5) num_threads(num_threads)
    for (unsigned int i = 0; i < names.size(); i++) {
      if (!interrupt) {
        try {
          SeqBuf seq_in(files[i], kmer_lengths.back());
          sketches[i] = Reference(names[i], seq_in, kmer_seeds, sketchsize64,
                                  codon_phased, use_rc, min_count, exact);
#pragma omp atomic
          ++done_count;
        } catch (const std::runtime_error &e) {
#pragma omp critical
          {
            errors.push_back(e);
            interrupt = true;
          }
        }
      }

      if (omp_get_thread_num() == 0) {
        sketch_progress.tick_count(done_count);
        if (PyErr_CheckSignals() != 0) {
          interrupt = true;
        }
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

  size_t dist_rows;
  // Set up progress meter
  if (ref_sketches == query_sketches) {
    // calculate dists
    dist_rows = static_cast<size_t>(0.5 * (ref_sketches.size()) *
                                           (ref_sketches.size() - 1));
  } else {
    dist_rows = ref_sketches.size() * query_sketches.size();
  }
  static const uint64_t n_progress_ticks = 1000;
  uint64_t update_every = 1;
  if (dist_rows > n_progress_ticks) {
    update_every = dist_rows / n_progress_ticks;
  }
  ProgressMeter dist_progress(n_progress_ticks, true);
  volatile int progress = 0;

  // These could be the same but out of order, which could be dealt with
  // using a sort, except the return order of the distances wouldn't be as
  // expected. self iff ref_names == query_names as input
  bool interrupt = false;
  if (ref_sketches == query_sketches) {
    // calculate dists
    distMat.resize(dist_rows, dist_cols);
    Eigen::MatrixXf kmer_mat = kmer2mat(kmer_lengths);

    // Iterate upper triangle
#pragma omp parallel for schedule(dynamic, 5) num_threads(num_threads) shared(progress)
    for (size_t i = 0; i < ref_sketches.size(); i++) {
      if (!interrupt) {
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
#pragma omp critical
              {
                progress += MAX(1, n_progress_ticks / dist_rows);
                dist_progress.tick_count(progress);
                if (omp_get_thread_num() == 0 && PyErr_CheckSignals() != 0) {
                  interrupt = true;
                }
              }
          }
        }
      }
    }
  } else {
    // If ref != query, make a thread queue, with each element one ref
    // calculate dists
    distMat.resize(dist_rows, dist_cols);

    // Prepare objects used in distance calculations
    Eigen::MatrixXf kmer_mat = kmer2mat(kmer_lengths);
    std::vector<size_t> query_lengths(query_sketches.size());
    std::vector<uint16_t> query_random_idxs(query_sketches.size());

#pragma omp parallel for simd schedule(static) num_threads(num_threads)
    for (unsigned int q_idx = 0; q_idx < query_sketches.size(); q_idx++) {
      query_lengths[q_idx] = query_sketches[q_idx].seq_length();
      query_random_idxs[q_idx] =
          random_chance.closest_cluster(query_sketches[q_idx]);
    }

#pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
    for (unsigned int q_idx = 0; q_idx < query_sketches.size(); q_idx++) {
      for (unsigned int r_idx = 0; r_idx < ref_sketches.size(); r_idx++) {
        if (!interrupt) {
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
          if ((q_idx * ref_sketches.size() + r_idx) % update_every == 0) {
#pragma omp critical
            {
              progress += MAX(1, n_progress_ticks / dist_rows);
              dist_progress.tick_count(progress);
              if (omp_get_thread_num() == 0 && PyErr_CheckSignals() != 0) {
                interrupt = true;
              }
            }
          }
        }
      }
    }
  }

  // Handle Ctrl-C from python
  if (interrupt) {
    throw py::error_already_set();
  }
  dist_progress.finalise();

  return (distMat);
}

void check_sparse_inputs(const std::vector<Reference> &ref_sketches,
                         const std::vector<size_t> &kmer_lengths,
                         RandomMC &random_chance, const bool jaccard,
                         const size_t dist_col) {
  if (ref_sketches.size() < 1) {
    throw std::runtime_error("Query with empty ref or query list!");
  }
  if (kmer_lengths[0] <
      random_chance.min_supported_k(ref_sketches[0].seq_length())) {
    throw std::runtime_error("Smallest k-mer has no signal above random "
                             "chance; increase minimum k-mer length");
  }
  if ((jaccard && dist_col >= kmer_lengths.size()) || (dist_col != 0 && dist_col != 1)) {
    throw std::runtime_error("dist_col out of range");
  }

  // Check all references are in the random object, add if not
  if (random_chance.check_present(ref_sketches, true)) {
    std::cerr
        << "Some members of the reference database were not found "
           "in its random match chances. Consider refreshing with addRandom"
        << std::endl;
  }
}

// Struct that allows sorting by dist but also keeping index
struct SparseDist {
  float dist;
  long j;
};
bool operator<(SparseDist const &a, SparseDist const &b)
{
  return a.dist < b.dist;
}
bool operator==(SparseDist const &a, SparseDist const &b)
{
  return a.dist == b.dist;
}

sparse_coo query_db_sparse(std::vector<Reference> &ref_sketches,
                     const std::vector<size_t> &kmer_lengths,
                     RandomMC &random_chance, const bool jaccard,
                     const int kNN, const size_t dist_col,
                     const size_t num_threads) {
  check_sparse_inputs(ref_sketches, kmer_lengths, random_chance,
                      jaccard, dist_col);

  std::cerr << "Calculating distances using " << num_threads << " thread(s)"
            << std::endl;

  std::vector<float> dists(ref_sketches.size() * kNN);
  std::vector<long> i_vec(ref_sketches.size() * kNN);
  std::vector<long> j_vec(ref_sketches.size() * kNN);

  bool interrupt = false;

  // Set up progress meter
  size_t dist_rows = static_cast<size_t>(ref_sketches.size() * ref_sketches.size());
  static const uint64_t n_progress_ticks = 1000;
  uint64_t update_every = 1;
  if (dist_rows > n_progress_ticks) {
    update_every = dist_rows / n_progress_ticks;
  }
  ProgressMeter dist_progress(n_progress_ticks, true);
  volatile int progress = 0;

  Eigen::MatrixXf kmer_mat = kmer2mat(kmer_lengths);
#pragma omp parallel for schedule(static) num_threads(num_threads) shared(progress)
  for (size_t i = 0; i < ref_sketches.size(); i++) {
    // Use a priority queue to efficiently track the smallest N dists
    std::priority_queue<SparseDist> min_dists;
    if (!interrupt) {
      for (size_t j = 0; j < ref_sketches.size(); j++) {
        float row_dist = std::numeric_limits<float>::infinity();
        if (i != j) {
          if (jaccard) {
            // Need 1-J here to sort correctly
            row_dist = 1.0f - ref_sketches[i].jaccard_dist(
                ref_sketches[j], kmer_lengths[dist_col], random_chance);
          } else {
            float core, acc;
            std::tie(core, acc) =
                ref_sketches[i].core_acc_dist<RandomMC>(
                    ref_sketches[j], kmer_mat, random_chance);
            if (dist_col == 0) {
              row_dist = core;
            } else {
              row_dist = acc;
            }
          }
        }
        // Add dist if it is in the smallest k
        if (min_dists.size() < kNN || row_dist < min_dists.top().dist) {
          SparseDist new_min = {row_dist, static_cast<long>(j)};
          min_dists.push(new_min);
          if (min_dists.size() > kNN) {
            min_dists.pop();
          }
        }
        if ((i * ref_sketches.size() + j) % update_every == 0) {
#pragma omp critical
          {
            progress += MAX(1, n_progress_ticks / dist_rows);
            dist_progress.tick_count(progress);
            if (omp_get_thread_num() == 0 && PyErr_CheckSignals() != 0) {
              interrupt = true;
            }
          }
        }
      }

      // For each sample/row/i, fill the ijk vectors
      // This goes 'backwards' for compatibility with numpy (so dists are ascending)
      long offset = i * kNN;
      std::fill_n(i_vec.begin() + offset, kNN, i);
      for (int k = kNN - 1; k >= 0; --k) {
        SparseDist entry = min_dists.top();
        j_vec[offset + k] = entry.j;
        dists[offset + k] = entry.dist;
        min_dists.pop();
      }
    }
  }
  dist_progress.finalise();

  // Handle Ctrl-C from python
  if (interrupt) {
    throw py::error_already_set();
  }

  // Revert to correct distance J = 1 - dist
  if (jaccard) {
    #pragma omp parallel for simd num_threads(num_threads)
    for (size_t i = 0; i < dists.size(); ++i) {
      dists[i] = 1.0f - dists[i];
    }
  }

  return (std::make_tuple(i_vec, j_vec, dists));
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
