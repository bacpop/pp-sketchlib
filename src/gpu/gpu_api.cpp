/*
 *
 * gpu_api.cpp
 * PopPUNK dists using CUDA
 * gcc compiled part (uses Eigen)
 *
 */

// std
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <tuple>
#include <vector>
#include <memory>

// internal headers
#include "api.hpp"
#include "database/database.hpp"
#include "gpu.hpp"
#include "sketch/bitfuncs.hpp"
#include "sketch/sketch.hpp"

static const float mem_epsilon = 0.05;

template <class T> inline T samples_to_rows(const T samples) {
  return ((samples * (samples - 1)) >> 1);
}

/*
 *
 *  Sketch functions
 *
 */

// Read in a batch of sequence data (in parallel)
std::vector<std::shared_ptr<SeqBuf>>
read_seq_batch(std::vector<std::vector<std::string>>::const_iterator &file_it,
               const size_t batch_size, const size_t max_kmer,
               const size_t cpu_threads) {
  std::vector<std::shared_ptr<SeqBuf>> seq_in_batch(batch_size);
#pragma omp parallel for schedule(static) num_threads(cpu_threads)
  for (size_t j = 0; j < batch_size; j++) {
    seq_in_batch[j] = std::make_shared<SeqBuf>(*(file_it + j), max_kmer);
  }
  file_it += batch_size;
  return seq_in_batch;
}

inline size_t cap_batch_size(const size_t idx, const size_t total_size,
                             size_t batch_size) {
  if (idx + batch_size >= total_size) {
    batch_size = total_size - idx;
  }
  return (batch_size);
}

std::vector<Reference> create_sketches_cuda(
    const std::string &db_name, const std::vector<std::string> &names,
    const std::vector<std::vector<std::string>> &files,
    const std::vector<size_t> &kmer_lengths, const size_t sketchsize64,
    const bool use_rc, size_t min_count, const size_t cpu_threads,
    const int device_id) {
  // Try loading sketches from file
  std::vector<Reference> sketches;
  bool resketch = true;
  if (file_exists(db_name + ".h5")) {
    sketches = load_sketches(db_name, names, kmer_lengths);
    if (sketches.size() == names.size()) {
      resketch = false;
    }
  }

  if (resketch) {
    Database sketch_db = new_db(db_name + ".h5", use_rc, false);
    sketches.resize(names.size());

    size_t mem_free, mem_total, shared_size;
    std::tie(mem_free, mem_total, shared_size) = initialise_device(device_id);
    std::cerr << "Sketching " << files.size() << " read sets on GPU device "
              << device_id << std::endl;
    std::cerr << "also using " << cpu_threads << " CPU cores" << std::endl;

    // memory for filter and nthash only need to be allocated once
    copyNtHashTablesToDevice();
    GPUCountMin countmin_filter;
    if (min_count > std::numeric_limits<unsigned int>::max()) {
      min_count = std::numeric_limits<unsigned int>::max();
    }

    size_t worker_threads = MAX(1, cpu_threads - 1);
    size_t n_batches =
        files.size() / worker_threads + (files.size() % worker_threads ? 1 : 0);
    size_t batch_size = cap_batch_size(0, files.size(), worker_threads);

    // CPU threads read in sequence (asynchronously)
    std::launch policy = std::launch::async;
    if (cpu_threads == 1) {
      // Gives serial behaviour if only one thread available
      policy = std::launch::deferred;
    }
    auto file_it = files.cbegin();
    std::future<std::vector<std::shared_ptr<SeqBuf>>> seq_reader =
        std::async(std::launch::deferred, &read_seq_batch, std::ref(file_it),
                   batch_size, kmer_lengths.back(), cpu_threads);

    for (size_t i = 0; i < files.size(); i += worker_threads) {
      std::cerr << "Sketching batch: " << i / worker_threads + 1 << " of "
                << n_batches << std::endl;
      batch_size = cap_batch_size(i, files.size(), worker_threads);

      // Get the next batch asynchronously
      std::vector<std::shared_ptr<SeqBuf>> seq_in_batch = seq_reader.get();
      if (file_it != files.cend()) {
        seq_reader = std::async(
            policy, &read_seq_batch, std::ref(file_it),
            cap_batch_size(i + worker_threads, files.size(), worker_threads),
            kmer_lengths.back(), worker_threads);
      }

      // Run the sketch on the GPU (serially over the batch)
      for (size_t j = 0; j < batch_size; j++) {
        robin_hood::unordered_map<int, std::vector<uint64_t>> usigs;
        size_t seq_length;
        bool densified;
        try {
          std::tie(usigs, seq_length, densified) = sketch_gpu(
              seq_in_batch[j], countmin_filter, sketchsize64, kmer_lengths,
              def_bbits, use_rc, min_count, i + j, cpu_threads, shared_size);

          // Make Reference object, and save in HDF5 DB
          sketches[i + j] =
              Reference(names[i + j], usigs, def_bbits, sketchsize64,
                        seq_length, seq_in_batch[j]->get_composition(),
                        seq_in_batch[j]->missing_bases(), use_rc, densified);
          sketch_db.add_sketch(sketches[i + j]);
          if (densified) {
            std::cerr << "NOTE: " << names[i + j] << " required densification"
                      << std::endl;
          }
        } catch (const std::runtime_error &e) {
          sketch_db.flush();
          throw std::runtime_error("Error when sketching " + names[i + j]);
        }
      }
      fprintf(stderr, "%cSample %lu\tk = %d  \n", 13, i + batch_size,
              static_cast<int>(kmer_lengths.back()));
    }
  }
  return (sketches);
}

/*
 *
 *  Distance functions
 *
 */

// Checks bbits, sketchsize and k-mer lengths are identical in
// all sketches
// throws runtime_error if mismatches (should be ensured in passing
// code)
void checkSketchParamsMatch(const std::vector<Reference> &sketches,
                            const std::vector<size_t> &kmer_lengths,
                            const size_t bbits, const size_t sketchsize64) {
  for (auto sketch_it = sketches.cbegin(); sketch_it != sketches.cend();
       sketch_it++) {
    if (sketch_it->bbits() != bbits) {
      throw std::runtime_error("Mismatching bbits in sketches");
    }
    if (sketch_it->sketchsize64() != sketchsize64) {
      throw std::runtime_error("Mismatching sketchsize64 in sketches");
    }
    if (sketch_it->kmer_lengths() != kmer_lengths) {
      throw std::runtime_error("Mismatching k-mer lengths in sketches");
    }
  }
}

// Functions to join matrices when doing computation in chunks
void longToSquareBlock(NumpyMatrix &coreSquare, NumpyMatrix &accessorySquare,
                       const SketchSlice &sketch_subsample,
                       NumpyMatrix &blockMat, const bool self,
                       const unsigned int num_threads) {
  // Convert each long form column of Nx2 matrix into square distance matrix
  // Add this square matrix into the correct submatrix (block) of the final
  // square matrix
  Eigen::VectorXf dummy_query_ref;
  Eigen::VectorXf dummy_query_query;

  NumpyMatrix square_form;
  if (self) {
    square_form = long_to_square(blockMat.col(0), dummy_query_ref,
                                 dummy_query_query, num_threads);
  } else {
    square_form = Eigen::Map<NumpyMatrix, 0, Eigen::InnerStride<2>>(
        blockMat.data(), sketch_subsample.ref_size,
        sketch_subsample.query_size);
  }
  // Only update the upper triangle
  coreSquare.block(sketch_subsample.query_offset, sketch_subsample.ref_offset,
                   sketch_subsample.query_size, sketch_subsample.ref_size) =
      square_form;

  if (self) {
    square_form = long_to_square(blockMat.col(1), dummy_query_ref,
                                 dummy_query_query, num_threads);
  } else {
    square_form = Eigen::Map<NumpyMatrix, 0, Eigen::InnerStride<2>>(
        blockMat.data() + 1, sketch_subsample.ref_size,
        sketch_subsample.query_size);
  }
  accessorySquare.block(
      sketch_subsample.query_offset, sketch_subsample.ref_offset,
      sketch_subsample.query_size, sketch_subsample.ref_size) = square_form;
}

// Gives strides aligned to the warp size (32)
inline size_t warpPad(const size_t stride) {
  return (stride + (stride % warp_size ? warp_size - stride % warp_size : 0));
}

// Turn a vector of references into a flattened vector of
// uint64 with strides bins * kmers * samples
std::vector<uint64_t> flatten_by_bins(const std::vector<Reference> &sketches,
                                      const std::vector<size_t> &kmer_lengths,
                                      SketchStrides &strides,
                                      const size_t start_sample_idx,
                                      const size_t end_sample_idx,
                                      const int cpu_threads) {
  // Input checks
  size_t num_sketches = end_sample_idx - start_sample_idx;
  const size_t num_bins = strides.sketchsize64 * strides.bbits;
  assert(num_bins == sketches[0].get_sketch(kmer_lengths[0]).size());
  assert(end_sample_idx > start_sample_idx);

  // Set strides structure
  strides.bin_stride = 1;
  strides.kmer_stride = warpPad(strides.bin_stride * num_bins);
  // warpPad not needed here, as k-mer stride already a multiple of warp size
  strides.sample_stride = strides.kmer_stride * kmer_lengths.size();

  // Iterate over each dimension to flatten
  std::vector<uint64_t> flat_ref(strides.sample_stride * num_sketches);
#pragma omp parallel for simd schedule(static) num_threads(cpu_threads)
  for (size_t sample_idx = start_sample_idx; sample_idx < end_sample_idx;
       sample_idx++) {
    auto flat_ref_it = flat_ref.begin() +
                       (sample_idx - start_sample_idx) * strides.sample_stride;
    for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend();
         kmer_it++) {
      std::copy(sketches[sample_idx].get_sketch(*kmer_it).cbegin(),
                sketches[sample_idx].get_sketch(*kmer_it).cend(), flat_ref_it);
      flat_ref_it += strides.kmer_stride;
    }
  }
  return flat_ref;
}

// Turn a vector of queries into a flattened vector of
// uint64 with strides samples * bins * kmers
std::vector<uint64_t>
flatten_by_samples(const std::vector<Reference> &sketches,
                   const std::vector<size_t> &kmer_lengths,
                   SketchStrides &strides, const size_t start_sample_idx,
                   const size_t end_sample_idx, const int cpu_threads) {
  // Set strides
  size_t num_sketches = end_sample_idx - start_sample_idx;
  const size_t num_bins = strides.sketchsize64 * strides.bbits;
  assert(num_bins == sketches[0].get_sketch(kmer_lengths[0]).size());
  strides.sample_stride = 1;
  strides.bin_stride = warpPad(num_sketches);
  strides.kmer_stride = strides.bin_stride * num_bins;

  // Stride by bins then restride by samples
  // This is 4x faster than striding by samples by looping over References
  // vector, presumably because many fewer dereferences are being used
  SketchStrides old_strides = strides;
  std::vector<uint64_t> flat_bins = flatten_by_bins(
      sketches, kmer_lengths, old_strides, start_sample_idx, end_sample_idx);
  std::vector<uint64_t> flat_ref(strides.kmer_stride * kmer_lengths.size());
#pragma omp parallel for simd collapse(2) schedule(static)                     \
    num_threads(cpu_threads)
  for (size_t kmer_idx = 0; kmer_idx < kmer_lengths.size(); kmer_idx++) {
    for (size_t bin_idx = 0; bin_idx < num_bins; bin_idx++) {
      auto flat_ref_it = flat_ref.begin() + kmer_idx * strides.kmer_stride +
                         bin_idx * strides.bin_stride;
      for (size_t sample_idx = 0; sample_idx < num_sketches; sample_idx++) {
        *flat_ref_it = flat_bins[sample_idx * old_strides.sample_stride +
                                 bin_idx * old_strides.bin_stride +
                                 kmer_idx * old_strides.kmer_stride];
        flat_ref_it++;
      }
    }
  }

  return flat_ref;
}

// Main function callable via API
// Checks inputs
// Flattens sketches
// Copies flattened sketches to device
// Runs kernel function across distance elements
// Copies and returns results
NumpyMatrix query_db_cuda(std::vector<Reference> &ref_sketches,
                          std::vector<Reference> &query_sketches,
                          const std::vector<size_t> &kmer_lengths,
                          RandomMC &random_match, const int device_id,
                          const unsigned int num_cpu_threads) {
  size_t mem_free, mem_total, shared_size;
  std::tie(mem_free, mem_total, shared_size) = initialise_device(device_id);
  std::cerr << "Calculating distances on GPU device " << device_id << std::endl;

  // Check sketches are compatible
  bool self = false;
  size_t bbits = ref_sketches[0].bbits();
  size_t sketchsize64 = ref_sketches[0].sketchsize64();
  checkSketchParamsMatch(ref_sketches, kmer_lengths, bbits, sketchsize64);

  // Set up sketch information and sizes
  SketchStrides ref_strides;
  ref_strides.bbits = bbits;
  ref_strides.sketchsize64 = sketchsize64;
  SketchStrides query_strides = ref_strides;

  long long dist_rows;
  long n_samples = 0;
  if (ref_sketches == query_sketches) {
    self = true;
    dist_rows = static_cast<long long>(0.5 * (ref_sketches.size()) *
                                       (ref_sketches.size() - 1));
    n_samples = ref_sketches.size();
  } else {
    // Also check query sketches are compatible
    checkSketchParamsMatch(query_sketches, kmer_lengths, bbits, sketchsize64);
    dist_rows = ref_sketches.size() * query_sketches.size();
    n_samples = ref_sketches.size() + query_sketches.size();
  }
  double est_size =
      (bbits * sketchsize64 * kmer_lengths.size() * n_samples *
           sizeof(uint64_t) + // Size of sketches
       kmer_lengths.size() * std::pow(random_match.n_clusters(), 2) *
           sizeof(float) + // Size of random matches
       n_samples * sizeof(uint16_t) +
       dist_rows * 2 * sizeof(float)); // Size of distance matrix
  std::cerr << "Estimated device memory required: " << std::fixed
            << std::setprecision(0) << est_size / (1048576) << "Mb"
            << std::endl;
  std::cerr << "Total device memory: " << std::fixed << std::setprecision(0)
            << mem_total / (1048576) << "Mb" << std::endl;
  std::cerr << "Free device memory: " << std::fixed << std::setprecision(0)
            << mem_free / (1048576) << "Mb" << std::endl;

  if (est_size > mem_free * (1 - mem_epsilon) && !self) {
    throw std::runtime_error(
        "Using greater than device memory is unsupported for query mode. "
        "Split your input into smaller chunks");
  }

  // Turn the random matches into an array (same for any ref, query or subsample
  // thereof)
  bool missing = random_match.check_present(ref_sketches, true);
  if (missing) {
    std::cerr
        << "Some members of the reference database were not found "
           "in its random match chances. Consider refreshing with addRandom"
        << std::endl;
  }
  FlatRandom flat_random =
      random_match.flattened_random(kmer_lengths, ref_sketches[0].seq_length());
  std::vector<uint16_t> ref_random_idx =
      random_match.lookup_array(ref_sketches);

  // Ready to run dists on device
  SketchSlice sketch_subsample;
  unsigned int chunks = 1;
  std::vector<float> dist_results(dist_rows * 2);
  NumpyMatrix coreSquare, accessorySquare;
  if (self) {
    // To prevent memory being exceeded, total distance matrix is split up into
    // chunks which do fit in memory. The most is needed in the 'corners' where
    // two separate lots of sketches are loaded, hence the factor of two below

    // These are iterated over in the same order as a square distance matrix.
    // The i = j chunks are 'self', i < j can be skipped
    // as they contain only lower triangle values, i > j work as query vs ref
    if (est_size > mem_free * (1 - mem_epsilon)) {
      chunks = floor((est_size * 2) / (mem_free * (1 - mem_epsilon))) + 1;
    }
    size_t calc_per_chunk = n_samples / chunks;
    unsigned int num_big_chunks = n_samples % chunks;

    // Only allocate these square matrices if they are needed
    if (chunks > 1) {
      coreSquare.resize(n_samples, n_samples);
      accessorySquare.resize(n_samples, n_samples);
    }
    unsigned int total_chunks = (chunks * (chunks + 1)) >> 1;
    unsigned int chunk_count = 0;

    sketch_subsample.query_offset = 0;
    for (unsigned int chunk_i = 0; chunk_i < chunks; chunk_i++) {
      sketch_subsample.query_size = calc_per_chunk;
      if (chunk_i < num_big_chunks) {
        sketch_subsample.query_size++;
      }

      sketch_subsample.ref_offset = sketch_subsample.query_offset;
      for (unsigned int chunk_j = chunk_i; chunk_j < chunks; chunk_j++) {
        if (total_chunks > 1) {
          std::cerr << "Running chunk " << ++chunk_count << " of "
                    << total_chunks << std::endl;
        }

        sketch_subsample.ref_size = calc_per_chunk;
        if (chunk_j < num_big_chunks) {
          sketch_subsample.ref_size++;
        }

        dist_results = dispatchDists(
            ref_sketches, ref_sketches, ref_strides, query_strides, flat_random,
            ref_random_idx, ref_random_idx, sketch_subsample, kmer_lengths,
            chunk_i == chunk_j, num_cpu_threads, shared_size);

        // Read intermediate dists out
        if (chunks > 1) {
          NumpyMatrix blockMat = Eigen::Map<
              Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::ColMajor>>(
              dist_results.data(), dist_results.size() / 2, 2);

          // Convert each long form column of Nx2 matrix into square distance
          // matrix Add this square matrix into the correct submatrix (block) of
          // the final square matrix
          longToSquareBlock(coreSquare, accessorySquare, sketch_subsample,
                            blockMat, chunk_i == chunk_j, num_cpu_threads);
        }
        sketch_subsample.ref_offset += sketch_subsample.ref_size;
      }
      sketch_subsample.query_offset += sketch_subsample.query_size;
    }
  } else {
    std::vector<uint16_t> query_random_idx =
        random_match.lookup_array(query_sketches);

    sketch_subsample.ref_size = ref_sketches.size();
    sketch_subsample.query_size = query_sketches.size();
    sketch_subsample.ref_offset = 0;
    sketch_subsample.query_offset = 0;

    dist_results = dispatchDists(
        ref_sketches, query_sketches, ref_strides, query_strides, flat_random,
        ref_random_idx, query_random_idx, sketch_subsample, kmer_lengths, false,
        num_cpu_threads, shared_size);
  }

  NumpyMatrix dists_ret_matrix;
  if (self && chunks > 1) {
    // Chunked computation yields square matrix, which needs to be converted
    // back to long form
    Eigen::VectorXf core_dists = square_to_long(coreSquare, num_cpu_threads);
    Eigen::VectorXf accessory_dists =
        square_to_long(accessorySquare, num_cpu_threads);

    dists_ret_matrix.resize(samples_to_rows(coreSquare.rows()), 2);
    dists_ret_matrix << core_dists, accessory_dists; // Join columns
  } else {
    dists_ret_matrix =
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::ColMajor>>(
            dist_results.data(), dist_results.size() / 2, 2);
  }

  return dists_ret_matrix;
}