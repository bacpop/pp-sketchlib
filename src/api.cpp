/*
 * api.cpp
 * Main functions for running sketches
 *
 */

#include <thread>
#include <algorithm>
#include <queue>
#include <limits>
#include <sys/stat.h>

#include <H5Cpp.h>

#include "api.hpp"
#include "gpu.hpp"
#include "reference.hpp"
#include "database.hpp"

using namespace Eigen;

inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// Internal function definitions
void self_dist_block(DistMatrix& distMat,
                     const std::vector<size_t>& kmer_lengths,
                     std::vector<Reference>& sketches,
                     const bool jaccard,
                     const size_t start_pos,
                     const size_t calcs);

void query_dist_row(DistMatrix& distMat,
                    Reference * ref_sketch_ptr,
                    std::vector<Reference>& query_sketches,
                    const std::vector<size_t>& kmer_lengths,
                    const bool jaccard,
                    const size_t row_start);

void sketch_block(std::vector<Reference>& sketches,
                                    const std::vector<std::string>& names, 
                                    const std::vector<std::vector<std::string>>& files, 
                                    const std::vector<size_t>& kmer_lengths,
                                    const size_t sketchsize64,
                                    const bool use_rc,
                                    const uint8_t min_count,
                                    const bool exact,
                                    const size_t start,
                                    const size_t end);

/*
 * Main functions
 * 1) Create new sketches (tries calling 3) first)
 * 2) Calculate distances from sketches
 * 3) Load skteches from a database
 */

// Create sketches, save to file
std::vector<Reference> create_sketches(const std::string& db_name,
                   const std::vector<std::string>& names, 
                   const std::vector<std::vector<std::string>>& files, 
                   const std::vector<size_t>& kmer_lengths,
                   const size_t sketchsize64,
                   const bool use_rc,
                   size_t min_count,
                   const bool exact,
                   const size_t num_threads)
{
    // Store sketches in vector
    std::vector<Reference> sketches;

    // Try loading sketches from file
    bool resketch = true;
    if (file_exists(db_name + ".h5"))
    {
        sketches = load_sketches(db_name, names, kmer_lengths);
        if (sketches.size() == names.size())
        {
            resketch = false;
        }
    }

    // If not found or not matching, sketch from scratch
    if (resketch)
    {
        sketches.resize(names.size());

        // Truncate min_count if above 8 bit range
        if (min_count > std::numeric_limits<uint8_t>::max())
        {
            min_count = std::numeric_limits<uint8_t>::max(); 
        }
        
        // Create threaded queue for distance calculations
        size_t num_sketch_threads = num_threads;
        if (sketches.size() < num_threads)
        {
            num_sketch_threads = sketches.size(); 
        } 
        unsigned long int calc_per_thread = (unsigned long int)sketches.size() / num_sketch_threads;
        unsigned int num_big_threads = sketches.size() % num_sketch_threads;
        std::vector<std::thread> sketch_threads;

        // Spawn worker threads
        size_t start = 0;
        std::cerr << "Sketching " << names.size() << " genomes using " << num_sketch_threads << " thread(s)" << std::endl;
        for (unsigned int thread_idx = 0; thread_idx < num_sketch_threads; ++thread_idx) // Loop over threads
        {
            // First 'big' threads have an extra job
            unsigned long int thread_jobs = calc_per_thread;
            if (thread_idx < num_big_threads)
            {
                thread_jobs++;
            }
            sketch_threads.push_back(std::thread(&sketch_block,
                                            std::ref(sketches),
                                            std::cref(names),
                                            std::cref(files),
                                            std::cref(kmer_lengths),
                                            sketchsize64,
                                            use_rc,
                                            min_count,
                                            exact,
                                            start,
                                            start + thread_jobs));
            start += thread_jobs;
        }
        // Wait for threads to complete
        for (auto it = sketch_threads.begin(); it != sketch_threads.end(); it++)
        {
            it->join();
        }

        // Save sketches and check for densified sketches
        std::cerr << "Writing sketches to file" << std::endl;
        Database sketch_db(db_name + ".h5");
        for (auto sketch_it = sketches.begin(); sketch_it != sketches.end(); sketch_it++)
        {
            sketch_db.add_sketch(*sketch_it);
            if (sketch_it->densified()) {
                std::cerr << "NOTE: " << sketch_it->name() << " required densification" << std::endl; 
            }
        }
    }

    return sketches;
}

// Calculates distances against another database
// Input is vectors of sketches
DistMatrix query_db(std::vector<Reference>& ref_sketches,
                    std::vector<Reference>& query_sketches,
                    const std::vector<size_t>& kmer_lengths,
                    const bool jaccard,
                    const size_t num_threads) 
{
    if (ref_sketches.size() < 1 or query_sketches.size() < 1)
    {
        throw std::runtime_error("Query with empty ref or query list!");
    }
    
    std::cerr << "Calculating distances using " << num_threads << " thread(s)" << std::endl;
    DistMatrix distMat;
    size_t dist_cols;
    if (jaccard) {
        dist_cols = kmer_lengths.size();
    } else {
        dist_cols = 2;
    }
    
    // Check if ref = query, then run as self mode
    // Note: this only checks names. Need to ensure k-mer lengths matching elsewhere 
    std::sort(ref_sketches.begin(), ref_sketches.end());
    std::sort(query_sketches.begin(), query_sketches.end());

    if (ref_sketches == query_sketches)
    {
        // calculate dists
        size_t dist_rows = static_cast<int>(0.5*(ref_sketches.size())*(ref_sketches.size() - 1));
        distMat.resize(dist_rows, dist_cols);
        
        size_t num_dist_threads = num_threads;
        if (dist_rows < num_threads)
        {
            num_dist_threads = dist_rows; 
        }
        unsigned long int calc_per_thread = (unsigned long int)dist_rows / num_dist_threads;
        unsigned int num_big_threads = dist_rows % num_dist_threads; 
        
        // Loop over threads
        std::vector<std::thread> dist_threads;
        size_t start = 0;
        for (unsigned int thread_idx = 0; thread_idx < num_dist_threads; ++thread_idx)
        {
            // First 'big' threads have an extra job
            unsigned long int thread_jobs = calc_per_thread;
            if (thread_idx < num_big_threads)
            {
                thread_jobs++;
            }

            dist_threads.push_back(std::thread(&self_dist_block,
                                            std::ref(distMat),
                                            std::cref(kmer_lengths),
                                            std::ref(ref_sketches),
                                            jaccard,
                                            start,
                                            thread_jobs));
            start += thread_jobs; 
        }
        // Wait for threads to complete
        for (auto it = dist_threads.begin(); it != dist_threads.end(); it++)
        {
            it->join();
        }
    }
    // If ref != query, make a thread queue, with each element one ref (see kmds.cpp in seer)
    else
    {
        // calculate dists
        size_t dist_rows = ref_sketches.size() * query_sketches.size();
        distMat.resize(dist_rows, dist_cols);
        
        size_t num_dist_threads = num_threads;
        if (dist_rows < num_threads)
        {
            num_dist_threads = dist_rows; 
        }
        
        // Loop over threads, one per query, with FIFO queue
        std::queue<std::thread> dist_threads;
        size_t row_start = 0;
        for (auto query_it = query_sketches.begin(); query_it < query_sketches.end(); query_it++)
        {
            // If all threads being used, wait for one to finish
            if (dist_threads.size() == num_dist_threads)
            {
                dist_threads.front().join();
                dist_threads.pop();            
            }

            dist_threads.push(std::thread(&query_dist_row,
                              std::ref(distMat),
                              &(*query_it),
                              std::ref(ref_sketches),
                              std::cref(kmer_lengths),
                              jaccard,
                              row_start));
            row_start += ref_sketches.size();
        }
        // Wait for threads to complete
        while(!dist_threads.empty())
        {
            dist_threads.front().join();
            dist_threads.pop();
        } 
    }
    
    return(distMat);
}

#ifdef GPU_AVAILABLE
DistMatrix query_db_gpu(std::vector<Reference>& ref_sketches,
	std::vector<Reference>& query_sketches,
	const std::vector<size_t>& kmer_lengths,
    const int device_id)
{
    // Calculate dists on GPU, which is returned as a flattened array
    // CUDA code now returns column major data (i.e. all core dists, then all accessory dists)
    // to try and coalesce writes.
    // NB: almost all other code is row major (i.e. sample core then accessory, then next sample)
    std::vector<float> dist_vec = query_db_cuda(ref_sketches, query_sketches, 
                                                kmer_lengths, device_id);
    
    // Map this memory into an eigen matrix
    DistMatrix dists_ret = \
		Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,2,Eigen::ColMajor> >(dist_vec.data(),dist_vec.size()/2,2);

    return dists_ret;
}
#endif

// Load sketches from a HDF5 file
// Returns empty vector on failure
std::vector<Reference> load_sketches(const std::string& db_name,
                                     const std::vector<std::string>& names,
                                     std::vector<size_t> kmer_lengths,
                                     const bool messages)
{
    // Vector of set size to store results
    std::vector<Reference> sketches(names.size());
    std::sort(kmer_lengths.begin(), kmer_lengths.end());

    /* Turn off HDF5 error messages */
    /* getAutoPrint throws and unknown exception when called from python, but is ok from C++ */
#ifndef PYTHON_EXT
    H5E_auto2_t errorPrinter;
    void** clientData = nullptr;
    H5::Exception::getAutoPrint(errorPrinter, clientData);
#endif
    H5::Exception::dontPrint();

    try
    {
        HighFive::File h5_db(db_name + ".h5");
        Database prev_db(h5_db);
        
        if (messages)
        {
            std::cerr << "Looking for existing sketches in " + db_name + ".h5" << std::endl;
        }
        size_t i = 0;
        for (auto name_it = names.cbegin(); name_it != names.end(); name_it++)
        {
            sketches[i] = prev_db.load_sketch(*name_it);

            // Remove unwanted k-mer lengths from sketch dict
            auto loaded_sizes = sketches[i].kmer_lengths();
            std::sort(loaded_sizes.begin(), loaded_sizes.end());
            auto kmer_it = kmer_lengths.begin(); auto loaded_it = loaded_sizes.begin();
            while (kmer_it != kmer_lengths.end() && loaded_it != loaded_sizes.end())
            {
                if (*kmer_it == *loaded_it)
                {
                    kmer_it++;
                }
                else
                {
                    sketches[i].remove_kmer_sketch(*loaded_it);
                }
                loaded_it++;
            }
            // throw if any of the requested k-mer lengths were not found
            if (kmer_it != kmer_lengths.end())
            {
                throw std::runtime_error("k-mer lengths in old database do not match those requested");
            }

            i++;
        }
    }
    catch (const HighFive::Exception& e)
    {
        // Triggered if sketch not found
        std::cerr << "Missing sketch: " << e.what() << std::endl;
        sketches.clear();
    }
    catch (const std::exception& e)
    {
        // Triggered if k-mer lengths mismatch
        std::cerr << "Mismatched data: " << e.what() << std::endl;
        sketches.clear();
    }
    // Other errors (likely not safe to continue)
    catch (...)
    {
        std::cerr << "Error in reading previous database" << std::endl;
        sketches.clear();
        throw std::runtime_error("Database read error");
    }

#ifndef PYTHON_EXT
    /* Restore previous error handler */
    H5::Exception::setAutoPrint(errorPrinter, clientData);
#endif
    return(sketches);
}

/* 
 * Internal functions used by main exported functions above
 * Loading from file
 * Simple in thread function definitions
 */

// Creates sketches 
// (run this function in a thread)
void sketch_block(std::vector<Reference>& sketches,
                  const std::vector<std::string>& names, 
                  const std::vector<std::vector<std::string>>& files, 
                  const std::vector<size_t>& kmer_lengths,
                  const size_t sketchsize64,
                  const bool use_rc,
                  const uint8_t min_count,
                  const bool exact,
                  const size_t start,
                  const size_t end)
{
    for (unsigned int i = start; i < end; i++)
    {
        sketches[i] = Reference(names[i], files[i], kmer_lengths, sketchsize64, use_rc, min_count, exact);
    }
}

// Calculates dists self v self
// (run this function in a thread)
void self_dist_block(DistMatrix& distMat,
                     const std::vector<size_t>& kmer_lengths,
                     std::vector<Reference>& sketches,
                     const bool jaccard,
                     const size_t start_pos,
                     const size_t calcs)
{
    arma::mat kmer_mat = kmer2mat(kmer_lengths);
    // Iterate upper triangle
    size_t done_calcs = 0;
    size_t pos = 0;
    for (size_t i = 0; i < sketches.size(); i++)
    {
        for (size_t j = i + 1; j < sketches.size(); j++)
        {
            if (pos >= start_pos)
            {
                if (jaccard) {
                    for (unsigned int kmer_idx = 0; kmer_idx < kmer_lengths.size(); kmer_idx++) {
                        distMat(pos, kmer_idx) = sketches[i].jaccard_dist(sketches[j], kmer_lengths[kmer_idx]);
                    }
                } else {
                    std::tie(distMat(pos, 0), distMat(pos, 1)) = sketches[i].core_acc_dist(sketches[j], kmer_mat);
                }
                done_calcs++;
                if (done_calcs >= calcs)
                {
                    break;
                }
            }
            pos++;
        }
        if (done_calcs >= calcs)
        {
            break;
        }
    }
}

// Calculates dists ref v query
// (run this function in a thread) 
void query_dist_row(DistMatrix& distMat,
                    Reference * query_sketch_ptr,
                    std::vector<Reference>& ref_sketches,
                    const std::vector<size_t>& kmer_lengths,
                    const bool jaccard,
                    const size_t row_start)
{
    arma::mat kmer_mat = kmer2mat(kmer_lengths);
    size_t current_row = row_start;
    for (auto ref_it = ref_sketches.begin(); ref_it != ref_sketches.end(); ref_it++)
    {
        if (jaccard) {
            for (unsigned int kmer_idx = 0; kmer_idx < kmer_lengths.size(); kmer_idx++) {
                distMat(current_row, kmer_idx) = query_sketch_ptr->jaccard_dist(*ref_it, kmer_lengths[kmer_idx]);
            }
        } else {
            std::tie(distMat(current_row, 0), distMat(current_row, 1)) = 
                     query_sketch_ptr->core_acc_dist(*ref_it, kmer_mat);
        }
        current_row++;
    }
}
