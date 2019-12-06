/*
 * api.cpp
 * Main functions for running sketches
 *
 */

#include <thread>
#include <algorithm>
#include <queue>
#include <sys/stat.h>

#include <H5Cpp.h>

#include "api.hpp"
#include "reference.hpp"
#include "database.hpp"

using namespace Eigen;

inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// Internal function definitions
void self_dist_block(DistMatrix& distMat,
                     const dlib::matrix<double,0,2>& kmer_lengths,
                     upperTriIterator refQueryIt,
                     size_t pos,
                     const size_t calcs);

void query_dist_row(DistMatrix& distMat,
                    const Reference * ref_sketch_ptr,
                    const std::vector<Reference>& query_sketches,
                    const dlib::matrix<double,0,2>& kmer_lengths,
                    const size_t row_start);

void sketch_block(std::vector<Reference>& sketches,
                                    const std::vector<std::string>& names, 
                                    const std::vector<std::string>& files, 
                                    const std::vector<size_t>& kmer_lengths,
                                    const size_t sketchsize64,
                                    const size_t start,
                                    const size_t end);

std::vector<Reference> load_sketches(const std::string& db_name,
                                     const std::vector<std::string>& names,
                                     const std::vector<size_t>& kmer_lengths);


/*
 * Routines for iteration over upper triangle
 *
 */ 
upperTriIterator::upperTriIterator(const std::vector<Reference>& sketches)
    :_query_forwards(true),
     _end_it(sketches.cend()),
     _ref_it(sketches.cbegin()),
     _query_it(sketches.cbegin() + 1)
{
}

upperTriIterator::upperTriIterator(const std::vector<Reference> & sketches,
                                   const std::vector<Reference>::const_iterator& ref_start,
                                   const std::vector<Reference>::const_iterator& query_start,
                                   const bool query_forwards)
    :_query_forwards(query_forwards),
     _end_it(sketches.cend()),
     _ref_it(ref_start),
     _query_it(query_start)
{
}

// Iterate upper triangle, alternately forwards and backwards along rows
void upperTriIterator::advance()
{
    if (_query_forwards)
    {
        _query_it++;
        if (_query_it == _end_it)
        {
            _query_it--;
            _ref_it++;
            _query_forwards = false;
        }
    }
    else
    {
        _query_it--;
        if (_query_it == _ref_it)
        {
            _ref_it++;
            _query_it = _ref_it + 1;
            _query_forwards = true;
        }
    }
}

/*
 * Main functions
 * 1) Create new sketches (tries calling 3) first)
 * 2) Calculate distances from sketches
 * 3) Load skteches from a database
 */

// Create sketches, save to file
std::vector<Reference> create_sketches(const std::string& db_name,
                   const std::vector<std::string>& names, 
                   const std::vector<std::string>& files, 
                   const std::vector<size_t>& kmer_lengths,
                   const size_t sketchsize64,
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
                                            start,
                                            start + thread_jobs));
            start += thread_jobs;
        }
        // Wait for threads to complete
        for (auto it = sketch_threads.begin(); it != sketch_threads.end(); it++)
        {
            it->join();
        }

        // Save sketches
        std::cerr << "Writing sketches to file" << std::endl;
        Database sketch_db(db_name + ".h5");
        for (auto sketch_it = sketches.begin(); sketch_it != sketches.end(); sketch_it++)
        {
            sketch_db.add_sketch(*sketch_it);
        }
    }

    return sketches;
}

// Calculates distances against another database
// Input is vectors of sketches
DistMatrix query_db(std::vector<Reference>& ref_sketches,
                    std::vector<Reference>& query_sketches,
                    const std::vector<size_t>& kmer_lengths,
                    const size_t num_threads) 
{
    if (ref_sketches.size() < 1 or query_sketches.size() < 1)
    {
        throw std::runtime_error("Query with empty ref or query list!");
    }
    
    std::cerr << "Calculating distances using " << num_threads << " thread(s)" << std::endl;
    DistMatrix distMat;
    dlib::matrix<double,0,2> kmer_mat = add_intercept(vec_to_dlib(kmer_lengths));
    
    // Check if ref = query, then run as self mode
    // Note: this only checks names. Need to ensure k-mer lengths matching elsewhere 
    std::sort(ref_sketches.begin(), ref_sketches.end());
    std::sort(query_sketches.begin(), query_sketches.end());

    if (ref_sketches == query_sketches)
    {
        // calculate dists
        size_t dist_rows = static_cast<int>(0.5*(ref_sketches.size())*(ref_sketches.size() - 1));
        distMat.resize(dist_rows, 2);
        
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
        upperTriIterator refQueryIt(ref_sketches);
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
                                            std::cref(kmer_mat),
                                            refQueryIt,
                                            start,
                                            thread_jobs));
            
            // Move to start point for next thread
            for (size_t i = 0; i < thread_jobs; i++)
            {
                start++;
                refQueryIt.advance();
            }
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
        distMat.resize(dist_rows, 2);
        
        size_t num_dist_threads = num_threads;
        if (dist_rows < num_threads)
        {
            num_dist_threads = dist_rows; 
        }
        
        // Loop over threads, one per ref, with FIFO queue
        std::queue<std::thread> dist_threads;
        size_t row_start = 0;
        for (auto ref_it = ref_sketches.cbegin(); ref_it < ref_sketches.cend(); ref_it++)
        {
            // If all threads being used, wait for one to finish
            if (dist_threads.size() == num_dist_threads)
            {
                dist_threads.front().join();
                dist_threads.pop();            
            }

            dist_threads.push(std::thread(&query_dist_row,
                              std::ref(distMat),
                              &(*ref_it),
                              std::cref(query_sketches),
                              std::cref(kmer_mat),
                              row_start));
            row_start += query_sketches.size();
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

// Load sketches from a HDF5 file
// Returns empty vector on failure
std::vector<Reference> load_sketches(const std::string& db_name,
                                     const std::vector<std::string>& names,
                                     const std::vector<size_t>& kmer_lengths)
{
    // Vector of set size to store results
    std::vector<Reference> sketches(names.size());
    
    /* Turn off HDF5 error messages */
    H5E_auto2_t errorPrinter;
    void** clientData = nullptr;
    H5::Exception::getAutoPrint(errorPrinter, clientData);
    H5::Exception::dontPrint();

    try
    {
        HighFive::File h5_db(db_name + ".h5");
        Database prev_db(h5_db);
        
        std::cerr << "Looking for existing sketches in " + db_name + ".h5" << std::endl;
        size_t i = 0;
        for (auto name_it = names.cbegin(); name_it != names.end(); name_it++)
        {
            sketches[i] = prev_db.load_sketch(*name_it);
            if (sketches[i].kmer_lengths() != kmer_lengths)
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
        
        /* Restore previous error handler */
        H5::Exception::setAutoPrint(errorPrinter, clientData);
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
                  const std::vector<std::string>& files, 
                  const std::vector<size_t>& kmer_lengths,
                  const size_t sketchsize64,
                  const size_t start,
                  const size_t end)
{
    for (unsigned int i = start; i < end; i++)
    {
        sketches[i] = Reference(names[i], files[i], kmer_lengths, sketchsize64);
    }
}

// Calculates dists self v self
// (run this function in a thread)
void self_dist_block(DistMatrix& distMat,
                     const dlib::matrix<double,0,2>& kmer_lengths,
                     upperTriIterator refQueryIt,
                     size_t pos,
                     const size_t calcs)
{
    // Iterate upper triangle, alternately forwards and backwards along rows
    size_t done_calcs = 0;
    while (done_calcs < calcs)
    {
        std::tie(distMat(pos, 0), distMat(pos, 1)) = refQueryIt.getRefIt()->core_acc_dist(*(refQueryIt.getQueryIt()), kmer_lengths);
        refQueryIt.advance();
        done_calcs++;
        pos++;
    }
}

// Calculates dists ref v query
// (run this function in a thread) 
void query_dist_row(DistMatrix& distMat,
                    const Reference * ref_sketch_ptr,
                    const std::vector<Reference>& query_sketches,
                    const dlib::matrix<double,0,2>& kmer_lengths,
                    const size_t row_start)
{
    size_t current_row = row_start;
    for (auto query_it = query_sketches.cbegin(); query_it != query_sketches.cend(); query_it++)
    {
        std::tie(distMat(current_row, 0), distMat(current_row, 1)) = 
                ref_sketch_ptr->core_acc_dist(*query_it, kmer_lengths);
        current_row++;
    }
}
