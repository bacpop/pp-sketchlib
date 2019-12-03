/*
 * api.cpp
 * Main functions for running sketches
 *
 */

#include <thread>
#include <sys/stat.h>

#include <H5Cpp.h>

#include "api.hpp"
#include "reference.hpp"
#include "database.hpp"

void self_dist_block(DistMatrix& distMat,
                     const std::vector<Reference>& sketches,
                     const dlib::matrix<double,0,2>& kmer_lengths,
                     const size_t start,
                     const size_t end);

void sketch_block(std::vector<Reference>& sketches,
                                    const std::vector<std::string>& names, 
                                    const std::vector<std::string>& files, 
                                    const std::vector<size_t>& kmer_lengths,
                                    const size_t sketchsize64,
                                    const size_t start,
                                    const size_t end);

inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// Create sketches, save to file
std::vector<Reference> create_sketches(const std::string& db_name,
                   const std::vector<std::string>& names, 
                   const std::vector<std::string>& files, 
                   const std::vector<size_t>& kmer_lengths,
                   const size_t sketchsize64,
                   const size_t num_threads)
{
    // Store sketches in vector
    std::vector<Reference> sketches();

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
        sketch.resize(names.size());
        
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
DistMatrix query_db(const std::vector<Reference>& ref_sketches,
                  const std::vector<Reference>& query_sketches,
                  const std::vector<size_t>& kmer_lengths,
                  const size_t num_threads) 
{
    std::cerr << "Calculating distances using " << num_threads << " thread(s)" << std::endl;
    DistMatrix distMat;
    dlib::matrix<double,0,2> kmer_mat = add_intercept(vec_to_dlib(kmer_lengths));
    
    // TODO
    // Check if ref = query, then run as self mode (sort first, need to add function to reference.hpp)
    // If ref != query, make a thread queue, with each element one ref (see kmds.cpp in seer)

    if (self)
    {
        // calculate dists
        size_t dist_rows = static_cast<int>(0.5*(names.size())*(names.size() - 1));
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
                                            std::cref(ref_sketches),
                                            std::cref(kmer_mat),
                                            start,
                                            start + thread_jobs));
            start += thread_jobs + 1;
        }
        // Wait for threads to complete
        for (auto it = dist_threads.begin(); it != dist_threads.end(); it++)
        {
            it->join();
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
    void** clientData;
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
        sketches.clear()
        
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
    catch ()
    {
        std::cerr << "Error in reading previous database" << std::endl;
        sketches.clear()
        throw std::runtime_error("Database read error");
    }

    return(sketches);
}

// Creates sketches (run in thread)
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

// Calculates dists (run in thread)
void self_dist_block(DistMatrix& distMat,
                     const std::vector<Reference>& sketches,
                     const dlib::matrix<double,0,2>& kmer_lengths,
                     const size_t start,
                     const size_t end)
{
    // Iterate upper triangle, alternately forwards and backwards along rows
    size_t calcs = 0;
    auto ref_sketch = sketches.cbegin();
    auto query_sketch = sketches.cbegin() + 1;
    size_t pos = 0;
    bool row_forward = true;
    while (calcs < (end - start))
    {
        if (pos >= start)
        {
            std::tie(distMat(pos, 0), distMat(pos, 1)) = ref_sketch->core_acc_dist(*query_sketch, kmer_lengths);
            calcs++;
        }
        
        // Move to next element
        if (row_forward)
        {
            query_sketch++;
            if (query_sketch == sketches.end())
            {
                row_forward = false;
                ref_sketch++;
                query_sketch--;
            }
        }
        else
        {
            query_sketch--;
            if (query_sketch == ref_sketch)
            {
                row_forward = true;
                ref_sketch++;
                query_sketch = ref_sketch + 1;
            }
        }
        pos++;
    }
}

