/*
 * api.cpp
 * Main functions for running sketches
 *
 */

#include <thread>

#include "api.hpp"
#include "reference.hpp"
#include "database.hpp"

void self_dist_block(MatrixXd& distMat,
                     const std::vector<Reference>& sketches,
                     const column_vector& kmer_lengths,
                     const size_t start,
                     const size_t end);

void sketch_block(std::vector<Reference>& sketches,
                                    const std::vector<std::string>& names, 
                                    const std::vector<std::string>& files, 
                                    const std::vector<size_t>& kmer_lengths,
                                    const size_t sketchsize64,
                                    const size_t start,
                                    const size_t end);

MatrixXd create_db(const std::string& db_name,
                   const std::vector<std::string>& names, 
                   const std::vector<std::string>& files, 
                   const std::vector<size_t>& kmer_lengths,
                   const size_t sketchsize64,
                   const size_t num_threads) 
{
    // Sketches a set of genomes
    std::cerr << "Sketching " << names.size() << " genomes using " << num_threads << " threads" << std::endl;
    // std::vector<size_t> kmer_lengths {13, 17};

    // Create threaded queue for distance calculations
    std::vector<Reference> sketches(names.size());
    //std::vector<Reference> sketches;
    std::vector<std::thread> sketch_threads(num_threads);
    const unsigned long int calc_per_thread = (unsigned long int)sketches.size() / num_threads;
    const unsigned int num_big_threads = sketches.size() % num_threads;

    // Spawn worker threads
    size_t start = 0;
    for (unsigned int thread_idx = 0; thread_idx < num_threads; ++thread_idx) // Loop over threads
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
        start += thread_jobs + 1;
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

    // calculate dists
    std::cerr << "Calculating distances using " << num_threads << " threads" << std::endl;
    column_vector kmer_vec = vec_to_dlib(kmer_lengths);
    size_t dist_rows = static_cast<int>(0.5*(names.size())*(names.size() - 1));
    MatrixXd distMat(dist_rows, 2);
    
    std::vector<std::thread> dist_threads(num_threads);
    start = 0;
    
    for (unsigned int thread_idx = 0; thread_idx < num_threads; ++thread_idx) // Loop over threads
    {
        // First 'big' threads have an extra job
        unsigned long int thread_jobs = calc_per_thread;
        if (thread_idx < num_big_threads)
        {
            thread_jobs++;
        }
        
        dist_threads.push_back(std::thread(&self_dist_block,
                                           std::ref(distMat),
                                           std::cref(sketches),
                                           std::cref(kmer_vec),
                                           start,
                                           start + thread_jobs));
        start += thread_jobs + 1;
    }
    // Wait for threads to complete
    for (auto it = dist_threads.begin(); it != dist_threads.end(); it++)
    {
        it->join();
    }

    return(distMat);
}

void query_db() 
{
    // Calculates distances against another database
    // Input is arbitrary lists of sample names
    // Check if ref = query, then run as self mode
    // If ref != query, make a thread queue, with each element one ref (see kmds.cpp in seer)
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
        sketches[start + i] = Reference(names[i], files[i], kmer_lengths, sketchsize64);
    }
}

// Calculates dists (run in thread)
void self_dist_block(MatrixXd& distMat,
                     const std::vector<Reference>& sketches,
                     const column_vector& kmer_lengths,
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

