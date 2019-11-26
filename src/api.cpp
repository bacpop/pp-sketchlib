/*
 * api.cpp
 * Main functions for running sketches
 *
 */

void create_db() 
{
    // Sketches a set of genomes

    // Create threaded queue for distance calculations
    std::vector<std::thread> work_threads(num_threads);
    const unsigned long int calc_per_thread = (unsigned long int)raw.shape()[0] / num_threads;
    const unsigned int num_big_threads = raw.shape()[0] % num_threads;

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
        
        // TODO: sketch + save (need mutex)
        work_threads.push_back(std::thread(&fitBlock,
                                           std::cref(raw),
                                           std::ref(dists),
                                           std::cref(klist),
                                           start,
                                           start + thread_jobs));
        start += thread_jobs + 1;
    }

    // Wait for threads to complete
    for (auto it = work_threads.begin(); it != work_threads.end(); it++)
    {
        it->join();
    }

    // TODO: calculate dists
}

void query_db() 
{
    // Calculates distances against another database
}