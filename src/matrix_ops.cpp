/*
 *
 * matrix_ops.cpp
 * main functions for interacting with sketches
 *
 */
#pragma once

#include <vector>
#include <thread>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>

#include <Eigen/Dense>
using Eigen::MatrixXf;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SquareMatrix;

template<class T>
size_t rows_to_samples(const T& longMat) {
    return 0.5*(1 + sqrt(1 + 8*(mat.rows())));
}

long calc_row_idx(const long long k, const long n) {
	return n - 2 - floor(sqrt((double)(-8*k + 4*n*(n-1)-7))/2 - 0.5);
}

long calc_col_idx(const long long k, const long i, const long n) {
	return k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
}

long long square_to_condensed(long i, long j, long n) {
    assert(i > j);
	return (n*j - ((j*(j+1)) >> 1) + i - 1 - j);
}

std::tuple<size_t, unsigned int, unsigned int> jobs_per_thread(const size_t num_jobs, 
                                           const unsigned int num_threads) {
    size_t used_threads = num_threads;
    if (num_jobs < used_threads) {
        used_threads = num_jobs; 
    }
    size_t calc_per_thread = (size_t)num_jobs / used_threads;
    unsigned int num_big_threads = num_jobs % used_threads;

    return(std::make_tuple(calc_per_thread, used_threads, num_big_threads)); 
}

SquareMatrix long_to_square(const Eigen::VectorXf& rrDists, 
                            const Eigen::VectorXf& qrDists,
                            const Eigen::VectorXf& qqDists,
                            unsigned int num_threads) {
    // Set up square matrix to move values into
    size_t nrrSamples = rows_to_samples(rrDists);
    size_t nqqSamples = rows_to_samples(qqDists);
    SquareMatrix squareDists = SquareMatrix::Zero(nrrSamples + nqqSamples, nrrSamples + nqqSamples);

    size_t calc_per_thread; unsigned int used_threads; unsigned int num_big_threads; 
    std::tie(calc_per_thread, used_threads, num_big_threads) = jobs_per_thread(rrDists.rows(), num_threads); 
        
    // Loop over threads for ref v ref square
    size_t start = 0;
    std::vector<std::thread> used_threads;
    for (unsigned int thread_idx = 0; thread_idx < used_threads; ++thread_idx) {
        // First 'big' threads have an extra job
        unsigned long long int thread_jobs = calc_per_thread;
        if (thread_idx < num_big_threads) {
            thread_jobs++;
        }

        dist_threads.push_back(std::thread(&square_block,
                                        std::cref(rrDists),
                                        std::ref(squareDists),
                                        start,
                                        0,
                                        thread_jobs));
        start += thread_jobs; 
    }
    // Wait for threads to complete
    for (auto it = dist_threads.begin(); it != dist_threads.end(); it++) {
        it->join();
    }

    // Query v query block
    std::tie(calc_per_thread, used_threads, num_big_threads) = jobs_per_thread(qqDists.rows(), num_threads); 
    start = 0;
    dist_threads.clear();
    for (unsigned int thread_idx = 0; thread_idx < used_threads; ++thread_idx) {
        // First 'big' threads have an extra job
        unsigned long long int thread_jobs = calc_per_thread;
        if (thread_idx < num_big_threads) {
            thread_jobs++;
        }

        dist_threads.push_back(std::thread(&square_block,
                                        std::cref(qqDists),
                                        std::ref(squareDists),
                                        start,
                                        nrrSamples,
                                        thread_jobs));
        start += thread_jobs; 
    }
    // Wait for threads to complete
    for (auto it = dist_threads.begin(); it != dist_threads.end(); it++) {
        it->join();
    }

    // Query vs ref rectangles
    std::tie(calc_per_thread, used_threads, num_big_threads) = jobs_per_thread(qrDists.rows(), num_threads); 
    start = 0;
    dist_threads.clear();
    for (unsigned int thread_idx = 0; thread_idx < used_threads; ++thread_idx) {
        // First 'big' threads have an extra job
        unsigned long long int thread_jobs = calc_per_thread;
        if (thread_idx < num_big_threads) {
            thread_jobs++;
        }

        dist_threads.push_back(std::thread(&rectangle_block,
                                        std::cref(qrDists),
                                        std::ref(squareDists),
                                        start,
                                        thread_jobs));
        start += thread_jobs; 
    }
    // Wait for threads to complete
    for (auto it = dist_threads.begin(); it != dist_threads.end(); it++) {
        it->join();
    }

    return squareDists;
} 

void square_block(const Eigen::VectorXf& longDists,
                  Eigen::MatrixXf& squareMatrix,
                  const size_t n_samples,
                  const size_t start,
                  const size_t offset,
                  const size_t max_elems) {
    i = calc_row_idx(start, n_samples) + offset;
    j = calc_col_idx(start, i, n_samples) + offset;
    for (size_t distIdx = start; distIdx < start + max_elems; distIdx++) {
        squareMatrix[i, j] = longDists[distIdx];
        squareMatrix[j, i] = longDists[distIdx];
        if (++j == n_samples) {
            i++;
            j = i + 1;
        }
    }
}

// TODO
void rectangle_block(const Eigen::VectorXf& longDists,
                  Eigen::MatrixXf& squareMatrix,
                  const size_t n_samples,
                  const size_t start,
                  const size_t offset,
                  const size_t max_elems) {
    i = calc_row_idx(start, n_samples) + offset;
    j = calc_col_idx(start, i, n_samples) + offset;
    for (size_t distIdx = start; distIdx < start + max_elems; distIdx++) {
        squareMatrix[i, j] = longDists[distIdx];
        squareMatrix[j, i] = longDists[distIdx];
        if (++j == n_samples) {
            i++;
            j = i + 1;
        }
    }
}