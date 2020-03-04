/*
 * sketchlib_bindings.cpp
 * Python bindings for pp-sketchlib
 *
 */

// pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include <highfive/H5Exception.hpp>

#include "api.hpp"

/*
 * Gives the same functions as mash.py 
 * Create reference - check for existing DB (move existing code), creates sketches if none, runs query_db
 * Query db - creates query sketches, loads ref sketches, runs query_db
 */

// Calls function, but returns void
void constructDatabase(std::string db_name,
                       std::vector<std::string> sample_names,
                       std::vector<std::vector<std::string>> file_names,
                       std::vector<size_t> kmer_lengths,
                       size_t sketch_size,
                       size_t min_count = 0,
                       size_t num_threads = 1)
{
    std::vector<Reference> ref_sketches = create_sketches(db_name,
                                                            sample_names, 
                                                            file_names, 
                                                            kmer_lengths,
                                                            sketch_size,
                                                            min_count,
                                                            num_threads);
}

DistMatrix queryDatabase(std::string ref_db_name,
                         std::string query_db_name,
                         std::vector<std::string> ref_names,
                         std::vector<std::string> query_names,
                         std::vector<size_t> kmer_lengths,
                         size_t num_threads = 1,
                         bool use_gpu = false,
                         size_t blockSize = 128,
                         int device_id = 0)
{
    std::vector<Reference> ref_sketches = load_sketches(ref_db_name, ref_names, kmer_lengths, false);
    std::vector<Reference> query_sketches = load_sketches(query_db_name, query_names, kmer_lengths, false);

    DistMatrix dists; 
#ifdef GPU_AVAILABLE
    if (use_gpu)
    {
        dists = query_db_gpu(ref_sketches,
	                        query_sketches,
                            kmer_lengths,
                            blockSize,
                            0,
                            device_id);
    }
    else
    {
        dists = query_db(ref_sketches,
                query_sketches,
                kmer_lengths,
                num_threads);
    }
#else
    dists = query_db(ref_sketches,
                    query_sketches,
                    kmer_lengths,
                    num_threads));
#endif
    return(dists);
}

DistMatrix constructAndQuery(std::string db_name,
                             std::vector<std::string> sample_names,
                             std::vector<std::vector<std::string>> file_names,
                             std::vector<size_t> kmer_lengths,
                             size_t sketch_size,
                             size_t min_count = 0,
                             size_t num_threads = 1,
                             bool use_gpu = false,
                             size_t blockSize = 128,
                             int device_id = 0)
{
    std::vector<Reference> ref_sketches = create_sketches(db_name,
                                                            sample_names, 
                                                            file_names, 
                                                            kmer_lengths,
                                                            sketch_size,
                                                            min_count,
                                                            num_threads);
DistMatrix dists; 
#ifdef GPU_AVAILABLE
    if (use_gpu)
    {
        dists = query_db_gpu(ref_sketches,
                             ref_sketches,
                             kmer_lengths,
                             blockSize,
                             0,
                             device_id);
    }
    else
    {
        dists = query_db(ref_sketches,
                ref_sketches,
                kmer_lengths,
                num_threads);
    }
#else
    dists = query_db(ref_sketches,
                     ref_sketches,
                     kmer_lengths,
                     num_threads));
#endif
    return(dists);
}

double jaccardDist(std::string db_name,
                   std::string sample1,
                   std::string sample2,
                   size_t kmer_size)
{
    auto sketch_vec = load_sketches(db_name, {sample1, sample2}, {kmer_size}, false);
    return(sketch_vec.at(0).jaccard_dist(sketch_vec.at(1), kmer_size));
}

PYBIND11_MODULE(pp_sketchlib, m)
{
  m.doc() = "Sketch implementation for PopPUNK";

  // Exported functions
  m.def("constructDatabase", &constructDatabase, "Create and save sketches", 
        py::arg("db_name"),
        py::arg("samples"),
        py::arg("files"),
        py::arg("klist"),
        py::arg("sketch_size"),
        py::arg("min_count") = 0,
        py::arg("num_threads") = 1);
  
  m.def("queryDatabase", &queryDatabase, py::return_value_policy::reference_internal, "Find distances between sketches", 
        py::arg("ref_db_name"),
        py::arg("query_db_name"),
        py::arg("rList"),
        py::arg("qList"),
        py::arg("klist"),
        py::arg("num_threads") = 1,
        py::arg("use_gpu") = false,
        py::arg("blockSize") = 128,
        py::arg("device_id") = 0);

  m.def("constructAndQuery", &constructAndQuery, py::return_value_policy::reference_internal, "Create and save sketches, and get pairwise distances", 
        py::arg("db_name"),
        py::arg("samples"),
        py::arg("files"),
        py::arg("klist"),
        py::arg("sketch_size"),
        py::arg("min_count") = 0,
        py::arg("num_threads") = 1,
        py::arg("use_gpu") = false,
        py::arg("blockSize") = 128,
        py::arg("device_id") = 0);

  m.def("jaccardDist", &jaccardDist, "Calculate a raw Jaccard distance",
        py::arg("db_name"),
        py::arg("ref_name"),
        py::arg("query_name"),
        py::arg("kmer_length"));

    // Exceptions
    py::register_exception<HighFive::Exception>(m, "HDF5Exception");
}