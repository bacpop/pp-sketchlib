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

/*
* 
* Here are the functions exported from api.hpp
*
* std::vector<Reference> create_sketches(const std::string& db_name,
*                   const std::vector<std::string>& names, 
*                    const std::vector<std::string>& files, 
*                    const std::vector<size_t>& kmer_lengths,
*                    const size_t sketchsize64,
*                    const size_t num_threads);
* 
* DistMatrix query_db(std::vector<Reference>& ref_sketches,
*                     std::vector<Reference>& query_sketches,
*                     const std::vector<size_t>& kmer_lengths,
*                     const size_t num_threads);
* std::vector<Reference> load_sketches(const std::string& db_name,
*                                     const std::vector<std::string>& names,
*                                     const std::vector<size_t>& kmer_lengths);
*
*
*/

// Calls function, but returns void
void constructDatabase(std::string db_name,
                       std::vector<std::string> sample_names,
                       std::vector<std::string> file_names,
                       std::vector<size_t> kmer_lengths,
                       size_t sketch_size,
                       size_t num_threads = 1)
{
    std::vector<Reference> ref_sketches = create_sketches(db_name,
                                                            sample_names, 
                                                            file_names, 
                                                            kmer_lengths,
                                                            sketch_size,
                                                            num_threads);
}

DistMatrix queryDatabase(std::string ref_db_name,
                         std::string query_db_name,
                         std::vector<std::string> ref_names,
                         std::vector<std::string> query_names,
                         std::vector<size_t> kmer_lengths,
                         size_t num_threads = 1)
{
    std::vector<Reference> ref_sketches = load_sketches(ref_db_name, ref_names, kmer_lengths);
    std::vector<Reference> query_sketches = load_sketches(query_db_name, query_names, kmer_lengths);
    return(query_db(ref_sketches,
                    query_sketches,
                    kmer_lengths,
                    num_threads));
}

DistMatrix constructAndQuery(std::string db_name,
                             std::vector<std::string> sample_names,
                             std::vector<std::string> file_names,
                             std::vector<size_t> kmer_lengths,
                             size_t sketch_size,
                             size_t num_threads = 1)
{
    std::vector<Reference> ref_sketches = create_sketches(db_name,
                                                            sample_names, 
                                                            file_names, 
                                                            kmer_lengths,
                                                            sketch_size,
                                                            num_threads);
    return(query_db(ref_sketches,
                    ref_sketches,
                    kmer_lengths,
                    num_threads));  
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
        py::arg("num_threads") = 1);
  
  m.def("queryDatabase", &queryDatabase, py::return_value_policy::reference_internal, "Find distances between sketches", 
        py::arg("ref_db_name"),
        py::arg("query_db_name"),
        py::arg("rList"),
        py::arg("qList"),
        py::arg("klist"),
        py::arg("num_threads") = 1);

  m.def("constructAndQuery", &constructAndQuery, py::return_value_policy::reference_internal, "Create and save sketches, and get pairwise distances", 
        py::arg("db_name"),
        py::arg("samples"),
        py::arg("files"),
        py::arg("klist"),
        py::arg("sketch_size"),
        py::arg("num_threads") = 1);

    // Exceptions
    py::register_exception<HighFive::Exception>(m, "HDF5Exception");
}