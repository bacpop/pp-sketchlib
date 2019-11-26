/*
 * File: database.cpp
 *
 * Interface between sketches and HDF5 store
 *
 */

#include <iostream>

#include "database.hpp"

using namespace H5;

// Initialisation
Database::Database(const std::string& filename)
    :_filename(filename)
{
    try
    {
        h5_file = h5::open(filename.c_str(),H5F_ACC_RDWR);
        try
        {
            sketch_group = Group(h5_file.openGroup("/sketches"));
        }
        catch (GroupIException not_found_error)
        {
            std::cerr << "Database " + _filename + " does not contain expected groups" << std::endl;
            throw std::runtime_error("HDF5 database misformatted");
        }
    }
    catch(const std::exception& e)
    {
        h5_file = h5::create(_filename.c_str(),H5F_ACC_EXCL);
        sketch_group = Group(h5_file.createGroup("/sketches"));
    }
}


Database::add_sketch(const Reference& ref)
{
    
    /*
    try 
    {
        sketch_group = Group(sketch_group.openGroup(ref.name()));
    }
    catch(GroupIException not_found_error)
    {
        sketch_group = Group(sketch_group.createGroup(ref.name()));
    }
    */

    const std::vector<int> kmer_lengths = ref.kmer_lengths();
    for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend(); kmer_it++)
    {
        std::string sketch_name = "/sketches/" + ref.name() + "/" + std::to_string(*kmer_it)
        ds_t h5_sketch = h5::write(h5_file, sketch_name, ref.get_sketch(*kmer_it));
        h5_sketch["kmer-size"] = *kmer_it;
        h5_sketch["sketchsize64"] = ref.sketchsize64();
        h5_sketch["bbits"] = ref.bbits();
        h5_sketch["seed"] = ref.seed();
    }
}
