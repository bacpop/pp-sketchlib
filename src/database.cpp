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
}

Database::~Database()
{
    delete sketch_group;
    delete dist_group;        
}

Database::create()
{
    h5_file = h5::create(_filename.c_str(),H5F_ACC_EXCL);
    sketch_group = new Group(h5_file.createGroup("/sketches"));
    dist_group = new Group(h5_file.createGroup("/distances"));
}

Database::open()
{
    h5_file = h5::open(_filename.c_str(),H5F_ACC_RDWR);
    sketch_group = new Group(h5_file.openGroup("/sketches"));
    dist_group = new Group(h5_file.openGroup("/distances"));
}

Database::add_sketch(const Reference& ref)
{
    try 
    {
        sketch_group = Group(sketch_group->openGroup(ref.name()));
    }
    catch( GroupIException not_found_error )
    {
        sketch_group = Group(sketch_group->createGroup(ref.name()));
    }

    kmer_lengths = ref.kmer_lengths();
    for (auto kmer_it = kmer_lengths.begin(); kmer_it != kmer_lengths.end(); kmer_it++)
    {
        h5::write(h5_file, "/sketches" + sketch_name + itos(*kmer_it), ref.get_sketch(*kmer_len));
    }
}