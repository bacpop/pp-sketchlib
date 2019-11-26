/*
 * File: database.cpp
 *
 * Interface between sketches and HDF5 store
 *
 */

#include <iostream>

#include "database.hpp"

#include <highfive/H5Group.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

// Initialisation
Database::Database(const std::string& filename)
    :_filename(filename), 
    _h5_file(HighFive::File(filename.c_str(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate))
{
    HighFive::Group sketch_group = _h5_file.createGroup("sketches");
}

Database::Database(HighFive::File& _h5_file)
    :_h5_file(h5_file)
{
    _filename = _h5_file.getName();
}

void Database::add_sketch(const Reference& ref)
{
    std::string sketch_name = "/sketches/" + ref.name();
    HighFive::Group sketch_group = _h5_file.createGroup(sketch_name, true);

    const std::vector<int> kmer_lengths = ref.kmer_lengths();
    for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend(); kmer_it++)
    {
        auto sketch = ref.get_sketch(*kmer_it);
        
        std::string dataset_name = sketch_name + "/" + std::to_string(*kmer_it);
        HighFive::DataSet sketch_dataset = _h5_file.createDataSet<uint64_t>(dataset_name, HighFive::DataSpace::From(sketch));
        sketch_dataset.write(sketch);

        HighFive::Attribute kmer_size_a = sketch_dataset.createAttribute<int>("kmer-size", HighFive::DataSpace::From(*kmer_it));
        kmer_size_a.write(*kmer_it);
        HighFive::Attribute sketch_size_a = sketch_dataset.createAttribute<size_t>("sketchsize64", HighFive::DataSpace::From(ref.sketchsize64()));
        sketch_size_a.write(ref.sketchsize64());
        HighFive::Attribute bbits_a = sketch_dataset.createAttribute<size_t>("bbits", HighFive::DataSpace::From(ref.bbits()));
        bbits_a.write(ref.bbits());
        HighFive::Attribute seed_a = sketch_dataset.createAttribute<int>("seed", HighFive::DataSpace::From(ref.seed()));
        seed_a.write(ref.seed());
    }
}

HighFive::File open_h5(const std::string& filename)
{
    return(HighFive::File(filename.c_str(), HighFive::File::ReadWrite)); 
}