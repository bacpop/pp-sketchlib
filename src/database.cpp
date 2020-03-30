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

const int deflate_level = 9;

// Initialisation
Database::Database(const std::string& filename)
    :_filename(filename), 
    _h5_file(HighFive::File(filename.c_str(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate))
{
    HighFive::Group sketch_group = _h5_file.createGroup("sketches");
}

Database::Database(HighFive::File& h5_file)
    :_h5_file(h5_file)
{
    _filename = _h5_file.getName();
}

/*
*
* NB: It is up to the user of this API to check that ref.name()
* does not already exist in the DB! This is not checked, and will
* be overwritten
*
*/
void Database::add_sketch(const Reference& ref)
{
    // Create group for sketches
    std::string sketch_name = "/sketches/" + ref.name();
    HighFive::Group sketch_group = _h5_file.createGroup(sketch_name, true);
    
    // Write group attributes
    HighFive::Attribute sketch_size_a = sketch_group.createAttribute<size_t>("sketchsize64", HighFive::DataSpace::From(ref.sketchsize64()));
    sketch_size_a.write(ref.sketchsize64());
    HighFive::Attribute bbits_a = sketch_group.createAttribute<size_t>("bbits", HighFive::DataSpace::From(ref.bbits()));
    bbits_a.write(ref.bbits());
    HighFive::Attribute length_a = sketch_group.createAttribute<size_t>("length", HighFive::DataSpace::From(ref.seq_length()));
    length_a.write(ref.seq_length())

    // Write k-mer length vector as another group attribute
    const std::vector<size_t> kmer_lengths = ref.kmer_lengths();
    HighFive::Attribute kmers_a = sketch_group.createAttribute<int>("kmers", HighFive::DataSpace::From(kmer_lengths));
    kmers_a.write(kmer_lengths); 

    
    /*
        // Chunking and compression doesn't help with small sketches    
        HighFive::DataSetCreateProps save_properties;
        save_properties.add(HighFive::Chunking(std::vector<hsize_t>{ref.sketchsize64()}));
        save_properties.add(HighFive::Shuffle());
        save_properties.add(HighFive::Deflate(deflate_level));
    */
    
    // Write a new dataset for each k-mer length within this group
    for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend(); kmer_it++)
    {
        std::string dataset_name = sketch_name + "/" + std::to_string(*kmer_it);
        
        auto sketch = ref.get_sketch(*kmer_it);
        //HighFive::DataSet sketch_dataset = _h5_file.createDataSet<uint64_t>(dataset_name, HighFive::DataSpace::From(sketch), save_properties);
        HighFive::DataSet sketch_dataset = _h5_file.createDataSet<uint64_t>(dataset_name, HighFive::DataSpace::From(sketch));
        sketch_dataset.write(sketch);
        
        HighFive::Attribute kmer_size_a = sketch_dataset.createAttribute<int>("kmer-size", HighFive::DataSpace::From(*kmer_it));
        kmer_size_a.write(*kmer_it);
    }
}

Reference Database::load_sketch(const std::string& name)
{
    // Read in attributes
    HighFive::Group sketch_group = _h5_file.getGroup("/sketches/" + name);
    std::vector<int> kmer_lengths;
    sketch_group.getAttribute("kmers").read(kmer_lengths);
    size_t sketchsize64;
    sketch_group.getAttribute("sketchsize64").read(sketchsize64);
    size_t bbits;
    sketch_group.getAttribute("bbits").read(bbits);
    size_t seq_size;
    sketch_group.getAttribute("length").read(seq_size)

    Reference new_ref(name, bbits, sketchsize64, seq_size);
    for (auto kmer_it = kmer_lengths.cbegin(); kmer_it != kmer_lengths.cend(); kmer_it++)
    {
        std::vector<uint64_t> usigs;
        sketch_group.getDataSet(std::to_string(*kmer_it)).read(usigs);
        new_ref.add_kmer_sketch(usigs, *kmer_it);
    }

    return(new_ref);
}

HighFive::File open_h5(const std::string& filename)
{
    return(HighFive::File(filename.c_str(), HighFive::File::ReadWrite)); 
}