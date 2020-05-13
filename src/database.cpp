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
// Create new file
Database::Database(const std::string& filename)
    :_filename(filename), 
    _h5_file(HighFive::File(filename.c_str(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate))
{
    HighFive::Group sketch_group = _h5_file.createGroup("sketches");
    
    _version_hash = SKETCH_VERSION; 
    HighFive::Attribute sketch_version_a = 
        sketch_group.createAttribute<std::string>("sketch_version", HighFive::DataSpace::From(_version_hash));
    sketch_version_a.write(_version_hash);
}

// Open an existing file
Database::Database(HighFive::File& h5_file)
    :_h5_file(h5_file)
{
    _filename = _h5_file.getName();
    
    _version_hash = DEFAULT_VERSION; 
    HighFive::Group sketch_group = _h5_file.getGroup("/sketches");
    std::vector<std::string> attributes_keys = sketch_group.listAttributeNames();
    for (const auto& attr : attributes_keys) {
        if (attr == "sketch_version") {
            sketch_group.getAttribute("sketch_version").read(_version_hash);
        }
    }
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
    length_a.write(ref.seq_length());
    HighFive::Attribute missing_a = sketch_group.createAttribute<unsigned long int>("missing_bases", HighFive::DataSpace::From(ref.missing_bases()));
    missing_a.write(ref.missing_bases());  

    // Write base composition and k-mer length vectors as further group attributes
    const std::vector<double> bases = ref.base_composition();
    HighFive::Attribute bases_a = sketch_group.createAttribute<double>("base_freq", HighFive::DataSpace::From(bases));
    bases_a.write(bases);
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

    // Attributes in all sketches
    std::vector<int> kmer_lengths; size_t sketchsize64; size_t bbits;
    sketch_group.getAttribute("kmers").read(kmer_lengths);
    sketch_group.getAttribute("sketchsize64").read(sketchsize64);
    sketch_group.getAttribute("bbits").read(bbits);

    // Attributes added later (set defaults if not found)
    size_t seq_size = DEFAULT_LENGTH;
    std::vector<double> bases{0.25, 0.25, 0.25, 0.25};
    unsigned long int missing_bases = 0;
    std::vector<std::string> attributes_keys = sketch_group.listAttributeNames();
    for (const auto& attr : attributes_keys) {
        if (attr == "length") {
            sketch_group.getAttribute("length").read(seq_size);
        } else if (attr == "base_freq") {
            sketch_group.getAttribute("base_freq").read(bases);
        } else if (attr == "missing_bases") {
            sketch_group.getAttribute("missing_bases").read(missing_bases);
        }
    }

    Reference new_ref(name, bbits, sketchsize64, seq_size, bases, missing_bases);
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
