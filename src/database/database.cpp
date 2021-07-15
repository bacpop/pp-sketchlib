/*
 * File: database.cpp
 *
 * Interface between sketches and HDF5 store
 *
 */

#include <iostream>
#include <utility>

#include "database.hpp"
#include "hdf5_funcs.hpp"
#include "random/random_match.hpp"

#include "robin_hood.h"

// const int deflate_level = 9;

// Helper function prototypes

// Initialisation
// Create new file

// Open an existing file
Database::Database(const std::string &h5_filename, const bool writable)
    : _h5_file(open_h5(h5_filename, writable)), _filename(_h5_file.getName()),
      _version_hash(SKETCH_VERSION), _codon_phased(false), _writable(writable)
{

  HighFive::Group sketch_group = _h5_file.getGroup("/sketches");
  std::vector<std::string> attributes_keys = sketch_group.listAttributeNames();
  for (const auto &attr : attributes_keys)
  {
    if (attr == "sketch_version")
    {
      sketch_group.getAttribute("sketch_version").read(_version_hash);
    }
    else if (attr == "codon_phased")
    {
      sketch_group.getAttribute("codon_phased").read(_codon_phased);
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
void Database::add_sketch(const Reference &ref)
{
  if (!_writable) {
    _h5_file = open_h5(_h5_file.getName(), true);
  }

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

Reference Database::load_sketch(const std::string &name)
{
  // Read in attributes
  HighFive::Group sketch_group = _h5_file.getGroup("/sketches/" + name);

  // Attributes in all sketches
  std::vector<int> kmer_lengths;
  size_t sketchsize64;
  size_t bbits;
  sketch_group.getAttribute("kmers").read(kmer_lengths);
  sketch_group.getAttribute("sketchsize64").read(sketchsize64);
  sketch_group.getAttribute("bbits").read(bbits);

  // Attributes added later (set defaults if not found)
  size_t seq_size = DEFAULT_LENGTH;
  std::vector<double> bases{0.25, 0.25, 0.25, 0.25};
  unsigned long int missing_bases = 0;
  std::vector<std::string> attributes_keys = sketch_group.listAttributeNames();
  for (const auto &attr : attributes_keys)
  {
    if (attr == "length")
    {
      sketch_group.getAttribute("length").read(seq_size);
    }
    else if (attr == "base_freq")
    {
      sketch_group.getAttribute("base_freq").read(bases);
    }
    else if (attr == "missing_bases")
    {
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

  return (new_ref);
}

// Save a RandomMC object to the database
void Database::save_random(const RandomMC &random)
{
  // Open or create the random group
  if (!_h5_file.exist("random"))
  {
    if (!_writable) {
      _h5_file = open_h5(_h5_file.getName(), true);
    }
    HighFive::Group random_group = _h5_file.createGroup("random");

    // Save the cluster table
    save_hash<std::string, uint16_t>(random.cluster_table(), random_group, "table");
    // Save the match matrices
    save_hash<size_t, NumpyMatrix>(random.matches(), random_group, "matches");

    // Save the cluster centroid matrix
    save_eigen(random.cluster_centroids(), random_group, "centroids");

    // Save attributes
    unsigned int k_min, k_max;
    std::tie(k_min, k_max) = random.k_range();
    HighFive::Attribute k_min_a = random_group.createAttribute<unsigned int>("k_min", HighFive::DataSpace::From(k_min));
    k_min_a.write(k_min);
    HighFive::Attribute k_max_a = random_group.createAttribute<unsigned int>("k_max", HighFive::DataSpace::From(k_max));
    k_max_a.write(k_max);
    HighFive::Attribute rc_a = random_group.createAttribute<bool>("use_rc", HighFive::DataSpace::From(random.use_rc()));
    rc_a.write(random.use_rc());
  }
  else
  {
    throw std::runtime_error("Random matches already exist in " + _filename);
  }
}

// Retrive a RandomMC object from the database
RandomMC Database::load_random(const bool use_rc_default)
{
  RandomMC random(use_rc_default); // Will use formula version if not in DB
  if (_h5_file.exist("random"))
  {
    HighFive::Group random_group = _h5_file.getGroup("/random");

    // Flattened hashes
    robin_hood::unordered_node_map<std::string, uint16_t> cluster_table =
        load_hash<std::string, uint16_t>(random_group, "table");
    robin_hood::unordered_node_map<size_t, NumpyMatrix> matches =
        load_hash<size_t, NumpyMatrix>(random_group, "matches");

    // Centroid matrix
    NumpyMatrix centroids = load_eigen(random_group, "centroids");

    // Read attributes
    unsigned int k_min, k_max;
    bool use_rc;
    random_group.getAttribute("k_min").read(k_min);
    random_group.getAttribute("k_max").read(k_max);
    random_group.getAttribute("use_rc").read(use_rc);

    // Constructor for reading database
    random = RandomMC(use_rc, k_min, k_max, cluster_table, matches, centroids);
  }
  else
  {
    std::cerr << "Could not find random match chances in database, "
                 "calculating assuming equal base frequencies"
              << std::endl;
  }
  return (random);
}

// Open an existing file
HighFive::File open_h5(const std::string &filename, const bool write)
{
  return (HighFive::File(filename.c_str(),
          write ? HighFive::File::ReadWrite : HighFive::File::ReadOnly));
}

// Create a new HDF5 file with headline attributes
Database new_db(const std::string &filename, const bool use_rc,
                const bool codon_phased)
{
  // Restrict scope of HDF5 so it is closed before reopening it
  {
    HighFive::File h5_file = HighFive::File(filename.c_str(),
                                            HighFive::File::Overwrite);
    HighFive::Group sketch_group = h5_file.createGroup("sketches");

    std::string version_hash = SKETCH_VERSION;
    HighFive::Attribute sketch_version_a =
        sketch_group.createAttribute<std::string>("sketch_version",
                                                  HighFive::DataSpace::From(version_hash));
    sketch_version_a.write(version_hash);
    HighFive::Attribute codon_phased_a =
        sketch_group.createAttribute<bool>("codon_phased",
                                          HighFive::DataSpace::From(codon_phased));
    codon_phased_a.write(codon_phased);
    HighFive::Attribute use_rc_a =
        sketch_group.createAttribute<bool>("reverse_complement",
                                          HighFive::DataSpace::From(use_rc));
    use_rc_a.write(use_rc);
  }
  return(Database(filename, true));
}
