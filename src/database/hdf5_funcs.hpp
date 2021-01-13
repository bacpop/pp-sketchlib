/*
 *
 * hdf5_funcs.hpp
 * Special functions for reading/writing to HDF5
 *
 */
#pragma once

#include "dist/matrix.hpp"

#include "robin_hood.h"
#include <highfive/H5File.hpp>

// HighFive does have support for reading/writing Eigen::Matrix
// (and std::vector<Eigen::Matrix>) which is turned on with a compile
// flag. Unfortunately the conda-forge version doesn't have this support,
// so matrices are dealt with manually (in a very similar way to the HighFive
// code)
// Save a hash into a HDF5 file by saving as array of keys and values
template <typename T, typename U>
void save_hash(const robin_hood::unordered_node_map<T, U> &hash,
               HighFive::Group &group,
               const std::string &dataset_name)
{
  std::vector<T> hash_keys;
  std::vector<U> hash_values;
  for (auto hash_it = hash.cbegin(); hash_it != hash.cend(); ++hash_it)
  {
    hash_keys.push_back(hash_it->first);
    hash_values.push_back(hash_it->second);
  }

  HighFive::DataSet key_dataset = group.createDataSet<T>(dataset_name + "_keys", HighFive::DataSpace::From(hash_keys));
  key_dataset.write(hash_keys);
  HighFive::DataSet value_dataset = group.createDataSet<U>(dataset_name + "_values", HighFive::DataSpace::From(hash_values));
  value_dataset.write(hash_values);
}

// Specialisation for saving Eigen matrix keys
template <>
void save_hash<size_t, NumpyMatrix>(const robin_hood::unordered_node_map<size_t, NumpyMatrix> &hash,
                                    HighFive::Group &group,
                                    const std::string &dataset_name)
{
  std::vector<size_t> hash_keys;
  std::vector<float> buffer;
  std::vector<size_t> dims = {(size_t)hash.cbegin()->second.rows(), (size_t)hash.cbegin()->second.cols()};
  for (auto hash_it = hash.cbegin(); hash_it != hash.cend(); ++hash_it)
  {
    hash_keys.push_back(hash_it->first);

    // Saving the vector of matrices is more annoying
    // Flatten out and save the dimensions
    if ((size_t)hash_it->second.rows() != dims[0] || (size_t)hash_it->second.cols() != dims[1])
    {
      throw std::runtime_error("Mismatching matrix sizes in save");
    }
    std::copy(hash_it->second.data(),
              hash_it->second.data() + hash_it->second.rows() * hash_it->second.cols(),
              std::back_inserter(buffer));
  }

  HighFive::DataSet key_dataset = group.createDataSet<size_t>(dataset_name + "_keys", HighFive::DataSpace::From(hash_keys));
  key_dataset.write(hash_keys);

  HighFive::DataSet dataset =
      group.createDataSet<float>(dataset_name + "_values", HighFive::DataSpace::From(buffer));
  dataset.write(buffer);
  dims.push_back(hash_keys.size());
  HighFive::Attribute dim_a = dataset.createAttribute<size_t>("dims", HighFive::DataSpace::From(dims));
  dim_a.write(dims);
}

// Load a hash from a HDF5 file by reading arrays of keys and values
// and re-inserting into a new hash
template <typename T, typename U>
robin_hood::unordered_node_map<T, U> load_hash(HighFive::Group &group,
                                               const std::string &dataset_name)
{
  std::vector<T> hash_keys;
  std::vector<U> hash_values;
  group.getDataSet(dataset_name + "_keys").read(hash_keys);
  group.getDataSet(dataset_name + "_values").read(hash_values);

  robin_hood::unordered_node_map<T, U> hash;
  for (size_t i = 0; i < hash_keys.size(); i++)
  {
    hash[hash_keys[i]] = hash_values[i];
  }
  return (hash);
}

// Specialisation for reading in Eigen matrices
template <>
robin_hood::unordered_node_map<size_t, NumpyMatrix> load_hash(
    HighFive::Group &group,
    const std::string &dataset_name)
{

  std::vector<size_t> hash_keys, dims;
  std::vector<float> buffer;
  group.getDataSet(dataset_name + "_keys").read(hash_keys);
  HighFive::DataSet values = group.getDataSet(dataset_name + "_values");
  values.read(buffer);
  values.getAttribute("dims").read(dims);

  robin_hood::unordered_node_map<size_t, NumpyMatrix> hash;
  float *buffer_pos = buffer.data();
  for (size_t i = 0; i < hash_keys.size(); i++)
  {
    NumpyMatrix mat = Eigen::Map<NumpyMatrix>(buffer_pos, dims[0], dims[1]);
    buffer_pos += mat.rows() * mat.cols();
    hash[hash_keys[i]] = mat;
  }
  return (hash);
}

// Save a single Eigen matrix to a HDF5 file
void save_eigen(const NumpyMatrix &mat,
                HighFive::Group &group,
                const std::string &dataset_name)
{
  std::vector<float> buffer;
  std::copy(mat.data(), mat.data() + mat.rows() * mat.cols(), std::back_inserter(buffer));
  std::vector<size_t> dims = {(size_t)mat.rows(), (size_t)mat.cols()};

  HighFive::DataSet dataset =
      group.createDataSet<float>(dataset_name, HighFive::DataSpace::From(buffer));
  dataset.write(buffer);
  HighFive::Attribute dim_a = dataset.createAttribute<size_t>("dims", HighFive::DataSpace::From(dims));
  dim_a.write(dims);
}

// Load a single Eigen matrix from an HDF5 file
NumpyMatrix load_eigen(HighFive::Group &group,
                       const std::string &dataset_name)
{
  std::vector<float> buffer;
  std::vector<size_t> dims;
  HighFive::DataSet dataset = group.getDataSet(dataset_name);
  dataset.read(buffer);
  dataset.getAttribute("dims").read(dims);
  NumpyMatrix mat = Eigen::Map<NumpyMatrix>(buffer.data(),
                                            dims[0],
                                            dims[1]);
  return mat;
}