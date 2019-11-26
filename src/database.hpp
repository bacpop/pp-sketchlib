/*
 *
 * database.hpp
 * Header file for database.cpp
 *
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>

#include <H5Cpp.h>

#include "reference.hpp"

class Database 
{
    public:
        Database(const std::string& filename); // Open or create H5 file
        
        void add_sketch(const Reference& ref); // Write a new sketch to the HDF5
        Reference load_sketch(); // Retrieve a sketch

    private:
        std::string _filename;

        h5::hid_t h5_file;
        Group sketch_group(); 

};
