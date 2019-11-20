/*
 *
 * reference.hpp
 * Header file for reference.cpp
 *
 */

#include <vector>
#include <map>
#include <string>

class Reference
{
    public:
        Reference(const std::string& _name, 
                  const std::string& filename, 
                  const std::vector<size_t>& kmer_lengths); // read and run sketch
        
        void save();
        void load();

    private:
        // Info
        std::string name;
        size_t bbits;
        size_t sketchsize64;
        bool isstrandpreserved;

        // sketch - map keys are k-mer length
        std::map<int, std::vector<uint64_t>> usigs;
};
