#include <vector>
#include <climits>

#define XOSHIRO_WIDTH 4

class Xoshiro {
    public:
        // Definitions to be used as URNG in C++11
        typedef size_t result_type;
        static size_t min() { return 0; }
        static size_t max() { return ULLONG_MAX; }
        uint64_t operator()(); // generate random number U(min, max)

        // Constructor
        Xoshiro(uint64_t seed);
        
        // Change internal state
        void set_seed(uint64_t seed);
        void jump();
        void long_jump();

    private:
        uint64_t _state[XOSHIRO_WIDTH];
};


