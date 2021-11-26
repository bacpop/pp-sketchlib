#pragma once

// This file can be used by gcc and nvcc (i.e. no Eigen)
// Compound types used by many functions in api, bindings etc

#include <vector>
#include <tuple>

using sparse_coo = std::tuple<std::vector<long>, std::vector<long>, std::vector<float>>;
using network_coo = std::tuple<std::vector<long>, std::vector<long>, std::vector<long>>;