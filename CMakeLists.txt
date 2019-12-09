cmake_minimum_required(VERSION 3.4)
project(pp_sketchlib)
set(CMAKE_CXX_STANDARD 14)

# Variable definitions
set(TARGET_NAME pp_sketchlib)

# Add -O0 to remove optimizations when using gcc
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

# Add libraries
find_package(pybind11 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Submodules
include_directories(${PROJECT_SOURCE_DIR}/dlib)
add_definitions(-D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1)
include_directories(${PROJECT_SOURCE_DIR}/rollinghashcpp)
include_directories(${PROJECT_SOURCE_DIR}/HighFive/include)
include_directories(/Users/jlees/miniconda3/envs/pp_sketch/include) # local testing only
link_directories(/Users/jlees/miniconda3/envs/pp_sketch/lib) # local testing only
#find_package(HDF5 COMPONENTS C CXX HL NO_MODULE REQUIRED shared)
#set(HighFive_DIR "${CMAKE_SOURCE_DIR}/HighFive/CMake")
#find_package (HighFive REQUIRED)

pybind11_add_module("${TARGET_NAME}" src/sketchlib_bindings.cpp
                                 src/dist.cpp
                                 src/reference.cpp
                                 src/seqio.cpp
                                 src/sketch.cpp
                                 src/database.cpp
                                 src/api.cpp
                                 src/linear_regression.cpp
                                 src/link_function.cpp)

# possible fix to version/module/exception name clash
#set_target_properties("${TARGET_NAME}" PROPERTIES CXX_VISIBILITY_PRESET "default")

target_link_libraries("${TARGET_NAME}" PRIVATE Threads::Threads Eigen3::Eigen z bz2 hdf5_cpp hdf5 lapack openblas)
