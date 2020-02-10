cmake_minimum_required(VERSION 3.12)
project(pp_sketchlib)
set(CMAKE_CXX_STANDARD 14)

# Variable definitions
set(TARGET_NAME pp_sketchlib)
add_compile_definitions(PYTHON_EXT)

# Add -O0 to remove optimizations when using gcc
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -m64")
endif()

# Add libraries
find_package(pybind11 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Submodules
include_directories(${PROJECT_SOURCE_DIR}/dlib)
add_definitions(-D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1)
include_directories(${PROJECT_SOURCE_DIR}/HighFive/include)

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
endif()

# HDF5/HighFive testing -- use conda installs in lib/ instead
#find_package(HDF5 COMPONENTS C CXX HL NO_MODULE REQUIRED shared)
#set(HighFive_DIR "${CMAKE_SOURCE_DIR}/HighFive/CMake")
#find_package (HighFive REQUIRED)

pybind11_add_module("${TARGET_NAME}" src/sketchlib_bindings.cpp
                                 src/dist.cpp
                                 src/reference.cpp
                                 src/seqio.cpp
                                 src/countmin.cpp
                                 src/sketch.cpp
                                 src/database.cpp
                                 src/api.cpp
                                 src/linear_regression.cpp)

# Not needed - possible fix to version/module/exception name clash
# set_target_properties("${TARGET_NAME}" PROPERTIES CXX_VISIBILITY_PRESET "default")

if(UNIX AND NOT APPLE)
    target_link_libraries("${TARGET_NAME}" PRIVATE Eigen3::Eigen z bz2 hdf5_cpp hdf5 mkl_rt Threads::Threads m dl)
else()
    target_link_libraries("${TARGET_NAME}" PRIVATE Eigen3::Eigen z bz2 hdf5_cpp hdf5 openblas Threads::Threads m dl)
endif()