cmake_minimum_required(VERSION 3.4)
project(pp_sketchlib)
set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)

add_definitions(-D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1)
include_directories(${PROJECT_SOURCE_DIR}/dlib)
include_directories(${PROJECT_SOURCE_DIR}/rollinghashcpp)
include_directories(${PROJECT_SOURCE_DIR}/HighFive/include)
#set(HighFive_DIR "${CMAKE_SOURCE_DIR}/HighFive/CMake")
#find_package (HighFive REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 COMPONENTS C CXX HL NO_MODULE REQUIRED shared)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

pybind11_add_module(pp_sketchlib src/sketchlib_bindings.cpp
                                 src/dist.cpp
                                 src/reference.cpp
                                 src/seqio.cpp
                                 src/sketch.cpp
                                 src/database.cpp
                                 src/api.cpp
                                 src/linear_regression.cpp
                                 src/link_function.cpp)
target_link_libraries(pp_sketchlib PRIVATE Threads::Threads Eigen3::Eigen z bz2 hdf5_cpp hdf5 lapack openblas)
