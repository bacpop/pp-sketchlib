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
    if(CMAKE_CXX_COMPILER STREQUAL "icpc")  
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast -xCASCADELAKE -DMKL_ILP64 -m64 -static-intel")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
        set(CMAKE_LD_FLAGS "${CMAKE_LDFLAGS} -Wl,--as-needed")
    endif()
endif()

# Add libraries
find_package(pybind11 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
endif()

# Check for CUDA and compiles GPU library
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found, compiling both GPU and CPU code")
    enable_language(CUDA)
    
    set(CUDA_OPTS "-Xcompiler -fPIC -Xptxas -dlcm=ca --cudart static --relocatable-device-code=true")
    string(APPEND CUDA_OPTS " -gencode arch=compute_35,code=compute_35 -gencode arch=compute_50,code=sm_50")
    string(APPEND CUDA_OPTS " -gencode arch=compute_75,code=sm_75")
    set(CMAKE_CUDA_FLAGS "${CUDA_OPTS}")
    
    add_compile_definitions(GPU_AVAILABLE)
    add_library("${TARGET_NAME}_CUDA" OBJECT src/dist.cu)
    # target_include_directories("${TARGET_NAME}_CUDA" PRIVATE "${EIGEN3_INCLUDE_DIR}")
    set_property(TARGET "${TARGET_NAME}_CUDA" 
                 PROPERTY POSITION_INDEPENDENT_CODE ON
                 CUDA_SEPARABLE_COMPILATION ON
                 CUDA_RESOLVE_DEVICE_SYMBOLS ON   # try and ensure device link with nvcc
                 CUDA_VISIBILITY_PRESET "hidden")
else()
    message(STATUS "CUDA not found, compiling CPU code only")
endif()

# Compile CPU library
add_library("${TARGET_NAME}" MODULE src/sketchlib_bindings.cpp
                                    src/dist.cpp
                                    src/matrix_ops.cpp
                                    src/reference.cpp
                                    src/seqio.cpp
                                    src/countmin.cpp
                                    src/sketch.cpp
                                    src/database.cpp
                                    src/api.cpp
                                    src/linear_regression.cpp)
set_target_properties("${TARGET_NAME}" PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Link libraries
if(CMAKE_CUDA_COMPILER)
    target_link_libraries("${TARGET_NAME}" PRIVATE "${TARGET_NAME}_CUDA")
endif()
target_link_libraries("${TARGET_NAME}" PRIVATE pybind11::module Eigen3::Eigen 
                                               z hdf5_cpp hdf5 openblas lapack gfortran m dl)
