cmake_minimum_required(VERSION 3.16)
project(pp_sketchlib)
set(CMAKE_CXX_STANDARD 14)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
  cmake_policy(SET CMP0104 OLD) # Can't get CUDA_ARCHITECTURES to work with NEW
endif()

# Set sketch version
file(GLOB VERSION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/sketch/*.hpp)
execute_process(COMMAND cat ${VERSION_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/sketch.cu
                COMMAND openssl sha1
                COMMAND cut -d " " -f 2
                OUTPUT_VARIABLE sketch_hash)
string(REGEX REPLACE "\n$" "" sketch_hash "${sketch_hash}")
file(WRITE src/version.h "\#define SKETCH_VERSION \"${sketch_hash}\"\n")

# Variable definitions
set(TARGET_NAME pp_sketchlib)
add_compile_definitions(PYTHON_EXT)

# gcc: Add openmp
# gcc: Add -O0 to remove optimizations when using debug
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
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

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
endif()

# Add libraries

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/highfive/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

execute_process(COMMAND pybind11-config --cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_DIR)
find_package(pybind11 2.6 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
#find_package(OpenMP) # This links system openmp if present - conda sorts out rpath but take care

# Define python library target
add_library("${TARGET_NAME}" MODULE)

# Check for CUDA and compiles GPU library
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found, compiling both GPU and CPU code")
    enable_language(CUDA)

    # PIC/relocatable-device-code needed as this is linked by gcc later
    # -Xptxas -dlcm=ca turns cache on, but not needed in recent nvcc versions
    # --cudart static: static linking of the CUDA libraries
    # -gencode arch=compute_35 etc compiles for each (minimum) device version listed (v3.5, v5.0, v7.5)
    set(CUDA_OPTS "-Xcompiler -fPIC -Xptxas -dlcm=ca --relocatable-device-code=true --expt-relaxed-constexpr")
    # Turn on link time optimisation if available
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 11.0 AND CMAKE_BUILD_TYPE MATCHES Release)
        string(APPEND CUDA_OPTS " -dlto -arch=sm_86")
    else()
        string(APPEND CUDA_OPTS " -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")
    endif()
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        string(APPEND CUDA_OPTS " -G")
    endif()

    set(CMAKE_CUDA_FLAGS "${CUDA_OPTS}")

    add_compile_definitions(GPU_AVAILABLE)
    add_library("${TARGET_NAME}_CUDA" OBJECT src/gpu/dist.cu
                                             src/gpu/sketch.cu
                                             src/gpu/device_memory.cu
                                             src/gpu/gpu_countmin.cu
                                             src/gpu/device_reads.cu)
    target_include_directories("${TARGET_NAME}_CUDA" PRIVATE "${EIGEN3_INCLUDE_DIR}" "${pybind11_INCLUDE_DIRS}")
    set_property(TARGET "${TARGET_NAME}_CUDA"
                 PROPERTY POSITION_INDEPENDENT_CODE ON
                 CUDA_SEPARABLE_COMPILATION ON
                 CUDA_RESOLVE_DEVICE_SYMBOLS ON   # try and ensure device link with nvcc
                 CUDA_VISIBILITY_PRESET "hidden"
		         CUDA_RUNTIME_LIBRARY Static)
	         #CUDA_ARCHITECTURES OFF) # set off as done explicitly above (due to dlto complexities)
    # CPU code/gcc compiled code needed by cuda lib
    target_sources("${TARGET_NAME}" PRIVATE src/gpu/gpu_api.cpp)
else()
    message(STATUS "CUDA not found, compiling CPU code only")
endif()

# Compile CPU library
target_sources("${TARGET_NAME}" PRIVATE src/sketchlib_bindings.cpp
                                    src/dist/dist.cpp
                                    src/dist/matrix_ops.cpp
                                    src/reference.cpp
                                    src/sketch/seqio.cpp
                                    src/sketch/countmin.cpp
                                    src/sketch/sketch.cpp
                                    src/database/database.cpp
                                    src/api.cpp
                                    src/dist/linear_regression.cpp
                                    src/random/rng.cpp
                                    src/random/random_match.cpp)
set_target_properties("${TARGET_NAME}" PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
if(UNIX AND (NOT APPLE OR NOT CMAKE_COMPILER_IS_GNUCC))
    set_target_properties("${TARGET_NAME}" PROPERTIES
                          INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Link libraries
if(CMAKE_CUDA_COMPILER)
    target_link_libraries("${TARGET_NAME}" PRIVATE "${TARGET_NAME}_CUDA")
    set_property(TARGET "${TARGET_NAME}" PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    #set_property(TARGET "${TARGET_NAME}" PROPERTY CUDA_ARCHITECTURES OFF)
endif()
target_link_libraries("${TARGET_NAME}" PRIVATE pybind11::module Eigen3::Eigen z ${HDF5_LIBRARIES} gomp openblas lapack gfortran m dl)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries("${TARGET_NAME}" PRIVATE OpenMP::OpenMP_CXX)
#endif()
