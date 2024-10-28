cmake_minimum_required(VERSION 3.15)
# Set the C++ compiler to the one from Conda before the project declaration
set(CMAKE_CXX_COMPILER "$ENV{CONDA_PREFIX}/bin/g++")
set(CMAKE_C_COMPILER "$ENV{CONDA_PREFIX}/bin/gcc")
project(simsearch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check for AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    add_definitions(-DAVX2_ENABLED)
else()
    message(WARNING "AVX2 is not supported on this machine.")
endif()

# Add conda environment paths
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{CONDA_PREFIX})
    include_directories(SYSTEM $ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    message(STATUS "Using Conda prefix: $ENV{CONDA_PREFIX}")
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python: ${Python3_VERSION}")

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11: ${pybind11_VERSION}")

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3: ${EIGEN3_VERSION_STRING}")

# Find Arrow
find_package(Arrow CONFIG REQUIRED)
message(STATUS "Found Arrow: ${ARROW_VERSION}")

# Find nlohmann_json
find_package(nlohmann_json REQUIRED)
message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")

# Configure OpenMP
option(USE_OPENMP "Use OpenMP" ON)
if(USE_OPENMP)
    if(APPLE)
        # Special handling for macOS if needed
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY ${CMAKE_C_COMPILER})
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY ${CMAKE_CXX_COMPILER})
        endif()
    endif()

    find_package(OpenMP REQUIRED)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
endif()

# Add pthread compile options
if(UNIX AND NOT APPLE)
    add_compile_options(-pthread)
endif()

# RPATH settings
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Add the include directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the include directory to the include path
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/embedding_search_bindings.cpp
    src/embedding_search_avx2.cpp
    src/embedding_search_float.cpp
    src/embedding_search_binary.cpp
    src/embedding_search_binary_avx2.cpp
    src/embedding_search_uint8_avx2.cpp
    src/embedding_io.cpp
)

# Create the Python module
pybind11_add_module(${PROJECT_NAME}_py MODULE ${SOURCES})

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/embedding_io.cpp
    src/embedding_search_avx2.cpp
    src/embedding_search_binary.cpp
    src/embedding_search_float.cpp
    src/embedding_search_binary_avx2.cpp
    src/embedding_search_uint8_avx2.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# Find Arrow/Parquet libraries in Conda environment
find_library(ARROW_LIB arrow PATHS $ENV{CONDA_PREFIX}/lib REQUIRED)
find_library(PARQUET_LIB parquet PATHS $ENV{CONDA_PREFIX}/lib REQUIRED)


# Link the library and any other dependencies
target_link_libraries(${PROJECT_NAME}
    PRIVATE 
        Eigen3::Eigen
        ${ARROW_LIB}
        ${PARQUET_LIB}
        nlohmann_json::nlohmann_json
        pthread
        dl
)

# Add OpenMP if enabled
if(USE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()


# Print configuration summary
message(STATUS "\nConfiguration summary:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  Python version: ${Python3_VERSION}")
message(STATUS "  Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "  Python libraries: ${Python3_LIBRARIES}")
message(STATUS "  Conda prefix: $ENV{CONDA_PREFIX}")
message(STATUS "  Installation prefix: ${CMAKE_INSTALL_PREFIX}")

# Optionally, you can set different flags for Debug and Release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Print include directories for debugging
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()