cmake_minimum_required(VERSION 3.15)

# Force using conda's compilers and libraries
set(ENV{CC} "$ENV{CONDA_PREFIX}/bin/gcc")
set(ENV{CXX} "$ENV{CONDA_PREFIX}/bin/g++")
set(CMAKE_C_COMPILER "$ENV{CONDA_PREFIX}/bin/gcc")
set(CMAKE_CXX_COMPILER "$ENV{CONDA_PREFIX}/bin/g++")

# Set linker flags to force using conda's runtime path
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,$ENV{CONDA_PREFIX}/lib")

project(simsearch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_DEBUG "Enable debug assertions and logging" ON)

if(ENABLE_DEBUG)
    add_definitions(-DDEBUG)
else()
    add_definitions(-DNDEBUG)
endif()

# Add conda environment paths first
if(DEFINED ENV{CONDA_PREFIX})
    # Ensure conda paths are added first
    include_directories(BEFORE SYSTEM $ENV{CONDA_PREFIX}/include)
    link_directories(BEFORE $ENV{CONDA_PREFIX}/lib)
    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
endif()

# Check for AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    add_definitions(-DAVX2_ENABLED)
endif()

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Arrow CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

# Configure OpenMP
option(USE_OPENMP "Use OpenMP" ON)
if(USE_OPENMP)
    # Explicitly set OpenMP flags and library
    set(OpenMP_gomp_LIBRARY "$ENV{CONDA_PREFIX}/lib/libgomp.so")
    set(OpenMP_pthread_LIBRARY "$ENV{CONDA_PREFIX}/lib/libpthread.so")
    
    # Set compilation flags
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    
    find_package(OpenMP REQUIRED)
    
    # Add the flags to the global compile flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add the include directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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

# Create the executable
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# Find Arrow/Parquet libraries
find_library(ARROW_LIB arrow PATHS $ENV{CONDA_PREFIX}/lib REQUIRED NO_DEFAULT_PATH)
find_library(PARQUET_LIB parquet PATHS $ENV{CONDA_PREFIX}/lib REQUIRED NO_DEFAULT_PATH)
#find_library(PTHREAD_LIB pthread PATHS $ENV{CONDA_PREFIX}/lib REQUIRED NO_DEFAULT_PATH)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE 
        Eigen3::Eigen
        ${ARROW_LIB}
        ${PARQUET_LIB}
        nlohmann_json::nlohmann_json
        ${OpenMP_gomp_LIBRARY}
        dl
)

# Print configuration summary
message(STATUS "\nConfiguration summary:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  OpenMP Library: ${OpenMP_gomp_LIBRARY}")
message(STATUS "  Pthread Library: ${PTHREAD_LIB}")
message(STATUS "  Conda prefix: $ENV{CONDA_PREFIX}")