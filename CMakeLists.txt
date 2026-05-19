cmake_minimum_required(VERSION 3.15)
#include(build/Release/generators/conan_toolchain.cmake)
project(simsearch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "include-what-you-use")

add_compile_options(-Wno-ignored-attributes)

include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)
FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(Eigen3)

find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
)
FetchContent_MakeAvailable(pybind11)
# find_package(Arrow REQUIRED)
# find_package(Parquet REQUIRED)

if (NOT DEFINED PYARROW_DIR)
    message(FATAL_ERROR "PYARROW_DIR must be defined")
endif()

include_directories(${PYARROW_DIR}/include)
link_directories(${PYARROW_DIR})

find_library(ARROW_LIB NAMES arrow libarrow.so libarrow.so.2400 libarrow.so.2200 PATHS ${PYARROW_DIR} NO_DEFAULT_PATH)
find_library(PARQUET_LIB NAMES parquet libparquet.so libparquet.so.2400 libparquet.so.2200 PATHS ${PYARROW_DIR} NO_DEFAULT_PATH)

if (NOT ARROW_LIB)
    message(FATAL_ERROR "Arrow library not found in ${PYARROW_DIR}")
endif()
if (NOT PARQUET_LIB)
    message(FATAL_ERROR "Parquet library not found in ${PYARROW_DIR}")
endif()

add_library(arrow SHARED IMPORTED)
set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})

add_library(parquet SHARED IMPORTED)
set_target_properties(parquet PROPERTIES IMPORTED_LOCATION ${PARQUET_LIB})

# Add the include directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# main executable (main.cpp)
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/embedding_io.cpp
    src/embedding_search_avx2.cpp
    src/embedding_search_binary.cpp
    src/embedding_search_float.cpp
    src/embedding_search_binary_avx2.cpp
    src/embedding_search_uint8_avx2.cpp
    src/embedding_utils.cpp
    src/optimized_embedding_search_uint8_avx2.cpp
    src/optimized_embedding_search_avx2.cpp
    src/optimized_embedding_search_binary_avx2.cpp
    src/embedding_search_float16.cpp
    src/embedding_search_mapped_float.cpp
)

# Create the Python module
pybind11_add_module(embedding_search_benchmark
    src/embedding_search_benchmark_bindings.cpp
    src/embedding_io.cpp
    src/embedding_search_avx2.cpp
    src/embedding_search_binary.cpp
    src/embedding_search_float.cpp
    src/embedding_search_binary_avx2.cpp
    src/embedding_search_uint8_avx2.cpp
    src/embedding_utils.cpp
    src/optimized_embedding_search_uint8_avx2.cpp
    src/optimized_embedding_search_avx2.cpp
    src/optimized_embedding_search_binary_avx2.cpp
    src/embedding_search_float16.cpp
    src/embedding_search_mapped_float.cpp
)

# Link libraries for main executable
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        arrow
        parquet
)

# Link libraries for Python module
target_link_libraries(embedding_search_benchmark
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        arrow
        parquet
)

# Set Python module properties
set_target_properties(embedding_search_benchmark PROPERTIES
    PREFIX ""  # Removes lib prefix on Unix
    SUFFIX ".so"  # Sets correct extension on Unix
)

# set different flags for Debug and Release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -march=native -DNO_THREADS")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -fopenmp -mavx2 -DNO_THREADS")
